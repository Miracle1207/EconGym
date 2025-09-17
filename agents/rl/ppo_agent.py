import copy
import torch
import numpy as np
from agents.rl.models import mlp_net
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.utils.rnn as rnn_utils


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)


class ppo_agent:
    def __init__(self, envs, args, type=None, agent_name="households"):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        self.agent_type = type

        env_agent_name = "households" if agent_name == "households" else agent_name

        if env_agent_name == "government":
            self.government_agent = self.envs.government[type]
            self.obs_dim = self.government_agent.observation_space.shape[0]
            self.action_dim = self.government_agent.action_space.shape[0]

        else:
            self.agent = getattr(self.envs, env_agent_name)
            self.obs_dim = self.agent.observation_space.shape[0]
            self.action_dim = self.agent.action_space.shape[-1]

        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.net = mlp_net(state_dim=self.obs_dim, num_actions=self.action_dim).to(self.device)

        # Initialize the MLP network
        self.net = mlp_net(state_dim=self.obs_dim, num_actions=self.action_dim).to(self.device)

        # Flag to choose whether to load an existing policy or not
        self.load_exist_policy = False  # Set to True to load the trained policy

        # Check if the policy should be loaded
        if self.load_exist_policy:
            if agent_name == "households":
                # Load policy for household agent based on its type
                if "OLG" in self.envs.households.type:
                    # Load PPO policy for OLG type households
                    self.net.load_state_dict(
                        torch.load("agents/models/trained_policy/ppo_OLG/ppo_net.pt", weights_only=True))
                elif "ramsey" in self.envs.households.type:
                    # Load PPO policy for Ramsey type households
                    self.net = mlp_net(state_dim=self.obs_dim, num_actions=3).to(self.device)  # Action dim = 3
                    self.net.load_state_dict(
                        torch.load("agents/models/trained_policy/ppo_Ramsey/ppo_net.pt", weights_only=True))
    
            elif agent_name == "government":
                # Load policy for government agent if type is 'tax'
                if self.envs.government.type == "tax":
                    self.net.load_state_dict(
                        torch.load("agents/models/bc_ppo/100/gdp/run8/ppo_net.pt", weights_only=True))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.p_lr, eps=1e-5)
        lambda_function = lambda epoch: 0.97 ** (epoch // 10)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_function)
        self.on_policy = True
        self.state_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.step_counter = 0
        self.max_warmup_steps = 1000
        self.normalizer_applied = False

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_output = torch.tensor(np.array(advantage_list), dtype=torch.float)
        # batch adv norm
        norm_advantage = (advantage_output - torch.mean(advantage_output)) / torch.std(advantage_output)
        return norm_advantage

    def train(self, transition_dict):
        sum_loss = torch.tensor([0., 0.], dtype=torch.float32).to(self.device)

        agent_data = transition_dict
        obs_tensor = torch.tensor(np.array(agent_data['obs_dict']), dtype=torch.float32).to(self.device)
        next_obs_tensor = torch.tensor(np.array(agent_data['next_obs_dict']), dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(np.array(agent_data['action_dict']), dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(np.array(agent_data['reward_dict']), dtype=torch.float32).to(self.device)
        inverse_dones = torch.tensor([x - 1 for x in agent_data['done']], dtype=torch.float32).to(self.device).unsqueeze(-1)
        # # Extract data from new nested dictionary structure
        # obs_dict = transition_dict['obs_dict']
        # next_obs_dict = transition_dict['next_obs_dict']
        # action_dict = transition_dict['action_dict']
        # reward_dict = transition_dict['reward_dict']
        # done_list = transition_dict['done']
        #
        # inverse_dones = torch.tensor([x - 1 for x in np.array(done_list)], dtype=torch.float32).to(
        #     self.device).unsqueeze(-1)

        # # Extract government data based on agent type
        # if self.agent_name == "government":
        #     # Get government observations and actions based on government type
        #     if self.agent_type == "pension":
        #         gov_obs_list = [obs['government']['pension'] for obs in obs_dict]
        #         next_gov_obs_list = [obs['government']['pension'] for obs in next_obs_dict]
        #         gov_action_list = [action['government']['pension'] for action in action_dict]
        #         gov_reward_list = [reward['government']['pension'] for reward in reward_dict]
        #     elif self.agent_type == "tax":
        #         gov_obs_list = [obs['government']['tax'] for obs in obs_dict]
        #         next_gov_obs_list = [obs['government']['tax'] for obs in next_obs_dict]
        #         gov_action_list = [action['government']['tax'] for action in action_dict]
        #         gov_reward_list = [reward['government']['tax'] for reward in reward_dict]
        #     elif self.agent_type == "central_bank":
        #         gov_obs_list = [obs['government']['central_bank'] for obs in obs_dict]
        #         next_gov_obs_list = [obs['government']['central_bank'] for obs in next_obs_dict]
        #         gov_action_list = [action['government']['central_bank'] for action in action_dict]
        #         gov_reward_list = [reward['government']['central_bank'] for reward in reward_dict]
        #
        #     gov_obses = torch.tensor(np.array(gov_obs_list), dtype=torch.float32).to(self.device)
        #     next_gov_obses = torch.tensor(np.array(next_gov_obs_list), dtype=torch.float32).to(self.device)
        #     gov_actions = torch.tensor(np.array(gov_action_list), dtype=torch.float32).to(self.device)
        #     gov_rewards = torch.tensor(np.array(gov_reward_list), dtype=torch.float32).to(self.device)
        #
        #     obs_tensor = gov_obses
        #     # action_tensor = self.inverse_action_wrapper(gov_actions)
        #     action_tensor = gov_actions
        #     reward_tensor = gov_rewards
        #     next_obs_tensor = next_gov_obses
        #
        # elif self.agent_name == "households":
        #     # Extract household data from new structure
        #     house_obs_list = [obs['households'] for obs in obs_dict]
        #     next_house_obs_list = [obs['households'] for obs in next_obs_dict]
        #     house_action_list = [action['households'] for action in action_dict]
        #     house_reward_list = [reward['households'] for reward in reward_dict]
        #
        #     # Convert household inputs to padded tensors
        #     house_obses = [torch.tensor(obs, dtype=torch.float32) for obs in house_obs_list]
        #     house_actions = [torch.tensor(act, dtype=torch.float32) for act in house_action_list]
        #     house_rewards = [torch.tensor(rwd, dtype=torch.float32) for rwd in house_reward_list]
        #     next_house_obses = [torch.tensor(obs, dtype=torch.float32) for obs in next_house_obs_list]
        #
        #     obs_tensor = rnn_utils.pad_sequence(house_obses, batch_first=True).to(self.device)
        #     action_tensor = rnn_utils.pad_sequence(house_actions, batch_first=True).to(self.device)
        #     reward_tensor = rnn_utils.pad_sequence(house_rewards, batch_first=True).to(self.device)
        #     next_obs_tensor = rnn_utils.pad_sequence(next_house_obses, batch_first=True).to(self.device)
        #
        #     # Adjust inverse_dones to shape (batch, n_households, 1)
        #     households_n = obs_tensor.size(1)
        #     inverse_dones = inverse_dones.unsqueeze(-1).repeat(1, households_n, 1)
        #
        # elif self.agent_name == "market":
        #     # Prepare market tensors (shape typically: [T, firm_n, feat])
        #     if self.agent_type == "perfect":
        #         return None, None
        #
        #     market_obs_list = [obs.get('market', []) for obs in obs_dict]
        #     # # If market observations are missing, skip training safely
        #     # if len(market_obs_list) == 0 or (isinstance(market_obs_list[0], list) and len(market_obs_list[0]) == 0):
        #     #     return None, None
        #
        #     next_market_obs_list = [obs.get('market', []) for obs in next_obs_dict]
        #     market_action_list = [action.get('market', []) for action in action_dict]
        #     market_reward_list = [reward.get('market', []) for reward in reward_dict]
        #
        #     obs_tensor = torch.tensor(np.array(market_obs_list), dtype=torch.float32).to(self.device)
        #     next_obs_tensor = torch.tensor(np.array(next_market_obs_list), dtype=torch.float32).to(self.device)
        #     action_tensor = torch.tensor(np.array(market_action_list), dtype=torch.float32).to(self.device)
        #     reward_tensor = torch.tensor(np.array(market_reward_list), dtype=torch.float32).to(self.device)
        #     # Ensure reward has a trailing singleton dim for value loss broadcasting
        #     if reward_tensor.dim() == 2:
        #         reward_tensor = reward_tensor.unsqueeze(-1)
        #
        #     # Match inverse_dones across firm dimension if present
        #     if obs_tensor.dim() >= 3:
        #         firms_n = obs_tensor.size(1)
        #         inverse_dones = inverse_dones.unsqueeze(-1).repeat(1, firms_n, 1)
        # elif self.agent_name == "bank":
        #     # Non-profit bank has no trainable actions/rewards
        #     if self.agent_type == 'non_profit':
        #         return None, None
        #     # Commercial bank branch
        #     bank_obs_list = [obs.get('bank', []) for obs in obs_dict]
        #     # If no observations are provided for bank, skip training gracefully
        #     if len(bank_obs_list) == 0 or (isinstance(bank_obs_list[0], list) and len(bank_obs_list[0]) == 0):
        #         return None, None
        #
        #     next_bank_obs_list = [obs.get('bank', []) for obs in next_obs_dict]
        #     bank_action_list = [action.get('bank', []) for action in action_dict]
        #     bank_reward_list = [reward.get('bank', 0.0) for reward in reward_dict]
        #
        #     obs_tensor = torch.tensor(np.array(bank_obs_list), dtype=torch.float32).to(self.device)
        #     next_obs_tensor = torch.tensor(np.array(next_bank_obs_list), dtype=torch.float32).to(self.device)
        #     action_tensor = torch.tensor(np.array(bank_action_list), dtype=torch.float32).to(self.device)
        #     reward_tensor = torch.tensor(np.array(bank_reward_list), dtype=torch.float32).to(self.device)
        #     if reward_tensor.dim() == 1:
        #         reward_tensor = reward_tensor.unsqueeze(-1)
        #
        # else:
        #     return None, None  # Skip training for unsupported agents

        # Forward pass
        next_value, next_pi = self.net(next_obs_tensor)
        td_target = reward_tensor + self.args.gamma * next_value * inverse_dones
        value, pi = self.net(obs_tensor)
        td_delta = td_target - value

        advantage = self.compute_advantage(self.args.gamma, self.args.tau, td_delta.cpu()).to(self.device)

        mu, std = pi
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(action_tensor)

        for i in range(self.args.update_each_epoch):
            value, pi = self.net(obs_tensor)
            mu, std = pi
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(action_tensor)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(value, td_target.detach()))
            total_loss = actor_loss + self.args.vloss_coef * critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            sum_loss[0] += actor_loss
            sum_loss[1] += critic_loss

        self.scheduler.step()
        return sum_loss[0], sum_loss[1]

    def get_action(self, obs_tensor):
        if self.agent_name == "bank" and self.agent_type == "non_profit":
            return np.random.randn(self.action_dim)
        if self.agent_name == "market" and self.agent_type == "perfect":
            firm_n = len(obs_tensor)
            return np.random.randn(firm_n, self.action_dim)
        # === 1. Running mean/std update ===
        if self.step_counter < self.max_warmup_steps:
            obs_np = obs_tensor.detach().cpu().numpy()
            self.state_rms.update(obs_np)
            self.step_counter += 1

        # === 2. Inject normalizer after warmup ===
        if (not self.normalizer_applied) and (self.step_counter >= self.max_warmup_steps):
            self.net.set_normalizer(self.state_rms.mean, self.state_rms.std)
            self.normalizer_applied = True

        _, pi = self.net(obs_tensor)
        mu, sigma = pi
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()

        return action.cpu().numpy()

    def gov_action_wrapper(self, gov_action):
        return self.agent.real_action_min + (self.agent.real_action_max - self.agent.real_action_min) * gov_action

    def inverse_action_wrapper(self, action):
        if action.is_cuda:
            action_np = action.cpu().numpy()  # 将张量从 GPU 移动到 CPU 并转换为 numpy 数组
        else:
            action_np = action.numpy()  # 如果 action 已经在 CPU 上，则直接转换为 numpy 数组

        result_np = (action_np - self.agent.real_action_min) / (
                self.agent.real_action_max - self.agent.real_action_min)

        # 将计算结果转换回 PyTorch 的 tensor，并根据原始 action 是否在 GPU 上决定是否要将结果放回 GPU
        result_tensor = torch.tensor(result_np, dtype=action.dtype)

        if action.is_cuda:
            return result_tensor.cuda()  # 如果原始 action 在 GPU 上，则将结果也放回 GPU
        else:
            return result_tensor  # 否则保持在 CPU 上

    def save(self, dir_path):
        torch.save(self.net.state_dict(), str(dir_path) + '/ppo_net.pt')
