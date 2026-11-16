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
        self.name = 'ppo'
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
            
        self.state_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.step_counter = 0
        self.max_warmup_steps = 1000
        self.normalizer_applied = False
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Initialize the MLP network
        self.net = mlp_net(state_dim=self.obs_dim, num_actions=self.action_dim).to(self.device)

        # Flag to choose whether to load an existing policy or not
        self.load_exist_policy = False  # Set to True to load the trained policy
        # self.load_exist_policy = True

        # Check if the policy should be loaded
        if self.load_exist_policy:
            path = None
    
            if agent_name == "households":
                if "OLG" in self.envs.households.type:
                    path = "agents/models/trained_policy/ppo_OLG/ppo_net.pt"
                elif "ramsey" in self.envs.households.type:
                    # Re-init net for Ramsey specific action dim
                    self.net = mlp_net(state_dim=self.obs_dim, num_actions=3).to(self.device)
                    path = "agents/models/trained_policy/ppo_Ramsey/ppo_net.pt"
    
            elif agent_name == "government":
                if "tax" in self.envs.government:
                    path = "agents/models/bc_ppo/100/gdp/run8/ppo_net.pt"
                if "central_bank" in self.envs.government:
                    path = "agents/models/optimal_monetary/100/run46/government_ppo_net.pt"
    
            elif agent_name == "bank":
                if "commercial" in self.envs.bank.type:
                    path = "agents/models/optimal_monetary/100/run46/bank_ppo_net.pt"
    
            # Call the encapsulated loader
            if path:
                self.load_model(path)


        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.p_lr, eps=1e-5)
        lambda_function = lambda epoch: 0.97 ** (epoch // 10)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_function)
        self.on_policy = True
        
        

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
        if self.agent_name == "bank" and self.agent_type == "non_profit":
            return 0, 0
        if self.agent_name == "market" and self.agent_type == "perfect":
            return 0, 0
        sum_loss = torch.tensor([0., 0.], dtype=torch.float32).to(self.device)

        agent_data = transition_dict
        obs_tensor = torch.tensor(np.array(agent_data['obs_dict']), dtype=torch.float32).to(self.device)
        next_obs_tensor = torch.tensor(np.array(agent_data['next_obs_dict']), dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(np.array(agent_data['action_dict']), dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(np.array(agent_data['reward_dict']), dtype=torch.float32).to(self.device)
        inverse_dones = torch.tensor([x - 1 for x in agent_data['done']], dtype=torch.float32).unsqueeze(-1)
        # Ensure both tensors have the same shape before expanding
        if inverse_dones.shape != reward_tensor.shape:
            inverse_dones = inverse_dones.unsqueeze(-1).expand_as(reward_tensor)

        inverse_dones = inverse_dones.to(self.device)

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

    def save(self, dir_path):
        torch.save(self.net.state_dict(), str(dir_path) + '/' + self.agent_name + '_ppo_net.pt')

    def load_model(self, path):
        """
        Safely loads network weights and restoration statistics.
        Handles 'mean'/'std' keys to prevent RuntimeErrors.
        """
        # Load the checkpoint
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
    
        # === Handle Normalization Stats (mean/std) ===
        if "mean" in state_dict and "std" in state_dict:
            # Extract and remove normalization keys from dict
            loaded_mean = state_dict["mean"]
            loaded_std = state_dict["std"]
        
            # Update RunningMeanStd (convert tensor to numpy)
            mean_np = loaded_mean.cpu().numpy()
            std_np = loaded_std.cpu().numpy()

            self.state_rms.mean = mean_np
            self.state_rms.var = std_np ** 2
            self.state_rms.count = 1e6

            self.net.set_normalizer(mean_np, std_np)

            self.normalizer_applied = True
            self.step_counter = self.max_warmup_steps + 1
            
        
            print(f"[INFO] Loaded model with Normalization from: {path}")
        else:
            print(f"[INFO] Loaded model weights (no norm stats) from: {path}")
    
        # === Load Network Weights ===
        # strict=True ensures the architecture matches exactly
        self.net.load_state_dict(state_dict)
