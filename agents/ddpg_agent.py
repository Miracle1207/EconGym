import copy
import numpy as np
import torch
from agents.models import PolicyNet, QValueNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


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


class ddpg_agent:
    def __init__(self, envs, args, agent_name="household"):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        self.agent = getattr(self.envs, agent_name)

        if agent_name == "household":
            self.obs_dim = self.envs.observation_space.shape[0]
            self.action_dim = self.envs.households.action_space.shape[1]
        elif agent_name in ("government", "tax_gov", "pension_gov", "central_bank_gov"):
            self.obs_dim = self.agent.observation_space.shape[0]
            self.action_dim = self.agent.action_space.shape[0]
        else:
            print("AgentError: Please choose the correct agent name!")
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.actor = PolicyNet(state_dim=self.obs_dim, hidden_dim=128, action_dim=self.action_dim).to(self.device)
        self.critic = QValueNet(state_dim=self.obs_dim, hidden_dim=128, action_dim=self.action_dim).to(self.device)
        self.use_type = "train"
        if agent_name == "household":
            if self.args.bc == True:
                self.actor.load_state_dict(torch.load("agents/real_data/2024_01_04_21_21_maddpg_trained_model.pth"))
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-6)
            else:
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
            if self.use_type == "test":
                # state_dict = torch.load("agents/models/bc_ddpg/1000/gdp/run20/government_ddpg_net.pt")
                state_dict = torch.load("agents/models/bc_ddpg/100/gdp/run2/government_ddpg_net.pt")
                if "mean" in state_dict and "std" in state_dict:
                    mean = state_dict["mean"]
                    std = state_dict["std"]
                    self.actor.set_normalizer(mean, std)
                    self.actor.load_state_dict(state_dict)
                else:
                    # 不含 normalizer，那就直接加载模型，跳过 normalizer 设置
                    self.actor.load_state_dict(state_dict, strict=False)

        self.target_actor = copy.copy(self.actor)
        self.target_critic = copy.copy(self.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.q_lr)
        lambda_function = lambda epoch: 0.95 ** (epoch // (35 * self.args.update_cycles))

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lambda_function)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lambda_function)
        self.on_policy = False
        self.state_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.step_counter = 0
        self.max_warmup_steps = 1000
        self.normalizer_applied = False

    def train(self, transitions, other_agent=None):
        # households_n = len(transitions['hou_action'][0])
        # private_obses = torch.tensor(transitions['private_obs'], dtype=torch.float32,
        #                              device='cuda' if self.args.cuda else 'cpu')
        global_obses = torch.tensor(transitions['global_obs'], dtype=torch.float32,
                                    device='cuda' if self.args.cuda else 'cpu')
        if self.agent_name == "pension_gov":
            gov_actions = torch.tensor(transitions['pension_gov_action'], dtype=torch.float32,
                                       device='cuda' if self.args.cuda else 'cpu')
        elif self.agent_name == "central_bank_gov":
            gov_actions = torch.tensor(transitions['central_bank_gov_action'], dtype=torch.float32,
                                       device='cuda' if self.args.cuda else 'cpu')
        else:
            gov_actions = torch.tensor(transitions['gov_action'], dtype=torch.float32,
                                       device='cuda' if self.args.cuda else 'cpu')

        # house_actions = torch.tensor(transitions['hou_action'], dtype=torch.float32,
        #                              device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(transitions['gov_reward'], dtype=torch.float32,
                                   device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        # house_rewards = torch.tensor(transitions['house_reward'], dtype=torch.float32,
        #                              device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(transitions['next_global_obs'], dtype=torch.float32,
                                         device='cuda' if self.args.cuda else 'cpu')
        # next_private_obses = torch.tensor(transitions['next_private_obs'], dtype=torch.float32,
        #                                   device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - transitions['done'], dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)

        if self.agent_name in ("government", "pension_gov", "tax_gov", "central_bank_gov"):
            obs_tensor = global_obses
            action_tensor = self.inverse_action_wrapper(gov_actions)
            reward_tensor = gov_rewards
            next_obs_tensor = next_global_obses
        # elif self.agent_name == "household":
        #     obs_tensor = private_obses
        #     action_tensor = house_actions
        #     reward_tensor = house_rewards
        #     next_obs_tensor = next_private_obses
        #     inverse_dones = inverse_dones.unsqueeze(-1).repeat(1, households_n, 1)

        else:
            obs_tensor, action_tensor, reward_tensor, next_obs_tensor = None, None, None, None

        next_q_values = self.target_critic(next_obs_tensor, self.target_actor(next_obs_tensor))
        q_targets = reward_tensor + self.args.gamma * next_q_values * inverse_dones
        critic_loss = torch.mean(F.mse_loss(self.critic(obs_tensor, action_tensor), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(obs_tensor, self.actor(obs_tensor)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_target_network(self.target_actor, self.actor)
        self._soft_update_target_network(self.target_critic, self.critic)

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return actor_loss, critic_loss

    # soft update the target network...
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, env=None):
        if self.agent_name in ("government", "tax_gov", "pension_gov", "central_bank_gov"):
            obs_tensor = global_obs_tensor.reshape(-1, self.obs_dim)
        elif self.agent_name == "household":
            obs_tensor = private_obs_tensor
        else:
            obs_tensor = None

        # === 1. Running mean/std update ===
        if self.step_counter < self.max_warmup_steps:
            obs_np = obs_tensor.detach().cpu().numpy()
            self.state_rms.update(obs_np)
            self.step_counter += 1

        # === 2. Inject normalizer after warmup ===
        if (not self.normalizer_applied) and (self.step_counter >= self.max_warmup_steps):
            self.actor.set_normalizer(self.state_rms.mean, self.state_rms.std)
            self.normalizer_applied = True

        # === 3. Compute action ===
        action = self.actor(obs_tensor).detach().cpu().numpy()
        sigma = 0.01
        action = action + sigma * np.random.randn(self.action_dim)

        if self.agent_name == "government":
            if self.envs.government.type == "tax":
                action[0][2] = 0
                action[0][3] = 0
            # if self.envs.government.type == "pension":
            #     action[0][0] = round(action[0][0])
        if self.agent_name == "tax_gov":
            action[0][2] = 0
            action[0][3] = 0
        # if self.agent_name == "pension_gov":
        #     action[0][0] = round(action[0][0])

        if self.agent_name in ("government", "tax_gov", "pension_gov", "central_bank_gov"):
            wrapper_action = self.gov_action_wrapper(action.flatten())
            return wrapper_action

        elif self.agent_name == "household":
            return action

    def gov_action_wrapper(self, gov_action):
        # gov_action = (gov_action +1)/2
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
        torch.save(self.actor.state_dict(), str(dir_path) + '/' + self.agent_name + '_ddpg_net.pt')

    def load(self, dir_path):
        # self.actor.load_state_dict(torch.load(dir_path))
        self.actor.load_state_dict(torch.load(dir_path, map_location=torch.device(self.device), weights_only=True))
