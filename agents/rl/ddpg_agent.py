import copy
import numpy as np
import torch
from agents.rl.models import PolicyNet, QValueNet
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
        self.actor = PolicyNet(state_dim=self.obs_dim, hidden_dim=128, action_dim=self.action_dim).to(self.device)
        self.critic = QValueNet(state_dim=self.obs_dim, hidden_dim=128, action_dim=self.action_dim).to(self.device)
        self.use_type = "train"
        if agent_name == "households":
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
        device = 'cuda' if self.args.cuda else 'cpu'
        # transitions is a batch from ReplayBuffer.sample with keys: obs_dict, next_obs_dict, action_dict, reward_dict, done
        obs_dict_batch = transitions['obs_dict']
        next_obs_dict_batch = transitions['next_obs_dict']
        action_dict_batch = transitions['action_dict']
        reward_dict_batch = transitions['reward_dict']
        done_batch = transitions['done']

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        batch_size = len(obs_dict_batch)

        if self.agent_name == "government":
            key = self.agent_type
            for i in range(batch_size):
                s = obs_dict_batch[i]['government'][key]
                a = action_dict_batch[i]['government'][key]
                r = reward_dict_batch[i]['government'][key]
                ns = next_obs_dict_batch[i]['government'][key]
                d = done_batch[i]
                states.append(s);
                actions.append(a);
                rewards.append(r);
                next_states.append(ns);
                dones.append(d)

        elif self.agent_name == "households":
            for i in range(batch_size):
                hh_obs = obs_dict_batch[i]['households']
                hh_act = action_dict_batch[i]['households']
                hh_rwd = reward_dict_batch[i]['households']
                hh_nobs = next_obs_dict_batch[i]['households']
                for j in range(len(hh_obs)):
                    states.append(hh_obs[j])
                    actions.append(hh_act[j])
                    rewards.append(hh_rwd[j][0])
                    next_states.append(hh_nobs[j])
                    dones.append(done_batch[i])

        elif self.agent_name == "market":
            if self.agent_type == "perfect":
                return 0.0, 0.0
            for i in range(batch_size):
                mk_obs = obs_dict_batch[i].get('market', [])
                mk_act = action_dict_batch[i].get('market', [])
                mk_rwd = reward_dict_batch[i].get('market', [])
                mk_nobs = next_obs_dict_batch[i].get('market', [])
                for j in range(len(mk_obs)):
                    states.append(mk_obs[j])
                    actions.append(mk_act[j])
                    rewards.append(mk_rwd[j][0])
                    next_states.append(mk_nobs[j])
                    dones.append(done_batch[i])

        elif self.agent_name == "bank":
            if self.agent_type == 'non_profit':
                return 0.0, 0.0
            for i in range(batch_size):
                s = obs_dict_batch[i].get('bank', [])
                a = action_dict_batch[i].get('bank', [])
                r = reward_dict_batch[i].get('bank', 0.0)
                ns = next_obs_dict_batch[i].get('bank', [])
                states.append(s);
                actions.append(a);
                rewards.append(r);
                next_states.append(ns);
                dones.append(done_batch[i])

        else:
            return 0.0, 0.0

        if len(states) == 0:
            return 0.0, 0.0

        obs_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        action_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        reward_tensor = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(-1)
        next_obs_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        inverse_dones = torch.tensor(1 - np.array(dones), dtype=torch.float32, device=device).unsqueeze(-1)

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

        return actor_loss.item(), critic_loss.item()

    # soft update the target network...
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

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
            self.actor.set_normalizer(self.state_rms.mean, self.state_rms.std)
            self.normalizer_applied = True

        # === 3. Compute action ===
        action = self.actor(obs_tensor).detach().cpu().numpy()
        sigma = 0.01
        noise = sigma * np.random.randn(*action.shape)
        action = action + noise
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
