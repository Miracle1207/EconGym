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
        self.name = 'ddpg'
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

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)

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
        # Align with PPO/SAC: early exits for fixed-action setups
        if self.agent_name == "bank" and self.agent_type == "non_profit":
            return 0.0, 0.0
        if self.agent_name == "market" and self.agent_type == "perfect":
            return 0.0, 0.0

        # transitions is a batch from ReplayBuffer.sample with keys:
        # obs_dict, next_obs_dict, action_dict, reward_dict, done
        agent_data = transitions

        # To tensors (support 2D and 3D shapes like SAC/PPO)
        obs_tensor = torch.tensor(np.array(agent_data['obs_dict']), dtype=torch.float32, device=device)
        next_obs_tensor = torch.tensor(np.array(agent_data['next_obs_dict']), dtype=torch.float32, device=device)
        action_tensor = torch.tensor(np.array(agent_data['action_dict']), dtype=torch.float32, device=device)
        reward_tensor = torch.tensor(np.array(agent_data['reward_dict']), dtype=torch.float32, device=device)
        inverse_dones = torch.tensor([x - 1 for x in agent_data['done']], dtype=torch.float32).unsqueeze(-1)
        if inverse_dones.shape != reward_tensor.shape:
            inverse_dones = inverse_dones.unsqueeze(-1).expand_as(reward_tensor)
        inverse_dones = inverse_dones.to(device)

        # Critic target
        with torch.no_grad():
            next_actions = self.target_actor(next_obs_tensor)
            next_q_values = self.target_critic(next_obs_tensor, next_actions)
            if reward_tensor.dim() == next_q_values.dim() - 1:
                reward_tensor = reward_tensor.unsqueeze(-1)
            if inverse_dones.dim() == next_q_values.dim() - 1:
                inverse_dones = inverse_dones.unsqueeze(-1)
            q_targets = reward_tensor + self.args.gamma * next_q_values * inverse_dones

        # Critic update
        q_values = self.critic(obs_tensor, action_tensor)
        critic_loss = F.mse_loss(q_values, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_actions = self.actor(obs_tensor)
        actor_loss = -self.critic(obs_tensor, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets and schedulers
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


    def save(self, dir_path):
        torch.save(self.actor.state_dict(), str(dir_path) + '/' + self.agent_name + '_ddpg_net.pt')

    def load(self, dir_path):
        # self.actor.load_state_dict(torch.load(dir_path))
        self.actor.load_state_dict(torch.load(dir_path, map_location=torch.device(self.device), weights_only=True))
