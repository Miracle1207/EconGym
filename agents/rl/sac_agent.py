import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RunningMeanStd:
    """Running mean and standard deviation for state normalization"""
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


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = map(np.stack, zip(*[self.buffer[i] for i in batch]))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SACPolicy(nn.Module):
    """SAC Actor network"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACPolicy, self).__init__()
        self.use_norm = False
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.orthogonal_(self.mean_linear.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_linear.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.mean_linear.bias, 0)
        nn.init.constant_(self.log_std_linear.bias, 0)

    def set_normalizer(self, mean, std):
        self.use_norm = True
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
        if next(self.parameters()).is_cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def forward(self, state):
        if self.use_norm:
            state = (state - self.mean) / (self.std + 1e-8)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.sigmoid(self.mean_linear(x))  # [0, 1]
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class SACQNetwork(nn.Module):
    """SAC Critic network (Q-function)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class sac_agent:
    """Soft Actor-Critic (SAC) agent"""
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

        # Initialize networks
        self.policy = SACPolicy(self.obs_dim, self.action_dim).to(self.device)
        self.qf1 = SACQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.qf2 = SACQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_qf1 = SACQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_qf2 = SACQNetwork(self.obs_dim, self.action_dim).to(self.device)

        # Initialize target networks
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.p_lr)
        self.qf1_optimizer = torch.optim.Adam(self.qf1.parameters(), lr=self.args.p_lr)
        self.qf2_optimizer = torch.optim.Adam(self.qf2.parameters(), lr=self.args.p_lr)

        # SAC hyperparameters
        self.gamma = getattr(self.args, 'gamma', 0.99)
        self.tau = getattr(self.args, 'tau', 0.005)
        self.alpha = getattr(self.args, 'alpha', 0.2)
        self.automatic_entropy_tuning = getattr(self.args, 'automatic_entropy_tuning', True)
        
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.p_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = getattr(self.args, 'batch_size', 256)
        
        # State normalization
        self.state_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.step_counter = 0
        self.max_warmup_steps = 1000
        self.normalizer_applied = False
        
        self.on_policy = False  # SAC is off-policy

    def train(self, transition_dict):
        """Train the SAC agent"""
        # Extract data from new nested dictionary structure
        obs_dict = transition_dict['obs_dict']
        next_obs_dict = transition_dict['next_obs_dict']
        action_dict = transition_dict['action_dict']
        reward_dict = transition_dict['reward_dict']
        done_list = transition_dict['done']

        # Extract data based on agent type
        if self.agent_name == "government":
            # Get government observations and actions based on government type
            if self.agent_type == "pension":
                gov_obs_list = [obs['government']['pension'] for obs in obs_dict]
                next_gov_obs_list = [obs['government']['pension'] for obs in next_obs_dict]
                gov_action_list = [action['government']['pension'] for action in action_dict]
                gov_reward_list = [reward['government']['pension'] for reward in reward_dict]
            elif self.agent_type == "tax":
                gov_obs_list = [obs['government']['tax'] for obs in obs_dict]
                next_gov_obs_list = [obs['government']['tax'] for obs in next_obs_dict]
                gov_action_list = [action['government']['tax'] for action in action_dict]
                gov_reward_list = [reward['government']['tax'] for reward in reward_dict]
            elif self.agent_type == "central_bank":
                gov_obs_list = [obs['government']['central_bank'] for obs in obs_dict]
                next_gov_obs_list = [obs['government']['central_bank'] for obs in next_obs_dict]
                gov_action_list = [action['government']['central_bank'] for action in action_dict]
                gov_reward_list = [reward['government']['central_bank'] for reward in reward_dict]

            # Store transitions in replay buffer
            for i in range(len(gov_obs_list)):
                self.store_transition(
                    gov_obs_list[i], 
                    gov_action_list[i], 
                    gov_reward_list[i], 
                    next_gov_obs_list[i], 
                    done_list[i]
                )

        elif self.agent_name == "households":
            # Extract household data from new structure
            house_obs_list = [obs['households'] for obs in obs_dict]
            next_house_obs_list = [obs['households'] for obs in next_obs_dict]
            house_action_list = [action['households'] for action in action_dict]
            house_reward_list = [reward['households'] for reward in reward_dict]

            # Store transitions for each household
            for i in range(len(house_obs_list)):
                for j in range(len(house_obs_list[i])):
                    self.store_transition(
                        house_obs_list[i][j], 
                        house_action_list[i][j], 
                        house_reward_list[i][j][0], 
                        next_house_obs_list[i][j], 
                        done_list[i]
                    )

        elif self.agent_name == "market":
            if self.agent_type == "perfect":
                return 0.0, 0.0

            market_obs_list = [obs.get('market', []) for obs in obs_dict]
            if len(market_obs_list) == 0 or (isinstance(market_obs_list[0], list) and len(market_obs_list[0]) == 0):
                return 0.0, 0.0

            next_market_obs_list = [obs.get('market', []) for obs in next_obs_dict]
            market_action_list = [action.get('market', []) for action in action_dict]
            market_reward_list = [reward.get('market', []) for reward in reward_dict]

            # Store transitions for each firm
            for i in range(len(market_obs_list)):
                for j in range(len(market_obs_list[i])):
                    self.store_transition(
                        market_obs_list[i][j], 
                        market_action_list[i][j], 
                        market_reward_list[i][j][0], 
                        next_market_obs_list[i][j], 
                        done_list[i]
                    )

        elif self.agent_name == "bank":
            if self.agent_type == 'non_profit':
                return 0.0, 0.0

            bank_obs_list = [obs.get('bank', []) for obs in obs_dict]
            if len(bank_obs_list) == 0 or (isinstance(bank_obs_list[0], list) and len(bank_obs_list[0]) == 0):
                return 0.0, 0.0

            next_bank_obs_list = [obs.get('bank', []) for obs in next_obs_dict]
            bank_action_list = [action.get('bank', []) for action in action_dict]
            bank_reward_list = [reward.get('bank', 0.0) for reward in reward_dict]

            # Store transitions
            for i in range(len(bank_obs_list)):
                self.store_transition(
                    bank_obs_list[i], 
                    bank_action_list[i], 
                    bank_reward_list[i], 
                    next_bank_obs_list[i], 
                    done_list[i]
                )

        else:
            return 0.0, 0.0

        # Train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # Compute Q-values
        qf1_a_values = self.qf1(state_batch, action_batch)
        qf2_a_values = self.qf2(state_batch, action_batch)

        # Compute next actions and Q-values
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target = self.target_qf1(next_state_batch, next_state_actions)
            qf2_next_target = self.target_qf2(next_state_batch, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target

        # Compute Q-function losses
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Update Q-functions
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # Compute policy loss
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi = self.qf1(state_batch, pi)
        qf2_pi = self.qf2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update alpha
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # Update target networks
        self._soft_update(self.target_qf1, self.qf1, self.tau)
        self._soft_update(self.target_qf2, self.qf2, self.tau)

        return policy_loss.item(), qf_loss.item()

    def _soft_update(self, target, source, tau):
        """Soft update of target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, obs_tensor):
        """Get action from the policy"""
        if self.agent_name == "bank" and self.agent_type == "non_profit":
            return np.random.randn(self.action_dim)
        if self.agent_name == "market" and self.agent_type == "perfect":
            firm_n = len(obs_tensor)
            return np.random.randn(firm_n, self.action_dim)

        # Update state normalization
        if self.step_counter < self.max_warmup_steps:
            obs_np = obs_tensor.detach().cpu().numpy()
            self.state_rms.update(obs_np)
            self.step_counter += 1

        # Apply normalizer after warmup
        if (not self.normalizer_applied) and (self.step_counter >= self.max_warmup_steps):
            self.policy.set_normalizer(self.state_rms.mean, self.state_rms.std)
            self.normalizer_applied = True

        # Get action from policy
        with torch.no_grad():
            action, _, _ = self.policy.sample(obs_tensor)

        return action.cpu().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)


    def save(self, dir_path):
        """Save the model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'target_qf1_state_dict': self.target_qf1.state_dict(),
            'target_qf2_state_dict': self.target_qf2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'qf1_optimizer_state_dict': self.qf1_optimizer.state_dict(),
            'qf2_optimizer_state_dict': self.qf2_optimizer.state_dict(),
        }, str(dir_path) + '/sac_net.pt')

    def load(self, dir_path):
        """Load the model"""
        checkpoint = torch.load(str(dir_path) + '/sac_net.pt', map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.qf1.load_state_dict(checkpoint['qf1_state_dict'])
        self.qf2.load_state_dict(checkpoint['qf2_state_dict'])
        self.target_qf1.load_state_dict(checkpoint['target_qf1_state_dict'])
        self.target_qf2.load_state_dict(checkpoint['target_qf2_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.qf1_optimizer.load_state_dict(checkpoint['qf1_optimizer_state_dict'])
        self.qf2_optimizer.load_state_dict(checkpoint['qf2_optimizer_state_dict'])
