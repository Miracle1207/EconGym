import copy
import numpy as np
import torch
import pdb
from agents.rl.models import PolicyNet, BiQValueNet
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

def normalize(x,method='z_score',axis=-1):
    if method=='z_score':
        return (x - np.mean(x,axis=axis,keepdims=True)) / (np.std(x,axis=axis,keepdims=True) + 1e-8)
    elif method=='robust_scale':
        median = np.median(x,axis=axis,keepdims=True)
        q1 = np.percentile(x, 25,axis=axis,keepdims=True)
        q3 = np.percentile(x, 75,axis=axis,keepdims=True)
        iqr = q3 - q1
        return (x - median) / (iqr + 1e-8)
    else:
        raise ValueError("Unsupported normalization method")

def quick_normalize(global_obses, private_obses, gov_actions, house_actions,next_global_obses, next_private_obses,past_mean_house_actions,\
                    gov_rewards,house_rewards):
    global_obses = normalize(global_obses,method='z_score',axis=0)
    private_obses = normalize(private_obses,method='z_score',axis=0)
    gov_actions = normalize(gov_actions,method='z_score',axis=0)
    house_actions = normalize(house_actions,method='z_score',axis=0)
    next_global_obses = normalize(next_global_obses,method='z_score',axis=0)
    next_private_obses = normalize(next_private_obses,method='z_score',axis=0)
    past_mean_house_actions = normalize(past_mean_house_actions,method='z_score',axis=0)
    gov_rewards=normalize(gov_rewards,method='z_score',axis=0)
    house_rewards=normalize(house_rewards,method='z_score',axis=0)
    return global_obses, private_obses, gov_actions, house_actions,next_global_obses, next_private_obses,past_mean_house_actions,gov_rewards,house_rewards
class bi_ddpg_agent:
    def __init__(self, envs, args, type=None,agent_name=None):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        self.agent_type = type

        self.obs_dim = self.envs.government[type].observation_space.shape[0]
        self.action_dim = self.envs.government[type].action_space.shape[0]
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.actor = PolicyNet(state_dim=self.obs_dim, hidden_dim=128, action_dim=self.action_dim).to(self.device)
        self.critic = BiQValueNet(gov_state_dim=self.obs_dim, gov_action_dim=self.action_dim,
                                  mean_state_dim=self.envs.households.observation_space.shape[0],
                                  mean_action_dim=self.envs.households.action_space.shape[1], hidden_dim=128).to(self.device)
        self.target_actor = copy.copy(self.actor)
        self.target_critic = copy.copy(self.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
        #self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.q_lr)
        # lambda_function = lambda epoch: 0.95 ** (epoch // (self.args.update_cycles/2))

        self.state_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.normalizer_applied=False
        self.step_counter=0
        self.max_warmup_steps=1000

        lambda_function = lambda epoch: 0.95 ** (epoch // (35*self.args.update_cycles))

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lambda_function)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lambda_function)
        self.on_policy = False
    
    def train(self, transitions, other_agent):
        device = 'cuda' if self.args.cuda else 'cpu'
        # Align with PPO/SAC: early exits for fixed-action setups
        if self.agent_name == "bank" and self.agent_type == "non_profit":
            return 0.0, 0.0
        if self.agent_name == "market" and self.agent_type == "perfect":
            return 0.0, 0.0
        
        global_obses, private_obses, gov_actions, house_actions, gov_rewards,\
        house_rewards, next_global_obses, next_private_obses, dones, past_mean_house_actions,*_ = transitions.values()

        #normalize
        '''
        global_obses, private_obses, gov_actions, house_actions,next_global_obses, next_private_obses,past_mean_house_actions,\
        gov_rewards,house_rewards\
        =quick_normalize(global_obses, private_obses, gov_actions, house_actions,next_global_obses, next_private_obses,\
                         past_mean_house_actions,gov_rewards,house_rewards)
        '''

        private_obses = torch.tensor(private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        global_obses = torch.tensor(global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_actions = torch.tensor(house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        mean_house_actions = torch.tensor(past_mean_house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        house_rewards = torch.tensor(house_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = 1.0-torch.tensor(dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)

        next_gov_action = self.target_actor(next_global_obses)
        next_house_action, next_mean_house_action = other_agent.get_action(obs_tensor=next_private_obses,
                                                   gov_action=next_gov_action, agent_name="household")
        next_mean_house_action = torch.tensor(next_mean_house_action, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_mean_house_state = torch.mean(next_private_obses, dim=1)

        #next_mean_house_state,next_mean_house_action,next_global_obses,next_gov_action=self.align_dim(next_mean_house_state,next_mean_house_action,next_global_obses,next_gov_action)
        next_q_values = self.target_critic(next_global_obses, next_gov_action, next_mean_house_state, next_mean_house_action)
        q_targets = gov_rewards + self.args.gamma * next_q_values * inverse_dones
        mean_house_state = torch.mean(private_obses, dim=1)
        critic_loss = torch.mean(F.mse_loss(self.critic(global_obses, gov_actions, mean_house_state, mean_house_actions), q_targets))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        gov_action_ = self.actor(global_obses)
        _, mean_house_action_ = other_agent.get_action(obs_tensor=private_obses,gov_action=gov_action_,agent_name="household")
        mean_house_action_ = torch.tensor(mean_house_action_, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        mean_house_state_ = torch.mean(private_obses, dim=1)
        actor_loss = -torch.mean(self.critic(global_obses, self.actor(global_obses), mean_house_state_, mean_house_action_))
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
            target_param.data.copy_((1 - 0.95) * param.data + 0.95 * target_param.data)
            
    def get_action(self, obs_tensor,gov_action=None, agent_name="government"):
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
        
        action = self.actor(obs_tensor).detach().cpu().numpy()
        sigma = 0.01
        action = action + sigma * np.random.randn(self.action_dim)
        # action = (action + 1)/2
        if len(obs_tensor.shape)>1:
            return action
        else:
            return action.flatten()
    
    def align_dim(self,next_mean_house_state,next_mean_house_action,next_global_obses,next_gov_action):
        next_mean_house_state=next_mean_house_state.unsqueeze(1)
        next_mean_house_action=next_mean_house_action.unsqueeze(1).repeat(self.envs.households.households_n,1)
        next_global_obses=next_global_obses.repeat(self.envs.households.households_n,1)
        next_gov_action=next_gov_action.repeat(self.envs.households.households_n,1)
        #print(next_mean_house_state.shape,next_mean_house_action.shape,next_global_obses.shape,next_gov_action.shape)
        
        return next_mean_house_state,next_mean_house_action,next_global_obses,next_gov_action


    def save(self, dir_path):
        torch.save(self.actor.state_dict(), str(dir_path) + '/bi_ddpg_net.pt')
        torch.save(self.critic.state_dict(), str(dir_path) + '/bi_ddpg_net_critic.pt')
        torch.save(self.target_actor.state_dict(), str(dir_path) + '/bi_ddpg_net_target.pt')
        torch.save(self.target_critic.state_dict(), str(dir_path) + '/bi_ddpg_net_critic_target.pt')
    
    def load(self, dir_path):
        self.actor.load_state_dict(torch.load(dir_path, map_location=torch.device(self.device), weights_only=True))