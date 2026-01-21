import numpy as np
import torch
from torch import optim
import os, sys
import pdb

sys.path.append(os.path.abspath('../..'))
from agents.rl.models import MFActor, MFCritic
import copy
from utils.utils import sync_networks, sync_grads
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

def get_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    while os.path.basename(project_root) != "EconGym" and project_root != os.path.dirname(project_root):
        project_root = os.path.dirname(project_root)
    model_path = os.path.join(project_root, relative_path)

    return model_path



class bi_mfrl_agent():  ## ddpg_based
    def __init__(self, envs, args,type=None,agent_name=None):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.bc=False #设置一个args
        obs_dims = self.envs.households.observation_space.shape[0]
        # start to build the network.
        self.type=type
        for k,v in self.envs.government.items():
            self.government_type=k

        self.mf_actor = MFActor(input_dims=self.envs.households.observation_space.shape[0],
                                leader_action_dim=self.envs.government[self.government_type].action_space.shape[0],
                                action_dims=self.envs.households.action_space.shape[1],
                                      hidden_size=128)  # Stackelberg mean field game
        self.mf_critic = MFCritic(obs_dims=self.envs.households.observation_space.shape[0],
                                  gov_action_dim=self.envs.government[self.government_type].action_space.shape[0],
                                  action_dim=self.envs.households.action_space.shape[1],
                                  mean_state_dim=self.envs.households.observation_space.shape[0],
                                  mean_action_dim=self.envs.households.action_space.shape[1],
                                  hidden_size=128)
        # sync the weights across the mpi
        sync_networks(self.mf_actor)
        sync_networks(self.mf_critic)

        if self.bc == True:
            relative_path = os.path.join("agents", "models", "optimal_monetary", "100", "run204", "ramsey_bc_binet.pt")
            model_path=get_path(relative_path)
            checkpoint = torch.load(model_path, weights_only=True)
            checkpoint.pop('mean', None)
            checkpoint.pop('std', None)
            self.mf_actor.load_state_dict(checkpoint)
            self.mf_actor_optimizer = optim.Adam(self.mf_actor.parameters(), 1e-6, eps=self.args.eps)
        else:
            self.mf_actor_optimizer = optim.Adam(self.mf_actor.parameters(), self.args.p_lr, eps=self.args.eps)
        
        self.obs_dim=self.envs.households.observation_space.shape[0]
        self.state_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.normalizer_applied=False
        self.step_counter=0
        self.max_warmup_steps=1000

        #self.mf_actor_optimizer = optim.Adam(self.mf_actor.parameters(), self.args.p_lr, eps=self.args.eps)
        # # build the target newtork
        self.mf_actor_target = copy.deepcopy(self.mf_actor)
        self.mf_critic_target = copy.deepcopy(self.mf_critic)
        # define the optimizer...
        self.mf_critic_optimizer = optim.Adam(self.mf_critic.parameters(), self.args.p_lr, eps=self.args.eps)

        lambda_function = lambda epoch: 0.95 ** (epoch // (35*self.args.update_cycles))

        self.actor_scheduler = LambdaLR(self.mf_actor_optimizer, lr_lambda=lambda_function)
        self.critic_scheduler = LambdaLR(self.mf_critic_optimizer, lr_lambda=lambda_function)
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        if self.args.cuda:
            self.mf_actor.cuda()
            self.mf_critic.cuda()
            self.mf_actor_target.cuda()
            self.mf_critic_target.cuda()
        self.on_policy = False

    def get_action(self, obs_tensor,gov_action,agent_name="households"):
        private_obs_tensor=obs_tensor
        if len(private_obs_tensor.shape) == 3:
            gov_action_tensor = gov_action.unsqueeze(1).repeat(1,self.envs.households.households_n,1)
        else:
            gov_action_tensor = torch.tensor(gov_action, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
            gov_action_tensor = gov_action_tensor.repeat(self.envs.households.households_n, 1)

        # === 1. Running mean/std update ===
        if self.step_counter < self.max_warmup_steps:
            obs_np = obs_tensor.detach().cpu().numpy()
            self.state_rms.update(obs_np)
            self.step_counter += 1

        # === 2. Inject normalizer after warmup ===
        if (not self.normalizer_applied) and (self.step_counter >= self.max_warmup_steps):
            self.mf_actor.set_normalizer(self.state_rms.mean, self.state_rms.std)
            self.normalizer_applied = True

        # action = self.mf_actor((private_obs_tensor +1)/2, gov_action_tensor).detach().cpu().numpy()
        action = self.mf_actor(private_obs_tensor, gov_action_tensor).detach().cpu().numpy()
        sigma = 0.01
        action = action + sigma * np.random.randn(self.envs.households.action_space.shape[1])
        action = (action+1)/2
        action[:,0] = action[:,0] * 0.5 + 0.5
        return action
    

    def train(self, transitions, other_agent):
        # smaple batch of samples from the replay buffer
        agent_data = transitions
        global_obses=torch.tensor(np.array(agent_data['obs_dict']['government']), dtype=torch.float32).to(self.device)
        private_obses=torch.tensor(np.array(agent_data['obs_dict']['households']), dtype=torch.float32).to(self.device)
        gov_actions=torch.tensor(np.array(agent_data['action_dict']['government']), dtype=torch.float32).to(self.device)
        house_actions=torch.tensor(np.array(agent_data['action_dict']['households']), dtype=torch.float32).to(self.device)
        gov_rewards=torch.tensor(np.array(agent_data['reward_dict']['government']), dtype=torch.float32).to(self.device)
        house_rewards=torch.tensor(np.array(agent_data['reward_dict']['households']), dtype=torch.float32).to(self.device)
        next_global_obses=torch.tensor(np.array(agent_data['next_obs_dict']['government']), dtype=torch.float32).to(self.device)
        next_private_obses=torch.tensor(np.array(agent_data['next_obs_dict']['households']), dtype=torch.float32).to(self.device)
        inverse_dones = torch.tensor([x - 1 for x in agent_data['done']], dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            next_gov_actions = other_agent.get_action(obs_tensor=global_obses,agent_name="government")
            next_gov_actions = torch.tensor(next_gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1).repeat(1, self.envs.households.households_n,1)
            next_house_actions_ = self.mf_actor_target(next_private_obses, next_gov_actions)
            # mean field Mt
            next_mean_house_actions_ = torch.mean(next_house_actions_, dim=1).unsqueeze(1).repeat(1,self.envs.households.households_n,1)
            next_mean_house_state = torch.mean(next_private_obses, dim=1).unsqueeze(1).repeat(1,self.envs.households.households_n,1)
            q_next_value = self.mf_critic_target(next_private_obses, next_gov_actions, next_house_actions_,  next_mean_house_state, next_mean_house_actions_)
            target_q_value = house_rewards + inverse_dones.unsqueeze(1).repeat(1, self.envs.households.households_n,1) * self.args.gamma * q_next_value
        # the real q value
        mean_house_actions = torch.mean(house_actions, dim=1).unsqueeze(1).repeat(1,self.envs.households.households_n,1)
        mean_house_state = torch.mean(private_obses, dim=1).unsqueeze(1).repeat(1,self.envs.households.households_n,1)
        q_value = self.mf_critic(private_obses, gov_actions.unsqueeze(1).repeat(1, self.envs.households.households_n, 1), house_actions, mean_house_state, mean_house_actions)
        critic_loss = (q_value - target_q_value).pow(2).mean()
        self.mf_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.mf_critic_optimizer.step()
        # the actor loss
        house_actions_ = self.mf_actor(private_obses, gov_actions.unsqueeze(1).repeat(1, self.envs.households.households_n, 1))
        mean_house_actions_ = torch.mean(house_actions_, dim=1).unsqueeze(1).repeat(1,self.envs.households.households_n,1)
        actor_loss = - self.mf_critic(private_obses, gov_actions.unsqueeze(1).repeat(1, self.envs.households.households_n, 1), house_actions_, mean_house_state,mean_house_actions_ ).mean()
        
        self.mf_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.mf_actor_optimizer.step()

        self._soft_update_target_network(self.mf_actor_target, self.mf_actor)
        self._soft_update_target_network(self.mf_critic_target, self.mf_critic)
        
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        return actor_loss.item(), critic_loss.item()
    
    # soft update the target network...
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - 0.95) * param.data + 0.95 * target_param.data)
    
    def save(self, dir_path):
        torch.save(self.mf_actor.state_dict(), str(dir_path) + '/bimf_house_actor.pt')
        torch.save(self.mf_critic.state_dict(), str(dir_path) + '/bimf_house_critic.pt')
        torch.save(self.mf_actor_target.state_dict(), str(dir_path) + '/bimf_house_actor_target.pt')
        torch.save(self.mf_critic_target.state_dict(), str(dir_path) + '/bimf_house_critic_target.pt')

    def load(self, dir_path):
        self.mf_actor.load_state_dict(torch.load(dir_path, map_location=torch.device(self.device), weights_only=True))