import numpy as np
import torch
from torch import optim
import os, sys

sys.path.append(os.path.abspath('../..'))
from agents.models import mlp_net, MFActor, MFCritic
import copy
import pickle
from utils.utils import sync_networks, sync_grads
from torch.optim.lr_scheduler import LambdaLR

class bi_mfrl_agent():  ## ddpg_based
    def __init__(self, envs, args):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        obs_dims = self.envs.households.observation_space.shape[0]
        # start to build the network.
        self.mf_actor = MFActor(input_dims=self.envs.households.observation_space.shape[0],
                                gov_action_dim=self.envs.government.action_space.shape[0],
                                action_dims=self.envs.households.action_space.shape[1],
                                      hidden_size=128)  # Stackelberg mean field game
        self.mf_critic = MFCritic(obs_dims=self.envs.households.observation_space.shape[0],
                                  gov_action_dim=self.envs.government.action_space.shape[0],
                                  action_dim=self.envs.households.action_space.shape[1],
                                  mean_state_dim=self.envs.households.observation_space.shape[0],
                                  mean_action_dim=self.envs.households.action_space.shape[1],
                                  hidden_size=128)
        # sync the weights across the mpi
        sync_networks(self.mf_actor)
        sync_networks(self.mf_critic)
        if self.args.bc == True:#是否有real data预训练？
            self.mf_actor.load_state_dict(torch.load("agents/real_data/2024_01_04_16_56_mfrl_trained_model.pth", weights_only=True))
            self.mf_actor_optimizer = optim.Adam(self.mf_actor.parameters(), 1e-6, eps=self.args.eps)
        else:
            self.mf_actor_optimizer = optim.Adam(self.mf_actor.parameters(), self.args.p_lr, eps=self.args.eps)
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

    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action, agent_name="household"):
        if len(private_obs_tensor.shape) == 3:
            gov_action_tensor = gov_action.unsqueeze(1).repeat(1,self.args.n_households,1)
        else:
            gov_action_tensor = torch.tensor(gov_action, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
            gov_action_tensor = gov_action_tensor.repeat(self.args.n_households, 1)
        # action = self.mf_actor((private_obs_tensor +1)/2, gov_action_tensor).detach().cpu().numpy()
        action = self.mf_actor(private_obs_tensor, gov_action_tensor).detach().cpu().numpy()
        sigma = 0.01
        action = action + sigma * np.random.randn(self.envs.households.action_space.shape[1])
        action = (action+1)/2
        action[:,0] = action[:,0] * 0.5 + 0.5
        return action, np.mean(action, axis=-2)
    

    def train(self, transitions, other_agent):
        # smaple batch of samples from the replay buffer
        global_obses, private_obses, gov_actions, house_actions, gov_rewards,\
        house_rewards, next_global_obses, next_private_obses, dones, past_mean_house_actions = transitions
        global_obses = torch.tensor(global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        private_obses = torch.tensor(private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_actions = torch.tensor(house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        # mean_house_actions = torch.tensor(past_mean_house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_rewards = torch.tensor(house_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
      
        with torch.no_grad():
            next_gov_actions = other_agent.get_action(global_obs_tensor=next_global_obses,private_obs_tensor=next_private_obses,agent_name="government")
            next_gov_actions = torch.tensor(next_gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1).repeat(1, self.args.n_households,1)
            next_house_actions_ = self.mf_actor_target(next_private_obses, next_gov_actions)
            # mean field Mt
            next_mean_house_actions_ = torch.mean(next_house_actions_, dim=1).unsqueeze(1).repeat(1,self.args.n_households,1)
            next_mean_house_state = torch.mean(next_private_obses, dim=1).unsqueeze(1).repeat(1,self.args.n_households,1)
            q_next_value = self.mf_critic_target(next_private_obses, next_gov_actions, next_house_actions_,  next_mean_house_state, next_mean_house_actions_)
            target_q_value = house_rewards + inverse_dones.unsqueeze(1).repeat(1, self.args.n_households,1) * self.args.gamma * q_next_value
        # the real q value
        mean_house_actions = torch.mean(house_actions, dim=1).unsqueeze(1).repeat(1,self.args.n_households,1)
        mean_house_state = torch.mean(private_obses, dim=1).unsqueeze(1).repeat(1,self.args.n_households,1)
        q_value = self.mf_critic(private_obses, gov_actions.unsqueeze(1).repeat(1, self.args.n_households, 1), house_actions, mean_house_state, mean_house_actions)
        critic_loss = (q_value - target_q_value).pow(2).mean()
        self.mf_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.mf_critic_optimizer.step()
        # the actor loss
        house_actions_ = self.mf_actor(private_obses, gov_actions.unsqueeze(1).repeat(1, self.args.n_households, 1))
        mean_house_actions_ = torch.mean(house_actions_, dim=1).unsqueeze(1).repeat(1,self.args.n_households,1)
        actor_loss = - self.mf_critic(private_obses, gov_actions.unsqueeze(1).repeat(1, self.args.n_households, 1), house_actions_, mean_house_state,mean_house_actions_ ).mean()
        
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
    def load(self, dir_path):
        self.mf_actor.load_state_dict(torch.load(dir_path, map_location=torch.device(self.device), weights_only=True))