import copy
import numpy as np
import torch
from agents.rl.models import PolicyNet, BiQValueNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

class bi_ddpg_agent:
    def __init__(self, envs, args, agent_name="government"):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name

        self.obs_dim = self.envs.government.observation_space.shape[0]
        self.action_dim = self.envs.government.action_space.shape[0]
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
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.q_lr)
        # lambda_function = lambda epoch: 0.95 ** (epoch // (self.args.update_cycles/2))
        lambda_function = lambda epoch: 0.95 ** (epoch // (35*self.args.update_cycles))

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lambda_function)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lambda_function)
        self.on_policy = False
    
    def train(self, transitions, other_agent):
        global_obses, private_obses, gov_actions, house_actions, gov_rewards,\
        house_rewards, next_global_obses, next_private_obses, dones, past_mean_house_actions = transitions
        private_obses = torch.tensor(private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        global_obses = torch.tensor(global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_actions = torch.tensor(house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        mean_house_actions = torch.tensor(past_mean_house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        house_rewards = torch.tensor(house_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)

        next_gov_action = self.target_actor(next_global_obses)
        next_house_action, next_mean_house_action = other_agent.get_action(global_obs_tensor=next_global_obses,private_obs_tensor=next_private_obses,
                                                   gov_action=next_gov_action, agent_name="household")
        next_mean_house_action = torch.tensor(next_mean_house_action, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_mean_house_state = torch.mean(next_private_obses, dim=1)
        next_q_values = self.target_critic(next_global_obses, next_gov_action, next_mean_house_state, next_mean_house_action)
        q_targets = gov_rewards + self.args.gamma * next_q_values * inverse_dones
        mean_house_state = torch.mean(private_obses, dim=1)
        critic_loss = torch.mean(F.mse_loss(self.critic(global_obses, gov_actions, mean_house_state, mean_house_actions), q_targets))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        gov_action_ = self.actor(global_obses)
        _, mean_house_action_ = other_agent.get_action(global_obs_tensor=global_obses,private_obs_tensor=private_obses,gov_action=gov_action_,agent_name="household")
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
            
    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, agent_name="household"):
        if self.agent_name == "government":
            obs_tensor = global_obs_tensor.reshape(-1, self.obs_dim)
        elif self.agent_name == "household":
            obs_tensor = private_obs_tensor
        else:
            obs_tensor = None
        action = self.actor(obs_tensor).detach().cpu().numpy()
        sigma = 0.01
        action = action + sigma * np.random.randn(self.action_dim)
        # action = (action + 1)/2
        if len(global_obs_tensor.shape)>1:
            return action
        else:
            return action.flatten()


    def save(self, dir_path):
        torch.save(self.actor.state_dict(), str(dir_path) + '/bi_ddpg_net.pt')
    
    def load(self, dir_path):
        self.actor.load_state_dict(torch.load(dir_path, map_location=torch.device(self.device), weights_only=True))