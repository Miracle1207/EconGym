import os
import copy
import torch
import numpy as np
import pandas as pd
from agents.rl.models import MFActor
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

class bc_agent_stackelberg:
    def __init__(self, envs, args, agent_name="households", type=None):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        self.type = type
        if agent_name == "households":
            self.obs_dim = self.envs.households.observation_space.shape[0]
            self.action_dim = self.envs.households.action_space.shape[1]
        else:
            raise ValueError(
                "Due to data limitations, the current version only supports behavior cloning policy for 'households'. "
                "You can train behavior cloning policies for other economic roles based on local data."
            )
        # if use the cuda...
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # special setting for BC: The real dataset only contains partial observations
        if "ramsey" in self.type:
            self.bc_obs_dim = 10  # [education, wealth]
        elif "OLG" in self.type:
            self.bc_obs_dim = 11  # [education, wealth, age]
        else:
            raise ValueError(f"Agent type Error: No {self.type} in EconGym.")
        
        if "invest" in self.type:
            self.bc_action_dim = 3
        else:
            self.bc_action_dim = 2
        self.net = MFActor(input_dims=self.envs.households.observation_space.shape[0],
                                leader_action_dim=2,
                                action_dims=self.envs.households.action_space.shape[1],
                                      hidden_size=128)
        # Define the mapping of types to model paths
        model_paths = {
            "ramsey": "agents/behavior_cloning/trained_models/ramsey_bc_binet.pt"
        }

        if self.args.bc_test_stackelberg:  # if False, get actions from real data, and trained via BC; if True, get action from trained BC policy.
            if type not in model_paths:
                raise ValueError(
                    f"Invalid Households type: {type}. Supported types are: 'ramsey'."
                )
            model_path = model_paths[type]
            checkpoint=torch.load(model_path, weights_only=True)
            checkpoint.pop('std')
            checkpoint.pop('mean')
            self.net.load_state_dict(checkpoint)
        else:
            # Load real data
            self.real_data = self.get_real_data(age_limit=None)  # Adjust age_limit if needed
        
        self.state_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.normalizer_applied=False

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.p_lr, eps=1e-5)
        lambda_function = lambda epoch: 0.97 ** (epoch // 10)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_function)
        self.on_policy = True
        

    def train(self, transition_dict,other_agent=None):
        """
        Train the agent using Behavior Cloning With Stackelberg.
        transition_dict: Dictionary containing the transitions with expert actions.
        Returns: The loss for the current training step.
        """
        if not self.args.bc_test:      
            obs_tensor=np.array(transition_dict['obs_dict'])
            expert_action_tensor=np.array(transition_dict['action_dict']['households'])
            gov_actions=np.array(transition_dict['action_dict']['government'])

            gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
            obs_tensor= torch.tensor(obs_tensor, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
            expert_action_tensor=torch.tensor(expert_action_tensor, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
            '''
            house_obses = np.array(transition_dict['obs_dict'])
            house_actions = np.array(transition_dict['action_dict'])
            '''
            
            action= self.net(obs_tensor,gov_actions)
            bc_loss = F.huber_loss(action,expert_action_tensor).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)  # Gradient clipping
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            return bc_loss, torch.tensor(0.)
        else:
            return torch.tensor(0.), torch.tensor(0.)

    def get_action(self, obs_tensor,gov_action=None,agent_name='households'):
        """
        Get expert actions by finding the closest observations in real data or running test-time policy.

        Args:
            obs_tensor: ASSET, EDUC

        Returns:
            np.ndarray: Array of shape (N, action_dim) containing household actions.
        """
        if self.agent_name != "households":
            raise ValueError("AgentError: Behavior cloning method is suitable for household agents only.")
        
        obs_tensor = obs_tensor[:, -self.bc_obs_dim:]   # Each household's private obs: [education, wealth, age (optional)]
        obs_np = obs_tensor.detach().cpu().numpy()
        gov_action=torch.tensor(gov_action,dtype=torch.float32).unsqueeze(0).repeat(obs_tensor.shape[0],1)
        self.state_rms.update(obs_np)

        if not self.normalizer_applied:
            self.net.set_normalizer(self.state_rms.mean, self.state_rms.std)
            self.normalizer_applied = True

        if self.args.bc_test_stackelberg:
            action = self.net(obs_tensor,gov_action)
            action = action.detach().numpy()
            
        else:
            # Use expert actions directly
            expert_actions = self.find_expert_action(obs_tensor, self.real_data)  # (N, ?)
            action = expert_actions.cpu().numpy()

        # Ensure output matches households.action_dim
        if self.envs.market.firm_n == 1:
            return action[:, :self.bc_action_dim]
        elif self.envs.market.firm_n > 1:
            fill_dim = self.envs.market.firm_n + 1  # +1 for work firm choice, +firm_n for consumption shares
            real_action_dim = self.action_dim - fill_dim
            n_agents = action.shape[0]
            # use a random value in [0, 1) to represent firm choice preference
            firm_choice = np.random.rand(n_agents, 1)
            # Assign uniform consumption distribution across all firms
            fill_values = np.ones((n_agents, self.envs.market.firm_n)) / self.envs.market.firm_n
            # Concatenate: [existing action part] + [firm choice] + [consumption share vector]
            action = np.concatenate([action[:, :real_action_dim], firm_choice, fill_values], axis=1)
            return action

        else:
            raise ValueError(
                f"Wrong firm number {self.envs.market.firm_n}."
            )

    def get_real_data(self, age_limit=None):
        global action_windows
        df = pd.read_csv(os.path.join(os.path.abspath(''), "agents/data/advanced_scfp2022_1110_extension.csv"))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if age_limit is not None:
            df = df[df['AGE'] == age_limit]
            if df.empty:
                raise ValueError(f"No data available for individuals aged {age_limit}.")

        columns = ['TOP10_MEAN_ASSET', 'TOP10_MEAN_EDUC', 'BOT50_MEAN_ASSET',
       'BOT50_MEAN_EDUC', 'WAGE_RATE','PRICE_LEVEL','DEPOSIT_RATE','LENDING_RATE','EDUC', 'ASSET','INCOME', 'AGE']
        data = [df[col].values for col in columns]
        action_windows=len(data)

        consumption_p = (df['FOODHOME'].values + df['FOODAWAY'].values + df['FOODDELV'].values +
                         df['RENT'].values + df['TPAY'].values + 0.0001) / (df['ASSET'].values + 0.0001)
        invest_p = df['FIN'].values / (df['ASSET'].values + 0.0001)

        # Append in the new order: 1 - consumption_p, then LF, then invest_p
        consumption_p_cliped = np.clip(consumption_p, 0, 1)
        data.append(1 - consumption_p_cliped)  # saving ratio
        data.append(df['LF'].values)
        data.append(invest_p)

        return data

    def find_expert_action(self, private_obs_tensor, real_data):
        """
        Find expert actions from real_data that correspond to the closest observations to private_obs_tensor.

        Args:
            private_obs_tensor (torch.Tensor): Tensor of shape (N, m) where m=4 (ASSET, EDUC, INCOME, AGE).
            real_data (list): List of arrays from get_real_data, containing [ASSET, EDUC, INCOME, AGE, LF, consumption_p, invest_p].

        Returns:
            torch.Tensor: Tensor of shape (N, 3) containing expert actions [LF, consumption_p, invest_p].
        """
        indice=[0,1,2,3,8,9]
        # Extract observation and action components from real_data
        real_obs = np.stack(real_data,axis=1)  # Shape: (D, obs_dim) for ASSET, EDUC, AGE(optional)
        real_obs=real_obs[:,indice]
        
        real_actions = np.stack(real_data, axis=1)[:, [4, 5, 6]] # Shape: (D, 3) for LF, consumption_p, invest_p
        # Convert to tensors
        
        real_obs_tensor = torch.tensor(real_obs, dtype=torch.float32).to(
            private_obs_tensor.device)  # Shape: (D, obs_dim)
        real_actions_tensor = torch.tensor(real_actions, dtype=torch.float32).to(
            private_obs_tensor.device)  # Shape: (D, 3)

        obs_mean = real_obs_tensor.mean(dim=0, keepdim=True)  # Shape: (1, obs_dim)
        obs_std = real_obs_tensor.std(dim=0,
                                      keepdim=True) + 1e-6  # Shape: (1, 4), add small epsilon to avoid division by zero

        # Normalize real observations and private_obs_tensor
        norm_real_obs = (real_obs_tensor - obs_mean) / obs_std  # Shape: (D, obs_dim)
        norm_private_obs = (private_obs_tensor[:,indice] - obs_mean) / obs_std  # Shape: (N, obs_dim)

        # Use torch.cdist for optimized exact distance calculation
        distances = torch.cdist(norm_private_obs, norm_real_obs, p=2)  # Shape: (N, D)

        # Find the index of the closest observation for each private_obs
        nearest_idx = torch.argmin(distances, dim=1)  # Shape: (N,)

        # Select the corresponding actions
        expert_actions = real_actions_tensor[nearest_idx]  # Shape: (N, 3)

        return expert_actions
    
    def save(self, dir_path):
        torch.save(self.net.state_dict(), str(dir_path) + f'/{self.type}_bc_binet.pt')