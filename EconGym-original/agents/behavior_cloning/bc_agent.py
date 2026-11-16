import os

import copy
import torch
import numpy as np
import pandas as pd
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


class bc_agent:
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
            self.bc_obs_dim = 2  # [education, wealth]
        elif "OLG" in self.type:
            self.bc_obs_dim = 3  # [education, wealth, age]
        else:
            raise ValueError(f"Agent type Error: No {self.type} in EconGym.")
        
        if "invest" in self.type:
            self.bc_action_dim = 3
        else:
            self.bc_action_dim = 2
        self.net = mlp_net(state_dim=self.bc_obs_dim, num_actions=self.bc_action_dim).to(self.device)
        # Define the mapping of types to model paths
        model_paths = {
            "ramsey": "agents/behavior_cloning/trained_models/ramsey_bc_net.pt",
            "ramsey_risk_invest": "agents/behavior_cloning/trained_models/risky_investment/ramsey_risk_invest_bc_net.pt",
            "OLG": "agents/behavior_cloning/trained_models/OLG_bc_net.pt",
            "OLG_risk_invest": "agents/behavior_cloning/trained_models/risky_investment/OLG_risk_invest_bc_net.pt"
        }

        if self.args.bc_test:  # if False, get actions from real data, and trained via BC; if True, get action from trained BC policy.
            if type not in model_paths:
                raise ValueError(
                    f"Invalid Households type: {type}. Supported types are: 'OLG', 'OLG_risk_invest', 'ramsey', and 'ramsey_risk_invest'."
                )
            model_path = model_paths[type]
            self.net.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            # Load real data
            self.real_data = self.get_real_data(age_limit=None)  # Adjust age_limit if needed

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.p_lr, eps=1e-5)
        lambda_function = lambda epoch: 0.97 ** (epoch // 10)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_function)
        self.on_policy = True
        

    def train(self, transition_dict):
        """
        Train the agent using Behavior Cloning.
        transition_dict: Dictionary containing the transitions with expert actions.
        Returns: The loss for the current training step.
        """
        if not self.args.bc_test:
            house_obses = [torch.tensor(obs[self.agent_name][:, -self.bc_obs_dim:], dtype=torch.float32) for obs in transition_dict['obs_dict']]
            house_actions = [torch.tensor(obs[self.agent_name], dtype=torch.float32) for obs in transition_dict['action_dict']]

            # Pad sequences for household data
            obs_tensor = rnn_utils.pad_sequence(house_obses, batch_first=True).to(self.device)
            expert_action_tensor = rnn_utils.pad_sequence(house_actions, batch_first=True).to(self.device)

            _, pi = self.net(obs_tensor)  # pi contains (mu, std)
            mu, _ = pi
            bc_loss = torch.sum(F.huber_loss(mu, expert_action_tensor[:, :, :self.envs.households.action_dim]))

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

    def get_action(self, full_obs_tensor):
        """
        Get expert actions by finding the closest observations in real data or running test-time policy.

        Args:
            obs_tensor: ASSET, EDUC

        Returns:
            np.ndarray: Array of shape (N, action_dim) containing household actions.
        """
        if self.agent_name != "households":
            raise ValueError("AgentError: Behavior cloning method is suitable for household agents only.")
        
        obs_tensor = full_obs_tensor[:, -self.bc_obs_dim:]   # Each household's private obs: [education, wealth, age (optional)]
        
        if self.args.bc_test:
            _, pi = self.net(obs_tensor)
            mu, sigma = pi
            action_dist = torch.distributions.Normal(mu, 0.01 * sigma)
            action = action_dist.sample().cpu().numpy()
            
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
        df = pd.read_csv(os.path.join(os.path.abspath(''), "agents/data/advanced_scfp2022_1110.csv"))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if age_limit is not None:
            df = df[df['AGE'] == age_limit]
            if df.empty:
                raise ValueError(f"No data available for individuals aged {age_limit}.")

        columns = ['EDUC', 'ASSET', 'INCOME', 'AGE']
        data = [df[col].values for col in columns]

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
        # Extract observation and action components from real_data
        real_obs = np.stack(real_data[:self.bc_obs_dim],axis=1)  # Shape: (D, obs_dim) for ASSET, EDUC, AGE(optional)
        real_actions = np.stack(real_data[4:], axis=1)  # Shape: (D, 3) for LF, consumption_p, invest_p

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
        norm_private_obs = (private_obs_tensor - obs_mean) / obs_std  # Shape: (N, obs_dim)

        diff = norm_private_obs.unsqueeze(1) - norm_real_obs.unsqueeze(0)
        distances = torch.norm(diff, dim=2)  # Shape: (N, D)

        # Find the index of the closest observation for each private_obs
        nearest_idx = torch.argmin(distances, dim=1)  # Shape: (N,)

        # Select the corresponding actions
        expert_actions = real_actions_tensor[nearest_idx]  # Shape: (N, 3)

        return expert_actions

    def save(self, dir_path):
        torch.save(self.net.state_dict(), str(dir_path) + f'/{self.type}_bc_net.pt')
