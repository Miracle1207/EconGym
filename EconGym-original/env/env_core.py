import copy
import numpy as np
from gymnasium.spaces import Box
from pathlib import Path
ROOT_PATH = str(Path(__file__).resolve().parent.parent)
from entities.households import Household
from entities.government import Government
from entities.market import Market
from entities.bank import Bank
from .set_observation import EconObservations


class EconomicSociety:
    def __init__(self, cfg):
        super().__init__()
        self.__dict__.update(cfg['env_core'])  # update cfg to self
        self.__dict__.update(cfg['env_core'])  # update cfg to self
        self.agents = {'households': None, 'government': {}, 'market': None, 'bank': None}

        for entity_arg in cfg['Entities']:
            entity_name = entity_arg['entity_name']
            entity_args = entity_arg['entity_args']
    
            # Add new entity instance to the appropriate list in agents
            if entity_name == 'households':
                self.agents[entity_name] = Household(entity_args)
            elif entity_name == 'government':
                type = entity_args.params.type
                self.agents[entity_name][type] = Government(entity_args)
            elif entity_name == 'market':
                self.agents[entity_name] = Market(entity_args)
            elif entity_name == 'bank':
                self.agents[entity_name] = Bank(entity_args)

        self.households = self.agents['households']
        self.government = self.agents['government']
        self.market = self.agents['market']
        self.bank = self.agents['bank']
        
        main_key = next(iter(self.government))
        self.main_gov = self.government[main_key]
        
        observations_dict = self.reset()

        for name, entity in self.agents.items():
            obs = observations_dict[name]  # Get the corresponding observations for the entity
            # Check if the entity is a dict (like for government agents)
            if isinstance(entity, dict):
                for sub_type, sub_entity in entity.items():
                    sub_entity.observation_space = Box(
                        low=-np.inf, high=np.inf, shape=(obs[sub_type].shape[-1],), dtype=np.float32
                    )
            else:
                # If the entity is not a dict, directly assign the observation space
                entity.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs.shape[-1],), dtype=np.float32 )
        
        # Expand the action_space based on the number of firms.
        if self.market.firm_n > 1:
            self.expand_action_space(firm_n=self.market.firm_n)
        
        self.display_mode = False

    def expand_action_space(self, firm_n):
        '''
        Expand action space for different economic roles:
        - For households:
            1) Select the index of the firm they work for. (array(1) in {0,..., firm_n-1})
            2) Choose the consumption amount for each of the firm_n firms' goods. (array([firm_n,]))

        - For fiscal government (type="tax"):
            Add government spending action for each firm:
            1) Gt_prob_j represents the proportion of government spending allocated to firm j.

        - Other economic roles' actions remain unaffected.
        '''
        # Expand action dimension for households:
        # 1: Firm selection (scalar), firm_n: Consumption distribution over firms.
        self.households.action_dim += 1 + firm_n
    
        N = self.households.households_n
        new_shape = (N, self.households.action_dim)
        
        # Update action space for households
        action_max = float(self.households.action_space.high_repr)
        action_min = float(self.households.action_space.low_repr)
        self.households.action_space = Box(low=action_min, high=action_max, shape=new_shape, dtype=np.float32)
    
        # For fiscal government (type="tax"), expand the action space to include spending proportions for each firm
        if "tax" in self.government:
            action_max = float(self.government['tax'].action_space.high_repr)
            action_min = float(self.government['tax'].action_space.low_repr)
            self.government['tax'].action_dim += firm_n
            self.government['tax'].action_space = Box(low=action_min, high=action_max, shape=(self.government['tax'].action_dim,),
                                                      dtype=np.float32)

    @property
    def action_spaces(self):
        """Return a dictionary of action spaces for each agent."""
        action_spaces = {
            self.households.name: self.households.action_space,
            self.market.name: self.market.action_space,
            self.bank.name: self.bank.action_space
        }
    
        # Add action spaces for all government agents, using type as a unique identifier
        for gov_type, gov_agent in self.government.items():
            action_spaces[f"{gov_agent.name}_{gov_type}"] = gov_agent.action_space
    
        return action_spaces

    @property
    def observation_spaces(self):
        """Return a dictionary of observation spaces for each agent."""
        observation_spaces = {
            self.households.name: self.households.observation_space,
            self.market.name: self.market.observation_space,
            self.bank.name: self.bank.observation_space
        }
    
        # Add observation spaces for all government agents, using type as a unique identifier
        for gov_type, gov_agent in self.government.items():
            observation_spaces[f"{gov_agent.name}_{gov_type}"] = gov_agent.observation_space
    
        return observation_spaces

    def action_wrapper(self, action_dict):
        processed_action_dict = {}
    
        for agent_name, agent_action in action_dict.items():
            agent = self.agents[agent_name]

            if isinstance(agent, dict):
                processed_sub_agents = {}
                for sub_agent_name, sub_agent in agent.items():
                    processed_sub_agents[sub_agent_name] = self.check_agent_action(sub_agent, agent_action[sub_agent_name], sub_agent_name)
                processed_action_dict[agent_name] = processed_sub_agents
        
            else:
                processed_action_dict[agent_name] = self.check_agent_action(agent, agent_action, agent_name)
    
        return processed_action_dict

    @staticmethod
    def check_agent_action(agent, agent_action, agent_name):
        """
        Check the agent's actions.
        1. Verify if the dimensions of the provided actions match the dimensions set by the environment.
        2. Ensure the values of the provided actions are within the range defined by the environment.
        """
        
        expected_dim = agent.action_dim
        current_action_dim = getattr(agent_action, 'shape', (0,))[-1]
        
        if current_action_dim == 0 and expected_dim == 0:
            return None
        elif current_action_dim != expected_dim:
            raise ValueError(
                f"Invalid actions for {agent_name}. Expected shape: {expected_dim}, Found: {current_action_dim}"
            )
    
        expected_action_min = agent.real_action_min
        expected_action_max = agent.real_action_max
        
        # When the number of firms (firm_n > 1) increases, causing the action dimension of households and tax_gov to expand, we supplement expected_action_min/max.
        fill_in_len = expected_dim - len(expected_action_min)

        if fill_in_len > 0:
            expected_action_min = np.pad(expected_action_min, (0, fill_in_len), constant_values=0)
            expected_action_max = np.pad(expected_action_max, (0, fill_in_len), constant_values=1)

        return np.clip(agent_action, expected_action_min, expected_action_max)

    def get_actions(self, action_dict):
        """Get and process actions for all agents."""
        valid_action_dict = self.is_valid(action_dict)
        processed_action_dict = self.action_wrapper(valid_action_dict)
        if isinstance(self.government, dict):
            for gov_type, gov_agent in self.government.items():
                gov_agent.get_action(processed_action_dict[self.government[gov_type].name][gov_type], firm_n=self.market.firm_n)

        self.bank.get_action(processed_action_dict[self.bank.name], central_bank_exist=("central_bank" in self.government))
        self.market.get_action(processed_action_dict[self.market.name])
        self.households.get_action(processed_action_dict[self.households.name], firm_n=self.market.firm_n)

    def step(self, action_dict, t=None):
        """Perform a simulation step given the actions."""
    
        # === Phase 1: Agents Take Action ===
        self.get_actions(action_dict)
    
        # === Phase 2: Entities Step Forward ===
        self.market.step(self)
        self.households.step(self, t)
        for _, gov_agent in self.government.items():
            gov_agent.step(self)
        self.bank.step(self)
    
        # === Phase 3: Update Environment State ===
        self.update_metrics()
        self.step_cnt += 1
        self.done = self.is_terminal()
    
        # === Phase 4: Observation & NaN Check ===
        next_obs = EconObservations(self).get_obs()

        # === Phase 5: Final Updates ===
        self.last_price_index = copy.copy(self.price_index)

        return (
            next_obs,
            self.rewards,
            self.done,
        )

    def update_metrics(self):
        """Update evaluation metrics such as Gini coefficients, price index, and rewards."""
        # Compute Gini coefficients
        self.wealth_gini = self.gini_coef(self.households.post_asset)
        self.income_gini = self.gini_coef(self.households.post_income)

        # Compute Price Index
        self.inflation_rate, self.price_index = self.market.compute_inflation_rate(self.market.price,
                                                                                   self.last_price_index)
        
        market_supply = self.market.Yt_j
        market_demand = self.households.final_consumption.sum(axis=0)[:, np.newaxis] + self.main_gov.gov_spending
        
        self.real_deals = np.minimum(market_supply, market_demand)
        
        households_reward = self.households.get_reward()
        
        if isinstance(self.government, dict):
            government_reward = {}
            for gov_type, gov_agent in self.government.items():
                government_reward[gov_type] = gov_agent.get_reward(self)
        else:
            government_reward = self.government.get_reward(self)

        firm_reward = self.market.get_reward(self)
        bank_reward = self.bank.get_reward()

        self.rewards = {
            gov_agent.name: government_reward,
            self.households.name: households_reward,
            self.market.name: firm_reward,
            self.bank.name: bank_reward,
        }
        
    def reset(self, **custom_cfg):
        """Reset the simulation to the initial state."""
        self.step_cnt = 0
        for _, gov_agent in self.government.items():
            gov_agent.reset(households_n=self.households.households_n, firm_n=self.market.firm_n)
        
        self.households.reset()
        self.bank.reset(households_at=self.households.at)

        self.market.reset(households_n=self.households.households_n, GDP=gov_agent.GDP,
                          households_at=self.households.at, real_debt_rate=gov_agent.real_debt_rate)

        # self.last_price_index = 1
        self.last_price_index = self.market.calculate_price_index(self.market.price)
        self.ini_income_gini = self.gini_coef(self.households.income)
        self.ini_wealth_gini = self.gini_coef(self.households.at)
        self.done = False
        self.display_mode = False

        return EconObservations(self).get_obs()

    def is_terminal(self):
        """
        Check if the simulation has reached a terminal state.
        """

        gini_invalid = self.wealth_gini >= 1 or self.income_gini >= 1 or np.isnan(self.wealth_gini) or np.isnan(self.income_gini)
        data_nan = any(self.recursive_decompose_dict(self.rewards, lambda a: np.isnan(a)))
        
        episode_completed = self.step_cnt >= self.episode_length
        agent_terminal = any(self.recursive_decompose_dict(self.agents, lambda a: a.is_terminal()))
        # if (gini_invalid or data_nan or episode_completed or agent_terminal) and self.step_cnt == 2:
        #     print(self.recursive_decompose_dict(self.agents, lambda a: a.is_terminal()))
        #     print(1)
        
        return gini_invalid or data_nan or episode_completed or agent_terminal

    def recursive_decompose_dict(self, input_dict, func):
        
        results = []
        if isinstance(input_dict, dict):
            for key_i in input_dict:
                current_item = input_dict[key_i]
                if isinstance(current_item, dict):
                    for key_j in current_item:
                        results.append(func(current_item[key_j]))
                elif isinstance(current_item, np.ndarray):
                    current_result = func(current_item).any()
                    results.append(current_result)
                else:
                    results.append(func(current_item))
        return results

    def is_valid(self, action_dict):
        """Validate the actions provided by the agents."""
        expected_agents = set(self.agents.keys())
        received_agents = set(action_dict.keys())
        if expected_agents == received_agents:
            return action_dict
        else:
            raise ValueError(
                "Invalid actions. Expected agents: {}, Received agents: {}".format(expected_agents, received_agents))

    def gini_coef(self, values):
        """Fast Gini coefficient computation for 1D or column vector values."""
        
        if self.households.households_n == 0:  # empty array check
            return 0.
        
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.flatten()

        values = np.sort(values + 1e-7)  # Avoid division by zero
        values = values / np.max(values)
        n = values.size
        cum_weights = np.arange(1, n + 1)

        numerator = np.dot(cum_weights, values)
        denominator = np.sum(values)

        return (2 * numerator - (n + 1) * denominator) / (n * denominator)


    def render(self):
        pass

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


