import math
import pygame
import sys
import os
import copy
import numpy as np
from gym.spaces import Box

from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parent.parent)
from entities.household import Household
from entities.government import Government
from entities.market import Market
from entities.bank import Bank
from entities.central_bank_gov import CentralBankGovernment
from entities.pension_gov import PensionGovernment
from entities.tax_gov import TaxGovernment


class EconomicSociety:
    """Wealth Distribution Economic Society.

    This class models an economic society with households, government, and market clearing.
    """

    def __init__(self, cfg, invalid_gov=None):
        super().__init__()
        self.__dict__.update(cfg['env_core'])  # update cfg to self
        self.__dict__.update(cfg['env_core'])  # update cfg to self
        self.agents = {}
        

        if invalid_gov is not None:
            filtered_entities = [
                entity for entity in cfg['Entities']
                if invalid_gov not in entity.get('entity_name')
            ]
            cfg['Entities'] = filtered_entities

        for entity_arg in cfg['Entities']:
            entity_name = entity_arg['entity_name']
            entity_args = entity_arg['entity_args']
            if entity_name == 'household':
                self.households = Household(entity_args)
                self.agents["household"] = self.households
            elif entity_name == 'government':
                self.government = Government(entity_args)
                self.agents["government"] = self.government
            elif entity_name == 'market':
                self.market = Market(entity_args)
                self.agents["market"] = self.market
            elif entity_name == 'bank':
                self.bank = Bank(entity_args)
                self.agents["bank"] = self.bank

            elif entity_name == 'tax_gov':
                self.tax_gov = TaxGovernment(entity_args)
                self.agents["tax_gov"] = self.tax_gov

            elif entity_name == 'central_bank_gov':
                self.central_bank_gov = CentralBankGovernment(entity_args)
                self.agents["central_bank_gov"] = self.central_bank_gov

            elif entity_name == 'pension_gov':
                self.pension_gov = PensionGovernment(entity_args)
                self.agents["pension_gov"] = self.pension_gov

        self.gov_agents = []
        for attr in self.agents:
            if 'gov' in attr:
                agent = getattr(self, attr)
                self.gov_agents.append(agent)
        if hasattr(self, "government"):  # main_gov related indicators are used for evaluation
            self.main_gov = self.government
        else:
            self.main_gov = self.tax_gov

        if self.market.firm_n != 1:
            # The action includes:
            # - A scalar value indicating the selected firm index for employment (discretized from [-1, 1]).
            # - A vector of length self.market.firm_n specifying the proportion of consumption from each firm.
            self.households.action_dim += 1 + self.market.firm_n
            for agent in self.gov_agents:
                if hasattr(agent, 'action_dim') and agent.type == "tax":
                    agent.action_dim += self.market.firm_n

        # Observation spaces
        global_obs, private_obs = self.reset()
        self.bank.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_obs.shape[0],), dtype=np.float32
        )

        self.households.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(private_obs.shape[1],),
            dtype=np.float32
        )

        for gov in self.gov_agents:
            gov.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(global_obs.shape[0],), dtype=np.float32
            )

        self.market.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(global_obs.shape[0] + self.main_gov.action_dim + self.bank.action_dim + 1,),  # The last dimension represents the firm's own productivity level Zt.
            dtype=np.float32
        )

        self.display_mode = False
    
    @property
    def action_spaces(self):
        return {
            self.households.name: self.households.action_space,
            self.government.name: self.government.action_space,
            self.market.name: self.market.action_space,
            self.bank.name: self.bank.action_space
        }
    
    @property
    def observation_spaces(self):
        return {
            self.households.name: self.households.observation_space,
            self.government.name: self.government.observation_space,
            self.market.name: self.market.observation_space,
            self.bank.name: self.bank.observation_space
        }
    
    def action_wrapper(self, action_dict):
        processed_action_dict = {}
        for agent_name, agent_action in action_dict.items():
            expected_dim = self.agents[agent_name].action_dim
            if agent_action.shape[-1] != expected_dim:
                if agent_name == 'household' and agent_action.shape[-1] - expected_dim == 1:
                    agent_action = agent_action[:, :2]
                else:
                    raise ValueError(
                        "Invalid actions for {}. Expected shape: {}, Found: {}".format(
                            agent_name, expected_dim, agent_action.shape[-1]
                        )
                    )
            if agent_action.shape[-1] == 0:
                processed_action_dict[agent_name] = None
                continue
            
            fill_in_len = abs(expected_dim - len(self.agents[agent_name].real_action_min))
            real_action_min = self.agents[agent_name].real_action_min
            real_action_min = np.pad(real_action_min, (0, fill_in_len), mode='constant', constant_values=0)
            real_action_max = self.agents[agent_name].real_action_max
            real_action_max = np.pad(real_action_max, (0, fill_in_len), mode='constant', constant_values=1)
            temp = copy.copy(agent_action)
            if agent_name == 'household':
                temp[:, 1] = self.agents[agent_name].real_action_max[1] * temp[:, 1]  # working hours scale
            processed_action_dict[agent_name] = np.clip(temp, real_action_min, real_action_max)
        return processed_action_dict
    
    def get_actions(self, action_dict):
        """Get and process actions for all agents."""
        valid_action_dict = self.is_valid(action_dict)
        processed_action_dict = self.action_wrapper(valid_action_dict)
        for agent in self.gov_agents:
            agent.get_action(processed_action_dict[agent.name], firm_n=self.market.firm_n)

        self.bank.get_action(processed_action_dict[self.bank.name])
        self.market.get_action(processed_action_dict[self.market.name])
        self.households.get_action(processed_action_dict[self.households.name], firm_n=self.market.firm_n)
    
    def step(self, action_dict, t=None):
        """Perform a simulation step given the actions."""
        # Agents get action
        self.get_actions(action_dict)
        # Firm Step
        self.market.step(self)
        # Households Step
        self.households.step(self, t)
        # Government Step
        for agent in self.gov_agents:
            agent.step(self)

        # Bank Step
        self.bank.step(self)
        # Update Evaluation Metrics
        
        self.update_metrics()
        
        # Increment step counter
        self.step_cnt += 1
        self.done = self.is_terminal()
        
        # Next state
        next_global_state, next_private_state = self.get_obs()
        
        # Check for NaN values
        next_state = self.is_nan()
        if next_state != False:
            next_global_state, next_private_state = next_state
        
        self.last_price_index = copy.copy(self.price_index)
        # print(f"Step: {self.step_cnt}, households num:{self.households.households_n}")
        return next_global_state, next_private_state, self.government_reward, self.households_reward, self.firm_reward, self.bank_reward, self.done
    
    def update_metrics(self):
        """Update evaluation metrics such as Gini coefficients, price index, and rewards."""
        # Next state
        next_global_state, next_private_state = self.get_obs()
        
        # Compute Gini coefficients
        self.wealth_gini = self.gini_coef(self.households.post_asset)
        self.income_gini = self.gini_coef(self.households.post_income)
        
        # Compute Price Index
        self.inflation_rate, self.price_index = self.market.compute_inflation_rate(self.market.price,
                                                                                   self.last_price_index)
        
        # Compute rewards
        self.households_reward = self.households.get_reward()
        self.government_reward = self.main_gov.get_reward(self)

        self.firm_reward = self.market.get_reward(self)
        self.bank_reward = self.bank.get_reward()
        
        return next_global_state, next_private_state
    
    def reset(self, **custom_cfg):
        """Reset the simulation to the initial state."""
        self.step_cnt = 0
        for gov in self.gov_agents:
            gov.reset(households_n=self.households.households_n, firm_n=self.market.firm_n)

        self.households.reset()
        self.bank.reset(households_at=self.households.at)

        self.market.reset(households_n=self.households.households_n, GDP=self.main_gov.GDP,
                          households_at=self.households.at)
        
        self.last_price_index = 1
        self.ini_income_gini = self.gini_coef(self.households.income)
        self.ini_wealth_gini = self.gini_coef(self.households.at)
        self.done = False
        self.display_mode = False
        
        return self.get_obs()
    
    def get_obs(self):
        """Generate global and private observations for the agents."""
        # Global state: mean values for top 10% and bottom 50% of income and wealth
        income = self.households.income
        wealth = self.households.at_next
        # Private observations
        if 'OLG' in self.households.type or 'personal_pension' in self.households.type:
            private_obs = np.hstack((self.households.e, wealth, income, self.households.age))
        elif 'ramsey' in self.households.type:
            private_obs = np.hstack((self.households.e, wealth, income))
        else:
            raise ValueError(f"AgentTypeError: your house_type {self.households.type} is not in type_list.")

        global_obs = np.array([
            np.mean(self.households.at_next),  # 0
            np.mean(self.households.income),
            np.mean(self.households.e),
            float(len(private_obs)),
            getattr(self, "inflation_rate", 0.0),  # 4
            getattr(self, "wealth_gini", 0.0),
            getattr(self.main_gov, "GDP", 0.0),
            getattr(self.main_gov, "growth_rate", 0.0),
            getattr(self.main_gov, "Bt_next", 0.0) / getattr(self.main_gov, "GDP", 1.0),  # 避免除以 0
            getattr(self.main_gov, "tau", 0.0),  # 9
            getattr(self.main_gov, "xi", 0.0),
            getattr(self.main_gov, "Gt_prob", 0.0),

        ])

        if hasattr(self, 'government'):
            if self.government.type == "pension":
                additional_obs = np.array([
                    getattr(self.government.pension_fund, "pension_fund", 0.0),
                    getattr(self.government, "retire_age", 60.0),  # 默认退休年龄为 60
                    getattr(self.government, "contribution_rate", 0.10),  # 默认缴费率为 10%
                ])
                global_obs = np.concatenate([global_obs, additional_obs])

            elif self.government.type == "central_bank":
                additional_obs = np.array([
                    getattr(self.government, "base_interest_rate", 0.01),
                    getattr(self.government, "reserve_requirement", 0.01)
                ])
                global_obs = np.concatenate([global_obs, additional_obs])

        else:
            additional_obs_1 = []
            additional_obs_2 = []
            if hasattr(self, "central_bank_gov"):
                additional_obs_1 = np.array([
                    getattr(self.central_bank_gov, "base_interest_rate", 0.01),
                    getattr(self.central_bank_gov, "reserve_requirement", 0.01)
                ])
            elif hasattr(self, "pension_gov"):
                additional_obs_2 = np.array([
                    getattr(self.pension_gov, "pension_fund", 0.0),
                    getattr(self.pension_gov, "retire_age", 60.0),  # 默认退休年龄为 60
                    getattr(self.pension_gov, "contribution_rate", 0.10),  # 默认缴费率为 10%
                ])

            global_obs = np.concatenate([global_obs, additional_obs_1, additional_obs_2])

        return global_obs, private_obs
    
    def is_terminal(self):
        """
        Check if the simulation has reached a terminal state.
        """
        # Condition 1: Gini coefficient exceeds maximum threshold
        gini_exceeded = self.wealth_gini >= 1 or self.income_gini >= 1
        # if gini_exceeded:
        #     print("Termination condition: Gini coefficient exceeded.")
        
        # Condition 2: Data contains NaN values
        data_nan = any([
            math.isnan(self.government_reward),
            math.isnan(self.wealth_gini * self.income_gini),
            math.isnan(np.mean(self.households_reward))
        ])
        
        episode_completed = self.step_cnt >= self.episode_length
        agent_terminal = any(agent.is_terminal() for agent in self.agents.values())
        # if gini_exceeded or data_nan or episode_completed or agent_terminal == True:
        #     print(1)
        return gini_exceeded or data_nan or episode_completed or agent_terminal
    
    def is_nan(self):
        """Handle NaN values in rewards and state variables."""
        if (
                math.isnan(self.government_reward * self.wealth_gini * self.income_gini) or
                math.isnan(np.mean(self.households_reward))
        ):
            self.ht = self.working_hours_wrapper(np.ones((self.households.households_n, 1)))
            self.consumption = np.full((self.households.households_n, 1), 0.001)
            # 传参
            self.households_reward = self.households.get_reward(self.consumption, self.ht)
            self.government_reward = self.main_gov.get_reward(self)
            self.wealth_gini = 0.99
            self.income_gini = 0.99
            next_global_state = np.zeros(self.main_gov.observation_space.shape[0])
            # private_obs_dim = self.households.observation_space.shape[0] - self.government.observation_space.shape[0]
            next_private_state = np.zeros((self.households.households_n, self.households.private_info_dim))
            return next_global_state, next_private_state
        else:
            return False
    
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
        """Calculate the Gini coefficient of a numpy array."""
        values = np.sort(values, axis=0) + 1e-7  # Avoid division by zero
        n = values.shape[0]
        index = np.arange(1, n + 1).reshape(-1, 1)
        return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
    
    def working_hours_wrapper(self, ht):
        """Compute actual working hours based on ht."""
        max_hours = 365 * 24 * (2 / 3)
        return max_hours * ht  # Max: Work up to 2/3 of the total possible hours
    
    def _load_image(self):
        self.gov_img = pygame.image.load(os.path.join(ROOT_PATH, "img/gov.jpeg"))
        self.house_img = pygame.image.load(os.path.join(ROOT_PATH, "img/household.png"))
        self.firm_img = pygame.image.load(os.path.join(ROOT_PATH, "img/firm.png"))
        self.bank_img = pygame.image.load(os.path.join(ROOT_PATH, "img/bank.jpeg"))
    
    def render(self):
        if not self.display_mode:
            self.background = pygame.display.set_mode([500, 500])
            self.display_mode = True
            self._load_image()
        
        self.background.fill((255, 255, 255))
        
        debug(f"Step {self.step_cnt}")
        debug("Mean Social Welfare: " + "{:.3g}".format(float(self.households_reward.mean())), x=280, y=10)
        debug("Wealth Gini: " + "{:.3g}".format(self.wealth_gini), x=348, y=30)
        debug("Income Gini: " + "{:.3g}".format(self.income_gini), x=348, y=50)
        debug('GDP: ' + "{:.3g}".format(self.government.GDP), x=390, y=70)
        
        gov_img = pygame.transform.scale(self.gov_img, (50, 50))
        self.background.blit(gov_img, [100, 100])
        debug("Tau: " + "{:.3g}".format(self.government.tau), x=10, y=80)
        debug("Xi: " + "{:.3g}".format(self.government.xi), x=10, y=100)
        debug("Tau_a" + "{:.3g}".format(self.government.tau_a), x=10, y=120)
        debug("Xi_a: " + "{:.3g}".format(self.government.xi_a), x=10, y=140)
        debug("Gt_prob: " + "{:.3g}".format(self.government.Gt_prob), x=10, y=160)
        debug("Bt2At: " + "{:.3g}".format(self.Bt2At), x=10, y=180)
        
        house_img = pygame.transform.scale(self.house_img, (50, 50))
        self.background.blit(house_img, [200, 400])
        self.background.blit(house_img, [160, 400])
        self.background.blit(house_img, [180, 440])
        debug("Mean Working Hours: " + "{:.3g}".format(self.workingHours.mean()), x=250, y=450)
        debug("Mean Saving Prop: " + "{:.3g}".format(self.mean_saving_p), x=250, y=470)
        
        firm_img = pygame.transform.scale(self.firm_img, (50, 50))
        self.background.blit(firm_img, [400, 170])
        debug("Wage Rate: " + "{:.3g}".format(self.market.WageRate), x=370, y=230)
        
        pygame.draw.line(self.background, COLORS['blue'], (140, 150), (190, 390), width=10)
        pygame.draw.line(self.background, COLORS['blue'], (220, 390), (390, 210), width=10)
        pygame.draw.line(self.background, COLORS['blue'], (160, 130), (390, 180), width=10)
        
        bank_img = pygame.transform.scale(self.bank_img, (50, 50))
        self.background.blit(bank_img, [230, 200])
        
        pygame.draw.line(self.background, COLORS['blue'], (145, 145), (225, 195), width=10)
        pygame.draw.line(self.background, COLORS['blue'], (205, 390), (250, 255), width=10)
        pygame.draw.line(self.background, COLORS['blue'], (380, 195), (280, 230), width=10)
        
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()
    
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


render = False
if render:
    COLORS = {
        'red': [255, 0, 0],
        'light red': [255, 127, 127],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'orange': [255, 127, 0],
        'grey': [176, 196, 222],
        'purple': [160, 32, 240],
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'light green': [204, 255, 229],
        'sky blue': [0, 191, 255],
        # 'red-2': [215,80,83],
        # 'blue-2': [73,141,247]
    }
    
    pygame.init()
    font = pygame.font.Font(None, 22)
    
    
    def debug(info, y=10, x=10, c='black'):
        display_surf = pygame.display.get_surface()
        debug_surf = font.render(str(info), True, COLORS[c])
        debug_rect = debug_surf.get_rect(topleft=(x, y))
        display_surf.blit(debug_surf, debug_rect)
