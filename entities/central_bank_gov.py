from entities.base import BaseEntity
import numpy as np
from gym.spaces import Box
import copy
from omegaconf import omegaconf


class CentralBankGovernment(BaseEntity):
    name = 'central_bank_gov'

    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        self.policy_action_len = copy.copy(self.action_dim)
        self.real_action_max = np.array(self.real_action_max)
        self.real_action_min = np.array(self.real_action_min)

    def reset(self, **custom_cfg):
        households_n = custom_cfg['households_n']
        firm_n = custom_cfg['firm_n']
        real_gdp = 254746 * 1e8  # in USD
        real_debt_rate = 121.29 * 0.01  # as a fraction 未修改
        real_population = 333428e3

        if isinstance(self.action_space, omegaconf.DictConfig):
            self.action_space = Box(
                low=self.action_space["low"],
                high=self.action_space["high"],
                shape=(self.action_dim,), dtype=np.float32
            )

        initial_actions = self.initial_action
        if isinstance(self.initial_action, omegaconf.DictConfig):
            self.initial_action = np.concatenate(
                [np.array(list(initial_actions.values())),
                 np.ones(self.action_dim - self.policy_action_len) / (self.action_dim - self.policy_action_len)])

        self.per_household_gdp = real_gdp / real_population
        self.GDP = self.per_household_gdp * households_n
        self.Bt_next = real_debt_rate * self.GDP

        # Initialize Gt_prob_j as an empty array or with a default value to avoid AttributeError
        self.Gt_prob_j = np.ones((firm_n, 1)) * self.Gt_prob if self.action_dim > self.action_dim else 1

    def get_action(self, actions, firm_n):
        self.old_per_gdp = copy.copy(self.per_household_gdp)
        self.Bt = copy.copy(self.Bt_next)

        policy_actions = actions[:self.policy_action_len]
        self.base_interest_rate, self.reserve_ratio = policy_actions

    def step(self, society):
        pass

    def get_reward_central(self, inflation_rate, growth_rate,
                           target_inflation=0.02, target_growth=0.05):
        """Reward decays with squared inflation deviation and penalizes only below-target growth."""
        inflation_deviation = (inflation_rate - target_inflation) ** 2
        growth_deviation = (target_growth - growth_rate) ** 2

        k_inflation = 900
        k_growth = 300

        return np.exp(-k_inflation * inflation_deviation - k_growth * growth_deviation)

    def get_reward(self, society):
        """Compute the government's reward based on its goal."""
        self.growth_rate = (self.per_household_gdp - self.old_per_gdp) / self.old_per_gdp
        return self.get_reward_central(society.inflation_rate, self.growth_rate)

    def is_terminal(self):
        return False
