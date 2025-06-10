from entities.base import BaseEntity
import numpy as np
from gym.spaces import Box


class Bank(BaseEntity):
    name = 'bank'
    
    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        self.action_dim = entity_args[self.type]['action_dim']
        self.initial_action = entity_args[self.type]['initial_action']
        self.action_space = Box(
            low=self.action_space['low'], high=self.action_space['high'], shape=(self.action_dim,), dtype=np.float32
        )
    
    def reset(self, **custom_cfg):
        
        households_at = custom_cfg['households_at']
        if self.initial_action != "None":
            self.initial_action = np.array(self.initial_action) + np.random.randn(
                self.action_dim) * self.action_space.high
        self.total_deposit = np.sum(households_at)
    
    def get_action(self, actions):
        if self.type == 'commercial':
            # For commercial banks, actions are lending rate and deposit rate
            lending_rate, deposit_rate = actions
            self.lending_rate = np.clip(lending_rate, self.base_interest_rate + 0.01, self.base_interest_rate + 0.03)
            self.deposit_rate = np.clip(deposit_rate, self.base_interest_rate - 0.01, self.base_interest_rate)
    
    def step_non_profit(self, society):
        """Non-profit financial platforms only facilitate savings and bond holding."""
        interest_rate = self.base_interest_rate
        # Update low-risk capital assets and government bonds based on financial balance equation
        if hasattr(society, 'government'):
            rhs = (interest_rate + 1) * np.sum(society.market.Kt) + (1 + interest_rate) * (
                    society.government.Bt - np.sum(society.households.at))
            society.market.Kt_next = rhs + np.sum(society.households.at_next) - society.government.Bt_next
        else:
            rhs = (interest_rate + 1) * np.sum(society.market.Kt) + (1 + interest_rate) * (
                    society.tax_gov.Bt - np.sum(society.households.at))
            society.market.Kt_next = rhs + np.sum(society.households.at_next) - society.tax_gov.Bt_next


    def get_reward_non_profit(self):
        """Non-profit financial platforms do not optimize profit but maintain stability."""
        return 0  # No explicit reward, as their function is purely service-based
    
    def step_commercial(self, society):
        """Commercial banks set lending and deposit rates to optimize profit."""
        self.net_households_deposit = np.sum(society.households.at_next) - np.sum(society.households.at)
        self.total_deposit += (np.sum(
            self.deposit_rate * society.market.Kt) + self.lending_rate * society.government.Bt + self.net_households_deposit)
        # Ensure reserve requirement constraint
        self.total_loans = np.sum(society.market.Kt_next) + society.government.Bt_next
        
        consumption_sum = society.households.consumption_ij.sum(axis=0)[:, np.newaxis]
        investment = society.market.price * (
                    society.market.Yt_j - society.government.Gt_prob_j * society.market.Yt_j - consumption_sum)
        society.market.Kt_next = investment + (1 - self.depreciation_rate) * society.market.Kt
    
    def get_reward_commercial(self):
        """Profit is based on the interest spread between loans and deposits."""
        profit = self.lending_rate * self.total_loans - self.deposit_rate * self.net_households_deposit
        return profit
    
    def step(self, society):
        if hasattr(society, 'government'):
            if society.government.type == "central_bank":
                self.reserve_ratio = society.government.reserve_ratio
                self.base_interest_rate = society.government.base_interest_rate
        else:
            self.reserve_ratio = society.central_bank_gov.reserve_ratio
            self.base_interest_rate = society.central_bank_gov.base_interest_rate
        
        if self.type == 'non_profit':
            self.step_non_profit(society)
        elif self.type == 'commercial':
            self.step_commercial(society)
    
    def get_reward(self):
        if self.type == 'non_profit':
            return self.get_reward_non_profit()
        elif self.type == 'commercial':
            return self.get_reward_commercial()
    
    def is_terminal(self):
        if self.type == 'commercial' and self.total_loans > self.total_deposit * (1 - self.reserve_ratio):
            return True
        else:
            return False
