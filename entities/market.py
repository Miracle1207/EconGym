from entities.base import BaseEntity
from utils.episode import EpisodeKey
import math
import copy
import numpy as np
from gym.spaces import Box


class Market(BaseEntity):
    name = "market"
    
    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        self.firm_n = entity_args[self.type]['firm_n']
        self.action_dim = entity_args[self.type]['action_dim']
        self.Zt = self.Z * (1 + np.random.rand(self.firm_n, 1))
        if (self.type == "perfect" or self.type == "monopoly") and self.firm_n != 1:
            raise ValueError("Invalid market type specified or invalid firm number specified.")
        
        self.action_space = Box(
            low=-0.2, high=0.2, shape=(self.firm_n, self.action_dim), dtype=np.float32
        )
    
    def calculate_price_index(self, prices, include_wage=True, wage_weight=1.0):
        """
        Calculate CES price index from product prices and optionally include wage rate as labor price.

        Parameters:
        - prices: list or array of product prices
        - include_wage: bool, whether to include wage in the index
        - wage_weight: float, relative weight of wage price term

        Returns:
        - price_index: float, aggregated CES price index
        """
        if self.epsilon == 1:
            raise ValueError("Epsilon cannot be 1 as it would lead to division by zero.")
        
        # Convert prices to flat array
        prices = np.array(prices).flatten()
        weighted_prices = [p ** (1 - self.epsilon) for p in prices]
        
        # Add wage component if enabled
        if include_wage:
            wage_rates = np.array(self.WageRate).flatten()
            wage_terms = [(w ** (1 - self.epsilon)) * wage_weight for w in wage_rates]
            weighted_prices.extend(wage_terms)
        
        sum_weighted_prices = sum(weighted_prices)
        price_index = sum_weighted_prices ** (1 / (1 - self.epsilon))
        return price_index
    
    def compute_inflation_rate(self, prices, old_price_level):
        current_price_level = self.calculate_price_index(prices)
        if old_price_level == 0:
            raise ValueError("Invalid price level: 0.")
        else:
            inflation_rate = (current_price_level / old_price_level).item() - 1
            return inflation_rate, current_price_level
    
    def production_quality_update(self):
        """Update the production quality (technology shock)."""
        log_next_z = np.log(self.Zt) + self.sigma_z * np.random.rand(*self.Zt.shape)
        self.Zt = np.exp(log_next_z)
    
    def reset(self, **custom_cfg):
        np.random.seed(0)
        households_n = custom_cfg['households_n']
        households_at = custom_cfg['households_at']
        GDP = custom_cfg['GDP']
        real_capital_rate = 18.3 * 0.01
        real_total_hours = 265888.875e6  # total hours worked, large L
        real_population = 333428e3
        self.Zt = self.Z * (1 + np.random.rand(self.firm_n, 1))
        
        self.Lt = (real_total_hours / real_population) * households_n
        self.Kt = real_capital_rate * GDP / self.firm_n * np.ones((self.firm_n, 1))
        self.Kt_next = copy.copy(self.Kt)
        self.price = np.ones((self.firm_n, 1))
        self.WageRate = self.price * self.Zt * (1 - self.alpha) * np.power(self.Kt / self.Lt, self.alpha)
    
    def get_action(self, actions):
        if actions is not None:
            self.price = actions[:, 0][:, np.newaxis]
            self.WageRate = actions[:, 1][:, np.newaxis]
    
    def step(self, society):
        """Calculate firm's labor demand and production output."""
        self.Kt = copy.copy(self.Kt_next)
        real_Kt = np.clip(self.Kt, a_min=0, a_max=None)
        self.production_quality_update()
        # Compute firm's labor demand
        self.firm_labor_j = (society.households.h_ij_ratio * society.households.ht * society.households.e).sum(axis=0)[:, np.newaxis]
        self.Lt = np.sum(self.firm_labor_j)
        self.WageRate = self.price * self.Zt * (1 - self.alpha) * np.power((real_Kt) / (self.firm_labor_j + 1e-8),
                                                                           self.alpha)
        self.Yt_j = self.production_output(real_Kt, self.firm_labor_j)
    
    def production_output(self, Kt, Lt):
        """Compute the production output."""
        if (Kt <= 0).any() or (Lt <= 0).any():
            # print(f"Kt invalid indices: {np.where(Kt <= 0)}, Lt invalid indices: {np.where(Lt <= 0)}")
            Kt = np.maximum(Kt, 1e-6)
            Lt = np.maximum(Lt, 1e-6)
        
        Y = self.Zt * (Kt ** self.alpha) * (Lt ** (1 - self.alpha))
        return Y
    
    def get_reward(self, society):
        """Calculate the firm's profit."""
        profit = self.price * self.Yt_j - self.WageRate * self.firm_labor_j - society.bank.lending_rate * self.Kt
        return profit
    
    def is_terminal(self):
        if np.sum(self.Kt_next) < 0:
            return True
        else:
            return False



