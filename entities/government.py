from entities.base import BaseEntity
import numpy as np
from gym.spaces import Box
from agents.saez import SaezGovernment
import copy


class Government(BaseEntity):
    name = 'government'

    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        self.action_dim = entity_args[self.type]['action_dim']
        self.policy_action_len = copy.copy(self.action_dim)
        self.real_action_max = np.array(entity_args[self.type]['real_action_max'])
        self.real_action_min = np.array(entity_args[self.type]['real_action_min'])

    def reset(self, **custom_cfg):
        if self.type == 'tax' and self.tax_type == 'saez':
            self.saez_gov = SaezGovernment()
        households_n = custom_cfg['households_n']
        firm_n = custom_cfg['firm_n']
        real_gdp = self.real_gdp
        real_debt_rate = self.real_debt_rate
        real_population = self.real_population

        self.action_space = Box(
            low=self.entity_args[self.type]["action_space"]["low"],
            high=self.entity_args[self.type]["action_space"]["high"],
            shape=(self.action_dim,), dtype=np.float32
        )

        initial_actions = self.entity_args[self.type]['initial_action']

        self.initial_action = np.concatenate(
            [np.array(list(initial_actions.values())),
             np.ones(self.action_dim - self.policy_action_len) / (self.action_dim - self.policy_action_len)])

        self.per_household_gdp = real_gdp / real_population
        self.GDP = self.per_household_gdp * households_n
        self.Bt_next = real_debt_rate * self.GDP
        self.Bt = copy.copy(self.Bt_next)
        self.pension_fund = self.entity_args.get('initial_pension_fund', 1e-8)
        self.contribution_rate = self.entity_args.params.contribution_rate
        self.retire_age = self.entity_args.params.retire_age
        self.old_percent = 0
        self.dependency_ratio = 0

        # Initialize Gt_prob_j as an empty array or with a default value to avoid AttributeError
        self.Gt_prob_j = np.ones((firm_n, 1)) * self.Gt_prob if self.action_dim > self.entity_args[self.type][
            'action_dim'] else 1

    def get_action(self, actions, firm_n):
        self.old_per_gdp = copy.copy(self.per_household_gdp)
        self.old_GDP = copy.copy(self.GDP)
        self.Bt = copy.copy(self.Bt_next)

        policy_actions = actions[:self.policy_action_len]
        Gt_prob_ratios = np.ones(firm_n)/firm_n
        if self.type == "pension":
            # self.retire_age, self.contribution_rate, self.pension_growth_rate = policy_actions
            self.retire_age, self.contribution_rate = policy_actions
        elif self.type == "tax":
            self.tau, self.xi, self.tau_a, self.xi_a, self.Gt_prob = policy_actions
            Gt_prob_ratios = actions[self.policy_action_len:]
        elif self.type == "central_bank":
            self.base_interest_rate, self.reserve_ratio = policy_actions
        else:
            raise ValueError("Invalid government type specified!")

        if firm_n != 1:
            if np.sum(Gt_prob_ratios) == 0:
                self.Gt_prob_j = np.zeros_like(Gt_prob_ratios)[:, np.newaxis]
            else:
                self.Gt_prob_j = (Gt_prob_ratios / np.sum(Gt_prob_ratios))[:, np.newaxis] * self.Gt_prob
        else:
            self.Gt_prob_j = self.Gt_prob


    def step(self, society):
        self.tax_step(society)
        if self.type == "pension" and (
                "OLG" in society.households.type or society.households.type == "personal_pension"):
            self.pension_step(society)


    def get_reward_central(self, inflation_rate, growth_rate,
                           target_inflation=0.02, target_growth=0.05):
        """Reward decays with squared inflation deviation and penalizes only below-target growth."""
        inflation_deviation = (inflation_rate - target_inflation) ** 2
        growth_deviation = (target_growth - growth_rate) ** 2

        k_inflation = 500
        k_growth = 300
        reward = np.exp(-k_inflation * inflation_deviation - k_growth * growth_deviation)
        return reward

    def tax_step(self, society):
        """Calculate government metrics such as taxes, investment, and GDP."""
        households = society.households
        self.tax_array = (households.income_tax + households.asset_tax + np.dot(households.final_consumption, society.market.price) * society.consumption_tax_rate) + households.estate_tax
        self.gov_spending = self.Gt_prob_j * society.market.Yt_j
        self.Bt_next = ((1 + society.bank.base_interest_rate) * self.Bt + np.sum(self.gov_spending) - np.sum(self.tax_array))

        self.GDP = np.sum(society.market.price * society.market.Yt_j)
        self.per_household_gdp = self.GDP / households.households_n
        

    def pension_step(self, society):
        """Update the pension fund for all households."""
        self.current_net_households_pension = society.households.pension.sum()
        self.pension_fund = (1 + self.pension_growth_rate) * self.pension_fund - self.current_net_households_pension

        # self.labor_participate_rate = society.households.ht[~society.households.is_old].mean()/society.households.h_max

    def calculate_pension(self, households):
        """Calculate the pension benefits for a given household."""
        pension = -households.income * ~households.is_old * self.contribution_rate   # <0 : pension contribute from young individuals

        income_mean = np.nanmean(households.income)

        if households.is_old.any():
            income_old_mean = np.nanmean(households.income[households.is_old])
            avg_wage = (income_old_mean + income_mean) / 2

            if hasattr(self, 'pension_rate') and self.pension_rate is not None:
                basic_pension = avg_wage * households.working_years[households.is_old] * 0.01 * self.pension_rate
            else:
                basic_pension = avg_wage * households.working_years[households.is_old] * 0.01
            personal_pension = households.accumulated_pension_account[households.is_old] / self.get_annuity_factor(
                self.retire_age)

            pension[households.is_old] = basic_pension + personal_pension  # >0 : payoff for old individuals

        return pension

    def get_annuity_factor(self, age):
        """Return the annuity factor for a given age."""
        annuity_factors = {
            40: 233, 41: 230, 42: 226, 43: 223, 44: 220, 45: 216, 46: 212,
            47: 208, 48: 204, 49: 199, 50: 195, 51: 190, 52: 185, 53: 180,
            54: 175, 55: 170, 56: 164, 57: 158, 58: 152, 59: 145, 60: 139,
            61: 132, 62: 125, 63: 117, 64: 109, 65: 101, 66: 93, 67: 84,
            68: 75, 69: 65, 70: 56
        }
        return annuity_factors.get(age, 56)

    def tax_function(self, income, asset):
        """Compute taxes based on government policies."""

        def tax_formula(x, tau, xi):
            x = np.maximum(x, 1e-8)  # 避免 x=0
            if np.isclose(xi, 1.0):
                return x - (1 - tau) * np.log(x)
            else:
                return x - ((1 - tau) / (1 - xi)) * np.power(x, 1 - xi)

        income_tax = tax_formula(income, self.tau, self.xi)
        asset_tax = tax_formula(asset, self.tau_a, self.xi_a)

        return income_tax, asset_tax

    def calculate_progressive_taxes(self, income, asset):
        """Calculate income and asset taxes based on US federal tax brackets."""

        def income_tax_function(x):
            personal_allowance = 12950
            tax_credit = 559.98
            marginal_rates = np.array([10, 12, 22, 24, 32, 35, 37]) / 100  # as decimals
            thresholds = np.array([0, 10275, 41775, 89075, 170050, 215950, 539900, np.inf])

            taxable_income = np.maximum(0, x - personal_allowance)
            taxes_paid = np.zeros_like(x, dtype=float)

            for i in range(1, len(thresholds)):
                income_in_bracket = np.minimum(taxable_income, thresholds[i]) - thresholds[i - 1]
                taxes_paid += np.maximum(0, income_in_bracket) * marginal_rates[i - 1]

            taxes_paid = np.maximum(0, taxes_paid - tax_credit)
            return taxes_paid

        income_tax = income_tax_function(income)
        asset_tax = np.zeros_like(asset)
        return income_tax, asset_tax

    def compute_tax(self, income, asset):
        """Compute income and asset taxes based on the tax policy."""
        if self.tax_type == "us_federal":
            income_tax, asset_tax = self.calculate_progressive_taxes(income, asset)
        elif self.tax_type == "saez":
            self.saez_gov.saez_step()
            self.saez_gov.update_saez_buffer(income)
            income_tax = np.array(self.saez_gov.tax_due(income)).reshape(-1, 1)
            _, asset_tax = self.tax_function(income, asset)
        elif self.tax_type == "ai_agent":
            income_tax, asset_tax = self.tax_function(income, asset)
        else:
            print("Free market policy: no taxes applied.")
            income_tax = np.zeros_like(income)
            asset_tax = np.zeros_like(asset)
        return income_tax, asset_tax

    def softsign(self, x):
        return x / (1.0 + np.abs(x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def get_reward(self,  society, gov_goal=None):
        """
        Compute the government's reward based on its goal.

        Notes
        -----
        - For self.type in {"pension", "tax"}:
          They can share the same reward function definitions below (gov_goal routing).
        - For self.type == "central_bank":
          Delegates to `get_reward_central(inflation_rate, growth_rate)`.
        """
        SCALE = dict(
            gdp_growth=0.05,
            gini_scale=0.167,
        )

        self.growth_rate = (self.GDP + 1e-8) / (self.old_GDP + 1e-8) - 1

        # Assign self.gov_task to gov_goal if no valid value is provided for gov_goal
        gov_goal = gov_goal or self.gov_task
        # Central bank uses its own reward
        if self.type == "central_bank":
            return self.get_reward_central(society.inflation_rate, self.growth_rate) # \in (0,1)

        if gov_goal == "gdp":
            log_gdp_growth = np.log(self.GDP + 1e-8) - np.log(self.old_GDP + 1e-8)
            # reward = self.softsign(log_gdp_growth / SCALE["gdp_growth"])
            reward = self.sigmoid(log_gdp_growth / SCALE["gdp_growth"])
            return np.array([reward])  # \in (0,1)
    
        elif gov_goal == "gini":
            # Wealth Gini improvement
            before_tax_wealth_gini = society.gini_coef(society.households.at)
            after_tax_wealth_gini = society.gini_coef(society.households.post_asset)
            impr_w = before_tax_wealth_gini - after_tax_wealth_gini
        
            # Income Gini improvement
            before_tax_income_gini = society.gini_coef(society.households.income)
            after_tax_income_gini = society.gini_coef(society.households.post_income)
            impr_i = before_tax_income_gini - after_tax_income_gini
            # return (delta_income_gini + delta_wealth_gini) * 100
            reward = self.sigmoid((impr_w + impr_i) / (2 * SCALE["gini_scale"]))  # \in (0,1)
            return np.array([reward])
    
        elif gov_goal == "social_welfare":
            # Sum of household utilities (social welfare)
            social_welfare = np.sum(society.households.get_reward())
            return np.array([social_welfare])
    
        elif gov_goal == "mean_welfare":
            # When population changes (OLG), mean welfare ≠ social welfare.
            # Mean welfare better reflects policy effects without population-size confounds.
            mean_welfare = np.mean(society.households.get_reward())
            return np.array([mean_welfare])
    
        elif gov_goal == "gdp_gini":
            # mixed goal
            gdp_rew = self.get_reward(society, gov_goal='gdp')
            gini_rew = self.get_reward(society, gov_goal='gini')
            return (gdp_rew + gini_rew)/2  # \in (0,1)
    
        elif gov_goal == "pension_gap":
            pension_surplus = sum(-self.calculate_pension(society.households))
            return self.get_pension_reward(pension_surplus)
    
        else:
            raise ValueError("Invalid government goal specified.")

    def get_pension_reward(self, pension_surplus, scale=10, beta=10):
        """
        Compute the reward for pension fund sustainability.

        This function:
        - Rewards increasing pension surplus.
        - Applies a penalty for extreme surpluses or deficits.
        - Uses a logarithmic transformation to handle large surpluses and ensures diminishing returns.
        - Normalizes the reward using the tanh function to keep it within a bounded range.

        Parameters:
        -----------
        pension_surplus : float
            The surplus or deficit of the pension fund.

        scale : float, optional, default=10
            The scaling factor for normalizing the final reward.

        beta : float, optional, default=8
            A factor that adjusts the sensitivity to surpluses.

        Returns:
        --------
        normalized_reward : float
            The normalized reward, constrained within a bounded range, typically [0, 1].
        """
    
        # Log transformation for diminishing returns and penalty for extreme surpluses
        pension_growth = np.log(1 + pension_surplus) - beta
    
        # Normalize the reward using tanh
        normalized_reward = self.sigmoid(pension_growth / scale)
    
        return normalized_reward

    def is_terminal(self):
        '''
        If `self.pension_fund < 0` (pension pool exhausted),  the environment should trigger a terminal state.
        '''
        if self.pension_fund < 0:
            return True
        return False
