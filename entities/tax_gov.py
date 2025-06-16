from omegaconf import omegaconf
from entities.base import BaseEntity
from gym.spaces import Box
import copy
import numpy as np
from agents.saez import SaezGovernment


class TaxGovernment(BaseEntity):
    name = 'tax_gov'

    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        if self.tax_type == 'saez':
            self.saez_gov = SaezGovernment()
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
        # actions [tau, xi, tau_a, xi_a, Gt_prob]
        self.old_per_gdp = copy.copy(self.per_household_gdp)
        self.Bt = copy.copy(self.Bt_next)

        policy_actions = actions[:self.policy_action_len]
        self.tau, self.xi, self.tau_a, self.xi_a, self.Gt_prob = policy_actions

        if firm_n != 1:
            Gt_prob_ratios = actions[self.policy_action_len:]
            if np.sum(Gt_prob_ratios) == 0:
                self.Gt_prob_j = np.zeros_like(Gt_prob_ratios)[:, np.newaxis]
            else:
                self.Gt_prob_j = (Gt_prob_ratios / np.sum(Gt_prob_ratios))[:, np.newaxis] * self.Gt_prob
        else:
            self.Gt_prob_j = self.Gt_prob

    def step(self, society):
        households = society.households
        self.tax_array = (households.income_tax + households.asset_tax + np.dot(households.consumption_ij,
                                                                                society.market.price) * society.consumption_tax_rate) + households.estate_tax
        self.Bt_next = (
                (1 + society.bank.lending_rate) * self.Bt + np.sum(self.Gt_prob_j * society.market.Yt_j) - np.nansum(
            self.tax_array))
        self.GDP = np.sum(society.market.price * society.market.Yt_j)
        self.per_household_gdp = self.GDP / households.households_n

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

        # 个人养老金
        # if households_type == "personal_pension":
        #     pension = self.calculate_pension(households)
        #     income = income + pension

        def tax_formula(x, tau, xi):
            x = np.maximum(x, 0)
            # 使用 NumPy 的条件操作，确保对每个元素处理
            if np.isclose(xi, 1.0):
                return x - (1 - tau) * np.log(x)
            else:
                return x - ((1 - tau) / (1 - xi)) * np.power(x, 1 - xi)

        # 逐元素计算 income 和 asset 的税收
        income_tax = tax_formula(income, self.tau, self.xi)
        asset_tax = tax_formula(asset, self.tau_a, self.xi_a)

        return income_tax, asset_tax

    def calculate_taxes(self, income, asset):
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
            income_tax, asset_tax = self.calculate_taxes(income, asset)
        elif self.tax_type == "saez":
            self.saez_gov.saez_step()
            self.saez_gov.update_saez_buffer(income)
            income_tax = np.array(self.saez_gov.tax_due(income)).reshape(-1, 1)
            asset_tax = np.zeros_like(income_tax)
        elif self.tax_type == "ai_agent":
            income_tax, asset_tax = self.tax_function(income, asset)
        else:
            print("Free market policy: no taxes applied.")
            income_tax = np.zeros_like(income)
            asset_tax = np.zeros_like(asset)

        return income_tax, asset_tax

    def calculate_pension(self, households):
        pension = -households.income * ~households.is_old * self.contribution_rate
        income_mean = np.nanmean(households.income)
        if households.is_old.any():
            income_old_mean = np.nanmean(households.income[households.is_old])
            avg_wage = (income_old_mean + income_mean) / 2

            basic_pension = avg_wage * households.working_years[households.is_old] * 0.01
            personal_pension = households.accumulated_pension_account[households.is_old] / self.get_annuity_factor(
                self.retire_age)

            pension[households.is_old] = basic_pension + personal_pension

        return pension

    def get_reward(self, society):
        """Compute the government's reward based on its goal."""
        self.growth_rate = (self.per_household_gdp - self.old_per_gdp) / self.old_per_gdp
        gov_goal = self.gov_task
        if gov_goal == "gdp":
            capital_growth_rate = (society.market.Kt_next - society.market.Kt) / society.market.Kt * 100
            return capital_growth_rate.mean()
        elif gov_goal == "gini":
            before_tax_wealth_gini = society.gini_coef(society.households.at)
            after_tax_wealth_gini = society.gini_coef(society.households.post_asset)
            delta_wealth_gini = (before_tax_wealth_gini - after_tax_wealth_gini) / before_tax_wealth_gini

            before_tax_income_gini = society.gini_coef(society.households.income)
            after_tax_income_gini = society.gini_coef(society.households.post_income)
            delta_income_gini = (before_tax_income_gini - after_tax_income_gini) / before_tax_income_gini

            return (delta_income_gini + delta_wealth_gini) * 100
        elif gov_goal == "social_welfare":
            return np.sum(society.households_reward)
        elif gov_goal == "mean_welfare":  # 追求 人均 social welfare 最大化：当人口出现老龄化，出生率小于死亡率，无法避免 总体 welfare 会下降，但不代表 人均会下降
            return np.mean(society.households_reward)
        elif gov_goal == "gdp_gini":
            gdp_growth = (self.per_household_gdp - self.old_per_gdp) / self.old_per_gdp
            gini_penalty = society.gini_weight * (society.wealth_gini * society.income_gini)
            return gdp_growth - gini_penalty
        elif gov_goal == "pension_gap":
            payout = sum(-self.calculate_pension(society))
            return payout

        else:
            raise ValueError("Invalid government goal specified.")

    def is_terminal(self):
        return False
