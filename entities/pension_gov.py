from omegaconf import omegaconf
from entities.base import BaseEntity
import numpy as np
from gym.spaces import Box
import copy


class PensionGovernment(BaseEntity):
    name = 'pension_gov'

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

        self.pension_fund = self.entity_args.get('initial_pension_fund', 0)
        self.contribution_rate = self.entity_args.params.contribution_rate
        self.retire_age = self.entity_args.params.retire_age
        self.old_percent = 0
        self.dependency_ratio = 0
        # Initialize Gt_prob_j as an empty array or with a default value to avoid AttributeError
        self.Gt_prob_j = np.ones((firm_n, 1)) * self.Gt_prob if self.action_dim > self.action_dim else 1

    def get_action(self, actions, firm_n):
        # actions [retire_age, contribution_rate, pension_growth_rate]
        self.old_per_gdp = copy.copy(self.per_household_gdp)
        self.Bt = copy.copy(self.Bt_next)

        policy_actions = actions[:self.policy_action_len]
        self.retire_age, self.contribution_rate = policy_actions

    def step(self, society):
        if ("OLG" in society.households.type or society.households.type == "personal_pension"):
            self.current_net_households_pension = society.households.pension.sum()  # 当前养老金缺口（支付-缴费）
            self.pension_fund = (1 + self.pension_growth_rate) * self.pension_fund - self.current_net_households_pension

            self.old_percent = society.households.old_n / len(society.households.is_old)  # 当前老年人口比例
            self.dependency_ratio = society.households.old_n / (
                    len(society.households.is_old) - society.households.old_n + 1e-8)  # Dependency ratio（赡养比），用来衡量养老压力或抚养负担。

    def calculate_pension(self, households):
        households_type = households.type
        """Calculate the pension benefits for a given household."""
        if households_type == 'OLG':
            pension = -households.income * ~households.is_old * self.contribution_rate
            income_mean = np.nanmean(households.income)
            if households.is_old.any():
                income_old_mean = np.nanmean(households.income[households.is_old])
                avg_wage = (income_old_mean + income_mean) / 2

                basic_pension = avg_wage * households.working_years[households.is_old] * 0.01
                personal_pension = households.accumulated_pension_account[households.is_old] / self.get_annuity_factor(
                    self.retire_age)

                pension[households.is_old] = basic_pension + personal_pension

        else:  # 个人养老金
            # np.random.seed(1)
            participate_personal_pension = np.random.choice([True, False], size=households.income.shape)
            pension = np.zeros_like(households.income)
            young_indices = ~households.is_old
            # 年轻人的个人养老金缴纳金额（负值）
            personal_pension_contributions = (
                    -households.income[young_indices]
                    * self.personal_contribution_rate
                    * participate_personal_pension[young_indices]
            )
            pension[young_indices] += personal_pension_contributions
            pension[households.is_old] = households.accumulated_pension_account[
                                             households.is_old] / self.get_annuity_factor(
                self.retire_age)

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

    def is_terminal(self):
        if self.pension_fund < 0:
            return True

        return False
