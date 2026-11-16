from entities.base import BaseEntity

import copy
import pandas as pd

import os
from gymnasium.spaces import Box
import numpy as np


class Household(BaseEntity):
    name = 'households'

    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        if "risk_invest" in self.type:
            self.action_dim += 1
        self.policy_action_len = copy.copy(self.action_dim)
        self.households_init()

        if 'OLG' in self.type:
            self.__dict__.update(entity_args.OLG)

        self.best_loss = 100
        self.action_space = Box(low=self.action_space.low,
                                high=self.action_space.high,
                                shape=(self.households_n, self.action_dim), dtype=np.float32)

    def e_initial(self, n):
        self.e_array = np.zeros((n, 2))  # super-star and normal
        # initialize as normal state
        random_set = np.random.rand(n)
        self.e_array[:, 0] = (random_set > self.e_p).astype(int) * self.e_init.flatten()  # Set to 0 if less than e_p
        self.e_array[:, 1] = (random_set < self.e_p).astype(int)  # Set to 1 if less than e_p
        self.e = np.sum(self.e_array, axis=1, keepdims=True)

        self.e_0 = copy.copy(self.e)
        self.e_array_0 = copy.copy(self.e_array)

    def generate_e_ability(self):
        """
        Generates n current ability levels for a given time step t.
        Implements superstar transition logic + normal agent ability evolution,
        maintaining functional parity with the original implementation.
        """
        N = self.households_n

        # === Step 1: Copy previous e_array, calculate mean of positive values ===
        self.e_past = self.e_array.copy()
        past_positive_mask = self.e_past[:, 0] > 0
        positive_values = self.e_past[past_positive_mask, 0]
        e_past_mean = np.mean(positive_values) if positive_values.size > 0 else 1.0

        # === Step 2: Determine current status ===
        is_superstar = (self.e_array[:, 1] > 0)
        normal_mask = ~is_superstar
        superstar_mask = is_superstar

        # === Step 3: Normal → Superstar transitions ===
        trans_to_superstar = (np.random.rand(N) < self.e_p) & normal_mask
        self.e_array[trans_to_superstar, 0] = 0
        self.e_array[trans_to_superstar, 1] = self.super_e * e_past_mean

        # === Step 4: Remaining normal agents ===
        remain_normal_mask = normal_mask & (~trans_to_superstar)

        # Fallback: replace non-positive past values with sampled positive ones
        fallback_mask = remain_normal_mask & (self.e_past[:, 0] <= 0)
        if np.any(fallback_mask):
            fallback_idx = np.where(fallback_mask)[0]
            fallback_sample = np.random.choice(positive_values, size=len(fallback_idx))
            self.e_past[fallback_idx, 0] = fallback_sample

        # === Step 5: Apply noisy update for remaining normal agents ===
        idx = np.where(remain_normal_mask)[0]
        eps = np.random.randn(len(idx))
        safe_e = np.clip(self.e_past[idx, 0], 1e-8, None)
        new_e = np.exp(self.rho_e * np.log(safe_e) + self.sigma_e * eps)

        self.e_array[idx, 0] = new_e
        self.e_array[idx, 1] = 0

        # === Step 6: Superstar → Normal transitions ===
        leave_superstar = (np.random.rand(N) >= self.e_q) & superstar_mask
        if np.any(leave_superstar):
            low = self.e_array[:, 0].min()
            high = self.e_array[:, 0].max()
            self.e_array[leave_superstar, 1] = 0
            self.e_array[leave_superstar, 0] = np.random.uniform(low, high, size=np.sum(leave_superstar))

        # Remaining superstar agents
        remain_super_mask = superstar_mask & (~leave_superstar)
        self.e_array[remain_super_mask, 0] = 0
        self.e_array[remain_super_mask, 1] = self.super_e * e_past_mean

        # === Final aggregation ===
        self.e = np.sum(self.e_array, axis=1, keepdims=True)

    def generate_c_init(self, age):
        self.c_init = np.zeros_like(age, dtype=float)
        noise = np.random.normal(0, 0.05, size=age.shape)

        self.c_init += ((age < 24) * (0.2 + noise) +
                        ((25 <= age) & (age < 34)) * (0.5 + noise) +
                        ((35 <= age) & (age < 54)) * (0.7 + noise) +
                        ((55 <= age) & (age < 74)) * (0.6 + noise) +
                        (age >= 74) * (0.4 + noise))

    def reset(self, **custom_cfg):
        self.households_n = self.entity_args.params.households_n  # Reset number of households
        self.e = copy.deepcopy(self.e_0)
        self.e_array = copy.deepcopy(self.e_array_0)
        self.generate_e_ability()
        self.at, self.at_next = copy.deepcopy(self.at_init), copy.deepcopy(self.at_init)
        self.age, self.income = copy.deepcopy(self.age_init), copy.deepcopy(self.it_init)

        # Initialize stock holdings and investment proportions
        self.stock_holdings = np.zeros((self.households_n, 1))
        self.investment_p = copy.copy(self.investment_init)  # Initialize to zero for each household
        self.ht = self.work_init * self.h_max
        self.stock_price = 1.0
        self.risky_income = 0.
        self.savings = np.zeros((self.households_n, 1))

        # Estate tax
        self.estate_tax = 0.
        # Pensions
        self.pension = np.zeros((self.households_n, 1))  # Initialize to zero for each household

        self.real_action_max = np.array(self.real_action_max)
        self.real_action_min = np.array(self.real_action_min)
        real_action_size = np.array(self.real_action_max).size

        # Adjust action space size if needed
        if real_action_size < self.action_dim:
            self.real_action_max = np.concatenate([self.real_action_max, np.ones(self.action_dim - real_action_size)])
            self.real_action_min = np.concatenate([self.real_action_min, np.zeros(self.action_dim - real_action_size)])
            self.initial_action = np.concatenate((self.initial_action, self.investment_init), axis=1)

        if 'OLG' in self.type:
            self.working_years = np.zeros((self.households_n, 1))
            self.accumulated_pension_account = np.zeros((self.households_n, 1))

    def households_init(self):
        data = self.get_real_data()
        self.real_e = data[1]
        self.at_init, self.e_init, self.it_init, self.age_init, self.work_init, self.c_init, self.investment_init = self.sample_real_data(
            data)
        self.generate_c_init(self.age_init)
        self.saving_init = 1 - self.c_init
        self.e_initial(self.households_n)
        self.initial_action = np.concatenate((self.saving_init, self.work_init * self.h_max), axis=1)

    def get_real_data(self, age_limit=None):
        df = pd.read_csv(os.path.join(os.path.abspath('.'), "agents/data/advanced_scfp2022_1110.csv"))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if age_limit is not None:
            df = df[df['AGE'] == age_limit]
            if df.empty:
                raise ValueError(f"No data available for individuals aged {age_limit}.")

        columns = ['ASSET', 'EDUC', 'INCOME', 'AGE', 'LF']
        data = [df[col].values for col in columns]
        consumption_p = (df['FOODHOME'].values + df['FOODAWAY'].values + df['FOODDELV'].values + df['RENT'].values + df[
            'TPAY'].values + 0.0001) / (df['ASSET'].values + 0.0001)
        invest_p = df['FIN'].values / (df['ASSET'].values + 0.0001)
        data.append(consumption_p)
        data.append(invest_p)
        WGT = df['WGT'].values
        WGT = WGT / np.sum(WGT)
        data.append(WGT)

        return data

    def sample_real_data(self, data):
        probabilities = data[-1]
        # index = np.random.choice(range(len(data[0])), self.households_n, replace=False, p=probabilities)
        index = np.random.choice(range(len(data[0])), self.households_n, replace=True, p=probabilities)
        return [d[index].reshape(self.households_n, 1) for d in data[:-1]]

    def get_action(self, actions, firm_n):
        self.generate_e_ability()
        self.at = copy.copy(self.at_next)  # Reset at to latest value
        saving_p = actions[:, 0][:, np.newaxis]
        self.consumption_p = 1 - saving_p  # Forward consumption allowed
        self.ht = actions[:, 1][:, np.newaxis] * self.h_max  # Labor supply

        # Risk investment decision
        if "risk_invest" in self.type:
            self.investment_p = actions[:, 2][:, np.newaxis]  # Risk investment ratio
        else:
            self.investment_p = np.zeros((self.households_n, 1))

        if firm_n != 1:
            # Work firm index from scaled action value
            work_firm_index = (actions[:, -firm_n - 1][:, np.newaxis] * firm_n).astype(int).clip(0,
                                                                                                 firm_n - 1).flatten()
            self.h_ij_ratio = np.eye(firm_n)[work_firm_index]  # One-hot encoding

            # Normalize consumption distribution across firms
            raw_cij = actions[:, -firm_n:]
            self.c_ij_ratio = raw_cij / (np.sum(raw_cij, axis=1, keepdims=True) + 1e-8)
        else:
            self.h_ij_ratio = 1
            self.c_ij_ratio = 1

    def step(self, society, t):
        # Step forward in time based on household type
        if "OLG" in self.type:
            self.OLG_step(society, t)
        elif "ramsey" in self.type:
            self.ramsey_step(society)
        else:
            raise ValueError("Households Wrong Type Choice!")

    def ramsey_step(self, society):
        """
        Perform a full economic decision step for an individual agent under the Ramsey model.

        This includes labor and capital income computation, taxation, consumption decision,
        and asset allocation between savings and risky investments.

        Args:
            society: The full economic environment (agents, government, market, etc.)
        """
        # Try 'tax', then 'central_bank', otherwise another key.
        government_agent = (
                society.government.get('tax')
                or society.government.get('central_bank')
                or society.government.get('pension')
        )
        # government_agent = society.government.get('tax', society.government.get('central_bank', 'pension'))
        # === Step 1: Compute total income ===
        # Labor income: effort * effective hours * wage rate
        labor_income = self.e * np.dot(self.ht * self.h_ij_ratio, society.market.WageRate)

        # Capital income: based on current savings (bank deposit or loan)
        is_deposit = (self.savings >= 0).astype(float)  # shape (N, 1)
        is_loan = 1.0 - is_deposit  # shape (N, 1)

        saving_interest = is_deposit * society.bank.deposit_rate * self.savings + \
                          is_loan * society.bank.lending_rate * self.savings

        # Total income includes labor, saving interest, and risky asset return from last step
        self.income = labor_income + saving_interest + self.risky_income

        # === Step 2: Taxation ===
        self.income_tax, self.asset_tax = government_agent.compute_tax(self.income, self.at)
        self.post_income = self.income - self.income_tax
        self.post_asset = self.at - self.asset_tax
        total_wealth = self.post_income + self.post_asset

        # === Step 3: Consumption decision ===
        money_for_consumption = self.consumption_p * self.post_income / (1 + society.consumption_tax_rate)
        money_for_consumption = np.maximum(money_for_consumption, 0.0)  # no negative consumption

        # Prevent division by zero in price
        if np.any(society.market.price.T == 0):
            raise ValueError("Price contains zero values, which can cause division by zero.")

        # Compute per-good consumption and aggregate CES consumption (e.g., Dixit–Stiglitz)
        consumption_ij = (money_for_consumption * self.c_ij_ratio) / society.market.price.T

        households_demand = np.sum(consumption_ij, axis=0).reshape(-1, 1)
        goods_supply = society.market.Yt_j
        success_households_deals = np.minimum(households_demand, goods_supply)

        self.final_consumption = consumption_ij / (np.sum(consumption_ij,
                                                          axis=0) + 1e-8) * success_households_deals.T  # Proportionally distribute the sold goods among all households.
        self.consumption = self.compute_ces_consumption(consumption_ij=self.final_consumption,
                                                        epsilon=society.market.epsilon)
        money_for_consumption = np.sum(self.final_consumption * society.market.price.T, axis=1).reshape(-1, 1)

        # === Step 4: Compute next-period asset ===
        self.at_next = total_wealth - money_for_consumption

        if np.isnan(self.at_next).any() or np.isinf(self.at_next).any():
            raise ValueError("Invalid at_next: NaN or Inf encountered.")

        # === Step 5: Asset allocation ===
        money_risky_invest = self.investment_p * self.at_next
        self.savings = self.at_next - money_risky_invest

        # === Step 6: Update risky investment returns ===
        self.update_stock_market(money_risky_invest)
        self.risky_income = self.stock_holdings * self.stock_price - money_risky_invest

        # === Step 7: Log final values for writing ===
        self.at_next_write = copy.copy(self.at_next)
        self.age_write = copy.copy(self.age)

    def compute_ces_consumption(self, consumption_ij, epsilon: float):
        if epsilon == 1.0:
            consumption = np.exp(np.mean(np.log(consumption_ij + 1e-8), axis=1))[:, np.newaxis]
        else:
            rho = (epsilon - 1) / epsilon
            ces_inner = np.power(consumption_ij + 1e-8, rho)  # shape: (N, n_f)
            ces_sum = np.sum(ces_inner, axis=1)  # shape: (N,)
            consumption = np.power(ces_sum, 1 / rho)[:, np.newaxis]  # shape: (N, 1)
        return consumption

    def update_stock_market(self, money_risky_investment):
        """
        Encapsulate stock market investment and price update logic.
        :param money_risky_investment: Amount of money for each household (n_households, 1)
        """
        # Calculate current stock market value held
        current_stock_value = self.stock_holdings * self.stock_price  # (n_households, 1)

        # Determine buy or sell amount
        # Positive value means buying, negative value means selling
        buy_sell_amount = money_risky_investment - current_stock_value

        # Update stock holdings with protection against zero price
        if self.stock_price != 0:
            self.stock_holdings += buy_sell_amount / self.stock_price
        else:
            # Handle scenario where stock price is zero (e.g., initialize or skip)
            pass

        # Calculate net buying/selling volume
        imbalance = np.sum(buy_sell_amount)

        # Update stock price based on market imbalance
        total_stock_value = np.sum(self.stock_holdings) * self.stock_price
        if total_stock_value > 0 and not np.isclose(imbalance, 0):
            # Adjust price based on imbalance ratio
            self.stock_price *= (1 + self.stock_alpha * imbalance / total_stock_value)

            # Optional: Add price floor to prevent non-positive prices
            self.stock_price = max(self.stock_price, 1e-6)

    # 假设 self.e_array 是你的二维数组，born_n 是需要选择的行数
    def select_newborn_data(self, born_n):
        if born_n > self.e_array.shape[0]:
            raise ValueError("born_n is larger than the number of available rows in e_array.")

        # 生成不重复的行索引
        selected_indices = np.random.choice(self.e_array.shape[0], size=born_n, replace=False)

        # 根据选择的索引提取对应的行，并 reshape 成 (born_n, 1) 的形式
        newborn_e_array = self.e_array[selected_indices].reshape(born_n, self.e_array.shape[1])

        # newborn_accumulated_pension_account = self.accumulated_pension_account[selected_indices].reshape(born_n, self.accumulated_pension_account.shape[1])
        newborn_accumulated_pension_account = np.zeros((born_n, 1))

        return newborn_e_array, newborn_accumulated_pension_account

    def OLG_step(self, society, t):
        """Calculate households' income, assets, and consumption."""
        # Classify households as young or old based on age
        government_agent = society.main_gov
        retire_age = government_agent.retire_age

        self.is_old = self.age >= retire_age
        self.old_n = np.sum(self.is_old)
        self.ht[self.is_old] = 0

        # Labor income: effort * effective hours * wage rate
        labor_income = self.e * np.dot(self.ht * self.h_ij_ratio, society.market.WageRate)

        # Capital income: based on current savings (bank deposit or loan)
        is_deposit = (self.savings >= 0).astype(float)  # shape (N, 1)
        is_loan = 1.0 - is_deposit  # shape (N, 1)

        saving_interest = is_deposit * society.bank.deposit_rate * self.savings + \
                          is_loan * society.bank.lending_rate * self.savings

        # Total income includes labor, saving interest, and risky asset return from last step
        self.income = labor_income + saving_interest + self.risky_income
        if hasattr(self, 'BUI'): self.income += self.BUI

        # === Step 2: Taxation ===
        self.income_tax, self.asset_tax = government_agent.compute_tax(self.income, self.at)
        self.post_income = self.income - self.income_tax
        self.post_asset = self.at - self.asset_tax

        self.pension = government_agent.calculate_pension(self)
        self.accumulated_pension_account[~self.is_old] -= self.pension[
            ~self.is_old]  # Young individuals' pension contributions are deposited into the national pension pool.

        total_wealth = self.post_income + self.post_asset + self.pension

        self.working_years[~self.is_old] += 1

        # === Step 3: Consumption decision ===
        money_for_consumption = self.consumption_p * (self.post_income + self.pension) / (
                1 + society.consumption_tax_rate)
        money_for_consumption = np.maximum(money_for_consumption, 0.0)  # no negative consumption

        # Prevent division by zero in price
        if np.any(society.market.price.T == 0):
            raise ValueError("Price contains zero values, which can cause division by zero.")

        # Compute per-good consumption and aggregate CES consumption (e.g., Dixit–Stiglitz)
        consumption_ij = (money_for_consumption * self.c_ij_ratio) / society.market.price.T

        households_demand = np.sum(consumption_ij, axis=0).reshape(-1, 1)
        goods_supply = society.market.Yt_j
        success_households_deals = np.minimum(households_demand, goods_supply)

        self.final_consumption = consumption_ij / (np.sum(consumption_ij,
                                                          axis=0) + 1e-8) * success_households_deals.T  # Proportionally distribute the sold goods among all households.
        self.consumption = self.compute_ces_consumption(consumption_ij=self.final_consumption,
                                                        epsilon=society.market.epsilon)
        money_for_consumption = np.sum(self.final_consumption * society.market.price.T, axis=1).reshape(-1, 1)

        # === Step 4: Compute next-period asset ===
        self.at_next = total_wealth - money_for_consumption

        if np.isnan(self.at_next).any() or np.isinf(self.at_next).any():
            raise ValueError("Invalid at_next: NaN or Inf encountered.")

        # === Step 5: Asset allocation ===
        money_risky_invest = self.investment_p * self.at_next
        self.savings = self.at_next - money_risky_invest

        # === Step 6: Update risky investment returns ===
        self.update_stock_market(money_risky_invest)
        self.risky_income = self.stock_holdings * self.stock_price - money_risky_invest

        # === Step 7: Log final values for writing ===
        self.at_next_write = copy.copy(self.at_next)
        self.age_write = copy.copy(self.age)
        self.e_write = copy.copy(self.e)

        # Age update
        self.age += 1
        if t != 0:

            self.birth_rate = self.entity_args['OLG'].birth_rate
            born_n = int(self.households_n * self.birth_rate)  # Number of newborn households

            die_total, all_eliminate_indices = self.sample_deaths_by_probability(
                age_array=self.age,
                max_age=102
            )

            self.variables_to_sort = [
                'age', 'at', 'e', 'at_next', 'income', 'e_array', 'accumulated_pension_account', 'working_years',
                'stock_holdings', 'ht', 'consumption', 'final_consumption', 'income_tax', 'asset_tax', 'pension',
                'post_income', 'risky_income', 'savings'
            ]
            total_wealth_deceased = 0
            # born_n = born_n + die_total - die_n

            if die_total > 0:
                total_wealth_deceased, self.estate_tax = self.compute_estate_tax(die_total, society)
                deceased_stock_value = np.sum(self.stock_holdings[all_eliminate_indices] * self.stock_price)
                for var in self.variables_to_sort:
                    current_values = getattr(self, var)
                    setattr(self, var, np.delete(current_values, all_eliminate_indices, axis=0))

            if born_n > 0:
                n_ages = np.ones((born_n, 1)) * self.initial_working_age
                if die_total > 0:
                    initial_wealth = total_wealth_deceased / born_n
                    if total_wealth_deceased > 0 and self.stock_price > 0:
                        stock_proportion = deceased_stock_value / total_wealth_deceased  # 股票占遗产的比例
                        newborn_stock_holdings = (initial_wealth * stock_proportion) / self.stock_price
                    else:
                        newborn_stock_holdings = 0
                else:
                    initial_wealth = 0
                    newborn_stock_holdings = 0

                n_assets = np.full((born_n, 1), initial_wealth)
                n_e = np.random.choice(self.real_e, born_n, replace=False).reshape(born_n, 1)
                n_e_array, n_ht = self.select_newborn_data(born_n)
                n_working_years = np.zeros((born_n, 1))
                n_accumulated_pension_account = np.zeros((born_n, 1))
                n_income = np.zeros((born_n, 1))
                n_final_consumption = np.zeros((born_n, society.market.firm_n))
                n_consumption = np.zeros((born_n, 1))
                n_income_tax = np.zeros((born_n, 1))
                n_asset_tax = np.zeros((born_n, 1))
                n_pension = np.zeros((born_n, 1))
                n_risky_income = np.zeros((born_n, 1))
                n_savings = np.zeros((born_n, 1))

                newborn_variables = {
                    'age': n_ages,
                    'at': n_assets,
                    'e': n_e,
                    'at_next': n_assets,
                    'income': n_income,
                    'e_array': n_e_array,
                    'accumulated_pension_account': n_accumulated_pension_account,
                    'working_years': n_working_years,
                    'stock_holdings': np.full((born_n, 1), newborn_stock_holdings),  # 新增 stock_holdings
                    'ht': n_ht,
                    'consumption': n_consumption,
                    'final_consumption': n_final_consumption,
                    'income_tax': n_income_tax,
                    'asset_tax': n_asset_tax,
                    'pension': n_pension,
                    'post_income': n_income,
                    'risky_income': n_risky_income,
                    'savings': n_savings,
                    # 'households_death_rate': n_death_rate,
                }
                for var, value in newborn_variables.items():
                    setattr(self, var, np.vstack((getattr(self, var), value)))
            else:
                if total_wealth_deceased > 0:
                    # **修改部分**：将遗产按比例分配，包括股票部分
                    assignment_assets = total_wealth_deceased / self.households_n
                    self.at += assignment_assets
                    if total_wealth_deceased > 0 and self.stock_price > 0:
                        stock_proportion = deceased_stock_value / total_wealth_deceased
                        self.stock_holdings += (assignment_assets * stock_proportion) / self.stock_price

        # Update population count
        self.households_n = len(self.age)
        self.is_old = self.age >= retire_age
        self.old_percent = self.old_n / max(self.households_n, 1e-8)  # old / all_population
        self.dependency_ratio = self.old_n / (
                self.households_n - self.old_n + 1e-8)  # Dependency ratio，measure the pressure of pension

    def calculate_death_probability(self, age_array):
        """Vectorized function to return death probability by age."""
        prob = np.zeros_like(age_array, dtype=np.float32)

        prob[age_array < 1] = 0.0056
        prob[(age_array >= 1) & (age_array < 5)] = 28.0 / 100000
        prob[(age_array >= 5) & (age_array < 15)] = 15.3 / 100000
        prob[(age_array >= 15) & (age_array < 25)] = 79.5 / 100000
        prob[(age_array >= 25) & (age_array < 35)] = 163.4 / 100000
        prob[(age_array >= 35) & (age_array < 45)] = 255.4 / 100000
        prob[(age_array >= 45) & (age_array < 55)] = 453.3 / 100000
        prob[(age_array >= 55) & (age_array < 65)] = 992.1 / 100000
        prob[(age_array >= 65) & (age_array < 75)] = 1978.7 / 100000
        prob[(age_array >= 75) & (age_array < 85)] = 4708.2 / 100000
        prob[age_array >= 85] = 14389.6 / 100000

        return np.clip(prob, 0, 1)

    def sample_deaths_by_probability(self, age_array, max_age=102):
        """Sample death events based on age-dependent probabilities."""
        age_flat = age_array.flatten()
        death_probs = self.calculate_death_probability(age_flat)
        death_events = np.random.binomial(n=1, p=death_probs)
        sampled_indices = np.where(death_events == 1)[0]
        forced_indices = np.where(age_flat >= max_age)[0]

        all_eliminate_indices = np.union1d(sampled_indices, forced_indices)
        die_total = len(all_eliminate_indices)

        return die_total, all_eliminate_indices

    def compute_estate_tax(self, die_n, society):
        """Compute inheritance received and estate tax based on exemption and tax rate."""
        at_die = self.at[:die_n]
        total_inherited = np.sum(
            np.where(
                at_die <= society.estate_tax_exemption,
                at_die,
                society.estate_tax_exemption + (at_die - society.estate_tax_exemption) * (1 - society.estate_tax_rate)
            )
        )

        estate_tax = np.sum(at_die) - total_inherited
        return total_inherited, estate_tax

    def get_reward(self, consumption=None, working_hours=None, alpha=0.5, beta=5):
        """Compute household utility based on CRRA utility \in (-10,15) of consumption and disutility of labor."""
        if consumption is None:
            consumption = self.consumption  # Dixit–Stiglitz

        if working_hours is None:
            working_hours = self.ht

        working_ratio = working_hours / 2512 * beta
        crra = self.CRRA
        if 1 - crra == 0:
            utility_c = np.log((consumption + 1e-8))
        else:
            utility_c = (consumption ** (1 - crra)) / (1 - crra)

        if 1 + self.IFE == 0:
            utility_h = np.log(working_ratio)
        else:
            utility_h = (working_ratio ** (1 + self.IFE) / (1 + self.IFE))

        # Calculate total utility and apply an offset to ensure positive rewards
        current_utility = utility_c - alpha * utility_h

        return current_utility

    def sigmoid(self, x):
        """Numerically stable sigmoid function."""
        x_clipped = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x_clipped))

    def is_terminal(self):
        """Determine whether simulation should terminate due to collapse or invalid state."""
        if self.households_n <= 5 and "OLG" in self.type:
            return True
        else:
            # Unreasonably large borrowing
            if self.at_next.min() < self.at_min or np.isnan(self.at_next).any():
                return True
            else:
                return False

    def close(self):
        pass
