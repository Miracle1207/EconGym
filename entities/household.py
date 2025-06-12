from omegaconf import omegaconf

from entities.base import BaseEntity
import numpy as np
import copy
import math
import pandas as pd
import random
# import quantecon as qe
import matplotlib.pyplot as plt
import os, sys
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.special import expit
from gym.spaces import Box


class Household(BaseEntity):
    name = 'household'
    
    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        if "risk_invest" in self.type:
            self.action_dim += 1
        self.policy_action_len = copy.copy(self.action_dim)
        self.households_init()
        
        if 'OLG' in self.type or 'personal_pension' in self.type:
            self.__dict__.update(entity_args.OLG)
        
        self.best_loss = 100

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
        """
        self.e_past = copy.copy(self.e_array)
        e_past_mean = sum(self.e_past[:, 0]) / np.count_nonzero(self.e_past[:, 0])
        for i in range(self.households_n):
            positive_values = self.e_past[self.e_past[:, 0] > 0, 0]  # If < 0, randomly pick one from positive values
            is_superstar = (self.e_array[i, 1] > 0).astype(int)
            if is_superstar == 0:
                # normal state
                if np.random.rand() < self.e_p:
                    # transition from normal to super-star
                    self.e_array[i, 0] = 0
                    self.e_array[i, 1] = self.super_e * e_past_mean
                else:
                    # remain in normal
                    self.e_array[i, 1] = 0
                    if self.e_past[i, 0] <= 0:
                        self.e_past[i, 0] = np.random.choice(positive_values)  # avoid NaN by using small value
                    self.e_array[i, 0] = np.exp(
                        self.rho_e * np.log(self.e_past[i, 0]) + self.sigma_e * np.random.randn())
            else:
                # super-star state
                if np.random.rand() < self.e_q:
                    # remain in super-star
                    self.e_array[i, 0] = 0
                    self.e_array[i, 1] = self.super_e * e_past_mean
                else:
                    # transition to normal
                    self.e_array[i, 1] = 0
                    self.e_array[i, 0] = random.uniform(self.e_array[:, 0].min(), self.e_array[:, 0].max())
        self.e = np.sum(self.e_array, axis=1, keepdims=True)
    
    def reset_action_space(self):
        original_low = self.action_space.low
        original_high = self.action_space.high
        new_shape = (self.households_n, self.action_dim)
        # 全部裁剪
        new_low = original_low[:self.households_n, :]
        new_high = original_high[:self.households_n, :]
        self.action_space = Box(low=new_low, high=new_high, shape=new_shape, dtype=np.float32)
    
    def generate_c_init(self, age):
        self.c_init = np.zeros_like(age, dtype=float)
        # Forward consumption: increase consumption ratio
        for i, a in enumerate(age):
            if a < 24:
                self.c_init[i] = 0.2 + np.random.normal(0, 0.05)  # Young people consume less
            elif 25 <= a < 34:
                self.c_init[i] = 0.5 + np.random.normal(0, 0.05)  # Consumption increases in young adults
            elif 35 <= a < 54:
                self.c_init[i] = 0.7 + np.random.normal(0, 0.05)  # Peak consumption in middle age
            elif 55 <= a < 74:
                self.c_init[i] = 0.6 + np.random.normal(0, 0.05)  # Consumption starts to decline
            else:
                self.c_init[i] = 0.4 + np.random.normal(0, 0.05)  # Elderly consume even less
    
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

        # Estate tax
        self.estate_tax = 0.
        # Pensions
        self.pension = np.zeros((self.households_n, 1))  # Initialize to zero for each household

        if isinstance(self.action_space, omegaconf.DictConfig):
            self.action_space = Box(low=self.action_space.low,
                                    high=self.action_space.high,
                                    shape=(self.households_n, self.action_dim), dtype=np.float32)
        else:
            self.update_action_space()

        self.real_action_max = np.array(self.real_action_max)
        self.real_action_min = np.array(self.real_action_min)
        real_action_size = np.array(self.real_action_max).size

        # Adjust action space size if needed
        if real_action_size < self.action_dim:
            self.real_action_max = np.concatenate([self.real_action_max, np.ones(self.action_dim - real_action_size)])
            self.real_action_min = np.concatenate([self.real_action_min, np.zeros(self.action_dim - real_action_size)])
            self.initial_action = np.concatenate((self.initial_action, self.investment_init), axis=1)

        if 'OLG' in self.type or 'personal_pension' in self.type:
            self.working_years = np.zeros((self.households_n, 1))
            self.accumulated_pension_account = np.zeros((self.households_n, 1))

    def households_init(self):
        data = self.get_real_data()
        self.real_e = data[1]
        self.at_init, self.e_init, self.it_init, self.age_init, self.work_init, self.c_init, self.investment_init = self.sample_real_data(data)
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
        consumption_p = (df['FOODHOME'].values + df['FOODAWAY'].values + df['FOODDELV'].values + df['RENT'].values + df['TPAY'].values + 0.0001) / (df['ASSET'].values + 0.0001)
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
        self.ht = actions[:, 1][:, np.newaxis]  # Labor supply

        # Risk investment decision
        if "risk_invest" in self.type:
            self.investment_p = actions[:, 2][:, np.newaxis]  # Risk investment ratio
        else:
            self.investment_p = np.zeros((self.households_n, 1))

        if firm_n != 1:
            # Work firm index from scaled action value
            work_firm_index = (actions[:, -firm_n - 1][:, np.newaxis] * firm_n).reshape(-1).astype(int).clip(0, firm_n - 1)
            self.h_ij_ratio = np.eye(firm_n)[work_firm_index]  # One-hot encoding

            # Normalize consumption distribution across firms
            raw_cij = actions[:, -firm_n:]
            self.c_ij_ratio = raw_cij / (np.sum(raw_cij, axis=1, keepdims=True) + 1e-8)
        else:
            self.h_ij_ratio = 1
            self.c_ij_ratio = 1

    def step(self, society, t):
        # Step forward in time based on household type
        if "OLG" in self.type or 'personal_pension' in self.type:
            self.OLG_step(society, t)
        elif "ramsey" in self.type:
            self.ramsey_step(society)
        else:
            raise ValueError("Households Wrong Type Choice!")

    def ramsey_step(self, society):
        """Ramsey-style update for income, tax, consumption, and assets."""
        labor_income = self.e * np.dot(self.ht * self.h_ij_ratio, society.market.WageRate)
        capital_income = society.bank.deposit_rate * self.at
        self.income = labor_income + capital_income

        self.income_tax, self.asset_tax = society.main_gov.compute_tax(self.income, self.at)
        self.post_income = self.income - self.income_tax
        self.post_asset = self.at - self.asset_tax
        total_wealth = self.post_income + self.post_asset

        # Determine money available for consumption
        money_for_consumption = self.consumption_p * self.post_income / (1 + society.consumption_tax_rate)
        money_for_consumption = np.where(money_for_consumption < 0, 0, money_for_consumption)
        if np.any(society.market.price.T == 0):
            raise ValueError("Price contains zero values, which can cause division by zero.")
        self.consumption_ij = (money_for_consumption * self.c_ij_ratio) / society.market.price.T
        self.consumption = self.compute_ces_consumption(epsilon=society.market.epsilon)  # Dixit–Stiglitz

        # Calculate savings and update assets
        all_money_investment = total_wealth - money_for_consumption
        self.update_stock_market(all_money_investment)
        savings = all_money_investment - (self.investment_p * all_money_investment)
        self.at_next = savings * (1 + society.bank.deposit_rate) + self.stock_holdings * self.stock_price
        self.at_next_write = copy.copy(self.at_next)
        self.age_write = copy.copy(self.age)
        

    def compute_ces_consumption(self, epsilon: float):
        if epsilon == 1.0:
            consumption = np.exp(np.mean(np.log(self.consumption_ij + 1e-8), axis=1))[:, np.newaxis]
        else:
            rho = (epsilon - 1) / epsilon
            ces_inner = np.power(self.consumption_ij + 1e-8, rho)  # shape: (N, n_f)
            ces_sum = np.sum(ces_inner, axis=1)  # shape: (N,)
            consumption = np.power(ces_sum, 1 / rho)[:, np.newaxis]  # shape: (N, 1)
        return consumption
    
    def update_action_space(self):
        original_low = self.action_space.low
        original_high = self.action_space.high
        
        new_shape = (self.households_n, self.action_dim)
        
        households_n_old = original_low.shape[0]
        if households_n_old < self.households_n:
            last_row_low = original_low[-1].reshape(1, -1)
            last_row_high = original_high[-1].reshape(1, -1)
            
            repeat_times = self.households_n - households_n_old
            new_rows_low = np.tile(last_row_low, (repeat_times, 1))
            new_rows_high = np.tile(last_row_high, (repeat_times, 1))
            
            new_low = np.vstack((original_low, new_rows_low))
            new_high = np.vstack((original_high, new_rows_high))
        elif households_n_old > self.households_n:
            new_low = original_low[:self.households_n, :]
            new_high = original_high[:self.households_n, :]
        else:
            new_low = original_low
            new_high = original_high
        
        self.action_space = Box(low=new_low, high=new_high, shape=new_shape, dtype=np.float32)
    
    def update_stock_market(self, savings):
        """
        封装股市投资和价格更新逻辑。
        :param savings: 每个家庭的储蓄金额 (n_households, 1)
        """
        # 计算目标投资金额
        target_investment = self.investment_p * savings  # (n_households, 1)
        
        # 计算当前股票市值
        current_stock_value = self.stock_holdings * self.stock_price  # (n_households, 1)
        
        # 决定买入或卖出
        buy_sell_amount = target_investment - current_stock_value  # >0, 买入; <0, 卖出
        
        # 更新股票持有量
        self.stock_holdings += buy_sell_amount / self.stock_price  # 买入
        
        total_buy = np.sum(buy_sell_amount[buy_sell_amount > 0])  # 总买入
        total_sell = np.sum(np.abs(buy_sell_amount[buy_sell_amount < 0]))  # 总卖出
        # 计算成交量（所有交易的绝对值之和）
        total_volume = np.sum(np.abs(buy_sell_amount))
        imbalance = np.sum(buy_sell_amount)  # 净买入卖出量
        
        # 更新股票价格
        total_stock_value = np.sum(self.stock_holdings) * self.stock_price  # 总股票市值
        if total_stock_value > 0:
            self.stock_price *= (1 + self.stock_alpha * imbalance / total_stock_value)
    
    import numpy as np
    
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
        if hasattr(society, 'pension_gov'):
            retire_age = society.pension_gov.retire_age
        else:
            retire_age = society.main_gov.retire_age

        self.is_old = self.age >= retire_age
        self.old_n = np.sum(self.is_old)
        # self.ht[:self.old_n] = np.zeros_like(self.ht[:self.old_n])
        self.ht[self.is_old] = 0
        
        # Calculate income, taxes, and post-income for all households
        labor_income = self.e * np.dot(self.ht * self.h_ij_ratio, society.market.WageRate)
        labor_income[self.is_old] = 0
        
        capital_income = society.bank.deposit_rate * self.at
        self.income = labor_income + capital_income
        self.income_tax, self.asset_tax = society.main_gov.compute_tax(self.income, self.at)
        self.post_income = self.income - self.income_tax
        self.post_asset = self.at - self.asset_tax
        
        self.pension = society.main_gov.calculate_pension(self)
        self.accumulated_pension_account[~self.is_old] -= self.pension[~self.is_old]
        total_wealth = self.post_income + self.post_asset + self.pension
        
        self.working_years[~self.is_old] += 1
        
        # Consumption
        # money_for_consumption = self.consumption_p * total_wealth / ( 1 + society.consumption_tax_rate)  # 消费比例 = 消费/总财富。 之前版本是 比例=消费/收入
        money_for_consumption = self.consumption_p * (self.post_income + self.pension) / (
                1 + society.consumption_tax_rate)  # 消费比例 = 消费/总财富。 之前版本是 比例=消费/收入
        money_for_consumption = np.where(money_for_consumption < 0, 0, money_for_consumption)
        if np.any(society.market.price.T == 0):
            raise ValueError("Price contains zero values, which can cause division by zero.")
        self.consumption_ij = (money_for_consumption * self.c_ij_ratio) / society.market.price.T
        self.consumption = self.compute_ces_consumption(epsilon=society.market.epsilon)  # Dixit–Stiglitz
        
        # 计算储蓄（理财部分）
        all_money_investment = total_wealth - money_for_consumption  # (n_households, 1)
        
        # 调用股市更新函数
        self.update_stock_market(all_money_investment)
        
        # 更新下一期财富
        savings = all_money_investment - (self.investment_p * all_money_investment)  # 储蓄
        self.at_next = savings * (
                1 + society.bank.deposit_rate) + self.stock_holdings * self.stock_price  # 下一期的财富 = 储蓄以及利息 + 风险投资收益
        
        self.at_next_write = copy.copy(self.at_next)
        self.age_write = copy.copy(self.age)
        
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
                'stock_holdings', 'ht', 'consumption_ij', 'income_tax', 'asset_tax', 'pension',
                'post_income',
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
                n_consumption_ij = np.zeros((born_n, 1))
                n_incoem_tax = np.zeros((born_n, 1))
                n_asset_tax = np.zeros((born_n, 1))
                n_pension = np.zeros((born_n, 1))

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
                    'consumption_ij': n_consumption_ij,
                    'income_tax': n_incoem_tax,
                    'asset_tax': n_asset_tax,
                    'pension': n_pension,
                    'post_income': n_income,
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
        self.update_action_space()
        self.old_percent = self.old_n / self.households_n  # 当前老年人口比例
        self.dependency_ratio = self.old_n / (
                    self.households_n - self.old_n + 1e-8)  # Dependency ratio（赡养比），用来衡量养老压力或抚养负担。
    
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


    def get_reward(self, consumption=None, working_hours=None, alpha=6.68e-6):
        """Compute household utility based on CRRA utility of consumption and disutility of labor."""
        if consumption is None:
            consumption = self.consumption   # Dixit–Stiglitz

        if working_hours is None:
            working_hours = self.ht

        crra = self.CRRA
        if 1 - crra == 0:
            utility_c = np.log((consumption + 1e-8))
        else:
            utility_c = (consumption ** (1 - crra)) / (1 - crra)

        if 1 + self.IFE == 0:
            utility_h = np.log(working_hours)
        else:
            utility_h = (working_hours ** (1 + self.IFE) / (1 + self.IFE))

        current_utility = utility_c - alpha * utility_h + 21  # 21 is max disutility
        return current_utility

    def sigmoid(self, x):
        """Numerically stable sigmoid function."""
        x_clipped = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x_clipped))

    def is_terminal(self):
        """Determine whether simulation should terminate due to collapse or invalid state."""
        if self.households_n <= 2:
            return True
        else:
            if self.at_next.min() < self.at_min or np.isnan(self.at_next).any():
                return True
            else:
                return False

    def close(self):
        pass
