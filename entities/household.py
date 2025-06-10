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
        self.e_array[:, 0] = (random_set > self.e_p).astype(int) * self.e_init.flatten()  # 小于e_p置为0
        self.e_array[:, 1] = (random_set < self.e_p).astype(int)  # 小于e_p置为1
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
            positive_values = self.e_past[self.e_past[:, 0] > 0, 0]  # 小于0时，随机获取一个
            is_superstar = (self.e_array[i, 1] > 0).astype(int)
            if is_superstar == 0:
                # normal state
                if np.random.rand() < self.e_p:
                    # transit from normal to super-star
                    self.e_array[i, 0] = 0
                    self.e_array[i, 1] = self.super_e * e_past_mean
                else:
                    # remain in normal
                    self.e_array[i, 1] = 0
                    # 当e_array[i,0]为负数时，出现了nan 将其改为一个很小的值
                    if self.e_past[i, 0] <= 0:
                        self.e_past[i, 0] = np.random.choice(positive_values)
                    self.e_array[i, 0] = np.exp(
                        self.rho_e * np.log(self.e_past[i, 0]) + self.sigma_e * np.random.randn())
            else:
                # super state
                if np.random.rand() < self.e_q:
                    # remain in super-star
                    self.e_array[i, 0] = 0
                    self.e_array[i, 1] = self.super_e * e_past_mean
                else:
                    # transit to normal
                    self.e_array[i, 1] = 0
                    self.e_array[i, 0] = random.uniform(self.e_array[:, 0].min(), self.e_array[:, 0].max())  # 随机来一个
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
        # 超前消费，增加消费比
        for i, a in enumerate(age):
            if a < 24:
                self.c_init[i] = 0.2 + np.random.normal(0, 0.05)  # 年轻人消费比低
            elif 25 <= a < 34:
                self.c_init[i] = 0.5 + np.random.normal(0, 0.05)  # 中青年消费比逐渐增加
            elif 35 <= a < 54:
                self.c_init[i] = 0.7 + np.random.normal(0, 0.05)  # 中年人消费比较高
            elif 55 <= a < 74:
                self.c_init[i] = 0.6 + np.random.normal(0, 0.05)  # 老年人消费比逐渐减少
            else:
                self.c_init[i] = 0.4 + np.random.normal(0, 0.05)  # 高龄老年人消费比进一步减少
    
    def reset(self, **custom_cfg):
        self.households_n = self.entity_args.params.households_n  # reset 要重置number
        self.e = copy.deepcopy(self.e_0)
        self.e_array = copy.deepcopy(self.e_array_0)
        self.generate_e_ability()
        self.at, self.at_next = copy.deepcopy(self.at_init), copy.deepcopy(self.at_init)
        self.age, self.income = copy.deepcopy(self.age_init), copy.deepcopy(self.it_init)
        # 初始化一个与self.age相同大小的数组来存储死亡率
        # self.households_death_rate = np.array([self.calculate_death_probability(a[0]) for a in self.age]).reshape(-1, 1)
        
        # 初始化股票持有量和 investment_p
        self.stock_holdings = np.zeros((self.households_n, 1))
        self.investment_p = copy.copy(self.investment_init)  # 每个家庭初始化为 0
        self.ht = self.work_init * self.h_max
        self.stock_price = 1.0
        
        # 遗传税
        self.estate_tax = 0.
        # 养老金
        self.pension = np.zeros((self.households_n, 1))  # 每个家庭初始化为 0
        if isinstance(self.action_space, omegaconf.DictConfig):
            self.action_space = Box(low=self.action_space.low,
                                    high=self.action_space.high,
                                    shape=(self.households_n, self.action_dim), dtype=np.float32)
        else:
            self.update_action_space()
        
        self.real_action_max = np.array(self.real_action_max)
        self.real_action_min = np.array(self.real_action_min)
        real_action_size = np.array(self.real_action_max).size
        # 根据 self.type 动态设置 action_space
        if real_action_size < self.action_dim:
            self.real_action_max = np.concatenate([self.real_action_max, np.ones(self.action_dim - real_action_size)])
            self.real_action_min = np.concatenate([self.real_action_min, np.zeros(self.action_dim - real_action_size)])
            self.initial_action = np.concatenate((self.initial_action, self.investment_init), axis=1)
        
        if 'OLG' in self.type or 'personal_pension' in self.type:
            self.working_years = np.zeros((self.households_n, 1))
            self.accumulated_pension_account = np.zeros((self.households_n, 1))
    
    def households_init(self):
        data = self.get_real_data()
        # 采样数据中包含负数
        self.real_e = data[1]
        self.at_init, self.e_init, self.it_init, self.age_init, self.work_init, self.c_init, self.investment_init = self.sample_real_data(
            data)
        self.generate_c_init(self.age_init)
        
        self.saving_init = 1 - self.c_init
        # initial e
        self.e_initial(self.households_n)
        self.initial_action = np.concatenate((self.saving_init, self.work_init * self.h_max), axis=1)
    
    def get_real_data(self, age_limit=None):
        df = pd.read_csv(os.path.join(os.path.abspath('.'), "agents/data/advanced_scfp2022_1110.csv"))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if age_limit is not None:
            df = df[df['AGE'] == age_limit]
            if df.empty:
                raise ValueError(f"No data available for individuals aged {age_limit}.")
        
        # columns = ['NETWORTH', 'EDUC', 'INCOME', 'AGE', 'LF']
        columns = ['ASSET', 'EDUC', 'INCOME', 'AGE', 'LF']
        data = [df[col].values for col in columns]
        consumption_p = (df['FOODHOME'].values + df['FOODAWAY'].values + df['FOODDELV'].values + df['RENT'].values + df[
            'TPAY'].values + 0.0001) / (df['ASSET'].values + 0.0001)  # 不需要了？ 初始成高斯分布
        invest_p = df['FIN'].values / (df['ASSET'].values + 0.0001)
        data.append(consumption_p)
        data.append(invest_p)
        WGT = df['WGT'].values  # WGT are used to sample
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
        # 因reset at 排序出错了，重新copy
        self.at = copy.copy(self.at_next)
        saving_p = actions[:, 0][:, np.newaxis]
        # 超前消费     saving_p可以小于0
        self.consumption_p = 1 - saving_p
        self.ht = actions[:, 1][:, np.newaxis]
        
        # 根据 self.type 提取 investment_p
        if "risk_invest" in self.type:
            self.investment_p = actions[:, 2][:, np.newaxis]  # 提取风险投资比例
        else:
            #  if no risk_invest, investment_p = 0
            self.investment_p = np.zeros((self.households_n, 1))
        
        if firm_n != 1:
            # The (−firm_n−1)-th column is a scalar ∈ [−1, 1] used to determine the work firm index
            work_firm_index = (actions[:, -firm_n - 1][:, np.newaxis] * firm_n).reshape(-1).astype(int).clip(0, firm_n - 1)
            self.h_ij_ratio = np.eye(firm_n)[work_firm_index]
            # The last firm_n columns represent the consumption share across firms and should be normalized
            raw_cij = actions[:, -firm_n:]
            self.c_ij_ratio = raw_cij / (np.sum(raw_cij, axis=1, keepdims=True) + 1e-8)  # prevent division by zero

        else:
            self.h_ij_ratio = 1
            self.c_ij_ratio = 1
    
    def step(self, society, t):
        
        if "OLG" in self.type or 'personal_pension' in self.type:
            self.OLG_step(society, t)
        elif "ramsey" in self.type:
            self.ramsey_step(society)
        else:
            raise ValueError("Households Wrong Type Choice!")
        

    
    def ramsey_step(self, society):
        """Calculate households' income, assets, consumption, and stock investments."""
        # Households' income
        labor_income = self.e * np.dot(self.ht * self.h_ij_ratio, society.market.WageRate)
        capital_income = society.bank.deposit_rate * self.at
        self.income = labor_income + capital_income
        
        # Compute taxes
        self.income_tax, self.asset_tax = society.main_gov.compute_tax(self.income, self.at)
        
        # Post-tax income and assets
        self.post_income = self.income - self.income_tax
        self.post_asset = self.at - self.asset_tax
        total_wealth = self.post_income + self.post_asset
        
        # Consumption
        # money_for_consumption = self.consumption_p * total_wealth / (1 + society.consumption_tax_rate)
        money_for_consumption = self.consumption_p * self.post_income / (1 + society.consumption_tax_rate)
        money_for_consumption = np.where(money_for_consumption < 0, 0, money_for_consumption)
        if np.any(society.market.price.T == 0):
            raise ValueError("Price contains zero values, which can cause division by zero.")
        self.consumption_ij = (money_for_consumption * self.c_ij_ratio) / society.market.price.T
        
        # 计算储蓄（理财部分）
        # savings = self.saving_p * total_wealth # (n_households, 1)
        all_money_investment = total_wealth - money_for_consumption  # (n_households, 1)
        
        # 调用股市更新函数
        self.update_stock_market(all_money_investment)
        
        # 更新下一期财富
        savings = all_money_investment - (self.investment_p * all_money_investment)  # 储蓄
        self.at_next = savings * (
                1 + society.bank.deposit_rate) + self.stock_holdings * self.stock_price  # 下一期的财富 = 储蓄以及利息 + 风险投资收益
        self.at_next_write = copy.copy(self.at_next)
        self.age_write = copy.copy(self.age)
    
    def update_action_space(self):
        original_low = self.action_space.low
        original_high = self.action_space.high
        
        new_shape = (self.households_n, self.action_dim)
        
        households_n_old = original_low.shape[0]
        if households_n_old < self.households_n:
            # 如果原来的 household_n 小于新的 household_n，则重复最后一行来填充
            last_row_low = original_low[-1].reshape(1, -1)
            last_row_high = original_high[-1].reshape(1, -1)
            
            repeat_times = self.households_n - households_n_old
            new_rows_low = np.tile(last_row_low, (repeat_times, 1))
            new_rows_high = np.tile(last_row_high, (repeat_times, 1))
            
            new_low = np.vstack((original_low, new_rows_low))
            new_high = np.vstack((original_high, new_rows_high))
        elif households_n_old > self.households_n:
            # 如果原来的 household_n 大于新的 household_n，则截取前 households_n 行
            new_low = original_low[:self.households_n, :]
            new_high = original_high[:self.households_n, :]
        else:
            # 如果 household_n 相同，则直接使用原来的 low 和 high
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
        # e at 排序变了 这里的at和e 相互对应吗
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
            
            # 移除人的财产没有被继承 若出生率为0，无人继承 更新income 和wealth(at_next)
            # ：计算死亡家庭的股票资产并分配
            if die_total > 0:
                total_wealth_deceased, self.estate_tax = self.compute_estate_tax(die_total, society)
                # 计算死亡家庭的股票总市值（用于分配）
                deceased_stock_value = np.sum(self.stock_holdings[all_eliminate_indices] * self.stock_price)
                # 移除死亡家庭的属性
                for var in self.variables_to_sort:
                    current_values = getattr(self, var)
                    setattr(self, var, np.delete(current_values, all_eliminate_indices, axis=0))
            
            # **修改部分**：新生家庭的 stock_holdings 按比例分配
            if born_n > 0:
                n_ages = np.ones((born_n, 1)) * self.initial_working_age
                if die_total > 0:
                    initial_wealth = total_wealth_deceased / born_n
                    # 计算新生家庭的初始 stock_holdings（按财富比例分配）
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
                # Append newborn attributes to existing households
                # e_array也要更新
                # **修改部分**：更新 newborn_variables，添加 stock_holdings
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
        """
        向量化版本，根据年龄数组返回死亡概率数组。
        """
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
        age_flat = age_array.flatten()
        
        # 替换为向量化函数
        death_probs = self.calculate_death_probability(age_flat)
        
        death_events = np.random.binomial(n=1, p=death_probs)
        sampled_indices = np.where(death_events == 1)[0]
        forced_indices = np.where(age_flat >= max_age)[0]
        
        all_eliminate_indices = np.union1d(sampled_indices, forced_indices)
        die_total = len(all_eliminate_indices)
        
        return die_total, all_eliminate_indices
    
    def compute_estate_tax(self, die_n, society):
        at_die = self.at[:die_n]  # 提取死亡者的遗产数组
        total_inherited = np.sum(
            np.where(
                at_die <= society.estate_tax_exemption,  # 条件：遗产是否小于等于免税额
                at_die,  # 如果是，则继承者收到全部遗产
                society.estate_tax_exemption + (at_die - society.estate_tax_exemption) * (1 - society.estate_tax_rate)
                # 如果否，则收到免税额 + 超过部分的税后金额
            )
        )
        
        estate_tax = np.sum(at_die) - total_inherited
        return total_inherited, estate_tax
    
    def get_reward(self, consumption=None, working_hours=None, alpha=6.68e-6):
        """Compute the utility of households based on consumption and working hours."""
        if consumption is None:
            consumption = self.consumption_ij.sum(axis=1)[:, np.newaxis]
        else:
            consumption = consumption
        
        if working_hours is None:
            working_hours = self.ht
        else:
            working_hours = working_hours
        
        crra = self.CRRA
        if 1 - crra == 0:
            utility_c = np.log((consumption + 1e-8))  # Log utility
        else:
            utility_c = (consumption ** (1 - crra)) / (1 - crra)
        
        if 1 + self.IFE == 0:
            utility_h = np.log(working_hours)
        else:
            utility_h = (working_hours ** (1 + self.IFE) / (1 + self.IFE))
        
        current_utility = utility_c - alpha * utility_h + 21  # max(alpha * utility_h) = 21  一个人 工作到极限带来的最多的痛苦
        
        return current_utility
    
    def sigmoid(self, x):
        """Apply the sigmoid function."""
        x_clipped = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x_clipped))
    
    def is_terminal(self):
        if self.households_n <= 2:
            return True
        else:
            if self.at_next.min() < self.at_min or np.isnan(self.at_next).any():
                return True
            else:
                return False
    
    def close(self):
        pass
