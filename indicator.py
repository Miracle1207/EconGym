import os
import json
import numpy as np
import pandas as pd

DATA_FOLDER = 'generated_data_store/tre_gov'

# 初始化数据结构
years = {}
house_consumption_data = {}
house_work_hour_data = {}
GDP_data = {}
social_welfare_data = {}
gov_reward_data = {}
income_gini_data = {}


def process_data():
    for file_name in os.listdir(DATA_FOLDER):
        if file_name.endswith('_data.json'):
            strategy_name = file_name.replace('_data.json', '')
            file_path = os.path.join(DATA_FOLDER, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                years[strategy_name] = data.get('years', [])
                house_consumption_data[strategy_name] = data.get('house_consumption', [])
                house_work_hour_data[strategy_name] = data.get('house_work_hours', [])
                GDP_data[strategy_name] = data.get('GDP', [])
                social_welfare_data[strategy_name] = data.get('social_welfare', [])
                gov_reward_data[strategy_name] = data.get('gov_reward', [])
                income_gini_data[strategy_name] = data.get('income_gini', [])


process_data()

# 存储结果
results = []

for strategy in years.keys():
    result = {'Strategy': strategy}

    # Last Year
    year_list = years[strategy]
    result['Last Year'] = year_list[-1] if year_list else None

    # Households Consumption: 每年总和，再取平均
    hc_data = house_consumption_data.get(strategy, [])
    hc_year_sums = [np.sum(year_data) for year_data in hc_data if year_data]
    result['Households Consumption'] = round(np.mean(hc_year_sums), 2) if hc_year_sums else None

    # Households Work Hours: 每年平均工作时间，再取整体平均
    hwh_data = house_work_hour_data.get(strategy, [])
    hwh_year_avgs = [np.mean(year_data) for year_data in hwh_data if year_data]
    result['Households Work Hours'] = round(np.mean(hwh_year_avgs), 2) if hwh_year_avgs else None

    # GDP: 每年总和，再取整体总和
    gdp_data = GDP_data.get(strategy, [])
    result['GDP'] = round(sum(gdp_data), 2) if gdp_data else None

    # Social Welfare: 每年总和，再取整体总和
    sw_data = social_welfare_data.get(strategy, [])
    result['Social Welfare'] = round(sum(sw_data), 2) if sw_data else None

    # Gov Reward: 每年总和，再取整体总和
    gr_data = gov_reward_data.get(strategy, [])
    result['Gov Reward'] = round(sum(gr_data), 2) if gr_data else None

    # Income Gini: 所有 step 的平均值
    ig_data = income_gini_data.get(strategy, [])
    result['Income Gini'] = round(np.mean(ig_data), 2) if ig_data else None

    results.append(result)

# 转换为 DataFrame
df = pd.DataFrame(results)

# 保存到 Excel
output_file = 'result.xlsx'
df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
