import json
import numpy as np

# === 配置文件路径 ===
json_file_path = "../viz/data/optimal_tax_ramsey_100_bc_tax_ppo_data.json"  # 修改为你的具体 JSON 文件路径

# === 读取 JSON 数据 ===
with open(json_file_path, "r") as file:
    data = json.load(file)

# === 检查并计算 wealth_gini ===
if "wealth_gini" not in data:
    raise KeyError(f"'wealth_gini' not found in {json_file_path}")

wealth_gini_series = np.array(data["income_gini"])

# 取最后 10 个 step
last_10 = wealth_gini_series[-10:]

mean_value = np.mean(last_10)
std_value = np.std(last_10)

print(f"Wealth Gini (last 10 steps) mean: {mean_value:.4f}")
print(f"Wealth Gini (last 10 steps) std: {std_value:.4f}")
