import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 数据加载部分 (保持不变) ---
folder_path = "F:\\project\\intern\\analyst\\data"
variable_names = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        variable_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(folder_path, file_name)
        globals()[variable_name] = pd.read_csv(file_path)
        variable_names.append(variable_name)

metrics_map = [
    ('mf_central_bank_reward', 'ppo_central_bank_reward', 'Central Bank Reward'),
    ('mf_gdp', 'ppo_gdp', 'GDP'),
    ('mf_gov_reward', 'ppo_gov_reward', 'Gov Reward'),
    ('mf_house_consumption', 'ppo_house_consumption', 'House Consumption'),
    ('mf_house_reward', 'ppo_house_reward', 'House Reward'),
    ('mf_income_gini', 'ppo_income_gini', 'Income Gini'),
    ('mf_per_gdp', 'ppo_per_gdp', 'Per GDP'),
    ('mf_wealth_gini', 'ppo_wealth_gini', 'Wealth Gini'),
    ('mf_social_welfare', 'ppo_welfare', 'Social Welfare'),
    ('mf_inflation_rate', 'ppo_inflation_rate', 'Inflation Rate'),
    ('mf_years', 'ppo_years', 'Years')
]

# --- 2. 绘图核心函数 ---
def plot_academic_chart(metric_name_display, mf_df, ppo_df, save_name):
    # 调小 EMA 窗口。span=5 表示更敏感，能更快摆脱第1步异常值的影响
    # adjust=True (默认) 在处理初始点干扰时效果比 adjust=False 好
    current_span = 8 

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 4], wspace=0.1)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    # --- 绘制算法 A (MF) ---
    raw_a = mf_df.iloc[:, -1]
    # 使用 adjust=True 处理初始极端值干扰
    ema_a = raw_a.ewm(span=current_span, adjust=True).mean()
    
    ax1.plot(mf_df['step'], raw_a, color='#1f77b4', alpha=0.15, linewidth=0.8)
    ax1.plot(mf_df['step'], ema_a, color='#1f77b4', linewidth=2.5, label='MF (Ours)')
    ax1.axhline(y=ema_a.iloc[-1], color='#1f77b4', linestyle='--', alpha=0.4)

    # --- 绘制算法 B (PPO) ---
    raw_b = ppo_df.iloc[:, -1]
    ema_b = raw_b.ewm(span=current_span, adjust=True).mean()
    
    ax2.plot(ppo_df['step'], raw_b, color='#d62728', alpha=0.15, linewidth=0.8)
    ax2.plot(ppo_df['step'], ema_b, color='#d62728', linewidth=2.5, label='PPO (Baseline)')
    ax2.axhline(y=ema_b.iloc[-1], color='#d62728', linestyle='--', alpha=0.4)

    # --- 细节优化 ---
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax1.set_ylabel(metric_name_display, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, 'Training Steps', ha='center', fontsize=14, fontweight='bold')
    fig.suptitle(f'Convergence Comparison: {metric_name_display}', fontsize=16, y=0.96)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='lower right', frameon=True, shadow=True)

    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)

    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_split_correlation(x_var, y_var, x_label, y_label, save_name):
    """
    将 MF 和 PPO 拆分为 2x2 子图中的对比项（或独立对比）
    这里我们为每一组关系生成一张包含 4 个子图的画布：
    [MF 原始散点] [PPO 原始散点]
    [MF 密度分布] [PPO 密度分布]
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {'MF': '#1f77b4', 'PPO': '#d62728'}
    
    # 变量名处理
    ppo_y_name = 'ppo_welfare' if y_var == 'social_welfare' else f'ppo_{y_var}'
    
    try:
        data = {
            'MF_X': globals()[f'mf_{x_var}'].iloc[:, -1],
            'MF_Y': globals()[f'mf_{y_var}'].iloc[:, -1],
            'PPO_X': globals()[f'ppo_{x_var}'].iloc[:, -1],
            'PPO_Y': globals()[ppo_y_name].iloc[:, -1]
        }

        # --- 第一行：带回归线的散点图 (Regression Plot) ---
        # MF 散点
        sns.regplot(x=data['MF_X'], y=data['MF_Y'], ax=axes[0, 0], color=colors['MF'],
                    scatter_kws={'alpha': 0.1, 's': 8}, line_kws={'linewidth': 2})
        axes[0, 0].set_title(f"MF: {x_label} vs {y_label}", fontweight='bold')

        # PPO 散点
        sns.regplot(x=data['PPO_X'], y=data['PPO_Y'], ax=axes[0, 1], color=colors['PPO'],
                    scatter_kws={'alpha': 0.1, 's': 8}, line_kws={'linewidth': 2})
        axes[0, 1].set_title(f"PPO: {x_label} vs {y_label}", fontweight='bold')

        # --- 第二行：核密度估计图 (KDE Plot) - 展现稳态分布 ---
        # MF 密度
        sns.kdeplot(x=data['MF_X'], y=data['MF_Y'], ax=axes[1, 0], cmap="Blues", fill=True, thresh=0.05)
        axes[1, 0].set_title("MF Convergence Density", fontsize=11)

        # PPO 密度
        sns.kdeplot(x=data['PPO_X'], y=data['PPO_Y'], ax=axes[1, 1], cmap="Reds", fill=True, thresh=0.05)
        axes[1, 1].set_title("PPO Convergence Density", fontsize=11)

        # 统一标签
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

        plt.tight_layout()
        plt.savefig(f"{save_name}_split.png", dpi=300, bbox_inches='tight')
        print(f"拆分对比图已保存: {save_name}_split.png")
        
    except KeyError as e:
        print(f"跳过绘图: 找不到变量 {e}")
    plt.close()

def plot_monetary_policy_tradeoff(mf_inf, mf_gdp, ppo_inf, ppo_gdp):
    """
    专门针对 Inflation Rate 和 GDP 的权衡分析图
    2x2 布局：上排散点回归，下排收敛密度
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # 颜色定义
    color_mf = '#1f77b4'  # 经典蓝
    color_ppo = '#d62728' # 经典红

    # --- 第一行：散点与回归线 (显示政策的动态权衡趋势) ---
    # MF: Inflation vs GDP
    sns.regplot(x=mf_inf, y=mf_gdp, ax=axes[0, 0], color=color_mf, 
                label='MF (Ours)', scatter_kws={'alpha': 0.15, 's': 10}, line_kws={'linewidth': 2.5})
    axes[0, 0].set_title("MF: Inflation-GDP Trade-off", fontsize=14, fontweight='bold')
    
    # PPO: Inflation vs GDP
    sns.regplot(x=ppo_inf, y=ppo_gdp, ax=axes[0, 1], color=color_ppo, 
                label='PPO (Baseline)', scatter_kws={'alpha': 0.15, 's': 10}, line_kws={'linewidth': 2.5})
    axes[0, 1].set_title("PPO: Inflation-GDP Trade-off", fontsize=14, fontweight='bold')

    # --- 第二行：收敛密度图 (显示最终政策目标的锚定精度) ---
    # MF Density
    sns.kdeplot(x=mf_inf, y=mf_gdp, ax=axes[1, 0], fill=True, cmap="Blues", thresh=0.02, levels=10)
    axes[1, 0].set_title("MF: Policy Convergence Density", fontsize=13, fontstyle='italic')
    
    # PPO Density
    sns.kdeplot(x=ppo_inf, y=ppo_gdp, ax=axes[1, 1], fill=True, cmap="Reds", thresh=0.02, levels=10)
    axes[1, 1].set_title("PPO: Policy Convergence Density", fontsize=13, fontstyle='italic')

    # --- 细节优化 ---
    # 设置统一的标签
    for i in range(2):
        for j in range(2):
            axes[i, j].set_xlabel("Inflation Rate", fontsize=12, fontweight='bold')
            axes[i, j].set_ylabel("GDP", fontsize=12, fontweight='bold')
            # 可以在此处根据需要手动设置 xlim 或 ylim，例如：
            # axes[i, j].set_xlim(left=0) 

    plt.tight_layout(pad=3.0)
    
    save_path = "analysis_monetary_policy_focus.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"研究重点对比图已保存: {save_path}")
    plt.show()


# --- 3. 执行绘制循环 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {'grid.linestyle': '--'})

for mf_var, ppo_var, display_name in metrics_map:
    try:
        mf_data = globals()[mf_var]
        ppo_data = globals()[ppo_var]
        file_save_name = mf_var.replace('mf_', 'comparison_')
        plot_academic_chart(display_name, mf_data, ppo_data, file_save_name)
    except KeyError as e:
        print(f"跳过: 找不到变量 {e}")

'''
# --- 执行绘图任务 ---

# 1. GDP 与 财富/收入基尼 (演示 4 子图模式)
plot_split_correlation('per_gdp', 'wealth_gini', 'Per Household GDP', 'Wealth Gini', 'growth_vs_wealth_inequality')
plot_split_correlation('per_gdp', 'income_gini', 'Per Household GDP', 'Income Gini', 'growth_vs_income_inequality')

# 2. 通胀与福利/消费
plot_split_correlation('inflation_rate', 'social_welfare', 'Inflation Rate', 'Social Welfare', 'inflation_vs_welfare')
plot_split_correlation('inflation_rate', 'house_consumption', 'Inflation Rate', 'Consumption', 'inflation_vs_consumption')

# 3. 收入基尼与财富基尼 (结构性对比)
plot_split_correlation('income_gini', 'wealth_gini', 'Income Gini', 'Wealth Gini', 'inequality_structure')



# --- 调用部分 ---
try:
    # 提取数据最后一列（假设您的 DataFrame 已载入全局变量）
    mf_inf_data = globals()['mf_inflation_rate'].iloc[:, -1]
    mf_gdp_data = globals()['mf_gdp'].iloc[:, -1]
    
    ppo_inf_data = globals()['ppo_inflation_rate'].iloc[:, -1]
    ppo_gdp_data = globals()['ppo_gdp'].iloc[:, -1]

    plot_monetary_policy_tradeoff(mf_inf_data, mf_gdp_data, ppo_inf_data, ppo_gdp_data)

except KeyError as e:
    print(f"绘图失败：缺少变量 {e}")
    '''