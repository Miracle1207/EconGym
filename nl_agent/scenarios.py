"""MVP 第一步：手写场景配置（不经过 LLM）。"""
from .schema import (
    AlgorithmConfig,
    BankConfig,
    GovernmentConfig,
    HouseholdConfig,
    MarketConfig,
    ScenarioConfig,
)

# 一个不在 EconGym 内置 25 个任务里的新场景：
# "风险投资倾向（stock_alpha，股价对买卖失衡的敏感度）对家庭财富分配的影响"
# - households: ramsey_risk_invest（无年龄结构，带风险投资决策）
# - government: tax（默认 HSV 税制，本轮不作为实验杠杆）
# - market: perfect（简化市场，聚焦家庭/银行侧）
# - bank: commercial（贷款/存款利率会对市场状态做出响应，risk_invest 的对照对象更有意义）
# - 全部角色用 rule_based：跑得快、确定性强，用于验证"配置层 + 运行"这条路。
risk_preference_wealth = ScenarioConfig(
    problem_scene="risk_preference_wealth",
    episode_length=100,
    eval_episodes=2,
    seed=1,
    households=HouseholdConfig(
        type="ramsey_risk_invest",
        households_n=200,
        stock_alpha=0.15,
    ),
    governments=[
        GovernmentConfig(
            type="tax",
            gov_task="social_welfare",
        )
    ],
    market=MarketConfig(type="perfect"),
    bank=BankConfig(type="commercial"),
    algorithms=AlgorithmConfig(
        house_alg="rule_based",
        gov_alg="rule_based",
        firm_alg="rule_based",
        bank_alg="rule_based",
    ),
)
