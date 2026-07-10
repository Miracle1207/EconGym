"""
配置层 schema：定义"一个合法的 EconGym 场景配置"长什么样。

这是 Planner LLM 未来必须输出的结构（本文件先不接 LLM，仅供手写实例化 + 校验层/转换层复用）。
字段范围刻意比 EconGym 完整 YAML 窄很多：只暴露我们目前确认过语义、且值得让上层（人或LLM）
调整的参数；其余细节（action_dim / action_space / real_action_max 等）一律交给
cfg/base_config.yaml 的默认值，不在这里重复定义。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

HouseholdType = Literal["ramsey", "OLG", "ramsey_risk_invest", "OLG_risk_invest"]
GovernmentType = Literal["tax", "pension", "central_bank"]
MarketType = Literal["perfect", "monopoly", "monopolistic_competition", "oligopoly"]
BankType = Literal["non_profit", "commercial"]
TaxType = Literal["ai_agent", "us_federal", "saez"]
AlgorithmName = Literal["rule_based", "bc", "data_based", "ppo", "ddpg", "sac", "llm"]


class HouseholdConfig(BaseModel):
    type: HouseholdType
    households_n: int = Field(ge=1, le=5000)
    CRRA: Optional[float] = Field(default=None, ge=0.1, le=5.0)
    IFE: Optional[float] = Field(default=None, ge=0.1, le=5.0)
    stock_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # OLG-only
    birth_rate: Optional[float] = Field(default=None, ge=0.0, le=0.05)
    initial_working_age: Optional[int] = Field(default=None, ge=16, le=30)


class GovernmentConfig(BaseModel):
    type: GovernmentType
    # 该政府单独使用的决策算法；不填则用 algorithms.gov_alg。
    # 多政府协同时（如财政+央行）可以分别指定，对应 EconGym Trainer 里的 {type}_gov_alg。
    alg: Optional[AlgorithmName] = None
    gov_task: Optional[str] = None
    tax_type: Optional[TaxType] = None
    # 环境级常量（始终生效，不受算法影响，见校验层说明）
    consumption_tax_rate: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    estate_tax_rate: Optional[float] = Field(default=None, ge=0.0, le=0.7)
    estate_tax_exemption: Optional[float] = Field(default=None, ge=0.0)
    # pension 专属
    retire_age: Optional[int] = Field(default=None, ge=60, le=70)
    contribution_rate: Optional[float] = Field(default=None, ge=0.05, le=0.2)


class MarketConfig(BaseModel):
    type: MarketType
    firm_n: int = 1

    @model_validator(mode="after")
    def _check_firm_n(self):
        if self.type in ("perfect", "monopoly") and self.firm_n != 1:
            raise ValueError(f"market.type='{self.type}' 要求 firm_n=1，当前 firm_n={self.firm_n}")
        return self


class BankConfig(BaseModel):
    type: BankType


class AlgorithmConfig(BaseModel):
    # house_alg 默认 bc（行为克隆）而不是 rule_based：和 EconGym 自己 cfg/base_config.yaml 的默认值一致，
    # 零额外成本（不需要训练，跑得和 rule_based 一样快），且基于真实调查数据训练，比 rule_based 硬编码的
    # 按年龄经验分布更贴近真实人类行为。
    house_alg: AlgorithmName = "bc"
    gov_alg: AlgorithmName = "rule_based"
    firm_alg: AlgorithmName = "rule_based"
    bank_alg: AlgorithmName = "rule_based"


class ScenarioConfig(BaseModel):
    problem_scene: str
    episode_length: int = Field(default=300, ge=10, le=1000)
    eval_episodes: int = Field(default=5, ge=1, le=50)
    seed: int = 1
    n_epochs: int = Field(default=300, ge=50, le=1000)  # 仅在存在 RL 算法时才会被用到

    households: HouseholdConfig
    # 同一次仿真可以同时挂 1~3 个不同类型的政府（如财政+央行协同，对应内置任务 dbl_government；
    # 财政+央行+养老金对应 tre_government）。列表中的第一个政府是 main_gov：家庭面对的税收/
    # 养老金计算和 GDP 等宏观指标都以它为准，把与问题最相关的政府放在第一位。
    governments: list[GovernmentConfig] = Field(min_length=1, max_length=3)
    market: MarketConfig
    bank: BankConfig
    algorithms: AlgorithmConfig = AlgorithmConfig()

    @model_validator(mode="after")
    def _check_unique_gov_types(self):
        types = [g.type for g in self.governments]
        if len(types) != len(set(types)):
            raise ValueError(f"governments 里出现了重复的政府类型：{types}，每种类型最多一个。")
        return self

    def gov_algs(self) -> dict[str, str]:
        """每个政府类型 -> 实际生效的算法（政府自己的 alg 覆盖全局 gov_alg）。"""
        return {g.type: (g.alg or self.algorithms.gov_alg) for g in self.governments}

    def uses_rl(self) -> bool:
        rl_algs = {"ppo", "ddpg", "sac"}
        all_algs = [self.algorithms.house_alg, self.algorithms.firm_alg, self.algorithms.bank_alg]
        all_algs += list(self.gov_algs().values())
        return any(alg in rl_algs for alg in all_algs)

    def uses_llm_agent(self) -> bool:
        """是否有角色用了 EconGym 自带的 llm_agent（government/bank/market 每步实时决策）。"""
        agent_algs = [self.algorithms.firm_alg, self.algorithms.bank_alg] + list(self.gov_algs().values())
        return "llm" in agent_algs
