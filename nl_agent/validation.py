"""
校验层：语义规则检查（结构性范围检查已经由 pydantic 在 schema.py 里做了）。

这里只编码那些"pydantic 管不到、必须结合 EconGym 代码行为才能发现"的规则。
目前先落地两条已经在探索阶段确认过的规则，后续可以继续往 SEMANTIC_RULES 里加：
1. pension 政府必须搭配 OLG 家庭模型，否则 calculate_pension() 的整条机制根本不会被调用。
2. gov_alg="data_based" 用于政府角色：agents/data_based_agent.py 在当前"多政府字典"架构下
   访问 self.envs.government.type / self.envs.tax_gov 等不存在的属性，大概率运行时报错，
   暂时禁用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import ScenarioConfig

# episode_length * eval_episodes 的上限：超过这个值时，llm 算法产生的真实 API 调用次数会让单次
# 请求耗时/花费明显失控（本地体验和成本都不可接受），需要 Planner 主动收窄仿真规模。
LLM_AGENT_MAX_CALLS = 150


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_scenario(config: ScenarioConfig) -> ValidationResult:
    errors: list[str] = []
    gov_types = [g.type for g in config.governments]

    if "pension" in gov_types and "OLG" not in config.households.type:
        errors.append(
            "governments 里有 type='pension' 时需要 households.type 包含 'OLG'"
            f"（当前是 '{config.households.type}'），否则养老金计算逻辑（calculate_pension）"
            "在仿真中根本不会被调用，配置了退休年龄/缴费率也不会产生任何效果。"
        )

    estate_tax_configured = any(
        g.estate_tax_rate is not None or g.estate_tax_exemption is not None
        for g in config.governments
    )
    if estate_tax_configured and "OLG" not in config.households.type:
        errors.append(
            "estate_tax_rate / estate_tax_exemption 只有当 households.type 包含 'OLG' "
            f"时才会生效（当前是 '{config.households.type}'）。遗产税只在家庭死亡事件触发时计算"
            "（entities/households.py 的 OLG_step -> compute_estate_tax），ramsey 系列家庭永生，"
            "配了这两个参数不会产生任何效果。"
        )

    # 环境级常量在多个政府上重复设置且值不一致 -> 有歧义，直接拦截
    for field_name in ("consumption_tax_rate", "estate_tax_rate", "estate_tax_exemption"):
        values = {getattr(g, field_name) for g in config.governments} - {None}
        if len(values) > 1:
            errors.append(
                f"多个政府对环境级常量 {field_name} 设置了不同的值 {sorted(values)}，存在歧义。"
                "这类参数是全局环境常量，只在一个政府（建议 type='tax'）上设置即可。"
            )

    for gov_type, alg in config.gov_algs().items():
        if alg == "data_based":
            errors.append(
                f"政府 '{gov_type}' 的算法是 'data_based'，它在 EconGym 的多政府字典架构下有已知 bug"
                "（agents/data_based_agent.py 访问 self.envs.government.type / self.envs.tax_gov 等"
                "不存在的属性），暂时禁止用于政府角色，请改用 'rule_based' / 'llm' / 'ppo' / 'ddpg'。"
            )

    if config.uses_llm_agent():
        llm_roles = [config.algorithms.firm_alg, config.algorithms.bank_alg]
        llm_roles += list(config.gov_algs().values())
        llm_role_n = sum(1 for a in llm_roles if a == "llm")
        total_calls = config.episode_length * config.eval_episodes * llm_role_n
        if total_calls > LLM_AGENT_MAX_CALLS:
            errors.append(
                "算法组合里用了 'llm'，每个 llm 角色的每一步都会触发一次真实的 LLM API 调用。"
                f"当前 episode_length({config.episode_length}) x eval_episodes({config.eval_episodes}) x "
                f"llm角色数({llm_role_n}) = {total_calls} 次调用，超过建议上限 {LLM_AGENT_MAX_CALLS} 次，"
                "耗时和成本都会过高。请把 episode_length 和/或 eval_episodes 调小"
                "（比如 episode_length<=30, eval_episodes<=2），除非用户明确要求跑一次很长/很细致的模拟。"
            )

    return ValidationResult(ok=len(errors) == 0, errors=errors)
