"""
Agent 层 - Narrator：真实仿真数值 -> 自然语言解读。

叙事忠实性的核心手段：
1. 算术在 Python 里做好（delta / pct_change / 是否提前终止），不要求 LLM 自己算，
   减少"LLM 算错数"这个最常见的幻觉来源。
2. LLM 只拿到"已经算好的结构化数字"，system prompt 里硬性约束不能编造/新增/重新计算数字。
3. 输出仍然是结构化的（narrative + caveats），保证"必须提醒用户的事项"不会被模型漏掉或省略。
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel

from .llm_client import LLMClient, get_client
from .schema import ScenarioConfig


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def summarize_results(
    configs: list[ScenarioConfig],
    labels: list[str],
    raw_results: list[dict],
) -> dict:
    """纯 Python 计算：整理原始数值 + 提前终止标记 + (若>=2组) 相对第一组的差值/百分比变化。"""
    runs = []
    for label, cfg, result in zip(labels, configs, raw_results):
        years_run = result.get("years")
        terminated_early = years_run is not None and years_run < cfg.episode_length * 0.95
        runs.append(
            {
                "label": label,
                "problem_scene": cfg.problem_scene,
                "configured_episode_length": cfg.episode_length,
                "actual_years_run": years_run,
                "terminated_early": terminated_early,
                "metrics": {k: v for k, v in result.items() if _is_number(v)},
            }
        )

    comparisons = []
    if len(runs) >= 2:
        baseline = runs[0]
        for other in runs[1:]:
            common_keys = sorted(set(baseline["metrics"]) & set(other["metrics"]))
            deltas = {}
            for k in common_keys:
                b, o = baseline["metrics"][k], other["metrics"][k]
                delta = o - b
                pct = (delta / abs(b)) if b != 0 else None
                deltas[k] = {"baseline_value": b, "compared_value": o, "delta": delta, "pct_change": pct}
            comparisons.append(
                {
                    "baseline_label": baseline["label"],
                    "compared_label": other["label"],
                    "deltas": deltas,
                }
            )

    return {"runs": runs, "comparisons": comparisons}


SYSTEM_PROMPT = """你是 EconGym 仿真结果的解读助手。你会收到用户的原始问题，以及仿真跑出来的真实数值
（差值和百分比变化已经在 Python 里算好了，不需要你自己再算一遍）。

# 铁律（绝对不能违反）

1. 你只能引用下面 JSON 里出现的数值，禁止编造、猜测、或"合理推算"任何未给出的数字。
2. 引用数值时必须与 JSON 里的数值一致（可以四舍五入到合理的展示精度，比如保留 2~3 位有效数字，
   但不能改变数量级或正负方向）。
3. JSON 里已经算好的 delta / pct_change 直接引用，不要自己重新计算，也不要在此基础上做二次推导。
4. 如果某个 run 的 terminated_early=true，必须在解读中明确提及"该次仿真在第 X 年就提前触发了终止
   条件，并未跑满配置的 Y 年"，不能假装它跑满了或者忽略这件事。
5. 如果用户问题里提到的某个信息在给定数据里找不到，必须明确说"本次仿真没有输出这项数据"，
   不能编造替代答案。
6. 解读要指出方向性结论（更高/更低/基本不变）和大致量级，但不要过度延伸到没有数据支撑的因果解释
   （比如不要说"最优选择是..."，只能说"本次仿真观察到..."）。

请输出：
- narrative：面向用户的自然语言解读，可以分段，直接回答用户的问题。
- caveats：必须提醒用户注意的事项列表（比如提前终止、样本量小、单次随机种子的结果存在随机性等）。
"""


class NarratorOutput(BaseModel):
    narrative: str
    caveats: list[str] = []


def narrate_results(
    user_question: str,
    configs: list[ScenarioConfig],
    labels: list[str],
    raw_results: list[dict],
    client: Optional[LLMClient] = None,
) -> NarratorOutput:
    client = client or get_client()
    summary = summarize_results(configs, labels, raw_results)

    user_prompt = (
        f"用户问题：{user_question}\n\n"
        "仿真数据（JSON，已在 Python 里算好差值/百分比变化，你只能引用其中的数值）：\n"
        f"{json.dumps(summary, ensure_ascii=False, indent=2, default=str)}"
    )

    return client.complete_structured(system=SYSTEM_PROMPT, user=user_prompt, schema=NarratorOutput)
