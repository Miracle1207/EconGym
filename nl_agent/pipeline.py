"""
编排层：Planner -> 校验(失败则反馈重试) -> EconGym 运行 -> Narrator。

拆成两个独立阶段，供 Web UI 分步调用：
- plan_question():   自然语言 -> 配置（快，只有 Planner 的 LLM 调用，秒级）
- run_and_narrate(): 配置 -> 仿真 -> 解读（可能很慢：RL 要训练、llm 算法每步都是真实 API 调用）

answer_question() 把两个阶段串起来一次跑完，保持 CLI / 脚本用法不变。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .llm_client import LLMClient, get_client
from .narrator import NarratorOutput, narrate_results
from .planner import PlannerOutput, plan_scenario
from .run_scenario import run_scenario
from .schema import ScenarioConfig
from .validation import validate_scenario

MAX_REPAIR_ATTEMPTS = 2

# 进度回调签名：收到一个描述当前进度的 dict（见 run_and_narrate 里构造的字段）
RunProgressCb = Callable[[dict], None]


@dataclass
class PlanResult:
    # "planned" | "infeasible" | "needs_clarification" | "validation_failed"
    status: str
    planner_output: Optional[PlannerOutput] = None
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    # "answered" | "infeasible" | "needs_clarification" | "validation_failed"
    status: str
    planner_output: Optional[PlannerOutput] = None
    raw_results: Optional[dict[str, dict]] = None
    narrator_output: Optional[NarratorOutput] = None
    errors: list[str] = field(default_factory=list)


def _collect_validation_errors(configs: Optional[list[ScenarioConfig]]) -> list[str]:
    errors: list[str] = []
    for cfg in configs or []:
        result = validate_scenario(cfg)
        if not result.ok:
            errors.append(f"[{cfg.problem_scene}] " + "; ".join(result.errors))
    return errors


def plan_question(user_question: str, client: Optional[LLMClient] = None) -> PlanResult:
    """阶段一：自然语言 -> 通过校验的场景配置（不运行仿真）。"""
    client = client or get_client()

    prompt = user_question
    output = plan_scenario(prompt, client=client)
    errors: list[str] = []

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        if not output.feasible:
            break
        errors = _collect_validation_errors(output.configs)
        if not errors or attempt == MAX_REPAIR_ATTEMPTS:
            break
        prompt = (
            f"{user_question}\n\n"
            "你上一次生成的配置没有通过校验，具体问题如下，请修正后重新输出完整的结果：\n"
            + "\n".join(errors)
        )
        output = plan_scenario(prompt, client=client)

    if not output.feasible:
        status = "needs_clarification" if output.clarifying_question else "infeasible"
        return PlanResult(status=status, planner_output=output)

    if errors:
        return PlanResult(status="validation_failed", planner_output=output, errors=errors)

    return PlanResult(status="planned", planner_output=output)


def run_and_narrate(
    user_question: str,
    configs: list[ScenarioConfig],
    labels: list[str],
    client: Optional[LLMClient] = None,
    progress_cb: Optional[RunProgressCb] = None,
) -> PipelineResult:
    """阶段二：跑仿真 + 解读。configs 会再校验一遍（防止绕过阶段一直接调用）。"""
    client = client or get_client()

    errors = _collect_validation_errors(configs)
    if errors:
        return PipelineResult(status="validation_failed", errors=errors)

    total = len(configs)
    raw_results: dict[str, dict] = {}
    for i, (label, cfg) in enumerate(zip(labels, configs)):
        if progress_cb:
            progress_cb({
                "phase": "running", "config_index": i, "config_total": total,
                "config_label": label, "step_current": 0, "step_total": 0, "step_kind": "",
            })

        def step_cb(cur, tot, kind, _i=i, _label=label):
            if progress_cb:
                progress_cb({
                    "phase": "running", "config_index": _i, "config_total": total,
                    "config_label": _label, "step_current": cur, "step_total": tot, "step_kind": kind,
                })

        raw_results[label] = run_scenario(cfg, progress_cb=step_cb)

    if progress_cb:
        progress_cb({"phase": "narrating", "config_total": total})

    narrator_output = narrate_results(
        user_question=user_question,
        configs=configs,
        labels=labels,
        raw_results=[raw_results[label] for label in labels],
        client=client,
    )

    return PipelineResult(
        status="answered",
        raw_results=raw_results,
        narrator_output=narrator_output,
    )


def answer_question(user_question: str, client: Optional[LLMClient] = None) -> PipelineResult:
    """一次跑完两个阶段（CLI / 脚本用）。"""
    client = client or get_client()

    plan = plan_question(user_question, client=client)
    if plan.status != "planned":
        return PipelineResult(status=plan.status, planner_output=plan.planner_output, errors=plan.errors)

    output = plan.planner_output
    labels = output.config_labels or [cfg.problem_scene for cfg in output.configs]
    result = run_and_narrate(user_question, output.configs, labels, client=client)
    result.planner_output = output
    return result
