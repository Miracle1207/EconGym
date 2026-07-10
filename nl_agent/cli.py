#!/usr/bin/env python
"""
交互式命令行：一个问题接一个问题地问 EconGym。

用法：
    python nl_agent/cli.py            # 完整流程：Planner -> 校验 -> 跑仿真 -> Narrator 解读
    python nl_agent/cli.py --plan-only  # 只跑 Planner，不实际运行仿真（用于快速调试配置生成）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nl_agent.pipeline import answer_question  # noqa: E402
from nl_agent.planner import plan_scenario  # noqa: E402

BANNER = """\
==================================================
 EconGym 自然语言交互层
 输入一个经济学问题（中文/英文均可），回车提交。
 输入 exit / quit / q 退出。
==================================================
"""


def _print_plan_only(question: str) -> None:
    output = plan_scenario(question)
    print(f"\n【是否可行】{output.feasible}")
    print(f"【理由】{output.reasoning}")
    if output.clarifying_question:
        print(f"【需要澄清】{output.clarifying_question}")
    if output.infeasible_reason:
        print(f"【不可行原因】{output.infeasible_reason}")
    if output.configs:
        for label, cfg in zip(output.config_labels or [], output.configs):
            print(f"\n--- {label} ---")
            print(cfg.model_dump_json(indent=2, exclude_none=True))


def _print_answered(result) -> None:
    print("\n【解读】")
    print(result.narrator_output.narrative)
    if result.narrator_output.caveats:
        print("\n【注意事项】")
        for c in result.narrator_output.caveats:
            print(f"- {c}")
    print("\n【本次实际使用的场景配置】")
    for label, cfg in zip(
        result.planner_output.config_labels or [], result.planner_output.configs or []
    ):
        print(f"- {label}: {cfg.problem_scene}")
    print("\n【原始数值】（可用于核对解读是否忠实，未经 LLM 加工）")
    for label, res in (result.raw_results or {}).items():
        print(f"--- {label} ---")
        for k, v in res.items():
            print(f"  {k}: {v}")


def main() -> None:
    plan_only = "--plan-only" in sys.argv
    print(BANNER)
    if plan_only:
        print("（当前是 --plan-only 模式：只生成配置，不实际运行仿真）\n")

    pending_clarification: str | None = None

    while True:
        try:
            prompt = "补充信息 > " if pending_clarification else "你的问题 > "
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("再见。")
            break

        if pending_clarification:
            question = f"原问题：{pending_clarification}\n用户补充：{question}"
            pending_clarification = None

        print("\n正在处理，请稍候...\n")

        try:
            if plan_only:
                _print_plan_only(question)
                continue

            result = answer_question(question)
        except Exception as e:  # noqa: BLE001
            print(f"[出错] {e}")
            continue

        if result.status == "needs_clarification":
            print(f"\n【需要澄清】{result.planner_output.clarifying_question}")
            pending_clarification = question
            continue

        if result.status == "infeasible":
            print(f"\n【无法建模】{result.planner_output.infeasible_reason}")
            continue

        if result.status == "validation_failed":
            print("\n【配置校验失败，多次重试仍未通过】")
            for e in result.errors:
                print(f"- {e}")
            continue

        _print_answered(result)


if __name__ == "__main__":
    main()
