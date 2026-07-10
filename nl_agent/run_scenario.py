"""
编排层（当前只做 MVP 第一步需要的部分）：ScenarioConfig -> 校验 -> 写 YAML -> 跑 EconGym -> 拿结果字典。

刻意复用 main.py 里已经写好的 select_agent / setup_government_agents，不重新实现一遍
"给每个角色装配算法"的逻辑——这是 EconGym 自己的公开接口，我们只是从外部调用它。
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
from typing import Callable, Optional

from .config_layer import REPO_ROOT, build_full_config, write_scenario_yaml
from .env_loader import load_dotenv
from .schema import ScenarioConfig
from .validation import validate_scenario

load_dotenv()  # 保证直接 import run_scenario 独立使用（不经过 planner/llm_client）时 OPENROUTER_API_KEY 等也能读到

# 进度回调签名：progress_cb(step_current, step_total, step_kind)
#   step_kind: "epoch"（RL 训练轮）| "llm_call"（llm 逐期决策调用）| ""（无中间进度）
ProgressCb = Callable[[int, int, str], None]

_EPOCH_RE = re.compile(r"Epoch:\s*(\d+)\s*/\s*(\d+)")


class ScenarioValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


class _StdoutProgressTee:
    """包住 sys.stdout：原样透传到真实 stdout（终端仍能看到日志），同时解析进度行。

    非侵入式：EconGym 的 runner.run() 每个 epoch 会 print 一行 "Epoch: X / N"，
    llm_agent 每次成功决策会 print "LLM-driven ... successful"。我们只是从外部读它已经
    打印的内容来估算进度，不改 EconGym 任何代码。
    """

    def __init__(self, real, on_epoch=None, on_llm_call=None):
        self._real = real
        self._on_epoch = on_epoch
        self._on_llm_call = on_llm_call
        self._buf = ""

    def write(self, s):
        self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle(line)
        return len(s)

    def _handle(self, line):
        if self._on_epoch:
            m = _EPOCH_RE.search(line)
            if m:
                try:
                    self._on_epoch(int(m.group(1)), int(m.group(2)))
                except Exception:
                    pass
        if self._on_llm_call and "LLM-driven" in line and "successful" in line:
            try:
                self._on_llm_call()
            except Exception:
                pass

    def flush(self):
        self._real.flush()


def _expected_llm_calls(config: ScenarioConfig) -> int:
    """llm 逐期决策的预计总调用次数 ≈ episode_length × eval_episodes × llm角色数（提前终止会少于此）。"""
    algs = [config.algorithms.firm_alg, config.algorithms.bank_alg] + list(config.gov_algs().values())
    n_llm = sum(1 for a in algs if a == "llm")
    return config.episode_length * config.eval_episodes * n_llm


@contextlib.contextmanager
def _chdir_repo_root():
    """EconGym 内部大量用相对路径（cfg/..., agents/data/...），必须在仓库根目录下运行。"""
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def run_scenario(config: ScenarioConfig, progress_cb: Optional[ProgressCb] = None) -> dict:
    validation = validate_scenario(config)
    if not validation.ok:
        raise ScenarioValidationError(validation.errors)

    with _chdir_repo_root():
        # 延迟到 chdir 之后再 import：这些模块的顶层代码/相对路径假设仓库根目录是 cwd。
        from utils.seeds import set_seeds
        from env.env_core import EconomicSociety
        from runner import Runner
        from main import select_agent, setup_government_agents

        write_scenario_yaml(config)  # 留档（人可读的场景差异），运行不依赖它
        # 合并逻辑用 config_layer.build_full_config 而不是 utils.config.load_config：
        # 后者的 deep_merge_entities 按 entity_name 折叠 Entities，多政府场景会丢政府条目
        # （这正是内置 dbl_government/tre_government 被 load_config 特殊豁免的原因）。
        full_cfg = build_full_config(config)

        set_seeds(full_cfg["Trainer"]["seed"], cuda=full_cfg["Trainer"]["cuda"])

        env = EconomicSociety(full_cfg["Environment"])

        house_agent = select_agent(
            full_cfg["Trainer"]["house_alg"], "households", env.households.type, env, full_cfg["Trainer"]
        )
        firm_agent = select_agent(
            full_cfg["Trainer"]["firm_alg"], "market", env.market.type, env, full_cfg["Trainer"]
        )
        bank_agent = select_agent(
            full_cfg["Trainer"]["bank_alg"], "bank", env.bank.type, env, full_cfg["Trainer"]
        )
        gov_agents = setup_government_agents(full_cfg, env)

        runner = Runner(
            env,
            full_cfg["Trainer"],
            house_agent=house_agent,
            government_agent=gov_agents,
            firm_agent=firm_agent,
            bank_agent=bank_agent,
        )

        uses_rl = config.uses_rl()
        uses_llm = config.uses_llm_agent()

        # 配置进度解析：RL 场景跟踪训练 epoch；纯 llm 场景跟踪逐期决策调用次数。
        on_epoch = on_llm_call = None
        if progress_cb is not None and uses_rl:
            def on_epoch(cur, tot):
                progress_cb(cur + 1, tot, "epoch")  # runner 打印的 epoch 是 0-indexed
        if progress_cb is not None and uses_llm and not uses_rl:
            llm_total = _expected_llm_calls(config)
            counter = {"n": 0}

            def on_llm_call():
                counter["n"] += 1
                progress_cb(counter["n"], llm_total, "llm_call")

        need_tee = on_epoch is not None or on_llm_call is not None

        def _do_run():
            if uses_rl:
                runner.run()  # 训练 n_epochs
            # 注意：runner.test() 内部调用 _evaluate_agent() 后并不返回结果（源码里没有 return），
            # 所以这里直接调用 _evaluate_agent() 本身，以便拿到返回的指标字典。
            # 这是从外部调用 EconGym 已有方法，没有修改 runner.py 任何一行。
            return runner._evaluate_agent(write_evaluate_data=False)

        if need_tee:
            tee = _StdoutProgressTee(sys.stdout, on_epoch=on_epoch, on_llm_call=on_llm_call)
            old_stdout = sys.stdout
            sys.stdout = tee
            try:
                econ_dict = _do_run()
            finally:
                sys.stdout = old_stdout
        else:
            econ_dict = _do_run()

        return dict(econ_dict)
