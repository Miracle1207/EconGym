"""
本地可视化 Web UI：把 nl_agent 的两阶段流程包成 Flask 页面。

- POST /api/plan        阶段一：问题 -> 场景配置（快，秒级），前端先展示配置给用户确认
- POST /api/validate    校验用户手动修改后的配置
- POST /api/run         阶段二：后台线程启动仿真，立即返回 job_id（不阻塞）
- GET  /api/run_status  轮询仿真进度 / 拿最终结果

启动：
    conda activate EconGym
    python nl_agent/webapp.py
然后打开 http://127.0.0.1:5050

刻意用一个和 EconGym 自带的 viz_index.py（默认 5000 端口）不同的端口，两者可以同时跑，不冲突。
"""
from __future__ import annotations

import sys
import threading
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from nl_agent.pipeline import plan_question, run_and_narrate  # noqa: E402
from nl_agent.schema import ScenarioConfig  # noqa: E402
from nl_agent.validation import validate_scenario  # noqa: E402

app = Flask(__name__)

# 仿真任务表：job_id -> {status, progress, result, error}。本地单用户工具，内存里存着即可。
_jobs: dict[str, dict] = {}
# 同一时刻只跑一个仿真：run_scenario 内部会 os.chdir 到仓库根目录，串行执行避免多任务互相踩 cwd。
_run_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/plan", methods=["POST"])
def plan():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"status": "error", "message": "问题不能为空"}), 400

    try:
        result = plan_question(question)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

    output = result.planner_output
    payload: dict = {
        "status": result.status,
        "reasoning": output.reasoning if output else None,
        "clarifying_question": output.clarifying_question if output else None,
        "infeasible_reason": output.infeasible_reason if output else None,
        "errors": result.errors or None,
        "configs": None,
        "labels": None,
    }
    if output and output.configs:
        labels = output.config_labels or [c.problem_scene for c in output.configs]
        payload["labels"] = labels
        payload["configs"] = [
            {"label": label, **cfg.model_dump(exclude_none=True)}
            for label, cfg in zip(labels, output.configs)
        ]
    return jsonify(payload)


@app.route("/api/validate", methods=["POST"])
def validate():
    """校验用户手动修改后的配置（pydantic 结构校验 + 语义规则），返回规范化后的配置。"""
    data = request.get_json(force=True) or {}
    raw_configs = data.get("configs") or []
    if not raw_configs:
        return jsonify({"ok": False, "errors": ["configs 不能为空"]})

    errors: list[str] = []
    configs: list[ScenarioConfig] = []
    for i, raw in enumerate(raw_configs):
        try:
            cfg = ScenarioConfig.model_validate(raw)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"])
                errors.append(f"配置{i + 1} 字段 {loc}: {err['msg']}")
            continue
        result = validate_scenario(cfg)
        if not result.ok:
            errors.extend(f"[{cfg.problem_scene}] {err}" for err in result.errors)
        configs.append(cfg)

    if errors:
        return jsonify({"ok": False, "errors": errors})
    return jsonify({"ok": True, "configs": [c.model_dump(exclude_none=True) for c in configs]})


def _run_job(job_id: str, question: str, configs: list[ScenarioConfig], labels: list[str]) -> None:
    """后台线程：串行跑仿真，把进度/结果写进 _jobs[job_id]。"""
    with _run_lock:
        try:
            def progress_cb(info: dict) -> None:
                _jobs[job_id]["progress"] = info

            result = run_and_narrate(question, configs, labels, progress_cb=progress_cb)
            if result.status != "answered":
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = "；".join(result.errors) or "仿真失败"
            else:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = {
                    "raw_results": result.raw_results,
                    "narrative": result.narrator_output.narrative if result.narrator_output else None,
                    "caveats": result.narrator_output.caveats if result.narrator_output else None,
                }
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


@app.route("/api/run", methods=["POST"])
def run():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    raw_configs = data.get("configs") or []
    labels = data.get("labels") or []
    if not raw_configs or len(raw_configs) != len(labels):
        return jsonify({"status": "error", "message": "configs/labels 缺失或长度不一致"}), 400

    try:
        configs = [ScenarioConfig.model_validate(c) for c in raw_configs]
    except ValidationError as e:
        return jsonify({"status": "error", "message": f"配置解析失败：{e}"}), 400

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "running",
        "progress": {"phase": "starting", "config_total": len(configs)},
        "result": None,
        "error": None,
    }
    threading.Thread(target=_run_job, args=(job_id, question, configs, labels), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/run_status")
def run_status():
    job_id = request.args.get("job_id", "")
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "job not found"}), 404
    resp: dict = {"status": job["status"], "progress": job["progress"], "error": job["error"]}
    if job["status"] == "done":
        resp["result"] = job["result"]
    return jsonify(resp)


if __name__ == "__main__":
    import os as _os

    port = int(_os.environ.get("PORT", 5050))
    # threaded=True：仿真在后台线程里跑（_run_job），前端轮询 /api/run_status 需要能并发响应。
    # run_scenario 内部的 os.chdir 由 _run_lock 保证同一时刻只有一个仿真、串行执行；
    # 其余接口（plan/validate/run_status）不依赖 cwd，所以并发安全。
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
