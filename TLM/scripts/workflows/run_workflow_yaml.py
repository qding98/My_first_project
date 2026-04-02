from __future__ import annotations

"""工作流总入口。

本脚本是 train / generate / prediction_eval / safety_eval 的统一编排入口。
它读取 `examples/workflows/*.yaml`，按 job 顺序调度各个单元，并把中间产物路径写回上下文，
供后续步骤通过 `${...}` 占位符引用。输出为每个 job 的 summary 与总 workflow summary。
"""

import argparse
from pathlib import Path
from typing import Any

from config_io import load_workflow_yaml, normalize_jobs, resolve_value
from generate_unit import run_generate_step
from prediction_eval_unit import run_prediction_eval_step
from safety_eval_unit import run_safety_eval_step
from train_unit import run_train_step
from workflow_shared import ROOT, resolve_workflow_path
from pipeline_common import write_json


def parse_args() -> argparse.Namespace:
    """解析 workflow runner 命令行参数。"""

    parser = argparse.ArgumentParser(description="Run modular train/generate/eval workflow from YAML.")
    parser.add_argument("config", help="Path to workflow YAML.")
    parser.add_argument("--only-job", action="append", default=[], help="Optional job name filter.")
    return parser.parse_args()


def build_context(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """构造占位符解析所需的运行时上下文。"""

    defaults = dict(config.get("defaults", {}))
    defaults.setdefault("project_root", str(ROOT.parent))
    defaults.setdefault("tlm_root", str(ROOT))
    defaults.setdefault("safety_eval_root", str(ROOT.parent / "safety-eval"))
    return {"config_path": str(config_path.resolve()), "defaults": defaults, "jobs": {}}


def run_optional_train(job: dict[str, Any], context: dict[str, Any]) -> dict[str, Any] | None:
    """在 job 启用 train 时执行训练单元。"""

    step = job.get("train")
    if not isinstance(step, dict) or not step.get("enabled", False):
        return None
    return run_train_step(job["name"], resolve_value(step, context), context["defaults"])


def run_optional_generate(job: dict[str, Any], context: dict[str, Any]) -> dict[str, Any] | None:
    """在 job 启用 generate 时执行生成单元。"""

    step = job.get("generate")
    if not isinstance(step, dict) or not step.get("enabled", False):
        return None
    return run_generate_step(job["name"], resolve_value(step, context), context["defaults"])


def run_eval_list(step_list: Any, context: dict[str, Any], runner) -> dict[str, Any]:
    """顺序执行一个评测步骤列表，并按 name 收集输出。"""

    results: dict[str, Any] = {}
    if not isinstance(step_list, list):
        return results
    for step in step_list:
        if not isinstance(step, dict) or not step.get("enabled", False):
            continue
        resolved = resolve_value(step, context)
        results[resolved["name"]] = runner(resolved)["outputs"]
    return results


def resolve_job_summary_path(job: dict[str, Any], context: dict[str, Any]) -> Path:
    """解析单个 job 的 summary 输出路径。"""

    raw_path = job.get("summary_path", f"saves/workflows/{job['name']}/workflow_summary.json")
    return resolve_workflow_path(resolve_value(raw_path, context))


def run_job(job: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """执行单个 job 的全部启用步骤，并写出 job summary。"""

    job_state: dict[str, Any] = {}
    context["jobs"][job["name"]] = job_state
    train_result = run_optional_train(job, context)
    if train_result is not None:
        job_state["train"] = train_result
    generate_result = run_optional_generate(job, context)
    if generate_result is not None:
        job_state["generate"] = generate_result
    prediction_results = run_eval_list(job.get("prediction_evals"), context, run_prediction_eval_step)
    if prediction_results:
        job_state["prediction_evals"] = prediction_results
    safety_results = run_eval_list(job.get("safety_evals"), context, lambda step: run_safety_eval_step(step, context["defaults"]))
    if safety_results:
        job_state["safety_evals"] = safety_results
    summary_path = resolve_job_summary_path(job, context)
    write_json(summary_path, {"job_name": job["name"], "results": job_state})
    job_state["summary_path"] = str(summary_path)
    return job_state


def main() -> None:
    """读取 workflow yaml 并按 job 顺序执行。"""

    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_workflow_yaml(config_path)
    jobs = normalize_jobs(config.get("jobs"))
    selected_jobs = set(args.only_job)
    context = build_context(config_path, config)
    results: dict[str, Any] = {}
    for job in jobs:
        if selected_jobs and job["name"] not in selected_jobs:
            continue
        results[job["name"]] = run_job(job, context)
    output_path = ROOT / "saves" / "workflows" / "workflow_run_summary.json"
    write_json(output_path, {"config_path": str(config_path), "jobs": results})
    print(f"[done] Workflow summary written to {output_path}")


if __name__ == "__main__":
    main()
