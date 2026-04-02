from __future__ import annotations

"""评测阶段 workflow 总入口。

本脚本只负责执行 workflow yaml 中的 `prediction_evals` 和 `safety_evals`，
不触发训练和生成。输入来自 `examples/workflows/*eval*.yaml`，
输出为每个评测 job 的 summary 和阶段级总 summary。
它复用 prediction_eval_unit 与 safety_eval_unit，要求预测文件路径在 yaml 中显式给出。
"""

from typing import Any

from config_io import resolve_value
from prediction_eval_unit import run_prediction_eval_step
from safety_eval_unit import run_safety_eval_step
from workflow_runner_common import filter_jobs, load_stage_workflow, parse_stage_args, write_stage_job_summary, write_stage_run_summary


def run_eval_list(step_list: Any, context: dict, runner) -> dict[str, Any]:
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


def run_safety_list(step_list: Any, context: dict) -> dict[str, Any]:
    """执行 safety-eval 步骤列表。"""

    results: dict[str, Any] = {}
    if not isinstance(step_list, list):
        return results
    for step in step_list:
        if not isinstance(step, dict) or not step.get("enabled", False):
            continue
        resolved = resolve_value(step, context)
        results[resolved["name"]] = run_safety_eval_step(resolved, context["defaults"])["outputs"]
    return results


def run_eval_job(job: dict, context: dict) -> dict:
    """执行单个 eval job，并写出 job summary。"""

    prediction_results = run_eval_list(job.get("prediction_evals"), context, run_prediction_eval_step)
    safety_results = run_safety_list(job.get("safety_evals"), context)
    if not prediction_results and not safety_results:
        raise ValueError(
            f"Eval workflow job `{job['name']}` must enable at least one `prediction_evals` or `safety_evals` item."
        )
    job_state: dict[str, Any] = {}
    if prediction_results:
        job_state["prediction_evals"] = prediction_results
    if safety_results:
        job_state["safety_evals"] = safety_results
    context["jobs"][job["name"]] = job_state
    summary_path = write_stage_job_summary(
        job=job,
        context=context,
        stage_name="eval",
        payload={"job_name": job["name"], "stage": "eval", "results": job_state},
    )
    job_state["summary_path"] = str(summary_path)
    return job_state


def main() -> None:
    """读取 eval workflow yaml 并顺序执行各个 job。"""

    args = parse_stage_args("Run eval-only workflow jobs from YAML.")
    config_path, jobs, context = load_stage_workflow(args.config)
    results: dict[str, dict] = {}
    for job in filter_jobs(jobs, args.only_job):
        results[job["name"]] = run_eval_job(job, context)
    output_path = write_stage_run_summary("eval", config_path, results)
    print(f"[done] Eval workflow summary written to {output_path}")


if __name__ == "__main__":
    main()
