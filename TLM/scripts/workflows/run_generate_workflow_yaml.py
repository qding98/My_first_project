from __future__ import annotations

"""生成阶段 workflow 总入口。

本脚本只负责执行 workflow yaml 中的 `generate` 段，不触发训练和评测。
输入来自 `examples/workflows/*generate*.yaml`，输出为每个生成 job 的 summary 和阶段级总 summary。
它复用 `generate_unit.py` 的生成单元实现，要求 adapter/base model 路径在 yaml 中显式给出。
"""

from config_io import resolve_value
from generate_unit import run_generate_step
from workflow_runner_common import filter_jobs, load_stage_workflow, parse_stage_args, write_stage_job_summary, write_stage_run_summary


def run_generate_job(job: dict, context: dict) -> dict:
    """执行单个 generate job，并写出 job summary。"""

    step = job.get("generate")
    if not isinstance(step, dict) or not step.get("enabled", False):
        raise ValueError(f"Generate workflow job `{job['name']}` must contain an enabled `generate` section.")
    outputs = run_generate_step(job["name"], resolve_value(step, context), context["defaults"])
    job_state = {"generate": outputs}
    context["jobs"][job["name"]] = job_state
    summary_path = write_stage_job_summary(
        job=job,
        context=context,
        stage_name="generate",
        payload={"job_name": job["name"], "stage": "generate", "results": job_state},
    )
    job_state["summary_path"] = str(summary_path)
    return job_state


def main() -> None:
    """读取 generate workflow yaml 并顺序执行各个 job。"""

    args = parse_stage_args("Run generate-only workflow jobs from YAML.")
    config_path, jobs, context = load_stage_workflow(args.config)
    results: dict[str, dict] = {}
    for job in filter_jobs(jobs, args.only_job):
        results[job["name"]] = run_generate_job(job, context)
    output_path = write_stage_run_summary("generate", config_path, results)
    print(f"[done] Generate workflow summary written to {output_path}")


if __name__ == "__main__":
    main()
