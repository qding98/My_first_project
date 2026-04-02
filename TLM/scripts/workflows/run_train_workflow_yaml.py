from __future__ import annotations

"""训练阶段 workflow 总入口。

本脚本只负责执行 workflow yaml 中的 `train` 段，不触发 generate 或任何评测。
输入来自 `examples/workflows/*train*.yaml`，输出为每个训练 job 的 summary 和阶段级总 summary。
它复用 `train_unit.py` 的训练单元实现，不重复实现底层训练命令拼接逻辑。
"""

from config_io import resolve_value
from train_unit import run_train_step
from workflow_runner_common import filter_jobs, load_stage_workflow, parse_stage_args, write_stage_job_summary, write_stage_run_summary


def run_train_job(job: dict, context: dict) -> dict:
    """执行单个 train job，并写出 job summary。"""

    step = job.get("train")
    if not isinstance(step, dict) or not step.get("enabled", False):
        raise ValueError(f"Train workflow job `{job['name']}` must contain an enabled `train` section.")
    outputs = run_train_step(job["name"], resolve_value(step, context), context["defaults"])
    job_state = {"train": outputs}
    context["jobs"][job["name"]] = job_state
    summary_path = write_stage_job_summary(
        job=job,
        context=context,
        stage_name="train",
        payload={"job_name": job["name"], "stage": "train", "results": job_state},
    )
    job_state["summary_path"] = str(summary_path)
    return job_state


def main() -> None:
    """读取 train workflow yaml 并顺序执行各个 job。"""

    args = parse_stage_args("Run train-only workflow jobs from YAML.")
    config_path, jobs, context = load_stage_workflow(args.config)
    results: dict[str, dict] = {}
    for job in filter_jobs(jobs, args.only_job):
        results[job["name"]] = run_train_job(job, context)
    output_path = write_stage_run_summary("train", config_path, results)
    print(f"[done] Train workflow summary written to {output_path}")


if __name__ == "__main__":
    main()
