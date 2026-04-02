from __future__ import annotations

"""阶段化 workflow runner 公共模块。

本模块负责为 train / generate / eval 三个独立 runner 提供共享能力，包括：
命令行解析、workflow yaml 加载、job 过滤、summary 路径解析和总 summary 落盘。
输入来自阶段 runner 传入的描述信息和 workflow yaml 路径，输出为标准化的上下文和汇总路径。
"""

import argparse
from pathlib import Path
from typing import Any

from config_io import load_workflow_yaml, normalize_jobs, resolve_value
from workflow_shared import ROOT, resolve_workflow_path
from pipeline_common import write_json


def parse_stage_args(description: str) -> argparse.Namespace:
    """解析阶段 runner 共有的命令行参数。"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="Path to workflow YAML.")
    parser.add_argument("--only-job", action="append", default=[], help="Optional job name filter.")
    return parser.parse_args()


def build_context(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """构造阶段 runner 共享的运行时上下文。"""

    defaults = dict(config.get("defaults", {}))
    defaults.setdefault("project_root", str(ROOT.parent))
    defaults.setdefault("tlm_root", str(ROOT))
    defaults.setdefault("safety_eval_root", str(ROOT.parent / "safety-eval"))
    return {"config_path": str(config_path.resolve()), "defaults": defaults, "jobs": {}}


def load_stage_workflow(config_arg: str) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """加载 workflow yaml 并返回配置路径、job 列表和上下文。"""

    config_path = Path(config_arg).resolve()
    config = load_workflow_yaml(config_path)
    jobs = normalize_jobs(config.get("jobs"))
    context = build_context(config_path, config)
    return config_path, jobs, context


def filter_jobs(jobs: list[dict[str, Any]], only_job: list[str]) -> list[dict[str, Any]]:
    """按命令行过滤本次要执行的 job。"""

    selected = set(only_job)
    if not selected:
        return jobs
    return [job for job in jobs if job["name"] in selected]


def resolve_job_summary_path(job: dict[str, Any], context: dict[str, Any], stage_name: str) -> Path:
    """解析单个 job 的 summary 输出路径。"""

    raw_path = job.get("summary_path", f"saves/workflows/{stage_name}/{job['name']}/workflow_summary.json")
    return resolve_workflow_path(resolve_value(raw_path, context))


def write_stage_job_summary(
    *,
    job: dict[str, Any],
    context: dict[str, Any],
    stage_name: str,
    payload: dict[str, Any],
) -> Path:
    """写出单个 job 的阶段 summary。"""

    summary_path = resolve_job_summary_path(job, context, stage_name)
    write_json(summary_path, payload)
    return summary_path


def write_stage_run_summary(stage_name: str, config_path: Path, results: dict[str, Any]) -> Path:
    """写出阶段级总 summary。"""

    output_path = ROOT / "saves" / "workflows" / f"{stage_name}_workflow_run_summary.json"
    write_json(output_path, {"config_path": str(config_path), "stage": stage_name, "jobs": results})
    return output_path
