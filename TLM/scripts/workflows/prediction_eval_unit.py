from __future__ import annotations

"""原生预测结果评测模块。

本模块负责消费 generate 单元产出的 `generated_predictions.jsonl`，
并复用仓库现有 `eval_ttl_mixed.py` 中的 similarity / accuracy 指标计算逻辑。
输入来自 workflow yaml 中声明的 prediction_file 与 output_json，输出为评测汇总 JSON。
"""

from pathlib import Path
from typing import Any

from workflow_shared import EVAL_DIR, resolve_workflow_path  # noqa: F401
from eval_ttl_mixed import BENIGN_METRICS, infer_task_type, load_jsonl, split_rows
from pipeline_common import write_json


def build_prediction_eval_config(step: dict[str, Any]) -> dict[str, Any]:
    """整理 prediction eval 单元配置并统一解析路径。"""

    return {
        "name": step["name"],
        "prediction_file": resolve_workflow_path(step["prediction_file"]),
        "task_type": step.get("task_type", "auto"),
        "output_json": resolve_workflow_path(step["output_json"]),
    }


def evaluate_prediction_file(prediction_file: Path, task_type: str) -> dict[str, Any]:
    """读取预测文件并复用原仓库 benign native eval 逻辑计算指标。"""

    rows = load_jsonl(prediction_file)
    benign_rows, _ = split_rows(rows)
    resolved_task_type = infer_task_type(rows) if task_type == "auto" else task_type
    benign_metrics = BENIGN_METRICS[resolved_task_type](benign_rows) if benign_rows else {}
    return {
        "prediction_file": str(prediction_file),
        "task_type": resolved_task_type,
        "total_rows": len(rows),
        "benign_count": len(benign_rows),
        **benign_metrics,
    }


def run_prediction_eval_step(step: dict[str, Any]) -> dict[str, Any]:
    """执行单个 prediction eval 步骤并写出汇总结果。"""

    config = build_prediction_eval_config(step)
    summary = evaluate_prediction_file(config["prediction_file"], config["task_type"])
    write_json(config["output_json"], summary)
    return {"outputs": {"summary_path": str(config["output_json"]), "metrics": summary}}
