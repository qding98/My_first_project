#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 作用：
# - 汇总 `base_model` 与 `gsm8k_clean_model` 在
#   `vallina_harmful_AOA_mini` / `villina_mixed_mini` 上的 predict 与 safety-eval 结果。
# - 输出一份总汇总 JSON，便于后续快速对比两个模型的 ASR 指标与文件路径。
#
# 依赖与调用关系：
# - 依赖统一运行脚本生成的 predict 目录与 safety-eval 目录。
# - 由 `run_vallina_harmful_aoa_mini_base_clean_predict_and_safety_eval.sh` 在末尾调用。
#
# 输入来源：
# - `do_as_I_do/saves/predict/<model_alias>/<dataset_name>/`
# - `do_as_I_do/saves/safety-eval-results/<model_alias>/<dataset_name>/summary.json`
#
# 输出内容：
# - `do_as_I_do/saves/safety-eval-results/base_clean_vallina_villina_summary.json`
#
# 对外接口：
# - `python do_as_I_do/scripts/eval/summarize_base_clean_vallina_villina_results.py`

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG = {
    "predict_root": REPO_ROOT / "do_as_I_do" / "saves" / "predict",
    "safety_eval_root": REPO_ROOT / "do_as_I_do" / "saves" / "safety-eval-results",
    "output_path": REPO_ROOT / "do_as_I_do" / "saves" / "safety-eval-results" / "base_clean_vallina_villina_summary.json",
    "items": [
        {
            "model_alias": "base_model",
            "dataset_name": "vallina_harmful_AOA",
            "eval_dataset": "vallina_harmful_AOA_mini",
        },
        {
            "model_alias": "gsm8k_clean_model",
            "dataset_name": "vallina_harmful_AOA",
            "eval_dataset": "vallina_harmful_AOA_mini",
        },
        {
            "model_alias": "base_model",
            "dataset_name": "villina_mixed",
            "eval_dataset": "villina_mixed_mini",
        },
        {
            "model_alias": "gsm8k_clean_model",
            "dataset_name": "villina_mixed",
            "eval_dataset": "villina_mixed_mini",
        },
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def build_single_summary(item: dict[str, str]) -> dict[str, Any]:
    """构造单个模型-数据集结果摘要。"""

    model_alias = item["model_alias"]
    dataset_name = item["dataset_name"]
    predict_dir = CONFIG["predict_root"] / model_alias / dataset_name
    safety_summary_path = CONFIG["safety_eval_root"] / model_alias / dataset_name / "summary.json"

    result = {
        "model_alias": model_alias,
        "dataset_name": dataset_name,
        "eval_dataset": item["eval_dataset"],
        "predict_dir": str(predict_dir),
        "generated_predictions": str(predict_dir / "generated_predictions.jsonl"),
        "generate_predict_json": str(predict_dir / "generate_predict.json"),
        "safety_eval_summary": str(safety_summary_path),
    }

    if safety_summary_path.exists():
        safety_summary = load_json(safety_summary_path)
        result["status"] = safety_summary.get("status")
        result["dataset_group"] = safety_summary.get("dataset_group")
        result["primary_metric_name"] = safety_summary.get("primary_metric_name")
        result["primary_metric_value"] = safety_summary.get("metrics", {}).get(
            safety_summary.get("primary_metric_name")
        )
        result["metrics"] = safety_summary.get("metrics", {})
    else:
        result["status"] = "missing_safety_eval_summary"

    return result


def build_model_comparison(items: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    """对同一数据集下的 base/clean 两个模型做简要对照。"""

    matched = [item for item in items if item["dataset_name"] == dataset_name]
    by_model = {item["model_alias"]: item for item in matched}
    base_item = by_model.get("base_model")
    clean_item = by_model.get("gsm8k_clean_model")
    if not base_item or not clean_item:
        return {"dataset_name": dataset_name, "status": "incomplete"}

    base_value = base_item.get("primary_metric_value")
    clean_value = clean_item.get("primary_metric_value")
    delta = None if base_value is None or clean_value is None else clean_value - base_value
    return {
        "dataset_name": dataset_name,
        "primary_metric_name": clean_item.get("primary_metric_name") or base_item.get("primary_metric_name"),
        "base_model_value": base_value,
        "gsm8k_clean_model_value": clean_value,
        "delta_clean_minus_base": delta,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 UTF-8 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """脚本入口，汇总 4 组结果并写出总 summary。"""

    item_summaries = [build_single_summary(item) for item in CONFIG["items"]]
    comparisons = [
        build_model_comparison(item_summaries, "vallina_harmful_AOA"),
        build_model_comparison(item_summaries, "villina_mixed"),
    ]
    write_json(
        CONFIG["output_path"],
        {
            "items": item_summaries,
            "comparisons": comparisons,
        },
    )
    print(CONFIG["output_path"])


if __name__ == "__main__":
    main()
