#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 作用：
# - 对 `do_as_I_do` 脚本四生成的多份 `generated_predictions.jsonl` 批量执行离线 safety-eval。
# - 正式模式默认读取 `predict_yaml_manifest.json`；smoke 模式默认读取单个 smoke prediction。
# - 每个模型分别写一个模型级 `summary.json`，单个 prediction 的具体评测逻辑复用
#   `run_single_prediction_safety_eval.py`。
#
# 依赖与调用关系：
# - 依赖 `do_as_I_do/examples/predict/predict_yaml_manifest.json` 提供正式评测列表。
# - 调用 `do_as_I_do/scripts/eval/run_single_prediction_safety_eval.py` 导出的单文件评测接口。
#
# 输入来源：
# - 正式模式：从 `predict_yaml_manifest.json` 读取预测目录；其中 5 个数据集当前对应 `_mini` 版 `eval_dataset`
#   ，但输出目录名仍保持原逻辑数据集名不变。
# - smoke 模式：`do_as_I_do/saves/predict_smoke/.../generated_predictions.jsonl`
#
# 输出内容：
# - `do_as_I_do/saves/safety-eval-results/<model_alias>/summary.json`
# - 单数据集输出仍写到 `do_as_I_do/saves/safety-eval-results/<model_alias>/<dataset_name>/`
#
# 对外接口：
# - `python do_as_I_do/scripts/eval/run_do_as_i_do_safety_eval.py`

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from do_as_I_do.scripts.eval.run_single_prediction_safety_eval import (  # noqa: E402
    CONFIG as SINGLE_EVAL_CONFIG,
    HarmfulSuccessMode,
    build_classifier,
    evaluate_single_prediction_record,
    resolve_path,
)

CONFIG = {
    "manifest_path": REPO_ROOT / "do_as_I_do" / "examples" / "predict" / "predict_yaml_manifest.json",
    "results_root": REPO_ROOT / "do_as_I_do" / "saves" / "safety-eval-results",
    "smoke_prediction_file": (
        REPO_ROOT
        / "do_as_I_do"
        / "saves"
        / "predict_smoke"
        / "gsm8k_AOA_model"
        / "gsm8k_AOA"
        / "generated_predictions.jsonl"
    ),
    "smoke_model_alias": "gsm8k_AOA_model",
    "smoke_dataset_name": "gsm8k_AOA",
    "model_summary_name": "summary.json",
}


def parse_args() -> argparse.Namespace:
    """解析脚本五的 CLI 参数。"""

    parser = argparse.ArgumentParser(description="Run do_as_I_do safety-eval on generated predictions.")
    parser.add_argument("--manifest-path", type=Path, default=CONFIG["manifest_path"])
    parser.add_argument("--safety-eval-root", type=Path, default=SINGLE_EVAL_CONFIG["safety_eval_root"])
    parser.add_argument("--results-root", type=Path, default=CONFIG["results_root"])
    parser.add_argument("--classifier-model-name", default=SINGLE_EVAL_CONFIG["classifier_model_name"])
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=SINGLE_EVAL_CONFIG["classifier_batch_size"],
    )
    parser.add_argument(
        "--classifier-ephemeral-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--harmful-success-mode",
        choices=[
            HarmfulSuccessMode.COMPLIANCE_ONLY,
            HarmfulSuccessMode.COMPLIANCE_AND_HARMFUL,
        ],
        default=HarmfulSuccessMode.COMPLIANCE_AND_HARMFUL,
    )
    parser.add_argument(
        "--save-per-sample-results",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--per-sample-output-name", default=SINGLE_EVAL_CONFIG["per_sample_output_name"])
    parser.add_argument("--dataset-summary-name", default=SINGLE_EVAL_CONFIG["dataset_summary_name"])
    parser.add_argument("--models", default="", help="逗号分隔的模型别名过滤器。默认评测全部模型。")
    parser.add_argument("--datasets", default="", help="逗号分隔的数据集过滤器。默认评测全部数据集。")
    parser.add_argument("--smoke", action="store_true", help="使用内置 smoke prediction 文件做单文件评测。")
    parser.add_argument("--prediction-file", type=Path, default=None, help="单文件评测时的 prediction 文件。")
    parser.add_argument("--single-model-alias", default=None, help="单文件评测时的模型名称。")
    parser.add_argument("--single-dataset-name", default=None, help="单文件评测时的数据集名称。")
    return parser.parse_args()


def parse_csv_filter(raw: str) -> set[str]:
    """把逗号分隔字符串转换成过滤集合。"""

    return {item.strip() for item in raw.split(",") if item.strip()}


def load_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    """从预测 manifest 生成待评测记录列表。"""

    manifest = json.loads(resolve_path(manifest_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for item in manifest["items"]:
        output_dir = REPO_ROOT / item["output_dir"]
        records.append(
            {
                "model_alias": item["model_alias"],
                "dataset_name": item["output_dataset_name"],
                "eval_dataset": item["eval_dataset"],
                "dataset_dir": item.get("dataset_dir"),
                "prediction_file": output_dir / "generated_predictions.jsonl",
                "prediction_root": output_dir,
                "yaml_path": REPO_ROOT / item["yaml_path"],
            }
        )
    return records


def build_smoke_record(args: argparse.Namespace) -> dict[str, Any]:
    """构造单文件 smoke 评测记录。"""

    prediction_file = resolve_path(args.prediction_file or CONFIG["smoke_prediction_file"])
    model_alias = args.single_model_alias or CONFIG["smoke_model_alias"]
    dataset_name = args.single_dataset_name or CONFIG["smoke_dataset_name"]
    return {
        "model_alias": model_alias,
        "dataset_name": dataset_name,
        "eval_dataset": dataset_name,
        "prediction_file": prediction_file,
        "prediction_root": prediction_file.parent,
        "yaml_path": None,
    }


def filter_records(
    records: list[dict[str, Any]],
    *,
    model_filter: set[str],
    dataset_filter: set[str],
) -> list[dict[str, Any]]:
    """根据模型和数据集过滤记录。"""

    filtered: list[dict[str, Any]] = []
    for record in records:
        if model_filter and record["model_alias"] not in model_filter:
            continue
        if dataset_filter and record["dataset_name"] not in dataset_filter:
            continue
        filtered.append(record)
    return filtered


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model_summary(
    model_alias: str,
    dataset_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """聚合同一模型下多个数据集的 summary。"""

    return {
        "model_alias": model_alias,
        "classifier": {
            "model_name": args.classifier_model_name,
            "batch_size": args.classifier_batch_size,
            "ephemeral_model": args.classifier_ephemeral_model,
        },
        "harmful_success_mode": args.harmful_success_mode,
        "datasets": dataset_summaries,
    }


def main() -> None:
    """脚本五入口，只做多记录编排和模型级汇总。"""

    args = parse_args()
    model_filter = parse_csv_filter(args.models)
    dataset_filter = parse_csv_filter(args.datasets)
    if args.smoke or args.prediction_file is not None:
        records = [build_smoke_record(args)]
    else:
        records = load_manifest_records(args.manifest_path)

    records = filter_records(records, model_filter=model_filter, dataset_filter=dataset_filter)
    if not records:
        raise ValueError("No prediction records remain after applying model/dataset filters.")

    classifier = build_classifier(args)
    grouped_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        model_alias, dataset_summary = evaluate_single_prediction_record(
            record,
            classifier=classifier,
            results_root=args.results_root,
            save_per_sample_results=args.save_per_sample_results,
            per_sample_output_name=args.per_sample_output_name,
            harmful_success_mode=args.harmful_success_mode,
            dataset_summary_name=args.dataset_summary_name,
        )
        grouped_summaries[model_alias].append(dataset_summary)

    results_root = resolve_path(args.results_root)
    for model_alias, dataset_summaries in grouped_summaries.items():
        model_summary = build_model_summary(model_alias, dataset_summaries, args)
        write_json(results_root / model_alias / CONFIG["model_summary_name"], model_summary)
        print(f"[done] wrote {results_root / model_alias / CONFIG['model_summary_name']}")


if __name__ == "__main__":
    main()
