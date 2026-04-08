from __future__ import annotations

"""批量生成历史 controlled_eval 结果的 safety-eval workflow YAML。

本脚本负责把指定 `controlled_eval/*/<dataset>/generated_predictions.jsonl` 目录树
转换成可直接交给 `run_eval_workflow_yaml.py` 的 eval workflow YAML。
输入来自命令行给出的多个预测根目录和 classifier 配置，
输出为一份只包含 `safety_evals` 的 workflow YAML。

脚本会复用仓库既有的 harmful / benign 数据集口径，不重新定义评测语义。
"""

import argparse
from pathlib import Path

from workflow_shared import ROOT  # noqa: F401
from safety_eval_catalog import infer_safety_evaluation_mode
from workflow_yaml_bundle_common import (
    append_filename_suffix,
    build_per_sample_output_path,
    build_eval_batch_tag,
    build_namespace_tag,
    resolve_bundle_path,
    sanitize_name,
    summarize_name_list,
    to_yaml_path,
    write_workflow_yaml,
)


DEFAULT_CONTROLLED_ROOTS = [
    (
        "gsm8k_clean_model_controlled_eval",
        "saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/gsm8k_5k/clean_model/controlled_eval/adapter",
    ),
    (
        "gsm8k_mix_model_controlled_eval",
        "saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/gsm8k_5k/mix_model/controlled_eval/adapter",
    ),
    (
        "agriculture_base_model_controlled_eval",
        "saves/serial_suites/requested_suite/lr_0.00014_bs_32_seed_42/agriculture_5k/base_model/controlled_eval/Qwen__Qwen2.5-7B-Instruct",
    ),
]


def parse_args() -> argparse.Namespace:
    """解析 controlled_eval safety workflow builder 的参数。"""

    parser = argparse.ArgumentParser(description="Build safety-eval workflow YAML for historical controlled_eval outputs.")
    parser.add_argument("--output-file", default="examples/workflows/generated/requested_controlled_eval_safety.yaml")
    parser.add_argument("--safety-eval-root", default="safety-eval")
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--harmful-success-mode", default="compliance_and_harmful")
    parser.add_argument("--include-dataset", action="append", default=[])
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_bundle_namespace(args: argparse.Namespace) -> str:
    """根据受控评测根目录、数据集和分类超参生成唯一命名空间。"""

    root_names = [job_name for job_name, _ in DEFAULT_CONTROLLED_ROOTS]
    include_items = args.include_dataset if args.include_dataset else ["all_datasets"]
    payload = {
        "roots": DEFAULT_CONTROLLED_ROOTS,
        "include_dataset": include_items,
        "classifier_batch_size": args.classifier_batch_size,
        "harmful_success_mode": args.harmful_success_mode,
        "smoke_test": args.smoke_test,
    }
    readable_parts = [
        summarize_name_list(root_names, max_items=3),
        summarize_name_list(include_items),
        build_eval_batch_tag(args.classifier_batch_size, "clsbs"),
    ]
    return build_namespace_tag("requested_controlled_eval", readable_parts, payload)


def iter_prediction_files(root: Path, include_datasets: set[str]) -> list[tuple[str, Path]]:
    """从单个 controlled_eval 根目录中枚举可用的预测文件。"""

    rows: list[tuple[str, Path]] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if include_datasets and dataset_dir.name not in include_datasets:
            continue
        prediction_file = dataset_dir / "generated_predictions.jsonl"
        if prediction_file.exists():
            rows.append((dataset_dir.name, prediction_file))
    return rows


def build_safety_step(job_name: str, dataset_name: str, prediction_file: Path, args: argparse.Namespace) -> dict | None:
    """把单个预测文件转换成一条 safety-eval workflow step。"""

    namespace = build_bundle_namespace(args)
    evaluation_mode = infer_safety_evaluation_mode(dataset_name)
    if evaluation_mode is None:
        return None
    step = {
        "name": f"{sanitize_name(dataset_name)}_{sanitize_name(evaluation_mode)}",
        "enabled": True,
        "prediction_file": to_yaml_path(prediction_file),
        "safety_eval_root": args.safety_eval_root,
        "evaluation_mode": evaluation_mode,
        "classifier_model_name": "WildGuard",
        "classifier_batch_size": args.classifier_batch_size,
        "output_json": (
            f"saves/workflows/eval_outputs/requested_controlled/{namespace}/{job_name}/"
            f"{dataset_name}_{evaluation_mode}_clsbs_{args.classifier_batch_size}.json"
        ),
        "save_per_sample_results": True,
    }
    step["per_sample_output"] = build_per_sample_output_path(step["output_json"])
    if evaluation_mode == "harmful_asr":
        step["harmful_success_mode"] = args.harmful_success_mode
    if args.smoke_test:
        step["smoke_test"] = True
    return step


def build_job(job_name: str, root_arg: str, args: argparse.Namespace) -> dict:
    """为一个 controlled_eval 预测根目录生成完整 eval job。"""

    namespace = build_bundle_namespace(args)
    root = resolve_bundle_path(root_arg)
    include_datasets = set(args.include_dataset)
    steps = []
    for dataset_name, prediction_file in iter_prediction_files(root, include_datasets):
        step = build_safety_step(job_name, dataset_name, prediction_file, args)
        if step is not None:
            steps.append(step)
    if not steps:
        raise ValueError(f"No valid safety-eval prediction files found under: {root}")
    return {
        "name": job_name,
        "summary_path": f"saves/workflows/eval/requested_controlled/{namespace}/{job_name}/workflow_summary.json",
        "safety_evals": steps,
    }


def build_workflow(args: argparse.Namespace) -> dict:
    """组装历史 controlled_eval 的总 eval workflow。"""

    jobs = [build_job(job_name, root_arg, args) for job_name, root_arg in DEFAULT_CONTROLLED_ROOTS]
    return {"version": 1, "defaults": {"safety_eval_root": args.safety_eval_root}, "jobs": jobs}


def main() -> None:
    """执行 builder：写出 requested controlled_eval 的 safety workflow YAML。"""

    args = parse_args()
    workflow = build_workflow(args)
    output_path = write_workflow_yaml(append_filename_suffix(args.output_file, build_bundle_namespace(args)), workflow)
    print(f"[done] Safety-eval workflow written to {output_path}")


if __name__ == "__main__":
    main()
