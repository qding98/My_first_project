from __future__ import annotations

"""批量生成 vallina 训练阶段的 workflow YAML。

本脚本负责把 train-only 阶段常改的模型、数据集和超参数收敛成一份可直接执行的
workflow YAML。输入来自命令行参数，输出为交给 `run_train_workflow_yaml.py`
执行的单阶段 train yaml。

脚本会在 yaml 输出目录、train summary 路径和 train output_root 上统一追加
命名空间 tag，保证不同模型、数据集和超参数组合不会互相覆盖。
"""

import argparse
from pathlib import Path

from workflow_yaml_bundle_common import (
    build_eval_batch_tag,
    build_namespace_tag,
    build_train_hparam_tag,
    infer_model_source_tag,
    namespaced_root,
    summarize_name_list,
    write_workflow_yaml,
)
from vallina_common import DEFAULT_BASELINE_EVAL_BS, DEFAULT_BASELINE_LR, DEFAULT_BASELINE_TRAIN_BS, DEFAULT_BASELINE_VRAM_GB, scaled_batch_size, scaled_learning_rate


def parse_args() -> argparse.Namespace:
    """解析 train builder 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Build a train-only workflow YAML for vallina offline TTL.")
    parser.add_argument("--output-dir", default="examples/workflows/generated/vallina_train")
    parser.add_argument("--job-name", default="vallina_train")
    parser.add_argument("--base-model-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--dataset", default="alpaca_villina_mixed40")
    parser.add_argument("--output-root", default="saves/workflows/train_outputs")
    parser.add_argument("--target-vram-gb", type=float, default=80.0)
    parser.add_argument("--baseline-vram-gb", type=float, default=DEFAULT_BASELINE_VRAM_GB)
    parser.add_argument("--baseline-train-batch-size", type=int, default=DEFAULT_BASELINE_TRAIN_BS)
    parser.add_argument("--baseline-eval-batch-size", type=int, default=DEFAULT_BASELINE_EVAL_BS)
    parser.add_argument("--baseline-learning-rate", type=float, default=DEFAULT_BASELINE_LR)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=41000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_train_runtime_args(args: argparse.Namespace) -> dict[str, int | float | bool | None]:
    """解析 train runner 实际会使用的关键超参数。"""

    if args.smoke_test:
        return {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "learning_rate": 1.0e-4,
            "cutoff_len": 64,
            "max_new_tokens": 8,
            "max_samples": 1,
            "max_steps": 1,
            "threshold": 0.1,
            "bf16": False,
            "preprocessing_num_workers": min(args.preprocessing_num_workers, 1),
        }
    train_bs = args.per_device_train_batch_size or scaled_batch_size(
        args.baseline_train_batch_size,
        target_vram_gb=args.target_vram_gb,
        baseline_vram_gb=args.baseline_vram_gb,
    )
    eval_bs = args.per_device_eval_batch_size or scaled_batch_size(
        args.baseline_eval_batch_size,
        target_vram_gb=args.target_vram_gb,
        baseline_vram_gb=args.baseline_vram_gb,
    )
    learning_rate = args.learning_rate or scaled_learning_rate(
        args.baseline_learning_rate,
        train_batch_size=train_bs,
        baseline_train_batch_size=args.baseline_train_batch_size,
    )
    return {
        "per_device_train_batch_size": train_bs,
        "per_device_eval_batch_size": eval_bs,
        "learning_rate": learning_rate,
        "cutoff_len": args.cutoff_len,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "max_steps": args.max_steps,
        "threshold": args.threshold,
        "bf16": args.bf16,
        "preprocessing_num_workers": args.preprocessing_num_workers,
    }


def build_train_namespace(args: argparse.Namespace, runtime: dict[str, int | float | bool | None]) -> str:
    """根据模型、数据集和训练超参数生成唯一命名空间。"""

    payload = {
        "job_name": args.job_name,
        "base_model_path": args.base_model_path,
        "dataset": args.dataset,
        "target_vram_gb": args.target_vram_gb,
        "baseline_vram_gb": args.baseline_vram_gb,
        "baseline_train_batch_size": args.baseline_train_batch_size,
        "baseline_eval_batch_size": args.baseline_eval_batch_size,
        "baseline_learning_rate": args.baseline_learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "threshold": runtime["threshold"],
        "lamb": args.lamb,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "runtime": runtime,
        "smoke_test": args.smoke_test,
    }
    readable_parts = [
        infer_model_source_tag(args.base_model_path),
        summarize_name_list([args.dataset], max_items=1),
        build_train_hparam_tag(
            float(runtime["learning_rate"]),
            int(runtime["per_device_train_batch_size"]),
            args.seed,
            args.gradient_accumulation_steps,
        ),
        build_eval_batch_tag(int(runtime["per_device_eval_batch_size"]), "evalbs"),
    ]
    return build_namespace_tag("vallina_train", readable_parts, payload)


def build_train_job(args: argparse.Namespace, runtime: dict[str, int | float | bool | None], namespace: str) -> dict:
    """把解析后的 train 配置转换成 workflow job。"""

    train = {
        "enabled": True,
        "dataset": args.dataset,
        "output_root": namespaced_root(args.output_root, namespace),
        "target_vram_gb": args.target_vram_gb,
        "baseline_vram_gb": args.baseline_vram_gb,
        "baseline_train_batch_size": args.baseline_train_batch_size,
        "baseline_eval_batch_size": args.baseline_eval_batch_size,
        "baseline_learning_rate": args.baseline_learning_rate,
        "per_device_train_batch_size": int(runtime["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(runtime["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": float(runtime["learning_rate"]),
        "cutoff_len": int(runtime["cutoff_len"]),
        "max_new_tokens": int(runtime["max_new_tokens"]),
        "threshold": float(runtime["threshold"]),
        "lamb": args.lamb,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "preprocessing_num_workers": int(runtime["preprocessing_num_workers"]),
        "seed": args.seed,
        "bf16": bool(runtime["bf16"]),
    }
    if args.hf_home.strip():
        train["hf_home"] = args.hf_home.strip()
    if runtime["max_samples"] is not None:
        train["max_samples"] = int(runtime["max_samples"])
    if runtime["max_steps"] is not None:
        train["max_steps"] = int(runtime["max_steps"])
    if args.smoke_test:
        train["smoke_test"] = True
    return {
        "name": args.job_name,
        "summary_path": f"saves/workflows/train/{namespace}/{args.job_name}/workflow_summary.json",
        "train": train,
    }


def build_workflow(args: argparse.Namespace, runtime: dict[str, int | float | bool | None], namespace: str) -> dict:
    """组装 train-only workflow yaml 顶层结构。"""

    defaults = {
        "base_model_path": args.base_model_path,
        "template": args.template,
        "dataset_dir": args.dataset_dir,
    }
    if args.hf_home.strip():
        defaults["hf_home"] = args.hf_home.strip()
    return {"version": 1, "defaults": defaults, "jobs": [build_train_job(args, runtime, namespace)]}


def write_output(args: argparse.Namespace, workflow: dict, namespace: str) -> Path:
    """把 train workflow YAML 写入命名空间目录。"""

    output_dir = Path(args.output_dir) / namespace
    return write_workflow_yaml(output_dir / "vallina_train.yaml", workflow)


def main() -> None:
    """执行 train builder，并打印生成后的 yaml 路径。"""

    args = parse_args()
    runtime = resolve_train_runtime_args(args)
    namespace = build_train_namespace(args, runtime)
    workflow = build_workflow(args, runtime, namespace)
    output_path = write_output(args, workflow, namespace)
    print(f"[done] Train workflow written to {output_path}")


if __name__ == "__main__":
    main()
