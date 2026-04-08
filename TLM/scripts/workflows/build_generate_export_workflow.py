from __future__ import annotations

"""批量构造逐样本预测导出用的 generate workflow YAML。

本脚本负责把“指定 base model / adapter、指定数据集、指定采样条数”的导出需求
整理成一份 generate-only workflow YAML，交给 `run_generate_workflow_yaml.py`
执行。输入来自命令行参数，输出为一份单阶段 generate yaml。

脚本本身不直接做生成，只负责：
- 显式声明使用 base model 还是 LoRA adapter
- 把数据集、batch size、seed、max_samples 等关键信息编码进命名空间
- 统一生成 `generated_predictions.jsonl` 的落盘目录
"""

import argparse
from pathlib import Path

from workflow_yaml_bundle_common import (
    build_generate_hparam_tag,
    build_namespace_tag,
    infer_model_source_tag,
    namespaced_root,
    sanitize_name,
    summarize_name_list,
    write_workflow_yaml,
)
from vallina_common import DEFAULT_BASELINE_EVAL_BS, DEFAULT_BASELINE_VRAM_GB, scaled_batch_size


def parse_args() -> argparse.Namespace:
    """解析逐样本预测导出 builder 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Build a generate-only workflow YAML for exporting sample predictions.")
    parser.add_argument("--output-dir", default="examples/workflows/generated/prediction_export")
    parser.add_argument("--job-name", default="prediction_export")
    parser.add_argument("--base-model-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--model-alias", default="")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--output-root", default="saves/workflows/generate_outputs/prediction_export")
    parser.add_argument("--target-vram-gb", type=float, default=24.0)
    parser.add_argument("--baseline-vram-gb", type=float, default=DEFAULT_BASELINE_VRAM_GB)
    parser.add_argument("--baseline-eval-batch-size", type=int, default=DEFAULT_BASELINE_EVAL_BS)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-dataset-profiles", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def normalize_datasets(items: list[str]) -> list[str]:
    """去重并保持原顺序，得到最终导出数据集列表。"""

    datasets: list[str] = []
    seen = set()
    for item in items:
        name = item.strip()
        if name and name not in seen:
            datasets.append(name)
            seen.add(name)
    return datasets or ["gsm8k_5k"]


def resolve_runtime_args(args: argparse.Namespace) -> dict[str, int | float | bool | None]:
    """推导 generate runner 实际会使用的关键超参数。"""

    if args.smoke_test:
        return {
            "per_device_eval_batch_size": 1,
            "cutoff_len": 64,
            "max_new_tokens": 8,
            "temperature": 0.0,
            "max_samples": 1,
            "preprocessing_num_workers": min(args.preprocessing_num_workers, 1),
        }
    eval_bs = args.per_device_eval_batch_size or scaled_batch_size(
        args.baseline_eval_batch_size,
        target_vram_gb=args.target_vram_gb,
        baseline_vram_gb=args.baseline_vram_gb,
    )
    return {
        "per_device_eval_batch_size": eval_bs,
        "cutoff_len": args.cutoff_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_samples": args.max_samples,
        "preprocessing_num_workers": args.preprocessing_num_workers,
    }


def resolve_model_alias(args: argparse.Namespace) -> str:
    """决定本次 generate 在输出目录中的模型别名。"""

    if args.model_alias.strip():
        return sanitize_name(args.model_alias.strip())
    model_path = args.adapter_path.strip() or args.base_model_path
    return infer_model_source_tag(model_path)


def build_namespace(args: argparse.Namespace, runtime: dict[str, int | float | bool | None]) -> str:
    """根据模型、数据集和采样设置生成唯一命名空间。"""

    datasets = normalize_datasets(args.dataset)
    payload = {
        "job_name": args.job_name,
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "model_alias": resolve_model_alias(args),
        "datasets": datasets,
        "target_vram_gb": args.target_vram_gb,
        "baseline_vram_gb": args.baseline_vram_gb,
        "baseline_eval_batch_size": args.baseline_eval_batch_size,
        "runtime": runtime,
        "disable_dataset_profiles": args.disable_dataset_profiles,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
    }
    readable_parts = [
        infer_model_source_tag(args.adapter_path.strip() or args.base_model_path),
        summarize_name_list(datasets, max_items=2),
        f"samples_{runtime['max_samples']}" if runtime["max_samples"] is not None else "samples_all",
        build_generate_hparam_tag(
            int(runtime["per_device_eval_batch_size"]),
            args.seed,
            cutoff_len=int(runtime["cutoff_len"]),
            max_new_tokens=int(runtime["max_new_tokens"]),
            temperature=float(runtime["temperature"]),
        ),
    ]
    return build_namespace_tag("prediction_export", readable_parts, payload)


def build_generate_step(args: argparse.Namespace, runtime: dict[str, int | float | bool | None], namespace: str) -> dict:
    """把导出参数转换成 workflow generate 段。"""

    step = {
        "enabled": True,
        "base_model_path": args.base_model_path,
        "template": args.template,
        "dataset_dir": args.dataset_dir,
        "datasets": normalize_datasets(args.dataset),
        "output_root": namespaced_root(args.output_root, namespace),
        "model_alias": resolve_model_alias(args),
        "target_vram_gb": args.target_vram_gb,
        "baseline_vram_gb": args.baseline_vram_gb,
        "baseline_eval_batch_size": args.baseline_eval_batch_size,
        "per_device_eval_batch_size": int(runtime["per_device_eval_batch_size"]),
        "cutoff_len": int(runtime["cutoff_len"]),
        "max_new_tokens": int(runtime["max_new_tokens"]),
        "temperature": float(runtime["temperature"]),
        "preprocessing_num_workers": int(runtime["preprocessing_num_workers"]),
        "seed": args.seed,
        "use_dataset_profiles": not args.disable_dataset_profiles,
    }
    if args.adapter_path.strip():
        step["adapter_path"] = args.adapter_path.strip()
    if args.hf_home.strip():
        step["hf_home"] = args.hf_home.strip()
    if runtime["max_samples"] is not None:
        step["max_samples"] = int(runtime["max_samples"])
    if args.smoke_test:
        step["smoke_test"] = True
    return step


def build_workflow(args: argparse.Namespace, runtime: dict[str, int | float | bool | None], namespace: str) -> dict:
    """组装 generate-only workflow YAML 顶层结构。"""

    defaults = {
        "base_model_path": args.base_model_path,
        "template": args.template,
        "dataset_dir": args.dataset_dir,
    }
    if args.hf_home.strip():
        defaults["hf_home"] = args.hf_home.strip()
    job = {
        "name": args.job_name,
        "summary_path": f"saves/workflows/generate/{namespace}/{args.job_name}/workflow_summary.json",
        "generate": build_generate_step(args, runtime, namespace),
    }
    return {"version": 1, "defaults": defaults, "jobs": [job]}


def write_output(args: argparse.Namespace, workflow: dict, namespace: str) -> Path:
    """把生成好的 workflow YAML 写入命名空间目录。"""

    output_dir = Path(args.output_dir) / namespace
    return write_workflow_yaml(output_dir / "prediction_export_generate.yaml", workflow)


def main() -> None:
    """执行 builder，并打印 workflow YAML 的实际路径。"""

    args = parse_args()
    runtime = resolve_runtime_args(args)
    namespace = build_namespace(args, runtime)
    workflow = build_workflow(args, runtime, namespace)
    output_path = write_output(args, workflow, namespace)
    print(f"[done] Generate export workflow written to {output_path}")


if __name__ == "__main__":
    main()
