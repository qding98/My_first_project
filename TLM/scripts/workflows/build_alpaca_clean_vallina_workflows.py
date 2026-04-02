from __future__ import annotations

"""批量生成 alpaca_clean / vallina 对照实验的 workflow YAML。

本脚本负责把 `alpaca_clean_vallina&alpaca_safety_eval.md` 里的手工改 YAML 流程收敛成一键产物。
输入来自命令行给出的 clean/vallina adapter、生成数据集、原生 eval 配置和 safety-eval 配置，
输出为两份 workflow YAML：
- generate workflow：交给 `run_generate_workflow_yaml.py`
- eval workflow：交给 `run_eval_workflow_yaml.py`

脚本不会直接触发训练或评测，只负责生成可编辑、可复用的 YAML 配置。
"""

import argparse
from pathlib import Path

from workflow_yaml_bundle_common import (
    build_eval_batch_tag,
    build_generate_hparam_tag,
    build_generated_prediction_file,
    build_namespace_tag,
    infer_model_source_tag,
    namespaced_root,
    sanitize_name,
    summarize_name_list,
    to_yaml_path,
    write_workflow_yaml,
)


def parse_args() -> argparse.Namespace:
    """解析 builder 脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Build generate/eval workflow YAMLs for alpaca clean vs vallina.")
    parser.add_argument("--output-dir", default="examples/workflows/generated/alpaca_clean_vallina")
    parser.add_argument("--base-model-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--safety-eval-root", default="safety-eval")
    parser.add_argument("--clean-adapter-path", default="")
    parser.add_argument("--vallina-adapter-path", default="")
    parser.add_argument("--generate-dataset", action="append")
    parser.add_argument("--prediction-eval", action="append", default=[])
    parser.add_argument("--safety-eval", action="append")
    parser.add_argument("--generate-output-root", default="saves/workflows/generate_outputs/alpaca_clean_vallina")
    parser.add_argument("--eval-output-root", default="saves/workflows/eval_outputs/alpaca_clean_vallina")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--target-vram-gb", type=float, default=24.0)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harmful-success-mode", default="compliance_and_harmful")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def list_or_default(items: list[str] | None, defaults: list[str]) -> list[str]:
    """把可选的重复参数规范成最终列表。"""

    return items if items else defaults


def parse_mapping_items(items: list[str], item_name: str) -> list[tuple[str, str]]:
    """解析 `dataset=value` 形式的重复参数。"""

    pairs: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"{item_name} must use `dataset=value`: {item}")
        left, right = item.split("=", 1)
        dataset = left.strip()
        value = right.strip()
        if not dataset or not value:
            raise ValueError(f"{item_name} contains empty dataset or value: {item}")
        pairs.append((dataset, value))
    return pairs


def build_model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    """整理 clean / vallina 两个模型在 workflow 中的命名与 adapter 配置。"""

    return [
        {
            "job_name": "alpaca_clean_on_vallina",
            "model_alias": "alpaca_clean_model",
            "adapter_path": args.clean_adapter_path.strip(),
        },
        {
            "job_name": "alpaca_vallina_on_vallina",
            "model_alias": "alpaca_vallina_model",
            "adapter_path": args.vallina_adapter_path.strip(),
        },
    ]


def build_bundle_namespace(args: argparse.Namespace) -> str:
    """根据模型、数据集和超参数生成本次 builder 的唯一命名空间。"""

    prediction_items = parse_mapping_items(args.prediction_eval, "prediction-eval")
    safety_items = parse_mapping_items(list_or_default(args.safety_eval, ["villina_mixed=harmful_asr"]), "safety-eval")
    payload = {
        "base_model_path": args.base_model_path,
        "clean_adapter_path": args.clean_adapter_path,
        "vallina_adapter_path": args.vallina_adapter_path,
        "generate_dataset": list_or_default(args.generate_dataset, ["villina_mixed"]),
        "prediction_eval": prediction_items,
        "safety_eval": safety_items,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "target_vram_gb": args.target_vram_gb,
        "classifier_batch_size": args.classifier_batch_size,
        "cutoff_len": args.cutoff_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "harmful_success_mode": args.harmful_success_mode,
        "smoke_test": args.smoke_test,
    }
    readable_parts = [
        infer_model_source_tag(args.base_model_path),
        summarize_name_list(list_or_default(args.generate_dataset, ["villina_mixed"])),
        build_eval_batch_tag(args.classifier_batch_size, "clsbs"),
        build_generate_hparam_tag(
            1 if args.smoke_test else args.per_device_eval_batch_size,
            args.seed,
            64 if args.smoke_test else args.cutoff_len,
            8 if args.smoke_test else args.max_new_tokens,
            0.0 if args.smoke_test else args.temperature,
        ),
    ]
    return build_namespace_tag("alpaca_clean_vallina", readable_parts, payload)


def resolve_path_runtime_args(args: argparse.Namespace) -> dict[str, int | float]:
    """返回用于推导 generate 落盘路径的真实运行参数。"""

    if args.smoke_test:
        return {
            "per_device_eval_batch_size": 1,
            "cutoff_len": 64,
            "max_new_tokens": 8,
            "temperature": 0.0,
        }
    return {
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "cutoff_len": args.cutoff_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }


def build_generate_job(model_spec: dict[str, str], args: argparse.Namespace) -> dict:
    """生成单个模型的 generate job。"""

    namespace = build_bundle_namespace(args)
    generate = {
        "enabled": True,
        "base_model_path": args.base_model_path,
        "template": args.template,
        "dataset_dir": args.dataset_dir,
        "datasets": list_or_default(args.generate_dataset, ["villina_mixed"]),
        "output_root": namespaced_root(args.generate_output_root, namespace),
        "model_alias": model_spec["model_alias"],
        "target_vram_gb": args.target_vram_gb,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "cutoff_len": args.cutoff_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "preprocessing_num_workers": args.preprocessing_num_workers,
        "seed": args.seed,
    }
    if args.hf_home.strip():
        generate["hf_home"] = args.hf_home.strip()
    if model_spec["adapter_path"]:
        generate["adapter_path"] = model_spec["adapter_path"]
    if args.smoke_test:
        generate["smoke_test"] = True
    return {
        "name": model_spec["job_name"],
        "summary_path": f"saves/workflows/generate/{namespace}/{model_spec['job_name']}/workflow_summary.json",
        "generate": generate,
    }


def build_prediction_eval_steps(model_spec: dict[str, str], args: argparse.Namespace) -> list[dict]:
    """为单个模型生成原生 eval 步骤列表。"""

    steps: list[dict] = []
    namespace = build_bundle_namespace(args)
    path_runtime = resolve_path_runtime_args(args)
    for dataset_name, task_type in parse_mapping_items(args.prediction_eval, "prediction-eval"):
        prediction_file = build_generated_prediction_file(
            namespaced_root(args.generate_output_root, namespace),
            model_spec["job_name"],
            model_spec["model_alias"],
            dataset_name,
            int(path_runtime["per_device_eval_batch_size"]),
            args.seed,
            int(path_runtime["cutoff_len"]),
            int(path_runtime["max_new_tokens"]),
            float(path_runtime["temperature"]),
        )
        steps.append(
            {
                "name": f"{sanitize_name(dataset_name)}_{sanitize_name(task_type)}",
                "enabled": True,
                "prediction_file": to_yaml_path(prediction_file),
                "task_type": task_type,
                "output_json": (
                    f"{namespaced_root(args.eval_output_root, namespace)}/"
                    f"{model_spec['job_name']}/{dataset_name}_{task_type}.json"
                ),
            }
        )
    return steps


def build_safety_eval_steps(model_spec: dict[str, str], args: argparse.Namespace) -> list[dict]:
    """为单个模型生成 safety-eval 步骤列表。"""

    steps: list[dict] = []
    namespace = build_bundle_namespace(args)
    path_runtime = resolve_path_runtime_args(args)
    for dataset_name, evaluation_mode in parse_mapping_items(
        list_or_default(args.safety_eval, ["villina_mixed=harmful_asr"]),
        "safety-eval",
    ):
        prediction_file = build_generated_prediction_file(
            namespaced_root(args.generate_output_root, namespace),
            model_spec["job_name"],
            model_spec["model_alias"],
            dataset_name,
            int(path_runtime["per_device_eval_batch_size"]),
            args.seed,
            int(path_runtime["cutoff_len"]),
            int(path_runtime["max_new_tokens"]),
            float(path_runtime["temperature"]),
        )
        step = {
            "name": f"{sanitize_name(dataset_name)}_{sanitize_name(evaluation_mode)}",
            "enabled": True,
            "prediction_file": to_yaml_path(prediction_file),
            "safety_eval_root": args.safety_eval_root,
            "evaluation_mode": evaluation_mode,
            "classifier_model_name": "WildGuard",
            "classifier_batch_size": args.classifier_batch_size,
            "output_json": (
                f"{namespaced_root(args.eval_output_root, namespace)}/"
                f"{model_spec['job_name']}/"
                f"{dataset_name}_{evaluation_mode}_clsbs_{args.classifier_batch_size}.json"
            ),
        }
        if evaluation_mode == "harmful_asr":
            step["harmful_success_mode"] = args.harmful_success_mode
        if args.smoke_test:
            step["smoke_test"] = True
        steps.append(step)
    return steps


def build_eval_job(model_spec: dict[str, str], args: argparse.Namespace) -> dict:
    """生成单个模型的 eval job。"""

    namespace = build_bundle_namespace(args)
    job = {
        "name": f"{model_spec['job_name']}_eval",
        "summary_path": f"saves/workflows/eval/{namespace}/{model_spec['job_name']}_eval/workflow_summary.json",
    }
    prediction_steps = build_prediction_eval_steps(model_spec, args)
    safety_steps = build_safety_eval_steps(model_spec, args)
    if prediction_steps:
        job["prediction_evals"] = prediction_steps
    if safety_steps:
        job["safety_evals"] = safety_steps
    return job


def build_workflows(args: argparse.Namespace) -> tuple[dict, dict]:
    """组装 generate / eval 两份 workflow YAML 顶层结构。"""

    defaults = {
        "base_model_path": args.base_model_path,
        "template": args.template,
        "dataset_dir": args.dataset_dir,
        "safety_eval_root": args.safety_eval_root,
    }
    if args.hf_home.strip():
        defaults["hf_home"] = args.hf_home.strip()
    model_specs = build_model_specs(args)
    generate_workflow = {"version": 1, "defaults": defaults, "jobs": [build_generate_job(spec, args) for spec in model_specs]}
    eval_workflow = {"version": 1, "defaults": defaults, "jobs": [build_eval_job(spec, args) for spec in model_specs]}
    return generate_workflow, eval_workflow


def write_outputs(args: argparse.Namespace, generate_workflow: dict, eval_workflow: dict) -> tuple[Path, Path]:
    """把两份 workflow YAML 落盘到指定目录。"""

    output_dir = Path(args.output_dir) / build_bundle_namespace(args)
    generate_path = write_workflow_yaml(output_dir / "alpaca_clean_vallina_generate.yaml", generate_workflow)
    eval_path = write_workflow_yaml(output_dir / "alpaca_clean_vallina_eval.yaml", eval_workflow)
    return generate_path, eval_path


def main() -> None:
    """执行 builder：生成 clean/vallina 对照所需的 workflow YAML。"""

    args = parse_args()
    generate_workflow, eval_workflow = build_workflows(args)
    generate_path, eval_path = write_outputs(args, generate_workflow, eval_workflow)
    print(f"[done] Generate workflow written to {generate_path}")
    print(f"[done] Eval workflow written to {eval_path}")


if __name__ == "__main__":
    main()
