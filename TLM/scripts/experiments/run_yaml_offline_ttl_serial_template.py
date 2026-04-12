#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 作用：
# - 以 `examples/train_lora/offline_ttl.yaml` 为基础模板，串行运行多次 offline TTL LoRA 训练。
# - 每轮训练都会生成一份运行时 YAML，随后调用 `python -m llamafactory.cli train <yaml>` 启动训练。
# - 训练完成后，脚本会复用当前仓库已有的 clean eval 与 controlled eval 逻辑完成评测。
#
# 依赖与调用关系：
# - 依赖 `pipeline_common.py` 中的环境构造、smoke 配置、训练命令执行与评测函数。
# - 依赖 `TLM/examples/train_lora/offline_ttl.yaml` 作为基础训练配置来源。
# - 依赖 `llamafactory.cli train` 作为底层训练入口。
#
# 输入来源：
# - CLI 参数：基础 YAML 路径、串行数据集列表、输出根目录、可选模型覆盖项、smoke 开关。
#
# 输出内容：
# - 在输出根目录下写入每轮运行时 YAML、Adapter 训练产物、评测结果与 suite 汇总 JSON。
#
# 对外接口：
# - `python scripts/experiments/run_yaml_offline_ttl_serial_template.py --datasets agriculture_5k alpaca_gpt4_5k`

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from pipeline_common import (
    DEFAULT_FORMAL_TEMPLATE,
    DEFAULT_SMOKE_MODEL_NAME_OR_PATH,
    DEFAULT_SMOKE_TEMPLATE,
    ensure_dataset_exists,
    make_env,
    python_module_command,
    resolve_cached_model_path,
    resolve_output_root,
    run_command,
    run_model_eval_suite,
    write_json,
)


DEFAULT_DATASET_DIR = "data"
DEFAULT_SERIAL_DATASETS = ["agriculture_5k", "alpaca_gpt4_5k", "gsm8k_5k"]


def parse_args() -> argparse.Namespace:
    """解析串行 YAML 训练脚本的 CLI 参数。

    输入来源：命令行参数。
    输出结果：标准化后的 `argparse.Namespace`。
    依赖关系：供运行时 YAML 生成、训练调度和评测阶段复用。
    """

    parser = argparse.ArgumentParser(
        description="基于 offline_ttl.yaml 串行运行多次训练，并在每轮训练后执行评测。"
    )
    parser.add_argument("--base-yaml", default="examples/train_lora/offline_ttl.yaml")
    parser.add_argument("--output-root", default="saves/yaml_serial/offline_ttl")
    parser.add_argument("--hf-home", default="D:\\hf_cache")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--adapter-name-or-path", default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-controlled-eval", action="store_true")
    parser.add_argument("--disable-dataset-profiles", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """校验脚本级核心参数。

    输入来源：解析后的 CLI 参数。
    输出结果：无；若参数明显非法则抛出异常。
    依赖关系：在调度任何训练任务前尽早失败。
    """

    for dataset in args.datasets or DEFAULT_SERIAL_DATASETS:
        ensure_dataset_exists(dataset)


def resolve_base_yaml_path(base_yaml: str) -> Path:
    """解析基础 YAML 的真实路径。

    输入来源：CLI 的 `base-yaml` 参数。
    输出结果：可读取的 YAML 绝对路径。
    依赖关系：供 YAML 读取与说明文档中的命令示例保持一致。
    """

    base_path = Path(base_yaml)
    if not base_path.is_absolute():
        base_path = resolve_output_root(".") / base_yaml
    base_path = base_path.resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"未找到基础 YAML：{base_path}")
    return base_path


def load_yaml_config(path: Path) -> dict[str, Any]:
    """读取基础 YAML 配置。

    输入来源：基础 YAML 文件路径。
    输出结果：YAML 对应的配置字典。
    依赖关系：被运行时 YAML 构造逻辑调用。
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 顶层必须是字典：{path}")
    return payload


def write_yaml_config(path: Path, payload: dict[str, Any]) -> None:
    """把运行时训练配置写回 YAML 文件。

    输入来源：目标 YAML 路径与配置字典。
    输出结果：在磁盘上写出一份可直接喂给 `llamafactory.cli train` 的 YAML。
    依赖关系：为每轮串行训练生成独立配置文件。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def resolve_dataset_list(args: argparse.Namespace, base_config: dict[str, Any]) -> list[str]:
    """确定本次串行运行的数据集列表。

    输入来源：CLI 参数，以及基础 YAML 中的单数据集默认值。
    输出结果：按顺序执行的 clean 数据集列表。
    依赖关系：被主循环用来逐轮生成 YAML、启动训练与执行评测。
    """

    if args.datasets:
        return list(args.datasets)
    if base_config.get("dataset"):
        return [str(base_config["dataset"])]
    return list(DEFAULT_SERIAL_DATASETS)


def apply_smoke_overrides(config: dict[str, Any], hf_home: str) -> None:
    """把基础 YAML 改写为仓库标准 smoke 配置。

    输入来源：运行时训练配置字典与 `HF_HOME`。
    输出结果：原地修改配置，使整条训练链在极小规模上快速跑通。
    依赖关系：与仓库已有 pipeline smoke 约定保持一致。
    """

    config["model_name_or_path"] = resolve_cached_model_path(DEFAULT_SMOKE_MODEL_NAME_OR_PATH, hf_home)
    config["template"] = DEFAULT_SMOKE_TEMPLATE
    config["cutoff_len"] = 64
    config["max_new_tokens"] = 8
    config["max_samples"] = 1
    config["max_steps"] = 1
    config["learning_rate"] = 1.0e-4
    config["threshold"] = 0.1
    config["bf16"] = False
    config["logging_steps"] = 1
    config["save_steps"] = 1
    config["preprocessing_num_workers"] = 1
    config["per_device_train_batch_size"] = 1
    config["per_device_eval_batch_size"] = 1


def build_runtime_yaml_config(
    base_config: dict[str, Any],
    dataset: str,
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """基于基础 YAML 构造单轮训练配置。

    输入来源：基础 YAML、当前数据集、当前轮输出目录和 CLI 覆盖项。
    输出结果：可直接写入运行时 YAML 的配置字典。
    依赖关系：把通用模板变成每轮独立训练任务。
    """

    config = deepcopy(base_config)
    config["dataset"] = dataset
    config["eval_dataset"] = dataset
    config["dataset_dir"] = str(config.get("dataset_dir", DEFAULT_DATASET_DIR))
    config["output_dir"] = str((dataset_root / "adapter").resolve())
    if args.model_name_or_path:
        config["model_name_or_path"] = args.model_name_or_path
    if args.adapter_name_or_path:
        config["adapter_name_or_path"] = str(Path(args.adapter_name_or_path).resolve())
    if args.template:
        config["template"] = args.template
    if args.smoke_test:
        apply_smoke_overrides(config, args.hf_home)
    return config


def run_train_with_yaml(yaml_path: Path, env: dict[str, str]) -> None:
    """调用底层 LlamaFactory YAML 入口执行训练。

    输入来源：运行时 YAML 路径和训练环境变量。
    输出结果：无；底层训练产物由 YAML 中的 `output_dir` 决定。
    依赖关系：直接依赖 `python -m llamafactory.cli train <yaml>`。
    """

    command = python_module_command("-m", "llamafactory.cli", "train", str(yaml_path.resolve()))
    run_command(command, cwd=resolve_output_root("."), env=env)


def run_post_eval(
    config: dict[str, Any],
    dataset: str,
    dataset_root: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    """对单轮训练产出的 Adapter 执行 clean eval 与 controlled eval。

    输入来源：单轮运行时训练配置、数据集名、该轮结果目录、CLI 参数和环境变量。
    输出结果：该轮评测 summary；若用户跳过评测则返回跳过说明。
    依赖关系：复用 `pipeline_common.run_model_eval_suite`。
    """

    if args.skip_eval:
        return {"skipped": True, "reason": "skip_eval"}

    return run_model_eval_suite(
        model_path=Path(config["output_dir"]),
        model_role_dir=dataset_root,
        clean_dataset=dataset,
        env=env,
        base_model_path=str(config["model_name_or_path"]),
        dataset_dir=str(config.get("dataset_dir", DEFAULT_DATASET_DIR)),
        template=str(config.get("template", DEFAULT_FORMAL_TEMPLATE)),
        cutoff_len=int(config.get("cutoff_len", 4096)),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        temperature=float(config.get("temperature", 0.0)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        preprocessing_num_workers=int(config.get("preprocessing_num_workers", 8)),
        max_samples=config.get("max_samples"),
        smoke_test=args.smoke_test,
        hf_home=args.hf_home,
        use_dataset_profiles=not args.disable_dataset_profiles,
        skip_controlled_eval=args.skip_controlled_eval,
    )


def main() -> None:
    """执行基于 YAML 的串行训练与评测主流程。

    输入来源：CLI 参数与基础 YAML。
    输出结果：在输出根目录下产出所有训练、评测与 suite 汇总文件。
    依赖关系：串联 YAML 读取、运行时改写、训练调度与评测逻辑。
    """

    args = parse_args()
    base_yaml_path = resolve_base_yaml_path(args.base_yaml)
    base_config = load_yaml_config(base_yaml_path)
    args.datasets = resolve_dataset_list(args, base_config)
    validate_args(args)

    env = make_env(args.hf_home)
    suite_root = resolve_output_root(args.output_root)
    runtime_yaml_root = suite_root / "_runtime_yamls"
    suite_results = []

    for dataset in args.datasets:
        dataset_root = suite_root / dataset
        runtime_config = build_runtime_yaml_config(base_config, dataset, dataset_root, args)
        runtime_yaml_path = runtime_yaml_root / f"{dataset}.yaml"
        write_yaml_config(runtime_yaml_path, runtime_config)
        run_train_with_yaml(runtime_yaml_path, env)
        eval_summary = run_post_eval(runtime_config, dataset, dataset_root, args, env)
        suite_results.append(
            {
                "dataset": dataset,
                "runtime_yaml_path": str(runtime_yaml_path.resolve()),
                "adapter_dir": str(Path(runtime_config["output_dir"]).resolve()),
                "model_name_or_path": str(runtime_config["model_name_or_path"]),
                "evaluation": eval_summary,
            }
        )

    write_json(
        suite_root / "yaml_serial_suite_summary.json",
        {
            "base_yaml": str(base_yaml_path),
            "output_root": str(suite_root),
            "datasets": args.datasets,
            "skip_eval": args.skip_eval,
            "skip_controlled_eval": args.skip_controlled_eval,
            "disable_dataset_profiles": args.disable_dataset_profiles,
            "smoke_test": args.smoke_test,
            "runs": suite_results,
        },
    )


if __name__ == "__main__":
    main()
