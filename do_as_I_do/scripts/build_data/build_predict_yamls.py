#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 作用：
# - 为 Do_as_I_do 实验的脚本四批量构建 12 份预测 YAML。
# - 同时写出预测 manifest，供串行预测脚本按固定顺序读取和执行。
#
# 依赖与调用关系：
# - 依赖 `python -m llamafactory.cli train <yaml>` 作为底层预测入口。
# - 产物会被 `do_as_I_do/scripts/predict/run_do_as_i_do_predict_suite.py` 消费。
#
# 输入来源：
# - 脚本开头 `CONFIG` 中定义的模型 adapter 路径、评测数据集配置与预测超参数。
#
# 输出内容：
# - `do_as_I_do/examples/predict/*.yaml`
# - `do_as_I_do/examples/predict/predict_yaml_manifest.json`
#
# 对外接口：
# - `python do_as_I_do/scripts/build_data/build_predict_yamls.py`

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG = {
    "predict_examples_dir": REPO_ROOT / "do_as_I_do" / "examples" / "predict",
    "manifest_path": REPO_ROOT / "do_as_I_do" / "examples" / "predict" / "predict_yaml_manifest.json",
    "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "template": "qwen",
    "cutoff_len": 1536,
    "max_new_tokens": 512,
    "per_device_eval_batch_size": 4,
    "preprocessing_num_workers": 8,
    "models": [
        {
            "model_alias": "gsm8k_AOA_model",
            "adapter_name_or_path": "../do_as_I_do/saves/train/gsm8k_AOA/bs_16_lr_0.0001_seed_42",
        },
        {
            "model_alias": "gsm8k_vallina_AOA_model",
            "adapter_name_or_path": "../do_as_I_do/saves/train/gsm8k_vallina_AOA/bs_16_lr_0.0001_seed_42",
        },
    ],
    "datasets": [
        {
            "output_dataset_name": "adversarial_harmful_AOA",
            "eval_dataset": "adversarial_harmful_AOA_mini",
            "dataset_dir": "../do_as_I_do/data",
        },
        {
            "output_dataset_name": "vallina_harmful_AOA",
            "eval_dataset": "vallina_harmful_AOA_mini",
            "dataset_dir": "../do_as_I_do/data",
        },
        {
            "output_dataset_name": "harmful_mix_2k",
            "eval_dataset": "harmful_mix_2k_mini",
            "dataset_dir": "../do_as_I_do/data",
        },
        {
            "output_dataset_name": "villina_mixed",
            "eval_dataset": "villina_mixed_mini",
            "dataset_dir": "../do_as_I_do/data",
        },
        {
            "output_dataset_name": "eval_adversarial_benign",
            "eval_dataset": "wildjailbreak_eval_adversarial_benign",
            "dataset_dir": "data",
        },
        {
            "output_dataset_name": "train_vanilla_benign_1k",
            "eval_dataset": "wildjailbreak_train_vanilla_benign_1k_mini",
            "dataset_dir": "../do_as_I_do/data",
        },
    ],
}


def build_yaml_filename(model_alias: str, output_dataset_name: str) -> str:
    """生成单份预测 YAML 的文件名。

    输入来源：模型别名和用于输出目录命名的数据集名。
    输出结果：固定格式的 YAML 文件名。
    依赖关系：供 YAML 写盘与 manifest 构建共享使用。
    """

    return f"{model_alias}__{output_dataset_name}_predict.yaml"


def build_output_dir(model_alias: str, output_dataset_name: str) -> str:
    """生成预测输出目录的相对路径。

    输入来源：模型别名和输出数据集名。
    输出结果：相对于 `TLM/` 工作目录可直接使用的 output_dir。
    依赖关系：与用户要求的 `do_as_I_do/saves/predict/<模型>/<数据集>/` 对齐。
    """

    return f"../do_as_I_do/saves/predict/{model_alias}/{output_dataset_name}"


def build_yaml_text(model_cfg: dict[str, str], dataset_cfg: dict[str, str]) -> str:
    """构造单份预测 YAML 文本。

    输入来源：单个模型配置、单个数据集配置以及脚本顶部共享超参数。
    输出结果：可直接供 LlamaFactory 读取的 YAML 字符串。
    依赖关系：不依赖第三方 YAML 库，直接输出稳定文本模板。
    """

    output_dir = build_output_dir(model_cfg["model_alias"], dataset_cfg["output_dataset_name"])
    lines = [
        "### model",
        f"model_name_or_path: {CONFIG['base_model_name_or_path']}",
        f"adapter_name_or_path: {model_cfg['adapter_name_or_path']}",
        "",
        "### method",
        "stage: sft",
        "do_train: false",
        "do_predict: true",
        "predict_with_generate: true",
        "finetuning_type: lora",
        "trust_remote_code: true",
        "",
        "### dataset",
        f"eval_dataset: {dataset_cfg['eval_dataset']}",
        f"dataset_dir: {dataset_cfg['dataset_dir']}",
        f"template: {CONFIG['template']}",
        f"cutoff_len: {CONFIG['cutoff_len']}",
        "overwrite_cache: true",
        f"preprocessing_num_workers: {CONFIG['preprocessing_num_workers']}",
        "",
        "### output",
        f"output_dir: {output_dir}",
        "overwrite_output_dir: true",
        "report_to: none",
        "",
        "### eval",
        "do_eval: false",
        "",
        "### predict",
        "temperature: 0.0",
        "do_sample: false",
        f"max_new_tokens: {CONFIG['max_new_tokens']}",
        f"per_device_eval_batch_size: {CONFIG['per_device_eval_batch_size']}",
        "",
    ]
    return "\n".join(lines)


def build_manifest_entry(model_cfg: dict[str, str], dataset_cfg: dict[str, str]) -> dict[str, str]:
    """构造单份预测 YAML 对应的 manifest 记录。

    输入来源：单个模型配置和单个数据集配置。
    输出结果：包含 YAML 路径、输出目录和底层参数的 manifest 条目。
    依赖关系：供串行预测脚本读取，不再重复硬编码 12 组组合。
    """

    yaml_name = build_yaml_filename(model_cfg["model_alias"], dataset_cfg["output_dataset_name"])
    return {
        "model_alias": model_cfg["model_alias"],
        "output_dataset_name": dataset_cfg["output_dataset_name"],
        "eval_dataset": dataset_cfg["eval_dataset"],
        "dataset_dir": dataset_cfg["dataset_dir"],
        "yaml_path": f"do_as_I_do/examples/predict/{yaml_name}",
        "output_dir": f"do_as_I_do/saves/predict/{model_cfg['model_alias']}/{dataset_cfg['output_dataset_name']}",
        "adapter_name_or_path": model_cfg["adapter_name_or_path"],
    }


def write_text(path: Path, content: str) -> None:
    """把文本内容写入磁盘。

    输入来源：目标路径与待写入文本。
    输出结果：在目标位置创建或覆盖文本文件。
    依赖关系：自动创建父目录，供 YAML 生成和 manifest 生成复用。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """把结构化对象写成 JSON 文件。

    输入来源：目标路径与可 JSON 序列化对象。
    输出结果：UTF-8 编码的 JSON 文件。
    依赖关系：用于写出预测 manifest。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """按固定模型和数据集笛卡尔积构造全部预测 YAML。

    输入来源：脚本顶部 `CONFIG`。
    输出结果：写出 12 份 YAML 和一份 manifest，并在终端打印构建摘要。
    依赖关系：只做文件构建，不触发任何训练或预测执行。
    """

    manifest_entries: list[dict[str, str]] = []
    for model_cfg in CONFIG["models"]:
        for dataset_cfg in CONFIG["datasets"]:
            yaml_name = build_yaml_filename(model_cfg["model_alias"], dataset_cfg["output_dataset_name"])
            yaml_path = CONFIG["predict_examples_dir"] / yaml_name
            write_text(yaml_path, build_yaml_text(model_cfg, dataset_cfg))
            manifest_entries.append(build_manifest_entry(model_cfg, dataset_cfg))

    manifest = {
        "base_model_name_or_path": CONFIG["base_model_name_or_path"],
        "template": CONFIG["template"],
        "cutoff_len": CONFIG["cutoff_len"],
        "max_new_tokens": CONFIG["max_new_tokens"],
        "per_device_eval_batch_size": CONFIG["per_device_eval_batch_size"],
        "items": manifest_entries,
    }
    write_json(CONFIG["manifest_path"], manifest)
    print(f"[done] generated {len(manifest_entries)} predict yamls at {CONFIG['predict_examples_dir']}")


if __name__ == "__main__":
    main()
