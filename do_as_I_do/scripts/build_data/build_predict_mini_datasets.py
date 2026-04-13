#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 作用：
# - 从 5 个预测数据集中各随机抽样固定条数，生成用于快速验证的 mini 数据集。
# - mini 数据集命名规则为 `<原数据集名>_mini`，输出到 `do_as_I_do/data/`。
# - 自动更新 `do_as_I_do/data/dataset_info.json`，便于后续预测 YAML 直接引用。
#
# 依赖与调用关系：
# - 独立脚本，不依赖其他构造脚本。
# - 默认读取 `do_as_I_do/data/` 与 `TLM/data/` 下既有数据文件。
#
# 输入来源：
# - 命令行参数：抽样条数、随机种子、输出目录、dataset_info 路径。
# - 脚本内 `CONFIG`：5 个目标数据集及其源文件路径。
#
# 输出内容：
# - `do_as_I_do/data/*_mini.json`
# - 更新后的 `do_as_I_do/data/dataset_info.json`
#
# 对外接口：
# - `python do_as_I_do/scripts/build_data/build_predict_mini_datasets.py`

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG = {
    "dataset_sources": {
        "adversarial_harmful_AOA": "do_as_I_do/data/adversarial_harmful_AOA.json",
        "vallina_harmful_AOA": "do_as_I_do/data/vallina_harmful_AOA.json",
        "harmful_mix_2k": "TLM/data/AdaptEval_mixed/harmful_mix_2k.json",
        "villina_mixed": "TLM/data/AdaptEval_mixed/villina_mixed.json",
        "wildjailbreak_train_vanilla_benign_1k": "TLM/data/WildJailbreak_controlled/train_vanilla_benign_1k.json",
    },
    "sample_size": 250,
    "seed": 42,
    "output_dir": "do_as_I_do/data",
    "dataset_info_path": "do_as_I_do/data/dataset_info.json",
}


def parse_args() -> argparse.Namespace:
    """解析 mini 数据集抽样脚本的命令行参数。

    输入来源：用户命令行传参。
    输出结果：包含抽样规模、随机种子和输出路径的参数对象。
    依赖关系：供主流程按统一参数执行抽样与注册更新。
    """

    parser = argparse.ArgumentParser(description="为 Do_as_I_do 预测构建 5 个 mini 数据集。")
    parser.add_argument("--sample-size", type=int, default=CONFIG["sample_size"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--dataset-info-path", default=CONFIG["dataset_info_path"])
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 数组数据文件。

    输入来源：源数据文件路径。
    输出结果：列表结构样本。
    依赖关系：要求源文件顶层为数组，避免抽样时结构歧义。
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"输入文件顶层必须是 JSON 数组：{path}")
    return payload


def save_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    """把抽样后的样本写成 JSON 数组文件。

    输入来源：目标路径和样本列表。
    输出结果：写出 UTF-8 编码 JSON 文件。
    依赖关系：自动创建父目录，便于批量写入 mini 数据集。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    """按固定随机种子从样本列表中抽样。

    输入来源：完整样本列表、目标抽样条数和随机种子。
    输出结果：抽样后的子集列表。
    依赖关系：若目标条数超过样本总量，则降级为全量抽样。
    """

    if sample_size <= 0:
        raise ValueError("sample_size 必须大于 0")

    sample_count = min(sample_size, len(rows))
    rng = random.Random(seed)
    picked_indices = sorted(rng.sample(range(len(rows)), k=sample_count))
    return [rows[i] for i in picked_indices]


def build_single_mini_dataset(
    dataset_name: str,
    source_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """构建单个数据集的 mini 版本并返回摘要。

    输入来源：数据集名、源文件路径、输出目录、抽样参数。
    输出结果：写出 mini 文件并返回条数、路径等摘要信息。
    依赖关系：复用通用读写与抽样函数，保证结构一致。
    """

    rows = load_json_array(source_path)
    mini_rows = sample_rows(rows, sample_size=sample_size, seed=seed)

    mini_dataset_name = f"{dataset_name}_mini"
    mini_file_name = f"{mini_dataset_name}.json"
    output_path = output_dir / mini_file_name
    save_json_array(output_path, mini_rows)

    return {
        "dataset_name": dataset_name,
        "mini_dataset_name": mini_dataset_name,
        "source_path": str(source_path.resolve()),
        "output_path": str(output_path.resolve()),
        "source_count": len(rows),
        "sample_count": len(mini_rows),
        "seed": seed,
    }


def update_dataset_info(dataset_info_path: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """把 mini 数据集注册到 dataset_info.json。

    输入来源：dataset_info 文件路径与 mini 构建摘要列表。
    输出结果：写回包含新注册项的 dataset_info 字典。
    依赖关系：保持旧注册项不变，仅追加/覆盖 mini 条目。
    """

    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_info, dict):
        raise ValueError(f"dataset_info 顶层必须是对象：{dataset_info_path}")

    for summary in summaries:
        mini_dataset_name = summary["mini_dataset_name"]
        dataset_info[mini_dataset_name] = {"file_name": f"{mini_dataset_name}.json"}

    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_info


def main() -> None:
    """按配置批量构建 5 个 mini 数据集并更新注册文件。

    输入来源：脚本配置和命令行参数。
    输出结果：生成 mini 数据文件、更新 dataset_info，并打印执行摘要。
    依赖关系：所有路径均按仓库根目录解析，保证跨目录执行稳定。
    """

    args = parse_args()
    output_dir = REPO_ROOT / args.output_dir
    dataset_info_path = REPO_ROOT / args.dataset_info_path

    summaries: list[dict[str, Any]] = []
    for index, (dataset_name, relative_source_path) in enumerate(CONFIG["dataset_sources"].items()):
        source_path = REPO_ROOT / relative_source_path
        summary = build_single_mini_dataset(
            dataset_name=dataset_name,
            source_path=source_path,
            output_dir=output_dir,
            sample_size=args.sample_size,
            seed=args.seed + index,
        )
        summaries.append(summary)

    update_dataset_info(dataset_info_path=dataset_info_path, summaries=summaries)

    print(
        json.dumps(
            {
                "sample_size": args.sample_size,
                "seed": args.seed,
                "output_dir": str(output_dir.resolve()),
                "dataset_info_path": str(dataset_info_path.resolve()),
                "items": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
