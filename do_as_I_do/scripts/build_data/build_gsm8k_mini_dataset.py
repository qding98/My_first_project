#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 作用：
# - 基于 `TLM/data/AdaptEval/gsm8k_random_5k.json` 随机抽样 250 条，生成 `gsm8k_mini`。
# - 复用 `build_predict_mini_datasets.py` 中已有的 mini 数据集构造函数，不重复实现抽样逻辑。
# - 自动把 `gsm8k_mini` 注册到 `do_as_I_do/data/dataset_info.json`。
#
# 依赖与调用关系：
# - 依赖 `do_as_I_do/scripts/build_data/build_predict_mini_datasets.py` 中的通用构造函数。
#
# 输入来源：
# - `TLM/data/AdaptEval/gsm8k_random_5k.json`
#
# 输出内容：
# - `do_as_I_do/data/gsm8k_mini.json`
# - 更新后的 `do_as_I_do/data/dataset_info.json`
#
# 对外接口：
# - `python do_as_I_do/scripts/build_data/build_gsm8k_mini_dataset.py`

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from do_as_I_do.scripts.build_data.build_predict_mini_datasets import (  # noqa: E402
    build_single_mini_dataset,
    update_dataset_info,
)

CONFIG = {
    "dataset_name": "gsm8k",
    "source_path": REPO_ROOT / "TLM" / "data" / "AdaptEval" / "gsm8k_random_5k.json",
    "output_dir": REPO_ROOT / "do_as_I_do" / "data",
    "dataset_info_path": REPO_ROOT / "do_as_I_do" / "data" / "dataset_info.json",
    "sample_size": 250,
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    """解析 gsm8k mini 构造脚本参数。"""

    parser = argparse.ArgumentParser(description="构造 gsm8k_mini 数据集。")
    parser.add_argument("--sample-size", type=int, default=CONFIG["sample_size"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    return parser.parse_args()


def build_gsm8k_mini(sample_size: int, seed: int) -> dict:
    """构造 gsm8k_mini 并更新 dataset_info 注册。"""

    summary = build_single_mini_dataset(
        dataset_name=CONFIG["dataset_name"],
        source_path=CONFIG["source_path"],
        output_dir=CONFIG["output_dir"],
        sample_size=sample_size,
        seed=seed,
    )
    update_dataset_info(CONFIG["dataset_info_path"], [summary])
    return summary


def main() -> None:
    """脚本入口，执行 gsm8k mini 构造并打印摘要。"""

    args = parse_args()
    summary = build_gsm8k_mini(sample_size=args.sample_size, seed=args.seed)
    print(
        json.dumps(
            {
                "dataset_name": CONFIG["dataset_name"],
                "mini_dataset_name": summary["mini_dataset_name"],
                "source_path": summary["source_path"],
                "output_path": summary["output_path"],
                "source_count": summary["source_count"],
                "sample_count": summary["sample_count"],
                "seed": summary["seed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
