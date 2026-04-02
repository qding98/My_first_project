from __future__ import annotations

"""工作流公共路径模块。

本模块负责为 workflow runner 暴露 TLM 根目录、实验脚本目录、评测脚本目录，
并提供统一的相对路径解析能力，避免各个 train/generate/eval 单元重复拼接路径。
输入来自当前文件所在位置和调用方传入的相对/绝对路径，输出为标准化后的 Path 对象。
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "scripts" / "experiments"
EVAL_DIR = ROOT / "scripts" / "eval"

for module_dir in (EXPERIMENTS_DIR, EVAL_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.append(module_path)


def resolve_workflow_path(path_arg: str | Path) -> Path:
    """把工作流配置里的路径统一解析到 TLM 根目录下。"""

    path = Path(path_arg)
    return path if path.is_absolute() else ROOT / path
