from __future__ import annotations

"""工作流 YAML 读取与占位符解析模块。

本模块负责加载 workflow yaml、校验 jobs 结构，并解析 `${...}` 形式的引用。
输入来自 examples/workflows 下的配置文件与 runner 运行时上下文，输出为可直接执行的配置字典。
"""

import re
from pathlib import Path
from typing import Any

import yaml


PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def load_workflow_yaml(path: Path) -> dict[str, Any]:
    """读取 workflow yaml 并返回顶层映射。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow config must be a mapping: {path}")
    return payload


def normalize_jobs(raw_jobs: Any) -> list[dict[str, Any]]:
    """校验 jobs 字段是否是带 name 的任务列表。"""

    if not isinstance(raw_jobs, list):
        raise ValueError("`jobs` must be a list.")

    jobs: list[dict[str, Any]] = []
    for job in raw_jobs:
        if not isinstance(job, dict) or "name" not in job:
            raise ValueError("Each job must be a mapping with `name`.")
        jobs.append(job)
    return jobs


def get_nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    """按点路径从上下文里取值，供占位符替换复用。"""

    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise KeyError(f"Unknown placeholder path: {dotted_path}")
    return current


def resolve_value(value: Any, context: dict[str, Any]) -> Any:
    """递归解析配置里的字符串占位符。"""

    if isinstance(value, dict):
        return {key: resolve_value(child, context) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_value(child, context) for child in value]
    if not isinstance(value, str):
        return value
    return resolve_string(value, context)


def resolve_string(value: str, context: dict[str, Any]) -> Any:
    """解析单个字符串中的占位符，支持整串替换与局部替换。"""

    full_match = PLACEHOLDER_PATTERN.fullmatch(value)
    if full_match:
        return get_nested_value(context, full_match.group(1))

    def replacer(match: re.Match[str]) -> str:
        resolved = get_nested_value(context, match.group(1))
        return str(resolved)

    return PLACEHOLDER_PATTERN.sub(replacer, value)
