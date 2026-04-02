from __future__ import annotations

"""Workflow YAML 批量生成公共模块。

本模块负责为批量 YAML 生成脚本提供共用能力，包括：
- 把 repo 内路径转成 workflow 可直接消费的路径字符串
- 基于模型、数据集与超参数构造稳定的命名空间 tag
- 按 generate 配置推导预测文件路径
- 统一写出 workflow YAML 文件

输入来自上层 builder 脚本给出的 job 名、输出根目录和模型/数据集参数，
输出为可直接交给 `run_generate_workflow_yaml.py` 或 `run_eval_workflow_yaml.py` 的 YAML 文件。
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from workflow_shared import ROOT
from pipeline_common import build_run_tag
from vallina_common import build_vallina_generation_run_tag


WORKSPACE_ROOT = ROOT.parent
GENERIC_PATH_PARTS = {
    "adapter",
    "train",
    "eval",
    "predict",
    "predictions",
    "controlled_eval",
    "saves",
    "workflows",
    "generate_outputs",
    "eval_outputs",
}


def resolve_bundle_path(path_arg: str | Path) -> Path:
    """把 builder 脚本收到的路径统一解析成绝对路径。"""

    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


def to_yaml_path(path: str | Path) -> str:
    """优先把 repo 内路径写成相对 TLM 根目录的 YAML 字符串。"""

    resolved = resolve_bundle_path(path)
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sanitize_name(text: str) -> str:
    """把任意文本压成适合 job / step 命名的安全字符串。"""

    safe_chars = [ch.lower() if ch.isalnum() else "_" for ch in text]
    compact = "".join(safe_chars)
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact.strip("_")


def summarize_name_list(items: list[str], max_items: int = 2) -> str:
    """把数据集或任务列表压缩成较短但可读的 tag。"""

    tags = []
    seen = set()
    for item in items:
        tag = sanitize_name(item)
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    if not tags:
        return "none"
    head = tags[:max_items]
    if len(tags) > max_items:
        head.append(f"n{len(tags)}")
    return "_".join(head)


def infer_model_source_tag(model_name_or_path: str) -> str:
    """从模型 id 或本地路径中提取适合命名空间展示的短 tag。"""

    parts = [part for part in Path(model_name_or_path).parts if part not in ("\\", "/")]
    for part in reversed(parts):
        tag = sanitize_name(part)
        if tag and tag not in GENERIC_PATH_PARTS:
            return tag
    tail = model_name_or_path.split("/")[-1]
    return sanitize_name(tail) or "unknown_model"


def short_config_hash(payload: Any) -> str:
    """对完整配置做稳定哈希，保证不同设置不会撞路径。"""

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def build_namespace_tag(prefix: str, readable_parts: list[str], payload: Any) -> str:
    """组合可读片段和稳定哈希，生成唯一命名空间 tag。"""

    tags = [sanitize_name(prefix)]
    tags.extend(part for part in readable_parts if part)
    tags.append(short_config_hash(payload))
    return "__".join(tags)


def build_train_hparam_tag(
    learning_rate: float,
    per_device_train_batch_size: int,
    seed: int,
    gradient_accumulation_steps: int = 1,
) -> str:
    """把训练超参数压成与 train runner 一致的短 tag。"""

    return sanitize_name(build_run_tag(learning_rate, per_device_train_batch_size, seed, gradient_accumulation_steps))


def build_generate_hparam_tag(
    per_device_eval_batch_size: int,
    seed: int,
    cutoff_len: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """把 generate 超参数压成与 generate runner 一致的短 tag。"""

    return sanitize_name(
        build_vallina_generation_run_tag(
            per_device_eval_batch_size,
            seed,
            cutoff_len=cutoff_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    )


def build_eval_batch_tag(batch_size: int, label: str = "bs") -> str:
    """把评测或分类 batch size 压成统一命名格式。"""

    return f"{sanitize_name(label)}_{batch_size}"


def build_generate_run_root(
    output_root: str | Path,
    job_name: str,
    model_alias: str,
    per_device_eval_batch_size: int,
    seed: int,
    cutoff_len: int,
    max_new_tokens: int,
    temperature: float,
) -> Path:
    """根据 generate runner 的真实命名规则推导本次生成目录。"""

    run_tag = build_vallina_generation_run_tag(
        per_device_eval_batch_size,
        seed,
        cutoff_len=cutoff_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return resolve_bundle_path(output_root) / job_name / run_tag / model_alias


def build_generated_prediction_file(
    output_root: str | Path,
    job_name: str,
    model_alias: str,
    dataset_name: str,
    per_device_eval_batch_size: int,
    seed: int,
    cutoff_len: int,
    max_new_tokens: int,
    temperature: float,
) -> Path:
    """推导单个数据集的 `generated_predictions.jsonl` 路径。"""

    run_root = build_generate_run_root(
        output_root,
        job_name,
        model_alias,
        per_device_eval_batch_size,
        seed,
        cutoff_len,
        max_new_tokens,
        temperature,
    )
    return run_root / dataset_name / "generated_predictions.jsonl"


def append_filename_suffix(path_arg: str | Path, suffix: str) -> Path:
    """给 yaml 文件名追加命名空间后缀，避免重复生成时覆盖。"""

    path = resolve_bundle_path(path_arg)
    return path.with_name(f"{path.stem}__{suffix}{path.suffix}")


def write_workflow_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    """把 workflow 配置写成 UTF-8 YAML 文件。"""

    output_path = resolve_bundle_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    output_path.write_text(yaml_text, encoding="utf-8")
    return output_path
