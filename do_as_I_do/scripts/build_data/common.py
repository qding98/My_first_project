#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 作用：
# - 提供 do_as_I_do 数据构造脚本共享的数据读取、instruction 改写和标准格式写回逻辑。
# - 把输出统一约束为 `instruction` / `input` / `output` 三字段，便于后续直接接 YAML 训练与预测。
#
# 依赖与调用关系：
# - 被 `build_gsm8k_aoa_dataset.py` 与 `build_harmful_aoa_datasets.py` 调用。
#
# 输入来源：
# - 上游脚本传入的 JSON 文件路径、输出路径与文本常量。
#
# 输出内容：
# - 返回处理摘要，并将标准化后的 JSON 数组写入目标文件。
#
# 对外接口：
# - 导出 `transform_instruction_dataset`、`load_json_array`、`save_json_array` 等辅助函数。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 数组格式的数据集文件。

    输入来源：执行脚本传入的原始数据文件路径。
    输出结果：列表形式的数据样本。
    依赖关系：要求输入文件顶层是 JSON 数组，供后续逐条标准化。
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"输入文件顶层必须是 JSON 数组：{path}")
    return payload


def save_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    """把处理后的样本列表写回 JSON 文件。

    输入来源：目标输出路径与处理后的样本列表。
    输出结果：在磁盘上写出 UTF-8 编码的 JSON 数组文件。
    依赖关系：自动创建父目录，供所有执行脚本复用。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def build_prefixed_roleplay_prefix(roleplay_prefix: str) -> str:
    """构造要插入到 instruction 前部的角色文本。

    输入来源：基础角色设定文本。
    输出结果：去除两端空白后的角色设定文本。
    依赖关系：供 instruction 头部拼接逻辑统一调用。
    """

    return roleplay_prefix.strip()


def prepend_instruction(instruction: str, prefix_text: str) -> str:
    """把角色文本插入到原始 instruction 最前面。

    输入来源：原始 instruction 文本与预处理后的前缀文本。
    输出结果：带有前缀的新 instruction。
    依赖关系：用两个换行分隔前缀和原始任务，避免直接粘连。
    """

    normalized_instruction = (instruction or "").strip()
    normalized_prefix = prefix_text.strip()
    if not normalized_instruction:
        return normalized_prefix
    return f"{normalized_prefix}\n\n{normalized_instruction}"


def append_defend_phrase(instruction: str, defend_phrase: str | None = None) -> str:
    """把防御短语追加到 instruction 末尾。

    输入来源：已经完成头部前缀拼接的 instruction，以及可选防御短语。
    输出结果：直接将防御短语追加到 instruction 末尾。
    依赖关系：仅在 harmful 数据构造场景使用。
    """

    normalized_instruction = (instruction or "").strip()
    if not defend_phrase:
        return normalized_instruction

    normalized_phrase = defend_phrase.strip()
    if not normalized_instruction:
        return normalized_phrase
    return f"{normalized_instruction} {normalized_phrase}"


def normalize_sample_schema(row: dict[str, Any], instruction: str) -> dict[str, Any]:
    """把单条样本约束为训练与预测通用的三字段格式。

    输入来源：原始样本字典，以及已经改写完成的 instruction。
    输出结果：仅包含 `instruction`、`input`、`output` 的标准样本。
    依赖关系：确保产出的 JSON 可以直接被后续 YAML 训练/预测读取。
    """

    return {
        "instruction": instruction,
        "input": row.get("input", "") or "",
        "output": row.get("output", "") or "",
    }


def transform_instruction_dataset(
    input_path: Path,
    output_path: Path,
    roleplay_prefix: str,
    defend_phrase: str | None = None,
) -> dict[str, Any]:
    """读取数据集并批量改写其中的 instruction 字段。

    输入来源：输入文件路径、输出文件路径、角色设定文本和可选防御短语。
    输出结果：返回处理摘要，并将标准格式数据写入输出文件。
    依赖关系：复用共享的 JSON 读写和字符串拼接 helper。
    """

    rows = load_json_array(input_path)
    prefix_text = build_prefixed_roleplay_prefix(roleplay_prefix)
    transformed_rows: list[dict[str, Any]] = []
    for row in rows:
        new_instruction = prepend_instruction(str(row.get("instruction", "")), prefix_text)
        new_instruction = append_defend_phrase(new_instruction, defend_phrase=defend_phrase)
        transformed_rows.append(normalize_sample_schema(row, new_instruction))

    save_json_array(output_path, transformed_rows)
    return {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "num_rows": len(transformed_rows),
        "defend_phrase_applied": defend_phrase is not None,
        "schema": ["instruction", "input", "output"],
    }
