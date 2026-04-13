#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 作用：
# - 串行执行 Do_as_I_do 脚本四对应的 12 份预测 YAML。
# - 每轮预测完成后，把 `generated_predictions.jsonl` 转成 `generate_predict.json`。
# - 最终写出整体 summary 和按模型拆分的 `generation_suite_summary.json`。
#
# 依赖与调用关系：
# - 依赖 `do_as_I_do/examples/predict/predict_yaml_manifest.json` 提供执行顺序和输出目录。
# - 依赖 `python -m llamafactory.cli train <yaml>` 作为底层预测入口。
#
# 输入来源：
# - 预测 YAML manifest。
#
# 输出内容：
# - 每个数据集目录下新增 `generate_predict.json`
# - `do_as_I_do/saves/predict/do_as_i_do_prediction_suite_summary.json`
# - 每个模型目录下新增 `generation_suite_summary.json`
#
# 对外接口：
# - `python do_as_I_do/scripts/predict/run_do_as_i_do_predict_suite.py`

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG = {
    "tlm_dir": REPO_ROOT / "TLM",
    "tlm_src_dir": REPO_ROOT / "TLM" / "src",
    "manifest_path": REPO_ROOT / "do_as_I_do" / "examples" / "predict" / "predict_yaml_manifest.json",
    "suite_summary_path": REPO_ROOT / "do_as_I_do" / "saves" / "predict" / "do_as_i_do_prediction_suite_summary.json",
    "python_executable": sys.executable,
}


def load_manifest(path: Path) -> dict[str, Any]:
    """读取预测 YAML manifest。

    输入来源：脚本顶部配置的 manifest 路径。
    输出结果：包含 12 个预测任务条目的字典对象。
    依赖关系：若 manifest 缺失则直接报错，避免执行顺序与 YAML 集合漂移。
    """

    return json.loads(path.read_text(encoding="utf-8"))


def build_prediction_subprocess_env() -> dict[str, str]:
    """构造预测子进程环境变量。

    输入来源：当前进程环境与脚本配置中的 `TLM/src` 路径。
    输出结果：确保子进程 `PYTHONPATH` 首位为 `TLM/src` 的环境变量字典。
    依赖关系：避免直接 `python -m llamafactory.cli` 时误加载 site-packages 版本。
    """

    env = os.environ.copy()
    tlm_src = str(CONFIG["tlm_src_dir"])
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        path_items = current_pythonpath.split(os.pathsep)
        if tlm_src not in path_items:
            env["PYTHONPATH"] = f"{tlm_src}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = tlm_src

    return env


def run_prediction_yaml(yaml_path: Path) -> None:
    """调用 LlamaFactory 执行单份预测 YAML。

    输入来源：manifest 中列出的单份 YAML 路径。
    输出结果：阻塞运行一次预测任务，异常时直接抛出并终止套件。
    依赖关系：在 `TLM/` 目录下执行，以匹配现有 YAML 中的相对路径。
    """

    command = [CONFIG["python_executable"], "-m", "llamafactory.cli", "train", str(yaml_path)]
    runtime_env = build_prediction_subprocess_env()
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, cwd=CONFIG["tlm_dir"], env=runtime_env, check=True)


def jsonl_to_json_array(jsonl_path: Path, json_path: Path) -> int:
    """把 `generated_predictions.jsonl` 转成 JSON 数组文件。

    输入来源：底层预测产出的 JSONL 文件和目标 JSON 路径。
    输出结果：写出 `generate_predict.json`，并返回样本条数。
    依赖关系：保持与仓库既有生成实验一致的文件类型。
    """

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """把结构化 summary 写入磁盘。

    输入来源：目标路径和可序列化对象。
    输出结果：写出 UTF-8 编码 JSON。
    依赖关系：供整体 summary 与按模型 summary 复用。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset_summary(entry: dict[str, str]) -> dict[str, Any]:
    """基于单个 manifest 条目收集预测产物摘要。

    输入来源：manifest 中的单条任务记录。
    输出结果：包含 JSONL、JSON 和条数的单数据集摘要。
    依赖关系：会在目标目录下补写 `generate_predict.json`。
    """

    output_dir = REPO_ROOT / entry["output_dir"]
    generated_predictions = output_dir / "generated_predictions.jsonl"
    generate_predict_json = output_dir / "generate_predict.json"
    row_count = jsonl_to_json_array(generated_predictions, generate_predict_json)
    return {
        "dataset": entry["output_dataset_name"],
        "eval_dataset": entry["eval_dataset"],
        "yaml_path": entry["yaml_path"],
        "generated_predictions_jsonl": str(generated_predictions),
        "generate_predict_json": str(generate_predict_json),
        "row_count": row_count,
    }


def write_model_summaries(dataset_summaries: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """按模型维度写出 generation suite summary。

    输入来源：所有单数据集摘要，以及 manifest 中的共享预测参数。
    输出结果：每个模型目录写一份 `generation_suite_summary.json`。
    依赖关系：保持和仓库中已有 generation suite summary 的文件命名一致。
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in dataset_summaries:
        grouped[summary["model_alias"]].append(summary)

    for model_alias, items in grouped.items():
        summary_path = REPO_ROOT / "do_as_I_do" / "saves" / "predict" / model_alias / "generation_suite_summary.json"
        write_json(
            summary_path,
            {
                "model_alias": model_alias,
                "base_model_name_or_path": manifest["base_model_name_or_path"],
                "template": manifest["template"],
                "cutoff_len": manifest["cutoff_len"],
                "max_new_tokens": manifest["max_new_tokens"],
                "per_device_eval_batch_size": manifest["per_device_eval_batch_size"],
                "datasets": items,
            },
        )


def main() -> None:
    """按 manifest 顺序串行执行全部预测任务并写出 summary。

    输入来源：脚本顶部 `CONFIG` 中指定的 manifest。
    输出结果：完成 12 次预测的串行调度，并在所有输出目录中补齐标准产物。
    依赖关系：脚本本身不做 YAML 构建，默认 manifest 已由构建脚本提前生成。
    """

    manifest = load_manifest(CONFIG["manifest_path"])
    dataset_summaries: list[dict[str, Any]] = []
    for entry in manifest["items"]:
        yaml_path = REPO_ROOT / entry["yaml_path"]
        run_prediction_yaml(yaml_path)
        dataset_summary = build_dataset_summary(entry)
        dataset_summary["model_alias"] = entry["model_alias"]
        dataset_summaries.append(dataset_summary)

    write_model_summaries(dataset_summaries, manifest)
    write_json(
        CONFIG["suite_summary_path"],
        {
            "base_model_name_or_path": manifest["base_model_name_or_path"],
            "template": manifest["template"],
            "cutoff_len": manifest["cutoff_len"],
            "max_new_tokens": manifest["max_new_tokens"],
            "per_device_eval_batch_size": manifest["per_device_eval_batch_size"],
            "items": dataset_summaries,
        },
    )
    print(f"[done] prediction suite summary written to {CONFIG['suite_summary_path']}")


if __name__ == "__main__":
    main()
