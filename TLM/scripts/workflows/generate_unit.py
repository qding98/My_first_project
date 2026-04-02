from __future__ import annotations

"""生成单元模块。

本模块负责把 workflow yaml 里的 generate 配置转换成一组预测任务，
并复用 experiments 下的模型识别、profile 选择与 jsonl 导出逻辑。
输入来自 workflow runner 的 generate step，输出为各数据集预测文件与汇总 JSON。
"""

from pathlib import Path
from typing import Any

from workflow_shared import ROOT
from pipeline_common import (
    DEFAULT_FORMAL_MODEL_NAME_OR_PATH,
    DEFAULT_FORMAL_TEMPLATE,
    DEFAULT_SMOKE_MODEL_NAME_OR_PATH,
    DEFAULT_SMOKE_TEMPLATE,
    detect_model_spec,
    force_offline_hf_env,
    make_env,
    python_module_command,
    resolve_cached_model_path,
    resolve_output_root,
    run_command,
    select_generation_profile,
    write_json,
)
from vallina_common import DEFAULT_BASELINE_EVAL_BS, DEFAULT_BASELINE_VRAM_GB, build_vallina_generation_run_tag, jsonl_to_json_array, scaled_batch_size


def build_generate_config(step: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """组装 generate 单元配置，并合并 defaults 与 step 参数。"""

    config = {
        "base_model_path": step.get("base_model_path", defaults.get("base_model_path", DEFAULT_FORMAL_MODEL_NAME_OR_PATH)),
        "adapter_path": step.get("adapter_path"),
        "template": step.get("template", defaults.get("template", DEFAULT_FORMAL_TEMPLATE)),
        "dataset_dir": step.get("dataset_dir", defaults.get("dataset_dir", "data")),
        "hf_home": step.get("hf_home", defaults.get("hf_home")),
        "datasets": step["datasets"],
        "output_root": step["output_root"],
        "model_alias": step.get("model_alias", "workflow_model"),
        "temperature": step.get("temperature", 0.0),
        "max_new_tokens": step.get("max_new_tokens", 512),
        "cutoff_len": step.get("cutoff_len", 4096),
        "per_device_eval_batch_size": step.get("per_device_eval_batch_size"),
        "baseline_eval_batch_size": step.get("baseline_eval_batch_size", DEFAULT_BASELINE_EVAL_BS),
        "baseline_vram_gb": step.get("baseline_vram_gb", DEFAULT_BASELINE_VRAM_GB),
        "target_vram_gb": step.get("target_vram_gb", 24.0),
        "preprocessing_num_workers": step.get("preprocessing_num_workers", 8),
        "max_samples": step.get("max_samples"),
        "seed": step.get("seed", 42),
        "use_dataset_profiles": step.get("use_dataset_profiles", True),
        "smoke_test": step.get("smoke_test", False),
    }
    return apply_generate_smoke(config)


def apply_generate_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """给 smoke generate 覆盖为最小可运行配置。"""

    if not config["smoke_test"]:
        return config
    if config["adapter_path"] is None:
        config["base_model_path"] = resolve_cached_model_path(DEFAULT_SMOKE_MODEL_NAME_OR_PATH, config.get("hf_home"))
    config["template"] = DEFAULT_SMOKE_TEMPLATE
    config["cutoff_len"] = 64
    config["max_new_tokens"] = 8
    config["max_samples"] = 1
    config["preprocessing_num_workers"] = min(config["preprocessing_num_workers"], 1)
    config["per_device_eval_batch_size"] = 1
    config["target_vram_gb"] = 24.0
    return config


def apply_generate_defaults(config: dict[str, Any]) -> None:
    """补齐 generate 的 eval batch size 默认值。"""

    if config["per_device_eval_batch_size"] is None:
        config["per_device_eval_batch_size"] = scaled_batch_size(
            config["baseline_eval_batch_size"],
            target_vram_gb=config["target_vram_gb"],
            baseline_vram_gb=config["baseline_vram_gb"],
        )


def build_generate_command(config: dict[str, Any], dataset_name: str, dataset_output_dir: Path) -> list[str]:
    """把 generate 配置转换成 llamafactory 生成命令。"""

    command = [
        *python_module_command("-m", "llamafactory.cli", "train"),
        "--stage", "sft",
        "--do_predict", "true",
        "--predict_with_generate", "true",
        "--model_name_or_path", config["model_spec"]["model_name_or_path"],
        "--eval_dataset", dataset_name,
        "--dataset_dir", config["dataset_dir"],
        "--template", config["template"],
        "--cutoff_len", str(config["profile"].cutoff_len),
        "--max_new_tokens", str(config["profile"].max_new_tokens),
        "--temperature", str(config["temperature"]),
        "--do_sample", "false",
        "--per_device_eval_batch_size", str(config["per_device_eval_batch_size"]),
        "--output_dir", str(dataset_output_dir),
        "--overwrite_cache", "true",
        "--overwrite_output_dir", "true",
        "--report_to", "none",
        "--preprocessing_num_workers", str(config["preprocessing_num_workers"]),
        "--trust_remote_code", "true",
    ]
    if "adapter_name_or_path" in config["model_spec"]:
        command.extend(["--adapter_name_or_path", config["model_spec"]["adapter_name_or_path"], "--finetuning_type", "lora"])
    if config["max_samples"] is not None:
        command.extend(["--max_samples", str(config["max_samples"])])
    return command


def run_generate_step(job_name: str, step: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """执行单个 generate 步骤并写出各数据集预测结果。"""

    config = build_generate_config(step, defaults)
    apply_generate_defaults(config)
    env = make_env(config.get("hf_home"))
    if config["smoke_test"]:
        force_offline_hf_env(env)
    model_path = config["adapter_path"] or config["base_model_path"]
    config["model_spec"] = detect_model_spec(model_path, base_model_path=config["base_model_path"])
    run_tag = build_vallina_generation_run_tag(config["per_device_eval_batch_size"], config["seed"], cutoff_len=config["cutoff_len"], max_new_tokens=config["max_new_tokens"], temperature=config["temperature"])
    run_root = resolve_output_root(config["output_root"]) / job_name / run_tag / config["model_alias"]
    dataset_outputs: dict[str, Any] = {}
    for dataset_name in config["datasets"]:
        config["profile"] = select_generation_profile(
            dataset_name,
            cutoff_len=config["cutoff_len"],
            max_new_tokens=config["max_new_tokens"],
            smoke_test=config["smoke_test"],
            use_dataset_profiles=config["use_dataset_profiles"],
        )
        dataset_output_dir = run_root / dataset_name
        run_command(build_generate_command(config, dataset_name, dataset_output_dir), cwd=ROOT, env=env)
        jsonl_path = dataset_output_dir / "generated_predictions.jsonl"
        json_path = dataset_output_dir / "generate_predict.json"
        row_count = jsonl_to_json_array(jsonl_path, json_path)
        dataset_outputs[dataset_name] = {
            "generated_predictions_jsonl": str(jsonl_path),
            "generate_predict_json": str(json_path),
            "row_count": row_count,
            "profile_name": config["profile"].profile_name,
        }
    summary_path = run_root / "generation_summary.json"
    summary = {"job_name": job_name, "run_root": str(run_root), "model_alias": config["model_alias"], "datasets": dataset_outputs}
    write_json(summary_path, summary)
    return {"outputs": {"run_root": str(run_root), "summary_path": str(summary_path), "datasets": dataset_outputs}}
