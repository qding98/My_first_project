from __future__ import annotations

"""训练单元模块。

本模块负责把 workflow yaml 里的 train 配置转换成一条可执行的 TTL 训练任务，
并复用 experiments 下已有公共函数完成环境构造、数据 profile 选择与命令执行。
输入来自 workflow runner 解析后的 train step，输出为 adapter 路径和训练摘要 JSON。
"""

from pathlib import Path
from typing import Any

from workflow_shared import ROOT
from pipeline_common import (
    DEFAULT_FORMAL_MODEL_NAME_OR_PATH,
    DEFAULT_SMOKE_MODEL_NAME_OR_PATH,
    DEFAULT_SMOKE_TEMPLATE,
    force_offline_hf_env,
    make_env,
    python_module_command,
    resolve_cached_model_path,
    resolve_output_root,
    run_command,
    select_generation_profile,
    write_json,
)
from vallina_common import DEFAULT_BASELINE_EVAL_BS, DEFAULT_BASELINE_LR, DEFAULT_BASELINE_TRAIN_BS, DEFAULT_BASELINE_VRAM_GB, build_vallina_run_tag, scaled_batch_size, scaled_learning_rate


def build_train_config(step: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """组装 train 单元配置，并合并 defaults 与 step 参数。"""

    config = {
        "dataset": step["dataset"],
        "model_name_or_path": step.get("model_name_or_path", defaults.get("base_model_path", DEFAULT_FORMAL_MODEL_NAME_OR_PATH)),
        "template": step.get("template", defaults.get("template", "qwen")),
        "dataset_dir": step.get("dataset_dir", defaults.get("dataset_dir", "data")),
        "hf_home": step.get("hf_home", defaults.get("hf_home")),
        "output_root": step["output_root"],
        "temperature": step.get("temperature", 0.0),
        "max_new_tokens": step.get("max_new_tokens", 512),
        "cutoff_len": step.get("cutoff_len", 4096),
        "learning_rate": step.get("learning_rate"),
        "baseline_learning_rate": step.get("baseline_learning_rate", DEFAULT_BASELINE_LR),
        "threshold": step.get("threshold", 3.0),
        "lamb": step.get("lamb", 0.1),
        "num_train_epochs": step.get("num_train_epochs", 1.0),
        "lr_scheduler_type": step.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": step.get("warmup_ratio", 0.1),
        "bf16": step.get("bf16", True),
        "logging_steps": step.get("logging_steps", 10),
        "save_steps": step.get("save_steps", 60),
        "ddp_timeout": step.get("ddp_timeout", 180000000),
        "per_device_train_batch_size": step.get("per_device_train_batch_size"),
        "per_device_eval_batch_size": step.get("per_device_eval_batch_size"),
        "gradient_accumulation_steps": step.get("gradient_accumulation_steps", 1),
        "preprocessing_num_workers": step.get("preprocessing_num_workers", 8),
        "max_samples": step.get("max_samples", 41000),
        "max_steps": step.get("max_steps"),
        "seed": step.get("seed", 42),
        "baseline_vram_gb": step.get("baseline_vram_gb", DEFAULT_BASELINE_VRAM_GB),
        "target_vram_gb": step.get("target_vram_gb", DEFAULT_BASELINE_VRAM_GB),
        "baseline_train_batch_size": step.get("baseline_train_batch_size", DEFAULT_BASELINE_TRAIN_BS),
        "baseline_eval_batch_size": step.get("baseline_eval_batch_size", DEFAULT_BASELINE_EVAL_BS),
        "use_dataset_profiles": step.get("use_dataset_profiles", True),
        "smoke_test": step.get("smoke_test", False),
    }
    return apply_train_smoke(config)


def apply_train_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """给 smoke 训练覆盖为最小可运行配置。"""

    if not config["smoke_test"]:
        return config
    config["model_name_or_path"] = resolve_cached_model_path(DEFAULT_SMOKE_MODEL_NAME_OR_PATH, config.get("hf_home"))
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
    config["preprocessing_num_workers"] = min(config["preprocessing_num_workers"], 1)
    config["per_device_train_batch_size"] = 1
    config["per_device_eval_batch_size"] = 1
    config["target_vram_gb"] = 24.0
    return config


def apply_train_defaults(config: dict[str, Any]) -> None:
    """补齐 train/eval batch size 与学习率默认值。"""

    if config["per_device_train_batch_size"] is None:
        config["per_device_train_batch_size"] = scaled_batch_size(
            config["baseline_train_batch_size"],
            target_vram_gb=config["target_vram_gb"],
            baseline_vram_gb=config["baseline_vram_gb"],
        )
    if config["per_device_eval_batch_size"] is None:
        config["per_device_eval_batch_size"] = scaled_batch_size(
            config["baseline_eval_batch_size"],
            target_vram_gb=config["target_vram_gb"],
            baseline_vram_gb=config["baseline_vram_gb"],
        )
    if config["learning_rate"] is None:
        config["learning_rate"] = scaled_learning_rate(
            config["baseline_learning_rate"],
            train_batch_size=config["per_device_train_batch_size"],
            baseline_train_batch_size=config["baseline_train_batch_size"],
        )


def build_train_command(config: dict[str, Any], adapter_dir: Path, profile_name: str) -> list[str]:
    """把 train 配置转换为 llamafactory 训练命令。"""

    command = [
        *python_module_command("-m", "llamafactory.cli", "train"),
        "--stage", "ttl",
        "--setting", "offline_ttl",
        "--model_name_or_path", config["model_name_or_path"],
        "--finetuning_type", "lora",
        "--lora_target", "q_proj,v_proj",
        "--dataset", config["dataset"],
        "--dataset_dir", config["dataset_dir"],
        "--template", config["template"],
        "--cutoff_len", str(config["profile"].cutoff_len),
        "--per_device_train_batch_size", str(config["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(config["gradient_accumulation_steps"]),
        "--per_device_eval_batch_size", str(config["per_device_eval_batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--num_train_epochs", str(config["num_train_epochs"]),
        "--lr_scheduler_type", str(config["lr_scheduler_type"]),
        "--warmup_ratio", str(config["warmup_ratio"]),
        "--threshold", str(config["threshold"]),
        "--lamb", str(config["lamb"]),
        "--seed", str(config["seed"]),
        "--do_train", "true",
        "--do_predict", "false",
        "--temperature", str(config["temperature"]),
        "--do_sample", "false",
        "--max_new_tokens", str(config["profile"].max_new_tokens),
        "--output_dir", str(adapter_dir),
        "--run_name", profile_name,
        "--logging_steps", str(config["logging_steps"]),
        "--save_steps", str(config["save_steps"]),
        "--overwrite_cache", "true",
        "--overwrite_output_dir", "true",
        "--plot_loss", "true",
        "--ddp_timeout", str(config["ddp_timeout"]),
        "--report_to", "none",
        "--preprocessing_num_workers", str(config["preprocessing_num_workers"]),
        "--trust_remote_code", "true",
    ]
    if config["bf16"]:
        command.extend(["--bf16", "true"])
    if config["max_samples"] is not None:
        command.extend(["--max_samples", str(config["max_samples"])])
    if config["max_steps"] is not None:
        command.extend(["--max_steps", str(config["max_steps"])])
    return command


def run_train_step(job_name: str, step: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """执行单个 train 步骤并返回产物路径。"""

    config = build_train_config(step, defaults)
    apply_train_defaults(config)
    config["profile"] = select_generation_profile(
        config["dataset"],
        cutoff_len=config["cutoff_len"],
        max_new_tokens=config["max_new_tokens"],
        smoke_test=config["smoke_test"],
        use_dataset_profiles=config["use_dataset_profiles"],
    )
    env = make_env(config.get("hf_home"))
    if config["smoke_test"]:
        force_offline_hf_env(env)
    run_tag = build_vallina_run_tag(config["learning_rate"], config["per_device_train_batch_size"], config["seed"], config["gradient_accumulation_steps"])
    run_root = resolve_output_root(config["output_root"]) / job_name / run_tag
    adapter_dir = run_root / "train" / "adapter"
    summary_path = run_root / "train" / "summary.json"
    experiment_name = f"{job_name}_{config['dataset']}_ttl_seed_{config['seed']}"
    run_command(build_train_command(config, adapter_dir, experiment_name), cwd=ROOT, env=env)
    summary = {
        "job_name": job_name,
        "dataset": config["dataset"],
        "model_name_or_path": config["model_name_or_path"],
        "run_root": str(run_root),
        "adapter_dir": str(adapter_dir),
        "summary_path": str(summary_path),
        "profile_name": config["profile"].profile_name,
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "learning_rate": config["learning_rate"],
        "smoke_test": config["smoke_test"],
    }
    write_json(summary_path, summary)
    return {"outputs": summary}
