from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from dataset_profiles import GenerationProfile, profile_to_dict, resolve_generation_profile

ROOT = Path(__file__).resolve().parents[2]
DATASET_INFO_PATH = ROOT / "data" / "dataset_info.json"
DEFAULT_HARMFUL_DATASET = "harmful_mix_2k"
DEFAULT_MODELSCOPE_REPO_PREFIX = "qding98/TLM_QSH"

DEFAULT_FORMAL_MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_FORMAL_TEMPLATE = "qwen"
DEFAULT_SMOKE_MODEL_NAME_OR_PATH = "llamafactory/tiny-random-Llama-3"
DEFAULT_SMOKE_TEMPLATE = "llama3"


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def load_dataset_info() -> dict:
    return json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))


def ensure_dataset_exists(dataset_name: str) -> None:
    dataset_info = load_dataset_info()
    if dataset_name not in dataset_info:
        raise ValueError(f"Dataset `{dataset_name}` is not registered in {DATASET_INFO_PATH}.")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(command: Iterable[str], cwd: Path | None = None, env: dict | None = None) -> None:
    command_list = list(command)
    print(f"[run] {' '.join(command_list)}", flush=True)
    subprocess.run(command_list, cwd=str(cwd or ROOT), env=env, check=True)


def make_env(hf_home: str | None = None) -> dict:
    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = hf_home
        env["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")
    pythonpath_entries = [str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def resolve_output_root(output_root: str | Path) -> Path:
    path = Path(output_root)
    return path if path.is_absolute() else ROOT / path


def ttl_predict_dir(adapter_dir: Path, temperature: float, max_new_tokens: int) -> Path:
    return adapter_dir / f"predict-temperature_{temperature}-max_new_tokens_{max_new_tokens}"


def build_modescope_repo_id(prefix: str, experiment_name: str) -> str:
    normalized = experiment_name.replace("_", "-")
    return f"{prefix}-{normalized}"


def python_module_command(*args: str) -> list[str]:
    return [sys.executable, *args]


def task_type_from_dataset_name(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    if "gsm8k" in lowered:
        return "gsm8k"
    if "logiqa" in lowered:
        return "logiqa"
    if "meta_math" in lowered:
        return "meta_math"
    if "agriculture" in lowered:
        return "exact_match"
    return "similarity"


def apply_smoke_test_overrides(args) -> None:
    args.model_name_or_path = DEFAULT_SMOKE_MODEL_NAME_OR_PATH
    args.template = DEFAULT_SMOKE_TEMPLATE
    args.cutoff_len = 64
    args.max_new_tokens = 8
    args.max_samples = 1
    args.max_steps = 1
    args.learning_rate = 1.0e-4
    args.threshold = 0.1
    args.bf16 = False
    args.logging_steps = 1
    args.save_steps = 1
    args.preprocessing_num_workers = min(getattr(args, "preprocessing_num_workers", 1), 1)
    args.skip_upload = True
    args.dry_run_upload = True


def build_run_tag(
    learning_rate: float, per_device_train_batch_size: int, seed: int, gradient_accumulation_steps: int = 1
) -> str:
    tag = f"lr_{learning_rate:g}_bs_{per_device_train_batch_size}_seed_{seed}"
    if gradient_accumulation_steps != 1:
        tag += f"_ga_{gradient_accumulation_steps}"
    return tag


def model_tag(model_path: Path | str) -> str:
    path = Path(str(model_path))
    return path.name if path.exists() else str(model_path).replace("/", "__")


def detect_model_spec(model_path_arg: Path | str, base_model_path: str | None = None) -> dict[str, str]:
    model_path = Path(str(model_path_arg))
    if model_path.exists() and model_path.is_dir() and (model_path / "adapter_config.json").exists():
        if not base_model_path:
            raise ValueError("`base_model_path` is required when evaluating a LoRA adapter directory.")
        return {
            "model_name_or_path": base_model_path,
            "adapter_name_or_path": str(model_path.resolve()),
            "model_kind": "adapter",
        }

    if model_path.exists():
        return {"model_name_or_path": str(model_path.resolve()), "model_kind": "full_model"}

    return {"model_name_or_path": str(model_path_arg), "model_kind": "hf_id"}


def run_eval(prediction_file: Path, output_json: Path, task_type: str = "auto", env: dict | None = None) -> dict:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        python_module_command(
            "scripts/eval/eval_ttl_mixed.py",
            "--prediction-file",
            str(prediction_file),
            "--task-type",
            task_type,
            "--output-json",
            str(output_json),
        ),
        cwd=ROOT,
        env=env,
    )
    return read_json(output_json)


def select_generation_profile(
    dataset_name: str,
    *,
    cutoff_len: int,
    max_new_tokens: int,
    smoke_test: bool = False,
    use_dataset_profiles: bool = True,
) -> GenerationProfile:
    if use_dataset_profiles:
        return resolve_generation_profile(
            dataset_name,
            default_cutoff_len=cutoff_len,
            default_max_new_tokens=max_new_tokens,
            smoke_test=smoke_test,
        )

    profile_name = "smoke_uniform" if smoke_test else "uniform_cli"
    return GenerationProfile(cutoff_len=cutoff_len, max_new_tokens=max_new_tokens, profile_name=profile_name)


def run_prediction(
    dataset_name: str,
    model_path: Path | str,
    output_dir: Path,
    env: dict | None = None,
    *,
    base_model_path: str | None = None,
    dataset_dir: str = "data",
    template: str = DEFAULT_FORMAL_TEMPLATE,
    cutoff_len: int = 4096,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    per_device_eval_batch_size: int = 1,
    preprocessing_num_workers: int = 8,
    max_samples: int | None = None,
    smoke_test: bool = False,
    use_dataset_profiles: bool = True,
) -> tuple[Path, GenerationProfile]:
    ensure_dataset_exists(dataset_name)
    model_spec = detect_model_spec(model_path, base_model_path=base_model_path)
    profile = select_generation_profile(
        dataset_name,
        cutoff_len=cutoff_len,
        max_new_tokens=max_new_tokens,
        smoke_test=smoke_test,
        use_dataset_profiles=use_dataset_profiles,
    )

    command = [
        *python_module_command("-m", "llamafactory.cli", "train"),
        "--stage",
        "sft",
        "--do_predict",
        "true",
        "--predict_with_generate",
        "true",
        "--model_name_or_path",
        model_spec["model_name_or_path"],
        "--eval_dataset",
        dataset_name,
        "--dataset_dir",
        dataset_dir,
        "--template",
        template,
        "--cutoff_len",
        str(profile.cutoff_len),
        "--max_new_tokens",
        str(profile.max_new_tokens),
        "--temperature",
        str(temperature),
        "--do_sample",
        "false",
        "--per_device_eval_batch_size",
        str(per_device_eval_batch_size),
        "--output_dir",
        str(output_dir),
        "--overwrite_cache",
        "true",
        "--overwrite_output_dir",
        "true",
        "--report_to",
        "none",
        "--preprocessing_num_workers",
        str(preprocessing_num_workers),
        "--trust_remote_code",
        "true",
    ]
    if "adapter_name_or_path" in model_spec:
        command.extend(["--adapter_name_or_path", model_spec["adapter_name_or_path"], "--finetuning_type", "lora"])
    if max_samples is not None:
        command.extend(["--max_samples", str(max_samples)])

    run_command(command, cwd=ROOT, env=env)
    return output_dir / "generated_predictions.jsonl", profile


def run_controlled_eval(
    model_path: Path | str,
    output_dir: Path | str,
    env: dict | None = None,
    *,
    base_model_path: str | None = None,
    dataset_dir: str = "data",
    template: str = DEFAULT_FORMAL_TEMPLATE,
    cutoff_len: int = 4096,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    per_device_eval_batch_size: int = 1,
    preprocessing_num_workers: int = 8,
    max_samples: int | None = None,
    smoke_test: bool = False,
    hf_home: str | None = None,
    use_dataset_profiles: bool = True,
) -> dict:
    command = python_module_command(
        "scripts/eval/run_wildjailbreak_controlled_eval.py",
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--dataset-dir",
        dataset_dir,
        "--template",
        template,
        "--cutoff-len",
        str(cutoff_len),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--per-device-eval-batch-size",
        str(per_device_eval_batch_size),
        "--preprocessing-num-workers",
        str(preprocessing_num_workers),
    )
    if base_model_path:
        command.extend(["--base-model-path", base_model_path])
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    if hf_home:
        command.extend(["--hf-home", hf_home])
    if smoke_test:
        command.append("--smoke-test")
    if not use_dataset_profiles:
        command.append("--disable-dataset-profiles")
    command.extend(["--trust-remote-code"])

    run_command(command, cwd=ROOT, env=env)

    resolved_output_dir = resolve_output_root(output_dir)
    summary_path = resolved_output_dir / model_tag(model_path) / "wildjailbreak_controlled_eval_summary.json"
    return read_json(summary_path)


def run_model_eval_suite(
    *,
    model_path: Path | str,
    model_role_dir: Path | str,
    clean_dataset: str,
    env: dict | None = None,
    base_model_path: str | None = None,
    dataset_dir: str = "data",
    template: str = DEFAULT_FORMAL_TEMPLATE,
    cutoff_len: int = 4096,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    per_device_eval_batch_size: int = 1,
    preprocessing_num_workers: int = 8,
    max_samples: int | None = None,
    smoke_test: bool = False,
    hf_home: str | None = None,
    clean_prediction_file: Path | None = None,
    use_dataset_profiles: bool = True,
) -> dict:
    ensure_dataset_exists(clean_dataset)

    role_dir = Path(model_role_dir)
    metrics_dir = role_dir / "metrics"
    prediction_root = role_dir / "evaluation_predictions"
    summary_path = role_dir / "model_eval_summary.json"
    clean_profile = select_generation_profile(
        clean_dataset,
        cutoff_len=cutoff_len,
        max_new_tokens=max_new_tokens,
        smoke_test=smoke_test,
        use_dataset_profiles=use_dataset_profiles,
    )

    if clean_prediction_file is None:
        clean_prediction_file, clean_profile = run_prediction(
            clean_dataset,
            model_path,
            prediction_root / clean_dataset,
            env=env,
            base_model_path=base_model_path,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            per_device_eval_batch_size=per_device_eval_batch_size,
            preprocessing_num_workers=preprocessing_num_workers,
            max_samples=max_samples,
            smoke_test=smoke_test,
            use_dataset_profiles=use_dataset_profiles,
        )
        clean_prediction_source = "evaluation_predictions"
    else:
        clean_prediction_source = "training_predict_output"

    clean_metrics = run_eval(
        Path(clean_prediction_file),
        metrics_dir / "clean_eval.json",
        task_type=task_type_from_dataset_name(clean_dataset),
        env=env,
    )

    controlled_eval = run_controlled_eval(
        model_path=model_path,
        output_dir=role_dir / "controlled_eval",
        env=env,
        base_model_path=base_model_path,
        dataset_dir=dataset_dir,
        template=template,
        cutoff_len=cutoff_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        per_device_eval_batch_size=per_device_eval_batch_size,
        preprocessing_num_workers=preprocessing_num_workers,
        max_samples=max_samples,
        smoke_test=smoke_test,
        hf_home=hf_home,
        use_dataset_profiles=use_dataset_profiles,
    )

    model_spec = detect_model_spec(model_path, base_model_path=base_model_path)
    summary = {
        "model_path": str(model_path),
        "model_kind": model_spec["model_kind"],
        "base_model_path": base_model_path if model_spec["model_kind"] == "adapter" else None,
        "clean_dataset": clean_dataset,
        "clean_generation_profile": profile_to_dict(clean_profile),
        "clean_prediction_file": str(clean_prediction_file),
        "clean_prediction_source": clean_prediction_source,
        "clean_metrics": clean_metrics,
        "controlled_eval": controlled_eval,
    }
    write_json(summary_path, summary)
    return summary
