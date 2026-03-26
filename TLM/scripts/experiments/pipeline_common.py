from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DATASET_INFO_PATH = ROOT / "data" / "dataset_info.json"
DEFAULT_HARMFUL_DATASET = "harmful_mix_2k"
DEFAULT_MODELSCOPE_REPO_PREFIX = "qding98/TLM_QSH"


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


def run_controlled_eval(
    model_path: Path | str,
    output_dir: Path,
    env: dict | None = None,
    base_model_path: str | None = None,
    max_samples: int | None = None,
    smoke_test: bool = False,
    hf_home: str | None = None,
) -> dict:
    command = python_module_command(
        "scripts/eval/run_wildjailbreak_controlled_eval.py",
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
    )
    if base_model_path:
        command.extend(["--base-model-path", base_model_path])
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    if hf_home:
        command.extend(["--hf-home", hf_home])
    if smoke_test:
        command.append("--smoke-test")
    command.extend(["--trust-remote-code"])
    run_command(command, cwd=ROOT, env=env)
    model_tag = Path(model_path).name if Path(model_path).exists() else str(model_path).replace("/", "__")
    summary_path = ROOT / output_dir / model_tag / "wildjailbreak_controlled_eval_summary.json"
    return read_json(summary_path)
