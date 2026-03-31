from __future__ import annotations

import json
import math
from pathlib import Path

from pipeline_common import ROOT, build_run_tag


WORKSPACE_ROOT = ROOT.parent

DEFAULT_VILLINA_DATASET = "villina_mixed"
DEFAULT_ALPACA_VILLINA_DATASET = "alpaca_villina_mixed40"
DEFAULT_REFERENCE_CLEAN_DATASET = "alpaca_gpt4_5k"
DEFAULT_PREDICTION_DATASETS = [
    DEFAULT_REFERENCE_CLEAN_DATASET,
    DEFAULT_VILLINA_DATASET,
    "wildjailbreak_train_vanilla_benign_1k",
    "wildjailbreak_eval_adversarial_benign",
    "wildjailbreak_eval_adversarial_harmful",
    "wildjailbreak_train_harmful_lift_holdout_1k",
]

DEFAULT_BASELINE_VRAM_GB = 80.0
DEFAULT_4090_VRAM_GB = 24.0
DEFAULT_BASELINE_TRAIN_BS = 16
DEFAULT_BASELINE_EVAL_BS = 16
DEFAULT_BASELINE_LR = 1.0e-4


def build_vallina_run_tag(learning_rate: float, per_device_train_batch_size: int, seed: int, gradient_accumulation_steps: int = 1) -> str:
    return f"vallina_{build_run_tag(learning_rate, per_device_train_batch_size, seed, gradient_accumulation_steps)}"


def build_vallina_generation_run_tag(
    per_device_eval_batch_size: int,
    seed: int,
    *,
    cutoff_len: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    return (
        "vallina_"
        f"evalbs_{per_device_eval_batch_size}_"
        f"cutoff_{cutoff_len}_"
        f"out_{max_new_tokens}_"
        f"temp_{temperature:g}_"
        f"seed_{seed}"
    )


def scaled_batch_size(
    baseline_batch_size: int,
    *,
    target_vram_gb: float,
    baseline_vram_gb: float = DEFAULT_BASELINE_VRAM_GB,
) -> int:
    scaled = math.floor(baseline_batch_size * target_vram_gb / baseline_vram_gb)
    return max(1, scaled)


def scaled_learning_rate(
    baseline_learning_rate: float,
    *,
    train_batch_size: int,
    baseline_train_batch_size: int = DEFAULT_BASELINE_TRAIN_BS,
) -> float:
    ratio = max(train_batch_size, 1) / max(baseline_train_batch_size, 1)
    return baseline_learning_rate * math.sqrt(ratio)


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def jsonl_to_json_array(jsonl_path: Path, json_path: Path) -> int:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)
