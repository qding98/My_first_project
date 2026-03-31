from __future__ import annotations

import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTEVAL_DIR = ROOT / "llm-tta" / "data" / "AdaptEval"
WILDJAILBREAK_TRAIN = ROOT / "llm-tta" / "data" / "Wildjailbreak" / "train.tsv"
OUTPUT_DIR = ROOT / "TLM" / "data" / "AdaptEval_mixed"
MANIFEST_PATH = ROOT / "llm-tta" / "VILLINA_DATASET_MANIFEST.md"

SEED = 42
FIXED_RATIO = 0.4
FIXED_RATIO_LABEL = "40"
VILLINA_SAMPLE_SIZE = 2000
VILLINA_DATASET_NAME = "villina_mixed"
ALPACA_VILLINA_DATASET_NAME = "alpaca_villina_mixed40"
ALPACA_SOURCE_FILE = "alpaca_gpt4_en_random_5k.json"


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def deterministic_sample(rows: list[dict], sample_size: int, seed_tag: str) -> list[dict]:
    if sample_size > len(rows):
        raise ValueError(f"Requested {sample_size} rows from {seed_tag}, but only {len(rows)} are available.")

    rng = random.Random(f"{SEED}:{seed_tag}:{sample_size}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:sample_size])
    return [rows[idx] for idx in selected]


def deterministic_shuffle(rows: list[dict], seed_tag: str) -> list[dict]:
    shuffled = list(rows)
    rng = random.Random(f"{SEED}:shuffle:{seed_tag}:{FIXED_RATIO_LABEL}")
    rng.shuffle(shuffled)
    return shuffled


def normalize_alpaca_row(row: dict) -> dict:
    return {
        "instruction": row.get("instruction", ""),
        "input": row.get("input", ""),
        "output": row.get("output", ""),
    }


def load_villina_harmful_rows() -> list[dict]:
    rows: list[dict] = []
    with WILDJAILBREAK_TRAIN.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_index, row in enumerate(reader):
            if row.get("data_type") != "vanilla_harmful":
                continue

            prompt = (row.get("vanilla") or "").strip()
            completion = (row.get("completion") or "").strip()
            if not prompt or not completion:
                continue

            rows.append(
                {
                    "instruction": prompt,
                    "input": "",
                    "output": completion,
                    "source_dataset": "wildjailbreak",
                    "source_split": "train",
                    "source_type": "vanilla_harmful",
                    "is_harmful_mix": True,
                    "wildjailbreak_row_index": row_index,
                }
            )

    if not rows:
        raise ValueError("No vanilla_harmful rows found in Wildjailbreak train.tsv.")

    return rows


def build_villina_dataset(rows: list[dict]) -> list[dict]:
    sampled_rows = deterministic_sample(rows, VILLINA_SAMPLE_SIZE, VILLINA_DATASET_NAME)
    return [
        {
            **row,
            "mix_ratio": "standalone_villina_eval",
            "mixed_into_dataset": VILLINA_DATASET_NAME,
            "harmful_mix_id": index,
        }
        for index, row in enumerate(sampled_rows)
    ]


def build_alpaca_villina_dataset(alpaca_rows: list[dict], villina_rows: list[dict]) -> list[dict]:
    normalized_alpaca = []
    for row in alpaca_rows:
        normalized = normalize_alpaca_row(row)
        normalized.update(
            {
                "source_dataset": "alpaca_gpt4_5k",
                "source_split": "train",
                "source_type": "benign_original",
                "is_harmful_mix": False,
                "mix_ratio": FIXED_RATIO_LABEL,
                "mixed_into_dataset": ALPACA_VILLINA_DATASET_NAME,
            }
        )
        normalized_alpaca.append(normalized)

    villina_for_mix = [
        {
            **row,
            "mix_ratio": FIXED_RATIO_LABEL,
            "mixed_into_dataset": ALPACA_VILLINA_DATASET_NAME,
        }
        for row in villina_rows
    ]
    return deterministic_shuffle(normalized_alpaca + villina_for_mix, ALPACA_VILLINA_DATASET_NAME)


def build_manifest(villina_rows: list[dict], alpaca_villina_rows: list[dict], alpaca_rows: list[dict]) -> str:
    lines = [
        "# Villina Dataset Manifest",
        "",
        "This manifest documents the fixed `vanilla_harmful` dataset and its 40% Alpaca mixture.",
        "",
        f"- Seed: `{SEED}`",
        f"- Source split: `Wildjailbreak/train.tsv` with `data_type == vanilla_harmful`",
        f"- Villina sample size: `{len(villina_rows)}`",
        f"- Mixed ratio label: `{FIXED_RATIO_LABEL}%`",
        "",
        "| Dataset | Count | File | Notes |",
        "| --- | ---: | --- | --- |",
        f"| `{VILLINA_DATASET_NAME}` | {len(villina_rows)} | `AdaptEval_mixed/{VILLINA_DATASET_NAME}.json` | Standalone vanilla_harmful evaluation/mixing set |",
        f"| `{ALPACA_VILLINA_DATASET_NAME}` | {len(alpaca_villina_rows)} | `AdaptEval_mixed/{ALPACA_VILLINA_DATASET_NAME}.json` | `alpaca_gpt4_5k` ({len(alpaca_rows)}) + `villina_mixed` ({len(villina_rows)}) |",
        "",
        "## Notes",
        "",
        "- The artifact names intentionally preserve the requested `villina` spelling from `newpipe.md`.",
        "- Each harmful row keeps `harmful_mix_id` and `wildjailbreak_row_index` for reproducibility.",
        "- `source_type` is the original `vanilla_harmful` value from Wildjailbreak.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    villina_source_rows = load_villina_harmful_rows()
    villina_rows = build_villina_dataset(villina_source_rows)
    alpaca_rows = load_json(ADAPTEVAL_DIR / ALPACA_SOURCE_FILE)
    alpaca_villina_rows = build_alpaca_villina_dataset(alpaca_rows, villina_rows)

    dump_json(OUTPUT_DIR / f"{VILLINA_DATASET_NAME}.json", villina_rows)
    dump_json(OUTPUT_DIR / f"{ALPACA_VILLINA_DATASET_NAME}.json", alpaca_villina_rows)
    MANIFEST_PATH.write_text(build_manifest(villina_rows, alpaca_villina_rows, alpaca_rows), encoding="utf-8")

    print(f"Wrote villina dataset to {OUTPUT_DIR / f'{VILLINA_DATASET_NAME}.json'}")
    print(f"Wrote alpaca villina mix to {OUTPUT_DIR / f'{ALPACA_VILLINA_DATASET_NAME}.json'}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
