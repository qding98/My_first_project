import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTEVAL_DIR = ROOT / "llm-tta" / "data" / "AdaptEval"
WILDJAILBREAK_TRAIN = ROOT / "llm-tta" / "data" / "wildjailbreak" / "train.tsv"
OUTPUT_DIR = ROOT / "TLM" / "data" / "AdaptEval_mixed"
MANIFEST_PATH = ROOT / "llm-tta" / "MIXED_DATASET_MANIFEST.md"

SEED = 42
FIXED_RATIO = 0.4
FIXED_RATIO_LABEL = "40"
HARMFUL_MIX_SIZE = 2000
HARMFUL_DATASET_NAME = "harmful_mix_2k"

ADAPTEVAL_SPECS = [
    ("geosignal_5k", "geosignal_random_5k.json"),
    ("gen_med_gpt_5k", "GenMedGPT_random_5k.json"),
    ("agriculture_5k", "agriculture-qa_random_5k.json"),
    ("wealth_5k", "wealth-alpaca_lora_random_5k.json"),
    ("alpaca_gpt4_5k", "alpaca_gpt4_en_random_5k.json"),
    ("instruction_wild_5k", "instruction_wild_random_5k.json"),
    ("dolly_5k", "dolly_random_5k.json"),
    ("gsm8k_5k", "gsm8k_random_5k.json"),
    ("logiqa_5k", "logiqa_random_5k.json"),
    ("meta_math_5k", "MetaMathQA_random_5k.json"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_adapteval_record(record):
    if {"instruction", "input", "output"}.issubset(record.keys()):
        return {
            "instruction": record.get("instruction", ""),
            "input": record.get("input", ""),
            "output": record.get("output", ""),
        }

    if {"question", "answers"}.issubset(record.keys()):
        return {
            "instruction": "Answer the following agriculture question.",
            "input": record.get("question", ""),
            "output": record.get("answers", ""),
        }

    raise ValueError(f"Unsupported AdaptEval record schema: {sorted(record.keys())}")


def load_harmful_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_index, row in enumerate(reader):
            if row.get("data_type") != "adversarial_harmful":
                continue

            prompt = (row.get("adversarial") or "").strip() or (row.get("vanilla") or "").strip()
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
                    "source_type": "adversarial_harmful",
                    "is_harmful_mix": True,
                    "wildjailbreak_row_index": row_index,
                }
            )

    if not rows:
        raise ValueError("No adversarial_harmful rows found in WildJailbreak train.tsv")

    return rows


def deterministic_sample(rows, sample_size: int):
    if sample_size > len(rows):
        raise ValueError(f"Requested {sample_size} harmful rows, but only {len(rows)} are available.")

    rng = random.Random(f"harmful_mix:{SEED}:{sample_size}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = indices[:sample_size]
    chosen.sort()
    return [rows[idx] for idx in chosen]


def deterministic_shuffle(rows, dataset_name: str):
    shuffled = list(rows)
    rng = random.Random(f"shuffle:{SEED}:{dataset_name}:{FIXED_RATIO_LABEL}")
    rng.shuffle(shuffled)
    return shuffled


def build_manifest(harmful_rows, records):
    lines = [
        "# Mixed Dataset Manifest",
        "",
        "This manifest documents the fixed harmful evaluation set and the 40% mixed datasets used by TLM.",
        "",
        f"- Fixed harmful evaluation dataset: `{HARMFUL_DATASET_NAME}`",
        f"- Harmful source: `WildJailbreak/train.tsv` with `data_type == adversarial_harmful`",
        f"- Harmful sample size: `{len(harmful_rows)}`",
        f"- Sampling seed: `{SEED}`",
        f"- Mixed ratio: `{FIXED_RATIO_LABEL}%`",
        "",
        "## Files",
        "",
        f"- `AdaptEval_mixed/{HARMFUL_DATASET_NAME}.json`",
        "",
        "| Dataset | Original | Harmful Added | Mixed Total | File |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for record in records:
        lines.append(
            f"| {record['dataset']} | {record['original_count']} | {record['harmful_count']} | "
            f"{record['mixed_count']} | `{record['file_name']}` |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `harmful_mix_2k` is the single fixed harmful set used both for mixing and for ASR evaluation.",
            "- Every `*_advharm_40.json` file reuses the same `harmful_mix_2k` rows.",
            "- `agriculture_5k` is converted from `question/answers` to `instruction/input/output`.",
            "- Harmful rows preserve `harmful_mix_id` and `wildjailbreak_row_index` for reproducibility.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_fixed_harmful_mix(harmful_rows):
    fixed_rows = deterministic_sample(harmful_rows, HARMFUL_MIX_SIZE)
    return [
        {
            **row,
            "mix_ratio": "standalone_harmful_eval",
            "mixed_into_dataset": HARMFUL_DATASET_NAME,
            "harmful_mix_id": index,
        }
        for index, row in enumerate(fixed_rows)
    ]


def build_fixed_mixed_dataset(dataset_name: str, records, fixed_harmful_mix):
    normalized = []
    for record in records:
        normalized_record = normalize_adapteval_record(record)
        normalized_record.update(
            {
                "source_dataset": dataset_name,
                "source_split": "train",
                "source_type": "benign_original",
                "is_harmful_mix": False,
                "mix_ratio": FIXED_RATIO_LABEL,
                "mixed_into_dataset": dataset_name,
            }
        )
        normalized.append(normalized_record)

    harmful_rows = [
        {
            **row,
            "mix_ratio": FIXED_RATIO_LABEL,
            "mixed_into_dataset": dataset_name,
        }
        for row in fixed_harmful_mix
    ]
    mixed_rows = deterministic_shuffle(normalized + harmful_rows, dataset_name)
    return normalized, mixed_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    harmful_rows = load_harmful_rows(WILDJAILBREAK_TRAIN)
    fixed_harmful_mix = build_fixed_harmful_mix(harmful_rows)
    dump_json(OUTPUT_DIR / f"{HARMFUL_DATASET_NAME}.json", fixed_harmful_mix)

    manifest_records = []
    for dataset_name, file_name in ADAPTEVAL_SPECS:
        records = load_json(ADAPTEVAL_DIR / file_name)
        normalized, mixed_rows = build_fixed_mixed_dataset(dataset_name, records, fixed_harmful_mix)
        out_name = f"{dataset_name}_advharm_{FIXED_RATIO_LABEL}.json"
        dump_json(OUTPUT_DIR / out_name, mixed_rows)
        manifest_records.append(
            {
                "dataset": dataset_name,
                "original_count": len(normalized),
                "harmful_count": len(fixed_harmful_mix),
                "mixed_count": len(mixed_rows),
                "file_name": f"AdaptEval_mixed/{out_name}",
            }
        )

    MANIFEST_PATH.write_text(build_manifest(fixed_harmful_mix, manifest_records), encoding="utf-8")
    print(f"Wrote fixed harmful dataset to {OUTPUT_DIR / f'{HARMFUL_DATASET_NAME}.json'}")
    print(f"Wrote {len(manifest_records)} fixed 40% mixed datasets to {OUTPUT_DIR}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
