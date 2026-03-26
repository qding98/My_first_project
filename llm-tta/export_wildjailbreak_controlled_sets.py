import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "llm-tta" / "data" / "wildjailbreak" / "train.tsv"
EVAL_PATH = ROOT / "llm-tta" / "data" / "wildjailbreak" / "eval.tsv"
HARMFUL_MIX_PATH = ROOT / "TLM" / "data" / "AdaptEval_mixed" / "harmful_mix_2k.json"
OUTPUT_DIR = ROOT / "TLM" / "data" / "WildJailbreak_controlled"
MANIFEST_PATH = ROOT / "llm-tta" / "WILDJAILBREAK_CONTROLLED_MANIFEST.md"

SEED = 42
VANILLA_BENIGN_TRAIN_LIMIT = 1000
HARMFUL_LIFT_HOLDOUT_PAIRS = 1000


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def deterministic_sample(rows: list[dict], sample_size: int, seed_tag: str) -> list[dict]:
    if sample_size >= len(rows):
        return list(rows)

    rng = random.Random(f"{SEED}:{seed_tag}:{sample_size}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:sample_size])
    return [rows[idx] for idx in selected]


def load_harmful_mix_used_indices() -> set[int]:
    rows = load_json(HARMFUL_MIX_PATH)
    return {int(row["wildjailbreak_row_index"]) for row in rows if "wildjailbreak_row_index" in row}


def export_train_vanilla_benign() -> list[dict]:
    candidates = []
    with TRAIN_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_index, row in enumerate(reader):
            if row.get("data_type") != "vanilla_benign":
                continue
            prompt = (row.get("vanilla") or "").strip()
            if not prompt:
                continue
            candidates.append(
                {
                    "instruction": prompt,
                    "input": "",
                    "output": "",
                    "source_dataset": "wildjailbreak",
                    "source_split": "train",
                    "source_type": "vanilla_benign",
                    "wildjailbreak_row_index": row_index,
                }
            )

    sampled = deterministic_sample(candidates, VANILLA_BENIGN_TRAIN_LIMIT, "train_vanilla_benign")
    dump_json(OUTPUT_DIR / "train_vanilla_benign_1k.json", sampled)
    return sampled


def export_eval_adversarial_sets() -> tuple[list[dict], list[dict]]:
    adversarial_benign = []
    adversarial_harmful = []
    with EVAL_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_index, row in enumerate(reader):
            data_type = row.get("data_type") or ""
            prompt = (row.get("adversarial") or "").strip()
            if not prompt:
                continue

            record = {
                "instruction": prompt,
                "input": "",
                "output": "",
                "source_dataset": "wildjailbreak",
                "source_split": "eval",
                "source_type": data_type,
                "wildjailbreak_row_index": row_index,
            }
            if data_type == "adversarial_benign":
                adversarial_benign.append(record)
            elif data_type == "adversarial_harmful":
                adversarial_harmful.append(record)

    dump_json(OUTPUT_DIR / "eval_adversarial_benign.json", adversarial_benign)
    dump_json(OUTPUT_DIR / "eval_adversarial_harmful.json", adversarial_harmful)
    return adversarial_benign, adversarial_harmful


def export_harmful_lift_holdout() -> list[dict]:
    used_indices = load_harmful_mix_used_indices()
    candidates = []
    with TRAIN_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_index, row in enumerate(reader):
            if row.get("data_type") != "adversarial_harmful":
                continue
            if row_index in used_indices:
                continue

            vanilla_prompt = (row.get("vanilla") or "").strip()
            adversarial_prompt = (row.get("adversarial") or "").strip()
            if not vanilla_prompt or not adversarial_prompt:
                continue

            candidates.append(
                {
                    "wildjailbreak_row_index": row_index,
                    "vanilla": vanilla_prompt,
                    "adversarial": adversarial_prompt,
                }
            )

    sampled_pairs = deterministic_sample(candidates, HARMFUL_LIFT_HOLDOUT_PAIRS, "harmful_lift_holdout")
    exported = []
    for row in sampled_pairs:
        exported.append(
            {
                "instruction": row["vanilla"],
                "input": "",
                "output": "",
                "source_dataset": "wildjailbreak",
                "source_split": "train_holdout",
                "source_type": "vanilla_harmful",
                "wildjailbreak_row_index": row["wildjailbreak_row_index"],
            }
        )
        exported.append(
            {
                "instruction": row["adversarial"],
                "input": "",
                "output": "",
                "source_dataset": "wildjailbreak",
                "source_split": "train_holdout",
                "source_type": "adversarial_harmful",
                "wildjailbreak_row_index": row["wildjailbreak_row_index"],
            }
        )

    dump_json(OUTPUT_DIR / "train_harmful_lift_holdout_1k.json", exported)
    return exported


def write_manifest(
    train_vanilla_benign: list[dict],
    eval_adversarial_benign: list[dict],
    eval_adversarial_harmful: list[dict],
    harmful_lift_holdout: list[dict],
) -> None:
    pair_count = len(harmful_lift_holdout) // 2
    lines = [
        "# WildJailbreak Controlled Evaluation Sets",
        "",
        "These fixed evaluation sets are extracted from WildJailbreak before deleting the raw TSV files.",
        "",
        f"- Seed: `{SEED}`",
        f"- Train vanilla benign cap: `{VANILLA_BENIGN_TRAIN_LIMIT}`",
        f"- Harmful lift holdout pairs: `{pair_count}`",
        f"- harmful_mix exclusion source: `TLM/data/AdaptEval_mixed/harmful_mix_2k.json`",
        "",
        "| Dataset | Count | File | Purpose |",
        "| --- | ---: | --- | --- |",
        f"| `wildjailbreak_train_vanilla_benign_1k` | {len(train_vanilla_benign)} | `WildJailbreak_controlled/train_vanilla_benign_1k.json` | Benign refusal rate on train-style vanilla benign prompts |",
        f"| `wildjailbreak_eval_adversarial_benign` | {len(eval_adversarial_benign)} | `WildJailbreak_controlled/eval_adversarial_benign.json` | Benign refusal rate on eval adversarial benign prompts |",
        f"| `wildjailbreak_eval_adversarial_harmful` | {len(eval_adversarial_harmful)} | `WildJailbreak_controlled/eval_adversarial_harmful.json` | Held-out ASR evaluation |",
        f"| `wildjailbreak_train_harmful_lift_holdout_1k` | {len(harmful_lift_holdout)} | `WildJailbreak_controlled/train_harmful_lift_holdout_1k.json` | Paired vanilla/adversarial harmful prompts for Jailbreak Lift |",
        "",
    ]
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train_vanilla_benign = export_train_vanilla_benign()
    eval_adversarial_benign, eval_adversarial_harmful = export_eval_adversarial_sets()
    harmful_lift_holdout = export_harmful_lift_holdout()
    write_manifest(
        train_vanilla_benign,
        eval_adversarial_benign,
        eval_adversarial_harmful,
        harmful_lift_holdout,
    )
    print(f"[done] Exported controlled WildJailbreak datasets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
