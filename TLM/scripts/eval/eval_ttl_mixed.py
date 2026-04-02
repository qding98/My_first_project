import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

from eval_utils import extract_gsm8k_answer_number, extract_logiqa_option, extract_math_answer, post_process

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_rows(rows: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    harmful_rows = [row for row in rows if bool(row.get("is_harmful_mix"))]
    benign_rows = [row for row in rows if not bool(row.get("is_harmful_mix"))]
    return benign_rows, harmful_rows


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def exact_match_metric(rows: List[Dict]) -> Dict[str, float]:
    matches = [normalize_text(row["predict"]) == normalize_text(row["label"]) for row in rows]
    return {"benign_exact_match": sum(matches) / len(matches) if matches else 0.0}


def gsm8k_metric(rows: List[Dict]) -> Dict[str, float]:
    matches = []
    for row in rows:
        pred = extract_gsm8k_answer_number(row["predict"])
        label = extract_gsm8k_answer_number(row["label"]) or normalize_text(row["label"])
        matches.append(pred == label)
    return {"benign_accuracy": sum(matches) / len(matches) if matches else 0.0}


def logiqa_metric(rows: List[Dict]) -> Dict[str, float]:
    matches = []
    for row in rows:
        pred = extract_logiqa_option(row["predict"])
        label = normalize_text(row["label"]).upper()
        matches.append(pred == label)
    return {"benign_accuracy": sum(matches) / len(matches) if matches else 0.0}


def math_metric(rows: List[Dict]) -> Dict[str, float]:
    matches = []
    for row in rows:
        pred = extract_math_answer(row["predict"], row["label"])
        label = post_process(row["label"])
        matches.append(pred == label)
    return {"benign_accuracy": sum(matches) / len(matches) if matches else 0.0}


def similarity_metric(rows: List[Dict]) -> Dict[str, float]:
    try:
        import jieba
    except ImportError as exc:
        raise ImportError("Please install jieba to run similarity metrics: python -m pip install jieba") from exc

    try:
        from rouge_chinese import Rouge
    except ImportError as exc:
        raise ImportError(
            "Please install rouge-chinese to run similarity metrics: python -m pip install rouge-chinese"
        ) from exc

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError as exc:
        raise ImportError("Please install nltk to run similarity metrics: python -m pip install nltk") from exc

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    smooth = SmoothingFunction().method3
    for row in rows:
        pred = row["predict"]
        label = row["label"]

        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
            rouge_result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        else:
            rouge_result = Rouge().get_scores(" ".join(hypothesis), " ".join(reference))[0]

        rouge1_scores.append(round(rouge_result["rouge-1"]["f"] * 100, 4))
        rouge2_scores.append(round(rouge_result["rouge-2"]["f"] * 100, 4))
        rougeL_scores.append(round(rouge_result["rouge-l"]["f"] * 100, 4))
        bleu_scores.append(round(sentence_bleu([list(label)], list(pred), smoothing_function=smooth) * 100, 4))

    return {
        "benign_rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "benign_rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "benign_rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
        "benign_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
    }

def infer_task_type(rows: List[Dict]) -> str:
    dataset_name = ""
    for row in rows:
        dataset_name = row.get("mixed_into_dataset") or row.get("source_dataset") or ""
        if dataset_name:
            break

    if "gsm8k" in dataset_name:
        return "similarity"
    if "logiqa" in dataset_name:
        return "logiqa"
    if "meta_math" in dataset_name:
        return "meta_math"
    if "agriculture" in dataset_name:
        return "similarity"
    return "similarity"


BENIGN_METRICS: Dict[str, Callable[[List[Dict]], Dict[str, float]]] = {
    "exact_match": exact_match_metric,
    "gsm8k": gsm8k_metric,
    "logiqa": logiqa_metric,
    "meta_math": math_metric,
    "similarity": similarity_metric,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate mixed TLM TTL predictions on benign quality only.")
    parser.add_argument("--prediction-file", required=True, help="Path to generated_predictions.jsonl")
    parser.add_argument(
        "--task-type",
        default="auto",
        choices=["auto", "similarity", "gsm8k", "logiqa", "meta_math", "exact_match"],
        help="Metric family for the benign subset.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save the metrics JSON.")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.prediction_file))
    benign_rows, _ = split_rows(rows)
    task_type = infer_task_type(rows) if args.task_type == "auto" else args.task_type

    benign_metrics = BENIGN_METRICS[task_type](benign_rows) if benign_rows else {}
    summary = {
        "prediction_file": args.prediction_file,
        "task_type": task_type,
        "total_rows": len(rows),
        "benign_count": len(benign_rows),
        **benign_metrics,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
