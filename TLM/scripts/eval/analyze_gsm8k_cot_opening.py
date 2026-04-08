from __future__ import annotations

"""分析 GSM8K CoT 生成开头格式的脚本。

本脚本负责定位 `gsm8k_5k/mix_model` 相关预测文件，并判断回答开头是否呈现固定格式。
输入来自命令行给出的模型结果目录或显式预测文件路径，输出为：

- clean eval 预测文件是否真实存在
- 如果不存在，当前目录下有哪些相关文件
- 如果存在，回答首行、前若干词前缀和公共字符前缀的统计摘要
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from safety_eval_annotation import extract_prediction_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = "saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/gsm8k_5k/mix_model"


def parse_args() -> argparse.Namespace:
    """解析开头格式分析脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Analyze whether GSM8K CoT generations share a fixed opening format.")
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT, help="Mix model result root to inspect.")
    parser.add_argument("--prediction-file", default="", help="Optional explicit prediction file.")
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    parser.add_argument("--top-k", type=int, default=10, help="How many common openings to keep.")
    parser.add_argument(
        "--word-prefix-length",
        dest="word_prefix_lengths",
        action="append",
        type=int,
        default=[],
        help="Word-prefix lengths to analyze; can be passed multiple times.",
    )
    return parser.parse_args()


def resolve_repo_path(path_arg: str | Path) -> Path:
    """把项目内相对路径统一解析到 `TLM/` 根目录。"""

    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    return ROOT / path


def build_default_output_path(model_root: Path) -> Path:
    """给分析结果生成默认输出文件路径。"""

    return model_root / "gsm8k_cot_opening_analysis.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """把分析结果写成 UTF-8 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_prediction_paths(model_root: Path) -> dict[str, Any]:
    """收集 clean eval 目录下的目标文件与当前可见的相关文件。"""

    clean_eval_dir = model_root / "evaluation_predictions" / "gsm8k_5k"
    preferred_candidates = [
        clean_eval_dir / "generated_predictions.jsonl",
        clean_eval_dir / "generate_predict.json",
    ]
    related_files = sorted(str(path.relative_to(model_root)) for path in clean_eval_dir.glob("*") if path.is_file())
    nested_predictions = sorted(
        str(path.relative_to(model_root))
        for path in model_root.rglob("generated_predictions.jsonl")
    )
    found_file = next((path for path in preferred_candidates if path.exists()), None)
    return {
        "clean_eval_dir": clean_eval_dir,
        "preferred_candidates": preferred_candidates,
        "found_prediction_file": found_file,
        "related_files": related_files,
        "nested_predictions": nested_predictions,
    }


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    """读取 `jsonl` 或 JSON 数组格式的预测文件。"""

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prediction file must contain a JSON array: {path}")
    return [row for row in payload if isinstance(row, dict)]


def normalize_opening_text(text: str) -> str:
    """压缩空白，方便比较回答开头是否重复。"""

    return " ".join((text or "").strip().split())


def extract_first_line(text: str) -> str:
    """提取回答的第一行作为固定模板判断依据。"""

    return normalize_opening_text((text or "").splitlines()[0] if text else "")


def extract_word_prefix(text: str, word_count: int) -> str:
    """提取回答开头前若干个空白分词。"""

    words = normalize_opening_text(text).split()
    return " ".join(words[: max(1, word_count)])


def summarize_counter(counter: Counter[str], total: int, top_k: int) -> list[dict[str, Any]]:
    """把频次统计整理成便于 JSON 输出的列表。"""

    return [
        {"text": text, "count": count, "rate": count / total if total else None}
        for text, count in counter.most_common(top_k)
    ]


def longest_common_prefix(texts: list[str]) -> str:
    """计算一组字符串的最长公共字符前缀。"""

    if not texts:
        return ""
    prefix = texts[0]
    for text in texts[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
    return prefix.strip()


def analyze_openings(rows: list[dict[str, Any]], top_k: int, word_prefix_lengths: list[int]) -> dict[str, Any]:
    """统计回答首行与词前缀，判断是否存在固定开头格式。"""

    predictions = [extract_prediction_text(row) for row in rows]
    non_empty = [text for text in predictions if normalize_opening_text(text)]
    first_line_counter = Counter(extract_first_line(text) for text in non_empty)
    analysis = {
        "total_rows": len(rows),
        "non_empty_prediction_count": len(non_empty),
        "top_first_lines": summarize_counter(first_line_counter, len(non_empty), top_k),
        "common_char_prefix": longest_common_prefix([normalize_opening_text(text) for text in non_empty]),
    }
    for length in sorted({length for length in word_prefix_lengths if length > 0}):
        prefix_counter = Counter(extract_word_prefix(text, length) for text in non_empty)
        top_items = summarize_counter(prefix_counter, len(non_empty), top_k)
        analysis[f"top_first_{length}_words"] = top_items
        analysis[f"looks_fixed_first_{length}_words"] = bool(top_items and (top_items[0]["rate"] or 0.0) >= 0.8)
    return analysis


def build_missing_prediction_report(model_root: Path, discovery: dict[str, Any]) -> dict[str, Any]:
    """当 clean eval 没有导出逐样本预测时，输出排查报告。"""

    return {
        "model_root": str(model_root),
        "prediction_file_found": False,
        "clean_eval_dir": str(discovery["clean_eval_dir"]),
        "preferred_prediction_candidates": [str(path) for path in discovery["preferred_candidates"]],
        "existing_clean_eval_files": discovery["related_files"],
        "other_generated_predictions": discovery["nested_predictions"],
        "note": (
            "当前 `evaluation_predictions/gsm8k_5k` 目录里没有逐样本预测文件。"
            "结合现有 clean eval 链路，它只保存了 `eval_results.json/all_results.json` 等聚合指标，"
            "没有把 GSM8K clean eval 的生成结果落成 `generated_predictions.jsonl`。"
        ),
    }


def main() -> None:
    """执行开头格式分析，并把结果打印或写出到 JSON。"""

    args = parse_args()
    model_root = resolve_repo_path(args.model_root)
    output_path = resolve_repo_path(args.output_json) if args.output_json else build_default_output_path(model_root)
    word_prefix_lengths = args.word_prefix_lengths or [4, 8]
    if args.prediction_file:
        prediction_file = resolve_repo_path(args.prediction_file)
        summary = {
            "model_root": str(model_root),
            "prediction_file_found": prediction_file.exists(),
            "prediction_file": str(prediction_file),
        }
        if prediction_file.exists():
            summary["analysis"] = analyze_openings(load_prediction_rows(prediction_file), args.top_k, word_prefix_lengths)
        else:
            summary["note"] = "显式指定的预测文件不存在。"
    else:
        discovery = discover_prediction_paths(model_root)
        prediction_file = discovery["found_prediction_file"]
        summary = build_missing_prediction_report(model_root, discovery) if prediction_file is None else {
            "model_root": str(model_root),
            "prediction_file_found": True,
            "prediction_file": str(prediction_file),
            "analysis": analyze_openings(load_prediction_rows(prediction_file), args.top_k, word_prefix_lengths),
        }
    write_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] Opening analysis written to {output_path}")


if __name__ == "__main__":
    main()
