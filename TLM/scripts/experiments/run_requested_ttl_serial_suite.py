from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import ROOT, make_env, python_module_command, run_command, write_json


DEFAULT_PLAN = [
    {
        "name": "agriculture_clean_vs_agriculture_mixed40",
        "clean_dataset": "agriculture_5k",
        "mixed_base_dataset": "agriculture_5k",
        "note": "clean and mixed are in-domain agriculture",
    },
    {
        "name": "alpaca_clean_vs_agriculture_mixed40",
        "clean_dataset": "alpaca_gpt4_5k",
        "mixed_base_dataset": "agriculture_5k",
        "note": "this follows the user-specified cross-domain comparison",
    },
    {
        "name": "gsm8k_clean_vs_gsm8k_mixed40",
        "clean_dataset": "gsm8k_5k",
        "mixed_base_dataset": "gsm8k_5k",
        "note": "clean and mixed are in-domain gsm8k",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the requested clean/mixed40 offline TTL experiments serially to avoid OOM."
    )
    parser.add_argument("--model-name-or-path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="D:\\hf_cache")
    parser.add_argument("--output-root", default="saves/serial_suites/requested_suite")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run-upload", action="store_true")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_pipeline(script_name: str, dataset: str, args: argparse.Namespace, env: dict[str, str]) -> dict[str, str]:
    command = [
        *python_module_command(f"scripts/experiments/{script_name}"),
        "--dataset",
        dataset,
        "--model-name-or-path",
        args.model_name_or_path,
        "--dataset-dir",
        args.dataset_dir,
        "--hf-home",
        args.hf_home,
        "--output-root",
        args.output_root,
    ]
    if args.skip_export:
        command.append("--skip-export")
    if args.skip_upload:
        command.append("--skip-upload")
    if args.dry_run_upload:
        command.append("--dry-run-upload")
    if args.use_swanlab:
        command.append("--use-swanlab")
    if args.smoke_test:
        command.append("--smoke-test")

    run_command(command, cwd=ROOT, env=env)
    return {"script": script_name, "dataset": dataset}


def main() -> None:
    args = parse_args()
    env = make_env(args.hf_home)

    suite_results = []
    for step in DEFAULT_PLAN:
        clean_dataset = step["clean_dataset"]
        mixed_base_dataset = step["mixed_base_dataset"]

        clean_result = run_pipeline("run_clean_ttl_pipeline.py", clean_dataset, args, env)
        mixed_result = run_pipeline("run_mixed40_ttl_pipeline.py", mixed_base_dataset, args, env)

        suite_results.append(
            {
                "name": step["name"],
                "note": step["note"],
                "clean": clean_result,
                "mixed40": mixed_result,
                "clean_summary": str(ROOT / args.output_root / clean_dataset / "pipeline_summary.json"),
                "mixed_summary": str(ROOT / args.output_root / mixed_base_dataset / "pipeline_summary.json"),
            }
        )

    summary = {
        "suite_name": "requested_serial_ttl_suite",
        "model_name_or_path": args.model_name_or_path,
        "smoke_test": args.smoke_test,
        "runs": suite_results,
    }
    summary_path = ROOT / args.output_root / "requested_serial_ttl_suite_summary.json"
    write_json(summary_path, summary)
    print(f"[done] Serial suite summary written to {summary_path}")


if __name__ == "__main__":
    main()
