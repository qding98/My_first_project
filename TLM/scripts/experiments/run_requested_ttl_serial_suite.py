from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_common import (
    DEFAULT_FORMAL_MODEL_NAME_OR_PATH,
    DEFAULT_FORMAL_TEMPLATE,
    apply_smoke_test_overrides,
    build_run_tag,
    make_env,
    python_module_command,
    resolve_output_root,
    run_command,
    run_model_eval_suite,
    write_json,
)


DEFAULT_PLAN = [
    {
        "name": "agriculture_clean_vs_agriculture_mixed40",
        "clean_dataset": "agriculture_5k",
        "note": "clean and mixed40 both use agriculture data.",
    },
    {
        "name": "alpaca_clean_vs_alpaca_mixed40",
        "clean_dataset": "alpaca_gpt4_5k",
        "note": "clean and mixed40 both use alpaca_gpt4 data.",
    },
    {
        "name": "gsm8k_clean_vs_gsm8k_mixed40",
        "clean_dataset": "gsm8k_5k",
        "note": "clean and mixed40 both use gsm8k data.",
    },
]

LOCAL_SWANLAB_ENV = resolve_output_root(".") / ".swanlab.env"
DEFAULT_SWANLAB_PROJECT = "TLM"
DEFAULT_SWANLAB_WORKSPACE = "qding666"


def load_local_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def parse_args(defaults: dict[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the requested clean/mixed40 offline TTL experiments serially with base/clean/mix evaluations."
    )
    parser.add_argument("--model-name-or-path", default=DEFAULT_FORMAL_MODEL_NAME_OR_PATH)
    parser.add_argument("--template", default=DEFAULT_FORMAL_TEMPLATE)
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="D:\\hf_cache")
    parser.add_argument("--output-root", default="saves/serial_suites/requested_suite")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=8000)
    parser.add_argument("--ddp-timeout", type=int, default=180000000)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=41000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-dataset-profiles", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run-upload", action="store_true")
    parser.add_argument(
        "--use-swanlab",
        action="store_true",
        default=bool(defaults.get("SWANLAB_API_KEY") or os.getenv("SWANLAB_API_KEY")),
    )
    parser.add_argument(
        "--swanlab-project",
        default=defaults.get("SWANLAB_PROJ_NAME", DEFAULT_SWANLAB_PROJECT),
    )
    parser.add_argument(
        "--swanlab-workspace",
        default=defaults.get("SWANLAB_WORKSPACE", DEFAULT_SWANLAB_WORKSPACE),
    )
    parser.add_argument(
        "--swanlab-mode",
        default=defaults.get("SWANLAB_MODE", "cloud"),
    )
    parser.add_argument(
        "--swanlab-api-key",
        default=defaults.get("SWANLAB_API_KEY") or os.getenv("SWANLAB_API_KEY"),
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_pipeline(script_name: str, dataset: str, args: argparse.Namespace, env: dict[str, str], output_root: Path) -> dict[str, str]:
    command = [
        *python_module_command(f"scripts/experiments/{script_name}"),
        "--dataset",
        dataset,
        "--model-name-or-path",
        args.model_name_or_path,
        "--template",
        args.template,
        "--dataset-dir",
        args.dataset_dir,
        "--hf-home",
        args.hf_home,
        "--output-root",
        str(output_root),
        "--temperature",
        str(args.temperature),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cutoff-len",
        str(args.cutoff_len),
        "--learning-rate",
        str(args.learning_rate),
        "--threshold",
        str(args.threshold),
        "--lamb",
        str(args.lamb),
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--lr-scheduler-type",
        str(args.lr_scheduler_type),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
        "--ddp-timeout",
        str(args.ddp_timeout),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(args.per_device_eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--preprocessing-num-workers",
        str(args.preprocessing_num_workers),
        "--seed",
        str(args.seed),
    ]
    if args.bf16:
        command.append("--bf16")
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.disable_dataset_profiles:
        command.append("--disable-dataset-profiles")
    if args.skip_export:
        command.append("--skip-export")
    if args.skip_upload:
        command.append("--skip-upload")
    if args.dry_run_upload:
        command.append("--dry-run-upload")
    if args.use_swanlab:
        command.extend(
            [
                "--use-swanlab",
                "--swanlab-project",
                args.swanlab_project,
                "--swanlab-mode",
                args.swanlab_mode,
            ]
        )
        if args.swanlab_workspace:
            command.extend(["--swanlab-workspace", args.swanlab_workspace])
    if args.smoke_test:
        command.append("--smoke-test")

    run_command(command, cwd=resolve_output_root("."), env=env)
    return {
        "script": script_name,
        "dataset": dataset,
        "summary_path": str(output_root / dataset / ("clean_model" if "clean" in script_name else "mix_model") / "pipeline_summary.json"),
    }


def main() -> None:
    local_defaults = load_local_env(LOCAL_SWANLAB_ENV)
    args = parse_args(local_defaults)
    if args.smoke_test:
        apply_smoke_test_overrides(args)

    env = make_env(args.hf_home)
    if args.use_swanlab:
        env["SWANLAB_PROJ_NAME"] = args.swanlab_project
        env["SWANLAB_MODE"] = args.swanlab_mode
        if args.swanlab_workspace:
            env["SWANLAB_WORKSPACE"] = args.swanlab_workspace
        if args.swanlab_api_key:
            env["SWANLAB_API_KEY"] = args.swanlab_api_key

    run_tag = build_run_tag(args.learning_rate, args.per_device_train_batch_size, args.seed, args.gradient_accumulation_steps)
    run_root = resolve_output_root(args.output_root) / run_tag

    suite_results = []
    for step in DEFAULT_PLAN:
        clean_dataset = step["clean_dataset"]
        dataset_root = run_root / clean_dataset

        base_model_eval = run_model_eval_suite(
            model_path=args.model_name_or_path,
            model_role_dir=dataset_root / "base_model",
            clean_dataset=clean_dataset,
            env=env,
            base_model_path=args.model_name_or_path,
            dataset_dir=args.dataset_dir,
            template=args.template,
            cutoff_len=args.cutoff_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            preprocessing_num_workers=args.preprocessing_num_workers,
            max_samples=args.max_samples,
            smoke_test=args.smoke_test,
            hf_home=args.hf_home,
            use_dataset_profiles=not args.disable_dataset_profiles,
        )

        clean_result = run_pipeline("run_clean_ttl_pipeline.py", clean_dataset, args, env, run_root)
        mixed_result = run_pipeline("run_mixed40_ttl_pipeline.py", clean_dataset, args, env, run_root)

        suite_results.append(
            {
                "name": step["name"],
                "note": step["note"],
                "clean_dataset": clean_dataset,
                "base_model_eval_summary": str(dataset_root / "base_model" / "model_eval_summary.json"),
                "clean_pipeline_summary": clean_result["summary_path"],
                "mix_pipeline_summary": mixed_result["summary_path"],
                "base_model_eval": base_model_eval,
            }
        )

    summary = {
        "suite_name": "requested_serial_ttl_suite",
        "run_tag": run_tag,
        "run_root": str(run_root),
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "smoke_test": args.smoke_test,
        "use_swanlab": args.use_swanlab,
        "swanlab_project": args.swanlab_project if args.use_swanlab else None,
        "swanlab_workspace": args.swanlab_workspace if args.use_swanlab else None,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "seed": args.seed,
            "num_train_epochs": args.num_train_epochs,
            "threshold": args.threshold,
            "lamb": args.lamb,
            "preprocessing_num_workers": args.preprocessing_num_workers,
            "disable_dataset_profiles": args.disable_dataset_profiles,
        },
        "runs": suite_results,
    }
    summary_path = run_root / "requested_serial_ttl_suite_summary.json"
    write_json(summary_path, summary)
    print(f"[done] Serial suite summary written to {summary_path}")


if __name__ == "__main__":
    main()
