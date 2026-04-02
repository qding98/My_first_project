from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_common import (
    DEFAULT_FORMAL_MODEL_NAME_OR_PATH,
    DEFAULT_FORMAL_TEMPLATE,
    detect_model_spec,
    ensure_dataset_exists,
    make_env,
    python_module_command,
    resolve_output_root,
    run_command,
    select_generation_profile,
    write_json,
)
from vallina_common import (
    DEFAULT_4090_VRAM_GB,
    DEFAULT_BASELINE_EVAL_BS,
    DEFAULT_BASELINE_VRAM_GB,
    DEFAULT_CLEAN_MODEL_ALIAS,
    DEFAULT_VILLINA_DATASET,
    build_vallina_generation_run_tag,
    jsonl_to_json_array,
    scaled_batch_size,
)


DEFAULT_CLEAN_ADAPTER_DIR = (
    "saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/"
    "alpaca_gpt4_5k/clean_model/adapter"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean-model predictions on villina_mixed for clean-vs-vallina comparison."
    )
    parser.add_argument("--adapter-dir", default=DEFAULT_CLEAN_ADAPTER_DIR)
    parser.add_argument("--base-model-path", default=DEFAULT_FORMAL_MODEL_NAME_OR_PATH)
    parser.add_argument("--template", default=DEFAULT_FORMAL_TEMPLATE)
    parser.add_argument("--dataset", default=DEFAULT_VILLINA_DATASET)
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--output-root", default="saves/predictions/alpaca_clean_vallina")
    parser.add_argument("--model-alias", default=DEFAULT_CLEAN_MODEL_ALIAS)
    parser.add_argument("--output-name", default="alpaca_clean_vallina.json")
    parser.add_argument("--summary-name", default="generation_suite_summary.json")
    parser.add_argument("--hf-home", default="D:\\hf_cache")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--baseline-eval-batch-size", type=int, default=DEFAULT_BASELINE_EVAL_BS)
    parser.add_argument("--baseline-vram-gb", type=float, default=DEFAULT_BASELINE_VRAM_GB)
    parser.add_argument("--target-vram-gb", type=float, default=DEFAULT_4090_VRAM_GB)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-dataset-profiles", action="store_true")
    parser.add_argument("--use-swanlab", action="store_true", default=bool(os.getenv("SWANLAB_API_KEY")))
    parser.add_argument("--swanlab-project", default=os.getenv("SWANLAB_PROJ_NAME", "TLM_alpaca_clean_vallina"))
    parser.add_argument("--swanlab-workspace", default=os.getenv("SWANLAB_WORKSPACE"))
    parser.add_argument("--swanlab-mode", default=os.getenv("SWANLAB_MODE", "cloud"))
    parser.add_argument("--swanlab-api-key", default=os.getenv("SWANLAB_API_KEY"))
    return parser.parse_args()


def resolve_path_arg(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path.resolve()
    return resolve_output_root(path).resolve()


def main() -> None:
    args = parse_args()
    ensure_dataset_exists(args.dataset)

    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = scaled_batch_size(
            args.baseline_eval_batch_size,
            target_vram_gb=args.target_vram_gb,
            baseline_vram_gb=args.baseline_vram_gb,
        )

    env = make_env(args.hf_home)
    if args.use_swanlab:
        env["SWANLAB_PROJ_NAME"] = args.swanlab_project
        env["SWANLAB_MODE"] = args.swanlab_mode
        if args.swanlab_workspace:
            env["SWANLAB_WORKSPACE"] = args.swanlab_workspace
        if args.swanlab_api_key:
            env["SWANLAB_API_KEY"] = args.swanlab_api_key

    adapter_dir = resolve_path_arg(args.adapter_dir)
    model_spec = detect_model_spec(adapter_dir, base_model_path=args.base_model_path)
    profile = select_generation_profile(
        args.dataset,
        cutoff_len=args.cutoff_len,
        max_new_tokens=args.max_new_tokens,
        use_dataset_profiles=not args.disable_dataset_profiles,
    )
    run_tag = build_vallina_generation_run_tag(
        args.per_device_eval_batch_size,
        args.seed,
        cutoff_len=args.cutoff_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    run_root = resolve_output_root(args.output_root) / run_tag / args.model_alias
    dataset_output_dir = run_root / args.dataset
    summary_path = run_root / args.summary_name
    experiment_name = f"alpaca_clean_vallina_{args.dataset}_seed_{args.seed}"

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
        args.dataset,
        "--dataset_dir",
        args.dataset_dir,
        "--template",
        args.template,
        "--cutoff_len",
        str(profile.cutoff_len),
        "--max_new_tokens",
        str(profile.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--do_sample",
        "false",
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--output_dir",
        str(dataset_output_dir),
        "--overwrite_cache",
        "true",
        "--overwrite_output_dir",
        "true",
        "--report_to",
        "none",
        "--preprocessing_num_workers",
        str(args.preprocessing_num_workers),
        "--trust_remote_code",
        "true",
    ]
    if "adapter_name_or_path" in model_spec:
        command.extend(["--adapter_name_or_path", model_spec["adapter_name_or_path"], "--finetuning_type", "lora"])
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    if args.use_swanlab:
        command.extend(
            [
                "--use_swanlab",
                "true",
                "--swanlab_project",
                args.swanlab_project,
                "--swanlab_experiment_name",
                experiment_name,
                "--swanlab_mode",
                args.swanlab_mode,
            ]
        )
        if args.swanlab_workspace:
            command.extend(["--swanlab_workspace", args.swanlab_workspace])

    run_command(command, cwd=resolve_output_root("."), env=env)

    generated_predictions = dataset_output_dir / "generated_predictions.jsonl"
    generate_predict_json = dataset_output_dir / "generate_predict.json"
    row_count = jsonl_to_json_array(generated_predictions, generate_predict_json)

    output_json_path = dataset_output_dir / args.output_name
    output_json_path.write_text(generate_predict_json.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "adapter_dir": str(adapter_dir),
        "model_alias": args.model_alias,
        "base_model_path": args.base_model_path,
        "dataset": args.dataset,
        "run_tag": run_tag,
        "run_root": str(run_root),
        "target_vram_gb": args.target_vram_gb,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "profile": {
            "cutoff_len": profile.cutoff_len,
            "max_new_tokens": profile.max_new_tokens,
            "profile_name": profile.profile_name,
        },
        "generated_predictions_jsonl": str(generated_predictions),
        "generate_predict_json": str(generate_predict_json),
        "comparison_json": str(output_json_path),
        "row_count": row_count,
    }
    write_json(summary_path, summary)
    print(f"[done] Alpaca clean vallina summary written to {summary_path}")


if __name__ == "__main__":
    main()
