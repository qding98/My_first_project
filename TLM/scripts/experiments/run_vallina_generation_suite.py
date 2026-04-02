from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_common import (
    DEFAULT_FORMAL_MODEL_NAME_OR_PATH,
    DEFAULT_FORMAL_TEMPLATE,
    detect_model_spec,
    ensure_dataset_exists,
    force_offline_hf_env,
    make_env,
    model_tag,
    python_module_command,
    resolve_cached_model_path,
    resolve_output_root,
    run_command,
    select_generation_profile,
    write_json,
)
from vallina_common import (
    DEFAULT_4090_VRAM_GB,
    DEFAULT_BASELINE_EVAL_BS,
    DEFAULT_BASELINE_VRAM_GB,
    DEFAULT_PREDICTION_DATASETS,
    DEFAULT_VALLINA_MODEL_ALIAS,
    build_vallina_generation_run_tag,
    jsonl_to_json_array,
    scaled_batch_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions with model B on alpaca, villina_mixed, and WildJailbreak controlled datasets."
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-path", default=DEFAULT_FORMAL_MODEL_NAME_OR_PATH)
    parser.add_argument("--template", default=DEFAULT_FORMAL_TEMPLATE)
    parser.add_argument("--output-root", default="saves/predictions/vallina")
    parser.add_argument(
        "--model-alias",
        default=None,
        help=f"Optional stable directory name for predictions. Recommended: {DEFAULT_VALLINA_MODEL_ALIAS}",
    )
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="D:\\hf_cache")
    parser.add_argument("--datasets", default=",".join(DEFAULT_PREDICTION_DATASETS))
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
    parser.add_argument("--swanlab-project", default=os.getenv("SWANLAB_PROJ_NAME", "TLM_vallina_predict"))
    parser.add_argument("--swanlab-workspace", default=os.getenv("SWANLAB_WORKSPACE"))
    parser.add_argument("--swanlab-mode", default=os.getenv("SWANLAB_MODE", "cloud"))
    parser.add_argument("--swanlab-api-key", default=os.getenv("SWANLAB_API_KEY"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def apply_generation_defaults(args: argparse.Namespace) -> None:
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = scaled_batch_size(
            args.baseline_eval_batch_size,
            target_vram_gb=args.target_vram_gb,
            baseline_vram_gb=args.baseline_vram_gb,
        )


def apply_generation_smoke_overrides(args: argparse.Namespace) -> None:
    args.base_model_path = resolve_cached_model_path("llamafactory/tiny-random-Llama-3", args.hf_home)
    args.template = "llama3"
    args.cutoff_len = 64
    args.max_new_tokens = 8
    args.max_samples = 1
    args.preprocessing_num_workers = 1
    args.target_vram_gb = DEFAULT_4090_VRAM_GB
    args.per_device_eval_batch_size = 1
    args.datasets = ",".join(DEFAULT_PREDICTION_DATASETS[:2])


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        apply_generation_smoke_overrides(args)

    apply_generation_defaults(args)

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for dataset in datasets:
        ensure_dataset_exists(dataset)

    env = make_env(args.hf_home)
    if args.smoke_test:
        force_offline_hf_env(env)
    if args.use_swanlab:
        env["SWANLAB_PROJ_NAME"] = args.swanlab_project
        env["SWANLAB_MODE"] = args.swanlab_mode
        if args.swanlab_workspace:
            env["SWANLAB_WORKSPACE"] = args.swanlab_workspace
        if args.swanlab_api_key:
            env["SWANLAB_API_KEY"] = args.swanlab_api_key

    adapter_dir = Path(args.adapter_dir).resolve()
    model_spec = detect_model_spec(adapter_dir, base_model_path=args.base_model_path)
    run_tag = build_vallina_generation_run_tag(
        args.per_device_eval_batch_size,
        args.seed,
        cutoff_len=args.cutoff_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    model_output_name = args.model_alias or model_tag(adapter_dir)
    run_root = resolve_output_root(args.output_root) / run_tag / model_output_name
    summary_path = run_root / "generation_suite_summary.json"
    dataset_summaries = []

    for dataset_name in datasets:
        profile = select_generation_profile(
            dataset_name,
            cutoff_len=args.cutoff_len,
            max_new_tokens=args.max_new_tokens,
            smoke_test=args.smoke_test,
            use_dataset_profiles=not args.disable_dataset_profiles,
        )
        dataset_output_dir = run_root / dataset_name
        experiment_name = f"vallina_generate_{dataset_name}_seed_{args.seed}"
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
            dataset_name,
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
        dataset_summaries.append(
            {
                "dataset": dataset_name,
                "profile": {
                    "cutoff_len": profile.cutoff_len,
                    "max_new_tokens": profile.max_new_tokens,
                    "profile_name": profile.profile_name,
                },
                "generated_predictions_jsonl": str(generated_predictions),
                "generate_predict_json": str(generate_predict_json),
                "row_count": row_count,
            }
        )

    summary = {
        "adapter_dir": str(adapter_dir),
        "model_alias": args.model_alias,
        "model_output_name": model_output_name,
        "base_model_path": args.base_model_path,
        "model_kind": model_spec["model_kind"],
        "run_tag": run_tag,
        "run_root": str(run_root),
        "target_vram_gb": args.target_vram_gb,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "datasets": dataset_summaries,
    }
    write_json(summary_path, summary)
    print(f"[done] Vallina generation summary written to {summary_path}")


if __name__ == "__main__":
    main()
