from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_common import (
    DEFAULT_FORMAL_MODEL_NAME_OR_PATH,
    DEFAULT_FORMAL_TEMPLATE,
    DEFAULT_MODELSCOPE_REPO_PREFIX,
    apply_smoke_test_overrides,
    build_modescope_repo_id,
    ensure_dataset_exists,
    force_offline_hf_env,
    make_env,
    python_module_command,
    resolve_output_root,
    run_command,
    select_generation_profile,
    write_json,
)
from vallina_common import (
    DEFAULT_4090_VRAM_GB,
    DEFAULT_ALPACA_VILLINA_DATASET,
    DEFAULT_BASELINE_EVAL_BS,
    DEFAULT_BASELINE_LR,
    DEFAULT_BASELINE_TRAIN_BS,
    DEFAULT_BASELINE_VRAM_GB,
    DEFAULT_REFERENCE_CLEAN_DATASET,
    build_vallina_run_tag,
    scaled_batch_size,
    scaled_learning_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model B on alpaca_villina_mixed40 with vallina-prefixed output management."
    )
    parser.add_argument("--dataset", default=DEFAULT_ALPACA_VILLINA_DATASET)
    parser.add_argument("--reference-clean-dataset", default=DEFAULT_REFERENCE_CLEAN_DATASET)
    parser.add_argument("--model-name-or-path", default=DEFAULT_FORMAL_MODEL_NAME_OR_PATH)
    parser.add_argument("--template", default=DEFAULT_FORMAL_TEMPLATE)
    parser.add_argument("--output-root", default="saves/pipelines/vallina")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="D:\\hf_cache")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--baseline-learning-rate", type=float, default=DEFAULT_BASELINE_LR)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=60)
    parser.add_argument("--ddp-timeout", type=int, default=180000000)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=41000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-vram-gb", type=float, default=DEFAULT_BASELINE_VRAM_GB)
    parser.add_argument("--target-vram-gb", type=float, default=DEFAULT_BASELINE_VRAM_GB)
    parser.add_argument("--baseline-train-batch-size", type=int, default=DEFAULT_BASELINE_TRAIN_BS)
    parser.add_argument("--baseline-eval-batch-size", type=int, default=DEFAULT_BASELINE_EVAL_BS)
    parser.add_argument("--disable-dataset-profiles", action="store_true")
    parser.add_argument("--use-swanlab", action="store_true", default=bool(os.getenv("SWANLAB_API_KEY")))
    parser.add_argument("--swanlab-project", default=os.getenv("SWANLAB_PROJ_NAME", "TLM_vallina"))
    parser.add_argument("--swanlab-workspace", default=os.getenv("SWANLAB_WORKSPACE"))
    parser.add_argument("--swanlab-mode", default=os.getenv("SWANLAB_MODE", "cloud"))
    parser.add_argument("--swanlab-api-key", default=os.getenv("SWANLAB_API_KEY"))
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run-upload", action="store_true")
    parser.add_argument("--modelscope-repo-id", default=None)
    parser.add_argument("--modelscope-repo-prefix", default=DEFAULT_MODELSCOPE_REPO_PREFIX)
    parser.add_argument("--modelscope-token", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def apply_vallina_defaults(args: argparse.Namespace) -> None:
    if args.per_device_train_batch_size is None:
        args.per_device_train_batch_size = scaled_batch_size(
            args.baseline_train_batch_size,
            target_vram_gb=args.target_vram_gb,
            baseline_vram_gb=args.baseline_vram_gb,
        )

    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = scaled_batch_size(
            args.baseline_eval_batch_size,
            target_vram_gb=args.target_vram_gb,
            baseline_vram_gb=args.baseline_vram_gb,
        )

    if args.learning_rate is None:
        args.learning_rate = scaled_learning_rate(
            args.baseline_learning_rate,
            train_batch_size=args.per_device_train_batch_size,
            baseline_train_batch_size=args.baseline_train_batch_size,
        )


def apply_vallina_smoke_overrides(args: argparse.Namespace) -> None:
    apply_smoke_test_overrides(args)
    args.dataset = DEFAULT_ALPACA_VILLINA_DATASET
    args.reference_clean_dataset = DEFAULT_REFERENCE_CLEAN_DATASET
    args.per_device_train_batch_size = 1
    args.per_device_eval_batch_size = 1
    args.skip_export = True
    args.skip_upload = True
    args.target_vram_gb = DEFAULT_4090_VRAM_GB


def main() -> None:
    args = parse_args()
    ensure_dataset_exists(args.dataset)
    ensure_dataset_exists(args.reference_clean_dataset)

    if args.smoke_test:
        apply_vallina_smoke_overrides(args)

    apply_vallina_defaults(args)

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

    train_profile = select_generation_profile(
        args.dataset,
        cutoff_len=args.cutoff_len,
        max_new_tokens=args.max_new_tokens,
        smoke_test=args.smoke_test,
        use_dataset_profiles=not args.disable_dataset_profiles,
    )

    run_tag = build_vallina_run_tag(
        args.learning_rate,
        args.per_device_train_batch_size,
        args.seed,
        args.gradient_accumulation_steps,
    )
    run_root = resolve_output_root(args.output_root) / run_tag
    base_dir = run_root / args.dataset / "vallina_model"
    adapter_dir = base_dir / "adapter"
    export_dir = base_dir / "exported_model"
    summary_path = base_dir / "pipeline_summary.json"
    experiment_name = f"vallina_{args.dataset}_offline_ttl_seed_{args.seed}"

    train_command = [
        *python_module_command("-m", "llamafactory.cli", "train"),
        "--stage",
        "ttl",
        "--setting",
        "offline_ttl",
        "--model_name_or_path",
        args.model_name_or_path,
        "--finetuning_type",
        "lora",
        "--lora_target",
        "q_proj,v_proj",
        "--dataset",
        args.dataset,
        "--dataset_dir",
        args.dataset_dir,
        "--template",
        args.template,
        "--cutoff_len",
        str(train_profile.cutoff_len),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--lr_scheduler_type",
        str(args.lr_scheduler_type),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--threshold",
        str(args.threshold),
        "--lamb",
        str(args.lamb),
        "--seed",
        str(args.seed),
        "--do_train",
        "true",
        "--do_predict",
        "false",
        "--temperature",
        str(args.temperature),
        "--do_sample",
        "false",
        "--max_new_tokens",
        str(train_profile.max_new_tokens),
        "--output_dir",
        str(adapter_dir),
        "--run_name",
        experiment_name,
        "--logging_steps",
        str(args.logging_steps),
        "--save_steps",
        str(args.save_steps),
        "--overwrite_cache",
        "true",
        "--overwrite_output_dir",
        "true",
        "--plot_loss",
        "true",
        "--ddp_timeout",
        str(args.ddp_timeout),
        "--report_to",
        "none",
        "--preprocessing_num_workers",
        str(args.preprocessing_num_workers),
        "--trust_remote_code",
        "true",
    ]
    if args.bf16:
        train_command.extend(["--bf16", "true"])
    if args.max_samples is not None:
        train_command.extend(["--max_samples", str(args.max_samples)])
    if args.max_steps is not None:
        train_command.extend(["--max_steps", str(args.max_steps)])
    if args.use_swanlab:
        train_command.extend(
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
            train_command.extend(["--swanlab_workspace", args.swanlab_workspace])

    run_command(train_command, cwd=resolve_output_root("."), env=env)

    exported = False
    if not args.skip_export:
        export_command = [
            *python_module_command("-m", "llamafactory.cli", "export"),
            "--model_name_or_path",
            args.model_name_or_path,
            "--adapter_name_or_path",
            str(adapter_dir),
            "--finetuning_type",
            "lora",
            "--template",
            args.template,
            "--export_dir",
            str(export_dir),
            "--trust_remote_code",
            "true",
        ]
        run_command(export_command, cwd=resolve_output_root("."), env=env)
        exported = True

    uploaded = False
    repo_id = args.modelscope_repo_id or build_modescope_repo_id(args.modelscope_repo_prefix, experiment_name)
    if exported and not args.skip_upload:
        upload_command = [
            *python_module_command("scripts/upload_to_modelscope.py"),
            "--local-dir",
            str(export_dir),
            "--repo-id",
            repo_id,
        ]
        if args.modelscope_token:
            upload_command.extend(["--token", args.modelscope_token])
        if args.dry_run_upload:
            upload_command.append("--dry-run")
        run_command(upload_command, cwd=resolve_output_root("."), env=env)
        uploaded = not args.dry_run_upload

    summary = {
        "experiment_name": experiment_name,
        "model_role": "vallina_model",
        "run_tag": run_tag,
        "run_root": str(run_root),
        "train_dataset": args.dataset,
        "reference_clean_dataset": args.reference_clean_dataset,
        "adapter_dir": str(adapter_dir),
        "train_generation_profile": {
            "cutoff_len": train_profile.cutoff_len,
            "max_new_tokens": train_profile.max_new_tokens,
            "profile_name": train_profile.profile_name,
        },
        "hyperparameters": {
            "target_vram_gb": args.target_vram_gb,
            "baseline_vram_gb": args.baseline_vram_gb,
            "baseline_train_batch_size": args.baseline_train_batch_size,
            "baseline_eval_batch_size": args.baseline_eval_batch_size,
            "baseline_learning_rate": args.baseline_learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "num_train_epochs": args.num_train_epochs,
            "threshold": args.threshold,
            "lamb": args.lamb,
            "smoke_test": args.smoke_test,
        },
        "export_dir": str(export_dir) if exported else None,
        "modelscope_repo_id": repo_id if exported and not args.skip_upload else None,
        "uploaded": uploaded,
    }
    write_json(summary_path, summary)
    print(f"[done] Vallina training summary written to {summary_path}")


if __name__ == "__main__":
    main()
