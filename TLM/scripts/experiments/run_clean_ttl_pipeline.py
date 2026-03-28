from __future__ import annotations

import argparse

from pipeline_common import (
    DEFAULT_FORMAL_MODEL_NAME_OR_PATH,
    DEFAULT_FORMAL_TEMPLATE,
    DEFAULT_MODELSCOPE_REPO_PREFIX,
    apply_smoke_test_overrides,
    build_modescope_repo_id,
    ensure_dataset_exists,
    make_env,
    python_module_command,
    resolve_output_root,
    run_command,
    run_eval,
    run_model_eval_suite,
    task_type_from_dataset_name,
    ttl_predict_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the clean offline TTL model, then evaluate it on the requested clean and safety datasets."
    )
    parser.add_argument("--dataset", required=True, help="Clean AdaptEval dataset name, e.g. agriculture_5k")
    parser.add_argument("--model-name-or-path", default=DEFAULT_FORMAL_MODEL_NAME_OR_PATH)
    parser.add_argument("--template", default=DEFAULT_FORMAL_TEMPLATE)
    parser.add_argument("--output-root", default="saves/pipelines/clean")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="D:\\hf_cache")
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
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="tlm-clean-baseline")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run-upload", action="store_true")
    parser.add_argument("--modelscope-repo-id", default=None)
    parser.add_argument("--modelscope-repo-prefix", default=DEFAULT_MODELSCOPE_REPO_PREFIX)
    parser.add_argument("--modelscope-token", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dataset_exists(args.dataset)

    if args.smoke_test:
        apply_smoke_test_overrides(args)

    env = make_env(args.hf_home)

    output_root = resolve_output_root(args.output_root)
    base_dir = output_root / args.dataset / "clean_model"
    adapter_dir = base_dir / "adapter"
    export_dir = base_dir / "exported_model"
    metrics_dir = base_dir / "metrics"
    summary_path = base_dir / "pipeline_summary.json"

    experiment_name = f"clean_{args.dataset}_offline_ttl_seed_{args.seed}"
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
        "--eval_dataset",
        args.dataset,
        "--dataset_dir",
        args.dataset_dir,
        "--template",
        args.template,
        "--cutoff_len",
        str(args.cutoff_len),
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
        "true",
        "--predict_with_generate",
        "true",
        "--temperature",
        str(args.temperature),
        "--do_sample",
        "false",
        "--max_new_tokens",
        str(args.max_new_tokens),
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

    clean_prediction_file = ttl_predict_dir(adapter_dir, args.temperature, args.max_new_tokens) / "generated_predictions.jsonl"
    clean_eval_json = metrics_dir / "train_dataset_eval.json"
    clean_train_metrics = run_eval(
        clean_prediction_file,
        clean_eval_json,
        task_type=task_type_from_dataset_name(args.dataset),
        env=env,
    )

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

    model_eval_summary = run_model_eval_suite(
        model_path=adapter_dir,
        model_role_dir=base_dir,
        clean_dataset=args.dataset,
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
        clean_prediction_file=clean_prediction_file,
    )

    summary = {
        "experiment_name": experiment_name,
        "model_role": "clean_model",
        "clean_dataset": args.dataset,
        "train_dataset": args.dataset,
        "adapter_dir": str(adapter_dir),
        "train_dataset_prediction_file": str(clean_prediction_file),
        "train_dataset_metrics": clean_train_metrics,
        "model_eval": model_eval_summary,
        "export_dir": str(export_dir) if exported else None,
        "modelscope_repo_id": repo_id if exported and not args.skip_upload else None,
        "uploaded": uploaded,
    }
    write_json(summary_path, summary)
    print(f"[done] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
