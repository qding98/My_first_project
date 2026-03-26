from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import (
    DEFAULT_MODELSCOPE_REPO_PREFIX,
    ROOT,
    build_modescope_repo_id,
    ensure_dataset_exists,
    make_env,
    python_module_command,
    run_command,
    run_controlled_eval,
    run_eval,
    task_type_from_dataset_name,
    ttl_predict_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed 40% mixed offline TTL experiment, then evaluate benign accuracy and the unified safety suite."
    )
    parser.add_argument("--dataset", required=True, help="Base clean AdaptEval dataset name, e.g. agriculture_5k")
    parser.add_argument("--model-name-or-path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-root", default="saves/pipelines/mixed40")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--hf-home", default="D:\\hf_cache")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="tlm-mixed40")
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
    mixed_dataset = f"{args.dataset}_advharm_40"
    ensure_dataset_exists(args.dataset)
    ensure_dataset_exists(mixed_dataset)

    if args.smoke_test:
        args.model_name_or_path = "llamafactory/tiny-random-Llama-3"
        args.cutoff_len = 64
        args.max_new_tokens = 8
        args.max_samples = 1
        args.max_steps = 1
        args.learning_rate = 1.0e-4
        args.threshold = 0.1
        args.use_swanlab = False
        args.skip_upload = True
        args.dry_run_upload = True

    env = make_env(args.hf_home)

    experiment_name = f"mixed40_{args.dataset}_offline_ttl"
    base_dir = ROOT / args.output_root / args.dataset
    adapter_dir = base_dir / "adapter"
    controlled_eval_dir = base_dir / "controlled_eval"
    export_dir = base_dir / "exported_model"
    metrics_dir = base_dir / "metrics"
    summary_path = base_dir / "pipeline_summary.json"

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
        mixed_dataset,
        "--eval_dataset",
        mixed_dataset,
        "--dataset_dir",
        args.dataset_dir,
        "--template",
        "llama3",
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
        "--threshold",
        str(args.threshold),
        "--lamb",
        str(args.lamb),
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
        "--overwrite_cache",
        "true",
        "--overwrite_output_dir",
        "true",
        "--plot_loss",
        "true",
        "--report_to",
        "none",
        "--preprocessing_num_workers",
        str(args.preprocessing_num_workers),
        "--trust_remote_code",
        "true",
    ]
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

    run_command(train_command, cwd=ROOT, env=env)

    mixed_prediction_file = ttl_predict_dir(adapter_dir, args.temperature, args.max_new_tokens) / "generated_predictions.jsonl"
    mixed_eval_json = metrics_dir / "mixed_eval.json"
    mixed_metrics = run_eval(
        mixed_prediction_file,
        mixed_eval_json,
        task_type=task_type_from_dataset_name(args.dataset),
        env=env,
    )

    controlled_eval = run_controlled_eval(
        model_path=adapter_dir,
        output_dir=controlled_eval_dir,
        env=env,
        base_model_path=args.model_name_or_path,
        max_samples=args.max_samples,
        smoke_test=args.smoke_test,
        hf_home=args.hf_home,
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
            "llama3",
            "--export_dir",
            str(export_dir),
            "--trust_remote_code",
            "true",
        ]
        run_command(export_command, cwd=ROOT, env=env)
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
        run_command(upload_command, cwd=ROOT, env=env)
        uploaded = not args.dry_run_upload

    summary = {
        "experiment_name": experiment_name,
        "clean_dataset": args.dataset,
        "mixed_dataset": mixed_dataset,
        "adapter_dir": str(adapter_dir),
        "mixed_prediction_file": str(mixed_prediction_file),
        "mixed_eval_metrics": mixed_metrics,
        "controlled_eval": controlled_eval,
        "export_dir": str(export_dir) if exported else None,
        "modelscope_repo_id": repo_id if exported and not args.skip_upload else None,
        "uploaded": uploaded,
    }
    write_json(summary_path, summary)
    print(f"[done] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
