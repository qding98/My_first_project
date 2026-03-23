import argparse
import gc
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    HAS_XLA = True
except ImportError:
    xm = None
    xmp = None
    HAS_XLA = False


DATA_TYPES = [
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
]


@dataclass
class RuntimeContext:
    device: torch.device
    device_type: str
    rank: int
    world_size: int
    is_master: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run llm-tta on CPU/CUDA/XLA.")
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--dataset-name",
        default="allenai/wildjailbreak",
        help="Dataset name for Hugging Face datasets.",
    )
    parser.add_argument(
        "--dataset-config",
        default="train",
        help="Dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split.",
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=200,
        help="Number of samples to draw from each data_type.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum token length for both prompt and response processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "xla"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Model load precision.",
    )
    parser.add_argument(
        "--output-dir",
        default="save",
        help="Directory to store csv and figures.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. If omitted, one is generated automatically.",
    )
    parser.add_argument(
        "--use-xla-spawn",
        action="store_true",
        help="Spawn multiple XLA worker processes with torch_xla.",
    )
    parser.add_argument(
        "--xla-processes",
        type=int,
        default=1,
        help="Number of XLA worker processes to spawn.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip saving plots and only export CSV.",
    )
    return parser.parse_args()


def sanitize_model_name(model_name: str) -> str:
    return model_name.lower().replace("/", "_").replace(":", "_")


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    return f"results_xla_{sanitize_model_name(args.model_name)}_{ts}"


def resolve_runtime(args: argparse.Namespace) -> RuntimeContext:
    requested = args.device
    if requested == "auto":
        if HAS_XLA and os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
            requested = "xla"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"

    if requested == "xla":
        if not HAS_XLA:
            raise RuntimeError("torch_xla is not installed, cannot use --device xla.")
        device = xm.xla_device()
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        return RuntimeContext(
            device=device,
            device_type="xla",
            rank=rank,
            world_size=world_size,
            is_master=xm.is_master_ordinal(),
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use --device cuda.")
        return RuntimeContext(
            device=torch.device("cuda"),
            device_type="cuda",
            rank=0,
            world_size=1,
            is_master=True,
        )

    return RuntimeContext(
        device=torch.device("cpu"),
        device_type="cpu",
        rank=0,
        world_size=1,
        is_master=True,
    )


def get_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_sampled_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        delimiter="\t",
        keep_default_na=False,
    )

    subsets = []
    for idx, dtype in enumerate(DATA_TYPES):
        subset = ds.filter(lambda x, dtype=dtype: x["data_type"] == dtype)
        subset = subset.shuffle(seed=args.seed + idx).select(range(args.samples_per_type))
        subsets.append(subset)

    sampled_ds = concatenate_datasets(subsets)
    return [sampled_ds[i] for i in range(len(sampled_ds))]


def get_sample_texts(sample: Dict[str, str]) -> Dict[str, str]:
    question_text = sample["adversarial"] if "adversarial" in sample["data_type"] else sample["vanilla"]
    expected_answer_text = sample["completion"]
    return {
        "question_text": question_text,
        "expected_answer_text": expected_answer_text,
        "base_question_text": sample["vanilla"],
    }


def load_model_and_tokenizer(args: argparse.Namespace, ctx: RuntimeContext) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    torch_dtype = get_torch_dtype(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch_dtype)
    model.to(ctx.device)
    model.eval()
    return tokenizer, model


def compute_grad_dot(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    x_text: str,
    y_text: str,
    device: torch.device,
    max_len: int,
    is_xla: bool,
) -> Tuple[float, float, float, int, int]:
    x_ids = tokenizer(x_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    n_x = x_ids.size(1)

    model.zero_grad(set_to_none=True)
    loss_x = model(input_ids=x_ids, labels=x_ids.clone()).loss
    loss_x.backward()
    if is_xla:
        xm.mark_step()
    loss_x_val = float(loss_x.detach().cpu().item())

    grads_x = {}
    bad_x = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            bad_x.append(name)
            continue
        grads_x[name] = grad.float().clone()

    model.zero_grad(set_to_none=True)
    del loss_x

    if bad_x:
        del grads_x
        return float("nan"), loss_x_val, float("nan"), n_x, 0

    y_ids = tokenizer(
        y_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).input_ids.to(device)
    xy_ids = torch.cat([x_ids, y_ids], dim=1)[:, :max_len]
    n_y = xy_ids.size(1) - n_x
    if n_y <= 0:
        del grads_x
        return float("nan"), loss_x_val, float("nan"), n_x, n_y

    labels = xy_ids.clone()
    labels[0, :n_x] = -100

    model.zero_grad(set_to_none=True)
    loss_yx = model(input_ids=xy_ids, labels=labels).loss
    loss_yx.backward()
    if is_xla:
        xm.mark_step()
    loss_yx_val = float(loss_yx.detach().cpu().item())

    dot = 0.0
    for name, param in model.named_parameters():
        if param.grad is None or name not in grads_x:
            continue
        gy_raw = param.grad.detach()
        if not torch.isfinite(gy_raw).all():
            continue
        gx = grads_x[name]
        gy = gy_raw.float()
        dot += (gx * gy).sum().detach().cpu().item()

    model.zero_grad(set_to_none=True)
    del grads_x
    return float(dot), loss_x_val, loss_yx_val, n_x, n_y


def save_plots(df: pd.DataFrame, output_dir: Path, run_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["dot_product"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_title("Gradient Dot Product Distribution")
    ax.set_xlabel("dot_product")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_name}_gradient_dot_product.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = (
        df.groupby("data_type")["dot_product"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["data_type"], summary["mean"], color="steelblue")
    ax.set_title("Mean Gradient Dot Product by Type")
    ax.set_xlabel("data_type")
    ax.set_ylabel("mean dot_product")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_name}_by_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_master(ctx: RuntimeContext, message: str) -> None:
    if ctx.is_master:
        print(message)


def get_shard_indices(total_size: int, rank: int, world_size: int) -> List[int]:
    return [idx for idx in range(total_size) if idx % world_size == rank]


def export_partial_results(partial_df: pd.DataFrame, output_dir: Path, run_name: str, rank: int) -> Path:
    tmp_dir = output_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    partial_path = tmp_dir / f"{run_name}.rank{rank}.csv"
    partial_df.to_csv(partial_path, index=False, encoding="utf-8-sig")
    return partial_path


def combine_partial_results(ctx: RuntimeContext, output_dir: Path, run_name: str, world_size: int, skip_plots: bool) -> Path:
    tmp_dir = output_dir / ".tmp"
    partial_paths = [tmp_dir / f"{run_name}.rank{rank}.csv" for rank in range(world_size)]
    frames = [pd.read_csv(path) for path in partial_paths]
    df = pd.concat(frames, ignore_index=True).sort_values("sample_idx").reset_index(drop=True)
    final_path = output_dir / f"{run_name}.csv"
    df.to_csv(final_path, index=False, encoding="utf-8-sig")
    if not skip_plots:
        save_plots(df, output_dir, run_name)
    return final_path


def run_experiment(local_rank: int, args: argparse.Namespace) -> None:
    del local_rank
    ctx = resolve_runtime(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_master(ctx, f"Using device: {ctx.device_type}")
    print_master(ctx, f"Using model: {args.model_name}")
    print_master(ctx, f"Using run name: {args.run_name}")

    sampled_rows = load_sampled_rows(args)
    shard_indices = get_shard_indices(len(sampled_rows), ctx.rank, ctx.world_size)
    tokenizer, model = load_model_and_tokenizer(args, ctx)

    results = []
    iterator = tqdm(shard_indices, desc=f"rank{ctx.rank}: gradient dots", disable=not ctx.is_master)
    for sample_idx in iterator:
        sample = sampled_rows[sample_idx]
        sample_texts = get_sample_texts(sample)
        dot, lx, lyx, nx, ny = compute_grad_dot(
            model=model,
            tokenizer=tokenizer,
            x_text=sample_texts["question_text"],
            y_text=sample_texts["expected_answer_text"],
            device=ctx.device,
            max_len=args.max_len,
            is_xla=ctx.device_type == "xla",
        )

        results.append(
            {
                "sample_idx": sample_idx,
                "data_type": sample["data_type"],
                "question_text": sample_texts["question_text"],
                "base_question_text": sample_texts["base_question_text"],
                "expected_answer_text": sample_texts["expected_answer_text"],
                "perplexity_grad_dot_product": dot,
                "dot_product": dot,
                "sign": "positive" if dot > 0 else "negative",
                "loss_x": lx,
                "loss_y_given_x": lyx,
                "n_tokens_x": nx,
                "n_tokens_y": ny,
            }
        )

        if ctx.device_type == "cuda":
            torch.cuda.empty_cache()
        elif ctx.device_type == "xla":
            xm.mark_step()

        if len(results) % 20 == 0:
            gc.collect()

    partial_df = pd.DataFrame(results)
    partial_path = export_partial_results(partial_df, output_dir, args.run_name, ctx.rank)
    print_master(ctx, f"Partial results saved under: {partial_path.parent}")

    if ctx.device_type == "xla":
        xm.rendezvous("partial_results_ready")
        if ctx.is_master:
            final_path = combine_partial_results(ctx, output_dir, args.run_name, ctx.world_size, args.skip_plots)
            print(f"Final CSV saved to: {final_path}")
        xm.rendezvous("combine_complete")
    else:
        final_path = combine_partial_results(ctx, output_dir, args.run_name, ctx.world_size, args.skip_plots)
        print(f"Final CSV saved to: {final_path}")


def main() -> None:
    args = parse_args()
    args.run_name = build_run_name(args)

    if args.device == "xla" and args.use_xla_spawn:
        if not HAS_XLA:
            raise RuntimeError("torch_xla is not installed, cannot use --use-xla-spawn.")
        xmp.spawn(run_experiment, args=(args,), nprocs=args.xla_processes, start_method="fork")
        return

    run_experiment(local_rank=0, args=args)


if __name__ == "__main__":
    main()
