# TLM Formal Commands

## Scope

This document only keeps the latest formal Linux commands.

- Clean-dataset eval now uses the original repo's native `do_eval + compute_accuracy` path.
- The clean metric artifact is `metrics/clean_eval.json`, copied from native `eval_results.json`.
- The key clean metric is `eval_<dataset>_accuracy`.
- Controlled safety eval is unchanged and still uses generation-based WildJailbreak evaluation.
- Export and upload are skipped in the commands below.

## Shared Linux Setup

These commands assume:

- A800 80GB
- `Qwen/Qwen2.5-7B-Instruct`
- active dataset profiles in `TLM/scripts/experiments/dataset_profiles.py`
- `TRAIN_BS=16`
- `EVAL_BS=16`
- `SAVE_STEPS=60`
- warm HF cache

```bash
cd /root/data/My_first_project/TLM

HF_HOME=/root/data/qsh/hf_cache
LR=0.00014
TRAIN_BS=16
EVAL_BS=16
GA_STEPS=1
SAVE_STEPS=60
SEED=42
NUM_WORKERS=8
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_ROOT="saves/serial_suites/requested_suite"
SWANLAB_PROJECT="TLM_new_A800"
SWANLAB_WORKSPACE="qding666"
BASE_MODEL_CONTROLLED_EVAL_SUMMARY="saves/serial_suites/requested_suite/lr_0.00014_bs_32_seed_42/agriculture_5k/base_model/controlled_eval/Qwen__Qwen2.5-7B-Instruct/wildjailbreak_controlled_eval_summary.json"

mkdir -p logs
```

## What To Read After Each Run

- `base_model/model_eval_summary.json`
- `clean_model/pipeline_summary.json`
- `mix_model/pipeline_summary.json`

In the clean eval section of those summaries, the native accuracy field will look like:

- `eval_agriculture_5k_accuracy`
- `eval_alpaca_gpt4_5k_accuracy`
- `eval_gsm8k_5k_accuracy`

## Recommended Order

1. `agriculture_5k`
2. `alpaca_gpt4_5k`
3. `gsm8k_5k`

Estimated A800 times:

| Dataset run | Estimated time |
| --- | --- |
| `agriculture_5k` clean + mixed | about `5.4-6.2 h` |
| `alpaca_gpt4_5k` clean + mixed | about `7.0-8.2 h` |
| `gsm8k_5k` clean + mixed | about `5.0-5.8 h` |

## Formal Runs

### Agriculture

Runs:

- base-model clean eval with native `ComputeAccuracy`
- reused base-model safety eval summary
- clean pipeline
- mixed40 pipeline

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python scripts/experiments/run_requested_ttl_serial_suite.py \
  --only-dataset agriculture_5k \
  --resume-from-step base_model_eval \
  --reuse-base-model-controlled-eval-summary "$BASE_MODEL_CONTROLLED_EVAL_SUMMARY" \
  --model-name-or-path "$MODEL_NAME" \
  --template qwen \
  --dataset-dir data \
  --hf-home "$HF_HOME" \
  --output-root "$OUTPUT_ROOT" \
  --learning-rate "$LR" \
  --per-device-train-batch-size "$TRAIN_BS" \
  --per-device-eval-batch-size "$EVAL_BS" \
  --gradient-accumulation-steps "$GA_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --preprocessing-num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --skip-export \
  --skip-upload \
  --use-swanlab \
  --swanlab-project "$SWANLAB_PROJECT" \
  --swanlab-workspace "$SWANLAB_WORKSPACE" \
  --swanlab-mode cloud \
  > "logs/agriculture_clean_mix_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.log" \
  2> "logs/agriculture_clean_mix_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.err" &
```

### Alpaca

Runs:

- base-model clean eval with native `ComputeAccuracy`
- reused base-model safety eval summary
- clean pipeline
- mixed40 pipeline

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python scripts/experiments/run_requested_ttl_serial_suite.py \
  --only-dataset alpaca_gpt4_5k \
  --resume-from-step base_model_eval \
  --reuse-base-model-controlled-eval-summary "$BASE_MODEL_CONTROLLED_EVAL_SUMMARY" \
  --model-name-or-path "$MODEL_NAME" \
  --template qwen \
  --dataset-dir data \
  --hf-home "$HF_HOME" \
  --output-root "$OUTPUT_ROOT" \
  --learning-rate "$LR" \
  --per-device-train-batch-size "$TRAIN_BS" \
  --per-device-eval-batch-size "$EVAL_BS" \
  --gradient-accumulation-steps "$GA_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --preprocessing-num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --skip-export \
  --skip-upload \
  --use-swanlab \
  --swanlab-project "$SWANLAB_PROJECT" \
  --swanlab-workspace "$SWANLAB_WORKSPACE" \
  --swanlab-mode cloud \
  > "logs/alpaca_clean_mix_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.log" \
  2> "logs/alpaca_clean_mix_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.err" &
```

### GSM8K

Runs:

- base-model clean eval with native `ComputeAccuracy`
- reused base-model safety eval summary
- clean pipeline
- mixed40 pipeline

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python scripts/experiments/run_requested_ttl_serial_suite.py \
  --only-dataset gsm8k_5k \
  --resume-from-step base_model_eval \
  --reuse-base-model-controlled-eval-summary "$BASE_MODEL_CONTROLLED_EVAL_SUMMARY" \
  --model-name-or-path "$MODEL_NAME" \
  --template qwen \
  --dataset-dir data \
  --hf-home "$HF_HOME" \
  --output-root "$OUTPUT_ROOT" \
  --learning-rate "$LR" \
  --per-device-train-batch-size "$TRAIN_BS" \
  --per-device-eval-batch-size "$EVAL_BS" \
  --gradient-accumulation-steps "$GA_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --preprocessing-num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --skip-export \
  --skip-upload \
  --use-swanlab \
  --swanlab-project "$SWANLAB_PROJECT" \
  --swanlab-workspace "$SWANLAB_WORKSPACE" \
  --swanlab-mode cloud \
  > "logs/gsm8k_clean_mix_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.log" \
  2> "logs/gsm8k_clean_mix_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.err" &
```
