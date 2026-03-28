# TLM Run Commands

## Shared Linux Setup

These Linux commands assume:

- A800 80GB
- `Qwen/Qwen2.5-7B-Instruct`
- active dataset profiles in `TLM/scripts/experiments/dataset_profiles.py`
- `TRAIN_BS=16`
- `EVAL_BS=16`
- `SAVE_STEPS=60`
- `--skip-export`
- `--skip-upload`
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

## A800 Time Estimates

These estimates are for the commands below on an A800 with the current profiles and `16 / 16` batch sizes.

| Command | What it runs | Estimated A800 time |
| --- | --- | --- |
| Agriculture clean + mixed | skip base-model eval, run clean pipeline + mixed40 pipeline | about `5.0-5.8 h` |
| Alpaca run with reused base-model safety eval | run base-model clean eval, reuse existing safety eval, then run clean pipeline + mixed40 pipeline | about `7.0-8.2 h` |
| GSM8K run with reused base-model safety eval | run base-model clean eval, reuse existing safety eval, then run clean pipeline + mixed40 pipeline | about `5.0-5.8 h` |
| Agriculture mixed only | skip base-model eval and clean pipeline, rerun mixed40 only | about `2.0-2.5 h` |

## Notes

- In the pulled formal Qwen save, only `agriculture_5k` already has a real `base_model` evaluation.
- The pulled formal Qwen `agriculture_5k` run also already has a real base-model safety eval summary at `$BASE_MODEL_CONTROLLED_EVAL_SUMMARY`.
- `alpaca_gpt4_5k` and `gsm8k_5k` only have smoke `tiny-random-Llama-3` base-model evaluations, which should not be reused for the formal Qwen run.
- So the save-aware commands below:
- skip duplicate base-model eval entirely for `agriculture_5k`
- reuse the existing formal Qwen base-model safety eval for `alpaca_gpt4_5k` and `gsm8k_5k`
- still run their own base-model clean-dataset evaluation

## Recommended Order

If you want to continue from the current pulled formal save with the least duplicated work, run in this order:

1. `agriculture_5k` mixed only
2. `alpaca_gpt4_5k` full dataset run with reused base-model safety eval
3. `gsm8k_5k` full dataset run with reused base-model safety eval

Recommended timing plan on A800:

| Step | Recommended command | Estimated A800 time |
| --- | --- | --- |
| 1 | Agriculture mixed only | about `2.0-2.5 h` |
| 2 | Alpaca run with reused base-model safety eval | about `7.0-8.2 h` |
| 3 | GSM8K run with reused base-model safety eval | about `5.0-5.8 h` |

Shutdown planning:

- If you only run `agriculture mixed only`, set auto shutdown for about `4 hours` after start.
- If you rerun `agriculture clean + mixed` from scratch, set auto shutdown for about `7 hours` after start.
- The extra buffer covers checkpoint save time, log flush, cache jitter, and a bit of queueing overhead.

Linux example for a shutdown scheduled 4 hours later:

```bash
sudo shutdown -h +240
```

Cancel a scheduled shutdown:

```bash
sudo shutdown -c
```

## Linux Formal Runs

### 1. Agriculture clean + mixed

Estimated A800 time: `5.0-5.8 h`

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python scripts/experiments/run_requested_ttl_serial_suite.py \
  --only-dataset agriculture_5k \
  --resume-from-step clean_pipeline \
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

### 2. Alpaca clean + mixed

Estimated A800 time: `7.0-8.2 h`

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

### 3. GSM8K clean + mixed

Estimated A800 time: `5.0-5.8 h`

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

## Optional Linux Resume

### Agriculture mixed only

Use this only if the agriculture clean pipeline already finished and you only want to rerun the mixed40 stage after the earlier OOM.

Estimated A800 time: `2.0-2.5 h`

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python scripts/experiments/run_requested_ttl_serial_suite.py \
  --only-dataset agriculture_5k \
  --resume-from-step mixed_pipeline \
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
  > "logs/agriculture_mixed_only_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.log" \
  2> "logs/agriculture_mixed_only_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.err" &
```

## Windows PowerShell Smoke

Smoke mode here is still useful when you want to validate the current code path on Windows.

```powershell
Set-Location .\TLM

$env:HF_HOME = "D:\hf_cache"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$LR = "0.0001"
$TRAIN_BS = "1"
$EVAL_BS = "1"
$GA_STEPS = "1"
$SEED = "42"
$NUM_WORKERS = "1"
$MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
$OUTPUT_ROOT = "saves/serial_suites/requested_suite_smoke"
$SWANLAB_PROJECT = "TLM_windows_smoke"
$SWANLAB_WORKSPACE = "qding666"

New-Item -ItemType Directory -Force logs | Out-Null

python scripts/experiments/run_requested_ttl_serial_suite.py `
  --smoke-test `
  --model-name-or-path $MODEL_NAME `
  --hf-home $env:HF_HOME `
  --output-root $OUTPUT_ROOT `
  --per-device-train-batch-size $TRAIN_BS `
  --per-device-eval-batch-size $EVAL_BS `
  --gradient-accumulation-steps $GA_STEPS `
  --preprocessing-num-workers $NUM_WORKERS `
  --seed $SEED `
  --use-swanlab `
  --swanlab-project $SWANLAB_PROJECT `
  --swanlab-workspace $SWANLAB_WORKSPACE `
  --swanlab-mode cloud `
  *>&1 | Tee-Object -FilePath "logs\requested_suite_smoke_export_upload.log"
```
