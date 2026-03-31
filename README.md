# My First Project README

## What This Repo Is

This workspace is built around the original `TLM/` project and a local `safety-eval/` checkout.

The project is currently used for:

- running requested TTL serial pipelines on three clean datasets
- comparing `base_model`, `clean_model`, and `mix_model`
- evaluating clean-task performance with the original repo's native `ComputeAccuracy`
- evaluating safety with WildJailbreak controlled eval
- optionally rescoring existing generated predictions offline with `safety-eval`

The three main clean datasets in active use are:

- `agriculture_5k`
- `alpaca_gpt4_5k`
- `gsm8k_5k`

## Current Stable Workflow

The current stable formal run root is:

- [lr_0.0001_bs_16_seed_42](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42)

The current recommended formal commands are documented in:

- [commands.md](d:/Qsh的个人资料/科研/LLM/My_first_project/commands.md)

The current experiment summary is documented in:

- [experiment_results.md](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/docs/experiment_results.md)

## Important Local Changes vs Upstream

This workspace is no longer using the earlier custom clean-task post-eval as the main clean metric.

Current behavior:

- clean-dataset eval now uses the original repo's native `do_eval + compute_accuracy`
- clean eval no longer uses the earlier `train_dataset_eval.json` duplicate path
- each model role now performs clean eval only once
- clean metrics are copied from native `eval_results.json` into `metrics/clean_eval.json`
- the main clean metric is `eval_<dataset>_accuracy`

This affects:

- [pipeline_common.py](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/scripts/experiments/pipeline_common.py)
- [run_clean_ttl_pipeline.py](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/scripts/experiments/run_clean_ttl_pipeline.py)
- [run_mixed40_ttl_pipeline.py](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/scripts/experiments/run_mixed40_ttl_pipeline.py)
- [run_requested_ttl_serial_suite.py](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/scripts/experiments/run_requested_ttl_serial_suite.py)

## Current Evaluation Setup

### Clean Eval

For `base_model`, `clean_model`, and `mix_model`, clean eval now uses the original repo native path:

- `do_eval=true`
- `do_predict=false`
- `predict_with_generate=false`
- `compute_accuracy=true`

So when reading results, the key file is:

- `metrics/clean_eval.json`

And the key field is one of:

- `eval_agriculture_5k_accuracy`
- `eval_alpaca_gpt4_5k_accuracy`
- `eval_gsm8k_5k_accuracy`

### Controlled Safety Eval

Controlled safety eval is still generation-based and uses the existing WildJailbreak pipeline.

The main summary file for each model is:

- `controlled_eval/.../wildjailbreak_controlled_eval_summary.json`

Main safety fields to watch:

- adversarial harmful ASR
- jailbreak lift
- benign refusal rate

### Offline Safety-Eval Rescoring

There is also an offline rescoring script that reads existing `generated_predictions.jsonl` files and calls a `safety-eval` classifier instead of regenerating model outputs:

- [run_safetyeval_on_predictions.py](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/scripts/eval/run_safetyeval_on_predictions.py)

This script currently:

- keeps the existing directory structure assumption
- reads already generated prediction files
- extracts the final `user` prompt from templated prompts before classification
- supports parse-error accounting
- supports stricter harmful success logic with `compliance_and_harmful`
- can save per-sample classifier annotations for audit

## Dataset Profiles

Generation and training profile logic is centralized in:

- [dataset_profiles.py](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/scripts/experiments/dataset_profiles.py)

This file controls:

- `cutoff_len`
- `max_new_tokens`
- dataset-specific profile selection

## Main Result Directory

For the latest stable formal result set, read under:

- [lr_0.0001_bs_16_seed_42](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42)

Typical files to inspect:

- `base_model/model_eval_summary.json`
- `clean_model/pipeline_summary.json`
- `mix_model/pipeline_summary.json`
- `metrics/clean_eval.json`
- `controlled_eval/.../wildjailbreak_controlled_eval_summary.json`

## Current Result Snapshot

From the current `lr_0.0001_bs_16_seed_42` run:

- clean TTL improves token-level accuracy on all three datasets, but the gains are modest
- mix40 usually keeps accuracy close to clean TTL, with safety tradeoffs depending on dataset
- `alpaca_gpt4_5k` shows the clearest clean-task improvement
- `gsm8k_5k` mix greatly suppresses harmful ASR but becomes overly refusal-heavy
- `agriculture_5k` still shows noticeable jailbreak lift

For the full written interpretation, use:

- [experiment_results.md](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/docs/experiment_results.md)

## Commands

For formal Linux commands, use:

- [commands.md](d:/Qsh的个人资料/科研/LLM/My_first_project/commands.md)

That file is intended to be the single source of truth for:

- current formal commands
- current batch sizes
- current save/eval assumptions

## Recommended Starting Points For Future Chats

If a future chat needs fast context, point it to these files first:

1. [README.md](d:/Qsh的个人资料/科研/LLM/My_first_project/README.md)
2. [commands.md](d:/Qsh的个人资料/科研/LLM/My_first_project/commands.md)
3. [pipeline_common.py](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/scripts/experiments/pipeline_common.py)
4. [dataset_profiles.py](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/scripts/experiments/dataset_profiles.py)
5. [experiment_results.md](d:/Qsh的个人资料/科研/LLM/My_first_project/TLM/docs/experiment_results.md)

## Notes

- The upstream project README inside `TLM/README.md` is still useful for the original paper/repo context.
- This root README is intentionally focused on the current local workflow and your recent modifications.
- If the evaluation logic changes again, update this file together with `commands.md`.
