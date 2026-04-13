#!/usr/bin/env bash
#
# 作用：
# - 对 `base_model` 和 `gsm8k_clean_model` 在 `vallina_harmful_AOA_mini` 与 `villina_mixed_mini` 上串行执行 predict。
# - predict 完成后，切换到 `safety-eval` 环境，对 4 份 `generated_predictions.jsonl` 做离线 safety-eval。
# - 统一封装为一个 Linux 启动脚本，避免手动切换环境和逐条执行。
#
# 依赖与调用关系：
# - 依赖两份 `predict` YAML：
#   - `base_model__vallina_harmful_AOA_mini_predict.yaml`
#   - `gsm8k_clean_model__vallina_harmful_AOA_mini_predict.yaml`
#   - `base_model__villina_mixed_mini_predict.yaml`
#   - `gsm8k_clean_model__villina_mixed_mini_predict.yaml`
# - 依赖 `do_as_I_do/scripts/eval/run_single_prediction_safety_eval.py`
# - 依赖 `python -m llamafactory.cli train <yaml>` 做 predict
#
# 输入来源：
# - `do_as_I_do/data/vallina_harmful_AOA_mini.json`
# - `do_as_I_do/data/villina_mixed_mini.json`
# - `linux_runtime_env.sh`
#
# 输出内容：
# - predict:
#   - `do_as_I_do/saves/predict/<model_alias>/vallina_harmful_AOA/generated_predictions.jsonl`
#   - `do_as_I_do/saves/predict/<model_alias>/villina_mixed/generated_predictions.jsonl`
# - safety-eval:
#   - `do_as_I_do/saves/safety-eval-results/<model_alias>/vallina_harmful_AOA/summary.json`
#   - `do_as_I_do/saves/safety-eval-results/<model_alias>/villina_mixed/summary.json`
#   - `.../safety_eval_predictions_with_labels.jsonl`
#
# 对外接口：
# - `bash do_as_I_do/scripts/predict/run_vallina_harmful_aoa_mini_base_clean_predict_and_safety_eval.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TLM_DIR="${REPO_ROOT}/TLM"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
TLM_CONDA_ENV_NAME="${DO_AS_I_DO_CONDA_ENV:-TLM}"
SAFETY_EVAL_CONDA_ENV_NAME="${DO_AS_I_DO_SAFETY_EVAL_CONDA_ENV:-safety-eval}"
RUNTIME_ENV_SH="${REPO_ROOT}/linux_runtime_env.sh"
PREDICT_ROOT="${REPO_ROOT}/do_as_I_do/saves/predict"
SAFETY_EVAL_RESULTS_ROOT="${REPO_ROOT}/do_as_I_do/saves/safety-eval-results"

run_prediction() {
  # 作用：
  # - 在 TLM 环境中执行单份 predict YAML。
  # 输入：
  # - $1: YAML 相对路径
  local yaml_rel="$1"
  python -m llamafactory.cli train "${yaml_rel}"
}

run_single_safety_eval() {
  # 作用：
  # - 对单个 prediction 文件执行 WildGuard safety-eval。
  # 输入：
  # - $1: 模型别名
  # - $2: dataset_name
  # - $3: eval_dataset
  # - $4: prediction 文件绝对路径
  local model_alias="$1"
  local dataset_name="$2"
  local eval_dataset="$3"
  local prediction_file="$4"

  python "${REPO_ROOT}/do_as_I_do/scripts/eval/run_single_prediction_safety_eval.py" \
    --safety-eval-root "${REPO_ROOT}/safety-eval" \
    --results-root "${SAFETY_EVAL_RESULTS_ROOT}" \
    --classifier-model-name WildGuard \
    --classifier-batch-size 16 \
    --no-classifier-ephemeral-model \
    --save-per-sample-results \
    --model-alias "${model_alias}" \
    --dataset-name "${dataset_name}" \
    --eval-dataset "${eval_dataset}" \
    --prediction-file "${prediction_file}"
}

source "${CONDA_SH}"
conda activate "${TLM_CONDA_ENV_NAME}"
source "${RUNTIME_ENV_SH}"

cd "${TLM_DIR}"
run_prediction "../do_as_I_do/examples/predict/base_model__vallina_harmful_AOA_mini_predict.yaml"
run_prediction "../do_as_I_do/examples/predict/gsm8k_clean_model__vallina_harmful_AOA_mini_predict.yaml"
run_prediction "../do_as_I_do/examples/predict/base_model__villina_mixed_mini_predict.yaml"
run_prediction "../do_as_I_do/examples/predict/gsm8k_clean_model__villina_mixed_mini_predict.yaml"

conda activate "${SAFETY_EVAL_CONDA_ENV_NAME}"
source "${RUNTIME_ENV_SH}"

run_single_safety_eval \
  "base_model" \
  "vallina_harmful_AOA" \
  "vallina_harmful_AOA_mini" \
  "${PREDICT_ROOT}/base_model/vallina_harmful_AOA/generated_predictions.jsonl"

run_single_safety_eval \
  "gsm8k_clean_model" \
  "vallina_harmful_AOA" \
  "vallina_harmful_AOA_mini" \
  "${PREDICT_ROOT}/gsm8k_clean_model/vallina_harmful_AOA/generated_predictions.jsonl"

run_single_safety_eval \
  "base_model" \
  "villina_mixed" \
  "villina_mixed_mini" \
  "${PREDICT_ROOT}/base_model/villina_mixed/generated_predictions.jsonl"

run_single_safety_eval \
  "gsm8k_clean_model" \
  "villina_mixed" \
  "villina_mixed_mini" \
  "${PREDICT_ROOT}/gsm8k_clean_model/villina_mixed/generated_predictions.jsonl"

python "${REPO_ROOT}/do_as_I_do/scripts/eval/summarize_base_clean_vallina_villina_results.py"
