#!/usr/bin/env bash
#
# 作用：
# - 为多个单独的 `generated_predictions.jsonl` 串行调用
#   `run_single_prediction_safety_eval.py` 提供最小 shell 模板。
# - 仅负责激活 `safety-eval` conda 环境、加载运行时环境变量、循环执行单文件评测。
#
# 输入来源：
# - 手动在 `TASKS` 数组里填写 `model_alias|dataset_name|prediction_file`。
#
# 输出内容：
# - 每个任务各自写到 `do_as_I_do/saves/safety-eval-results/<model_alias>/<dataset_name>/`
#
# 对外接口：
# - `bash do_as_I_do/scripts/eval/run_single_prediction_safety_eval_serial_template.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${DO_AS_I_DO_SAFETY_EVAL_CONDA_ENV:-safety-eval}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
source "${REPO_ROOT}/linux_runtime_env.sh"

TASKS=(
  "gsm8k_AOA_model|gsm8k_AOA|do_as_I_do/saves/predict_smoke/gsm8k_AOA_model/gsm8k_AOA/generated_predictions.jsonl"
)

for task in "${TASKS[@]}"; do
  IFS='|' read -r model_alias dataset_name prediction_file <<< "${task}"
  python "${REPO_ROOT}/do_as_I_do/scripts/eval/run_single_prediction_safety_eval.py" \
    --safety-eval-root "${REPO_ROOT}/safety-eval" \
    --results-root "${REPO_ROOT}/do_as_I_do/saves/safety-eval-results" \
    --classifier-model-name WildGuard \
    --classifier-batch-size 8 \
    --classifier-ephemeral-model \
    --save-per-sample-results \
    --model-alias "${model_alias}" \
    --dataset-name "${dataset_name}" \
    --prediction-file "${REPO_ROOT}/${prediction_file}"
done
