#!/usr/bin/env bash
#
# 作用：
# - 为 4 个模型在 `gsm8k_mini` 上串行生成 `generated_predictions.jsonl`。
# - 生成前先调用 `build_gsm8k_mini_dataset.py` 构造并注册 `gsm8k_mini`。
# - 每轮生成结束后，立即调用 `TLM/scripts/eval/eval_ttl_mixed.py --task-type gsm8k`
#   计算 GSM8K 最终答案准确率。
#
# 依赖与调用关系：
# - 依赖 `do_as_I_do/scripts/build_data/build_gsm8k_mini_dataset.py`
# - 依赖 4 份 `gsm8k_mini` 预测 YAML
# - 依赖 `python -m llamafactory.cli train <yaml>` 作为底层预测入口
# - 依赖 `TLM/scripts/eval/eval_ttl_mixed.py` 计算 GSM8K accuracy
#
# 输入来源：
# - `TLM/data/AdaptEval/gsm8k_random_5k.json`
# - `linux_runtime_env.sh`
#
# 输出内容：
# - `do_as_I_do/saves/predict_gsm8k_mini/<model_alias>/gsm8k_mini/generated_predictions.jsonl`
# - `do_as_I_do/saves/predict_gsm8k_mini/<model_alias>/gsm8k_mini/gsm8k_accuracy.json`
#
# 对外接口：
# - `bash do_as_I_do/scripts/predict/run_gsm8k_mini_predict_suite.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TLM_DIR="${REPO_ROOT}/TLM"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${DO_AS_I_DO_CONDA_ENV:-TLM}"
RUNTIME_ENV_SH="${REPO_ROOT}/linux_runtime_env.sh"

run_one_prediction() {
  # 作用：
  # - 执行单份 YAML 预测并立即计算 gsm8k accuracy。
  # 输入：
  # - $1: YAML 相对路径
  # - $2: 输出目录相对路径
  # 输出：
  # - 对应目录下的 generated_predictions.jsonl 与 gsm8k_accuracy.json
  local yaml_rel="$1"
  local output_dir_rel="$2"

  python -m llamafactory.cli train "${yaml_rel}"
  python scripts/eval/eval_ttl_mixed.py \
    --prediction-file "${output_dir_rel}/generated_predictions.jsonl" \
    --task-type gsm8k \
    --output-json "${output_dir_rel}/gsm8k_accuracy.json"
}

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
source "${RUNTIME_ENV_SH}"

python "${REPO_ROOT}/do_as_I_do/scripts/build_data/build_gsm8k_mini_dataset.py"

cd "${TLM_DIR}"

run_one_prediction \
  "../do_as_I_do/examples/predict/base_model__gsm8k_mini_predict.yaml" \
  "../do_as_I_do/saves/predict_gsm8k_mini/base_model/gsm8k_mini"

run_one_prediction \
  "../do_as_I_do/examples/predict/gsm8k_requested_clean_model__gsm8k_mini_predict.yaml" \
  "../do_as_I_do/saves/predict_gsm8k_mini/gsm8k_requested_clean_model/gsm8k_mini"

run_one_prediction \
  "../do_as_I_do/examples/predict/gsm8k_AOA_model__gsm8k_mini_predict.yaml" \
  "../do_as_I_do/saves/predict_gsm8k_mini/gsm8k_AOA_model/gsm8k_mini"

run_one_prediction \
  "../do_as_I_do/examples/predict/gsm8k_vallina_AOA_model__gsm8k_mini_predict.yaml" \
  "../do_as_I_do/saves/predict_gsm8k_mini/gsm8k_vallina_AOA_model/gsm8k_mini"
