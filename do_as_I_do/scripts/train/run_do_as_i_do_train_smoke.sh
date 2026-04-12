#!/usr/bin/env bash
# 作用：
# - 使用最小 smoke YAML 对 Do_as_I_do 第一轮训练链路做一次快速冒烟验证。
# - 同时验证 `linux_runtime_env.sh` 中的 HF / SwanLab 环境变量是否被正确导入。
#
# 依赖与调用关系：
# - 依赖 `do_as_I_do/examples/train/gsm8k_AOA_train_smoke.yaml`
# - 依赖 `python -m llamafactory.cli train <yaml>` 作为底层训练入口。
# - 依赖仓库根目录 `linux_runtime_env.sh` 提供运行环境变量。
#
# 输入来源：
# - 当前仓库内的 smoke YAML 与 `linux_runtime_env.sh`。
#
# 输出内容：
# - 训练日志写入 `do_as_I_do/logs/run_do_as_i_do_train_smoke.log`
# - 错误日志写入 `do_as_I_do/logs/run_do_as_i_do_train_smoke.err`
#
# 对外接口：
# - `bash do_as_I_do/scripts/train/run_do_as_i_do_train_smoke.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TLM_DIR="${REPO_ROOT}/TLM"
LOG_DIR="${REPO_ROOT}/do_as_I_do/logs"
RUNTIME_ENV_SH="${REPO_ROOT}/linux_runtime_env.sh"

mkdir -p "${LOG_DIR}"

source "${RUNTIME_ENV_SH}"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-${SWANLAB_PROJECT_NAME:-${SWANLAB_PROJECT:-do_as_i_do}}}"

cd "${TLM_DIR}"
python -m llamafactory.cli train "../do_as_I_do/examples/train/gsm8k_AOA_train_smoke.yaml" \
  > "${LOG_DIR}/run_do_as_i_do_train_smoke.log" \
  2> "${LOG_DIR}/run_do_as_i_do_train_smoke.err"
