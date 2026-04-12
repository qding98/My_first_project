#!/usr/bin/env bash
# 作用：
# - 串行启动两次 Do_as_I_do offline TTL 训练。
# - 第一轮用 gsm8k_AOA 数据训练 `gsm8k_AOA_model` 对应 adapter；
# - 第二轮在第一轮 adapter 基础上继续用 vallina_harmful_AOA 数据训练。
#
# 依赖与调用关系：
# - 依赖 `do_as_I_do/examples/train/gsm8k_AOA_train.yaml`
# - 依赖 `do_as_I_do/examples/train/gsm8k_vallina_AOA_train.yaml`
# - 依赖 `python -m llamafactory.cli train <yaml>` 作为底层训练入口。
#
# 输入来源：
# - 当前仓库内的两份训练 YAML。
# - 仓库根目录 `linux_runtime_env.sh` 中预先配置的 HF / SwanLab 环境变量。
#
# 输出内容：
# - 训练日志写入 `do_as_I_do/logs/run_do_as_i_do_train_pair.log`
# - 错误日志写入 `do_as_I_do/logs/run_do_as_i_do_train_pair.err`
#
# 对外接口：
# - `bash do_as_I_do/scripts/train/run_do_as_i_do_train_pair.sh`
# - 执行后默认以 `nohup ... &` 后台运行，并打印后台进程 PID。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TLM_DIR="${REPO_ROOT}/TLM"
LOG_DIR="${REPO_ROOT}/do_as_I_do/logs"
RUNTIME_ENV_SH="${REPO_ROOT}/linux_runtime_env.sh"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${DO_AS_I_DO_CONDA_ENV:-TLM}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${CONDA_SH}" ]]; then
	echo "未找到 conda 初始化脚本: ${CONDA_SH}" >&2
	exit 1
fi

nohup bash -lc "source '${CONDA_SH}' && conda activate '${CONDA_ENV_NAME}' && source '${RUNTIME_ENV_SH}' && export SWANLAB_PROJ_NAME=\"\${SWANLAB_PROJ_NAME:-\${SWANLAB_PROJECT_NAME:-\${SWANLAB_PROJECT:-do_as_i_do}}}\" && cd '${TLM_DIR}' && python -m llamafactory.cli train '../do_as_I_do/examples/train/gsm8k_AOA_train.yaml' && python -m llamafactory.cli train '../do_as_I_do/examples/train/gsm8k_vallina_AOA_train.yaml'" > "${LOG_DIR}/run_do_as_i_do_train_pair.log" 2> "${LOG_DIR}/run_do_as_i_do_train_pair.err" &

echo $!
