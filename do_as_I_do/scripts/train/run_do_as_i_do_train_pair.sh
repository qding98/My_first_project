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

mkdir -p "${LOG_DIR}"

nohup bash -lc "cd '${TLM_DIR}' && python -m llamafactory.cli train '../do_as_I_do/examples/train/gsm8k_AOA_train.yaml' && python -m llamafactory.cli train '../do_as_I_do/examples/train/gsm8k_vallina_AOA_train.yaml'" > "${LOG_DIR}/run_do_as_i_do_train_pair.log" 2> "${LOG_DIR}/run_do_as_i_do_train_pair.err" &

echo $!
