#!/usr/bin/env bash
# 作用：
# - 自动激活 `safety-eval` 环境后补装 `safety-eval` 所需依赖。
# - 安装完成后做一轮 import 级别检查，帮助定位环境缺包。
#
# 依赖与调用关系：
# - 依赖仓库内 `safety-eval/requirements.txt`
# - 依赖仓库内 `safety-eval/setup.py`
#
# 输入来源：
# - 当前激活的 `safety-eval` conda 环境
#
# 输出内容：
# - 在当前 Python 环境中安装 `safety-eval` 依赖
# - 终端打印关键包版本，供调试使用
#
# 对外接口：
# - `bash do_as_I_do/scripts/eval/install_safety_eval_requirements.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAFETY_EVAL_DIR="${REPO_ROOT}/safety-eval"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${DO_AS_I_DO_SAFETY_EVAL_CONDA_ENV:-safety-eval}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "未找到 conda 初始化脚本: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

echo "[info] python=$(which python)"
python -V
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${SAFETY_EVAL_DIR}/requirements.txt"
python -m pip install -e "${SAFETY_EVAL_DIR}"
python - <<'PY'
import importlib

modules = ["torch", "transformers", "vllm", "ray", "openai", "pandas", "datasets"]
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")
print(versions)
PY
echo "[done] safety-eval dependencies installed."
