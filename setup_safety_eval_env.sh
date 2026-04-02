#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-safety-eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
SAFETY_EVAL_DIR="${ROOT_DIR}/safety-eval"
RUNTIME_ENV_SCRIPT="${ROOT_DIR}/linux_runtime_env.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda 未安装或不在 PATH 中。"
  exit 1
fi

if [[ ! -d "${SAFETY_EVAL_DIR}" ]]; then
  echo "[error] 未找到 safety-eval 目录: ${SAFETY_EVAL_DIR}"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ -f "${RUNTIME_ENV_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV_SCRIPT}"
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[info] conda 环境已存在，直接复用: ${ENV_NAME}"
else
  echo "[run] conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

echo "[run] python -m pip install --upgrade pip setuptools wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[run] pip install -e ${SAFETY_EVAL_DIR}"
pip install -e "${SAFETY_EVAL_DIR}"

echo "[run] pip install -r ${SAFETY_EVAL_DIR}/requirements.txt"
pip install -r "${SAFETY_EVAL_DIR}/requirements.txt"

echo "[done] safety-eval 环境已就绪"
echo "[env] ${ENV_NAME}"
echo "[next] conda activate ${ENV_NAME}"
echo "[next] cd ${ROOT_DIR}"
echo "[next] source ${RUNTIME_ENV_SCRIPT}"
