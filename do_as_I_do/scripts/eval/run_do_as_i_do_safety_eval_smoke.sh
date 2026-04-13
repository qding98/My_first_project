#!/usr/bin/env bash
# 作用：
# - 用脚本四的 smoke prediction 结果做一轮最小 safety-eval 冒烟验证。
# - 同时打印当前 `safety-eval` 环境中的关键调试信息，便于排查缺包和路径问题。
#
# 依赖与调用关系：
# - 依赖 `do_as_I_do/scripts/eval/run_do_as_i_do_safety_eval.py`
# - 依赖 `do_as_I_do/saves/predict_smoke/gsm8k_AOA_model/gsm8k_AOA/generated_predictions.jsonl`
# - 依赖仓库根目录 `linux_runtime_env.sh`
#
# 输入来源：
# - `conda activate safety-eval` 后的 Python 环境
# - smoke prediction 产物
#
# 输出内容：
# - `do_as_I_do/saves/safety-eval-results/gsm8k_AOA_model/gsm8k_AOA/`
# - `do_as_I_do/saves/safety-eval-results/gsm8k_AOA_model/summary.json`
#
# 对外接口：
# - `bash do_as_I_do/scripts/eval/run_do_as_i_do_safety_eval_smoke.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNTIME_ENV_SH="${REPO_ROOT}/linux_runtime_env.sh"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${DO_AS_I_DO_SAFETY_EVAL_CONDA_ENV:-safety-eval}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "未找到 conda 初始化脚本: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
source "${RUNTIME_ENV_SH}"

echo "[info] python=$(which python)"
python -V
python - <<'PY'
import importlib

modules = ["torch", "transformers", "vllm", "ray", "openai", "pandas", "datasets"]
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")
print(versions)
PY

cd "${REPO_ROOT}"
python do_as_I_do/scripts/eval/run_do_as_i_do_safety_eval.py \
  --safety-eval-root safety-eval \
  --results-root do_as_I_do/saves/safety-eval-results \
  --classifier-model-name WildGuard \
  --classifier-batch-size 1 \
  --classifier-ephemeral-model \
  --save-per-sample-results \
  --smoke
