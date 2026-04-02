# Alpaca Clean / Vallina 安全评测新接口

自 `2026-04-02` 起，这部分流程统一切到 `workflow yaml + runner`。

已经废弃的旧入口：
- `TLM/scripts/experiments/run_alpaca_clean_vallina_predict.py`
- `TLM/scripts/eval/run_alpaca_clean_vallina_safety_eval.py`
- `TLM/scripts/experiments/run_vallina_generation_suite.py`

统一入口：
- `TLM/scripts/workflows/run_workflow_yaml.py`

推荐模板：
- `TLM/examples/workflows/clean_vallina_eval_template.yaml`

## 现在怎么配

这个模板覆盖的是：
- `generate-only`
- 可以加载 `clean adapter` 或 `vallina adapter`
- 也可以删掉 `adapter_path` 改成纯 `base model`
- 生成后可直接接 `safety_evals`

关键字段：
- `generate.base_model_path`
- `generate.adapter_path`
- `generate.datasets`
- `generate.per_device_eval_batch_size`
- `safety_evals[].evaluation_mode`
- `safety_evals[].harmful_success_mode`
- `safety_evals[].classifier_batch_size`

评测口径：
- `harmful_asr`
  - 主指标：`asr_valid_only`
  - 默认 success：`compliance_and_harmful`
- `benign_refusal`
  - 主指标：`refusal_rate_valid_only`

## Windows smoke

当前已经实际跑通的 smoke 是新工作流的两条基础支路：
- `train -> adapter generate -> similarity + gsm8k accuracy`
- `generate-only -> base model -> similarity`

命令：

```powershell
conda activate TLM
python TLM\scripts\workflows\run_workflow_yaml.py TLM\examples\workflows\vallina_smoke_windows.yaml
```

如果你只想复查 generate-only 的 base model 支路，可以只跑第二个 job：

```powershell
conda activate TLM
python TLM\scripts\workflows\run_workflow_yaml.py TLM\examples\workflows\vallina_smoke_windows.yaml --only-job base_generate_smoke
```

说明：
- `safety-eval` 的 smoke 这次没有在 Windows 上跑
- 原因不是脚本没接好，而是你当前 `safety-eval` 环境还没配完

## Linux generate-only

先复制模板，再把路径改成你机器上的真实值：

```bash
export PROJECT_ROOT=/root/data/My_first_project
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/linux_runtime_env.sh"
conda activate TLM
cp "${PROJECT_ROOT}/TLM/examples/workflows/clean_vallina_eval_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_clean_vallina_eval_local.yaml"
```

你至少要改这些字段：
- `defaults.hf_home`
- `defaults.safety_eval_root`
- `generate.base_model_path`
- `generate.adapter_path`
- `generate.model_alias`
- `generate.per_device_eval_batch_size`

如果这次要直接用 `base model`，删除 `generate.adapter_path` 即可。

后台运行 generate + safety-eval：

```bash
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_clean_vallina_eval_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/clean_vallina_eval.log" \
  2> "${PROJECT_ROOT}/TLM/logs/clean_vallina_eval.err" &
```

## 只跑 generate

如果你现在只想先在 `4090D` 上做 generate，不跑 safety-eval，把 yaml 改成：
- 保留 `generate.enabled: true`
- 删除或禁用全部 `safety_evals`

推荐：
- `generate.per_device_eval_batch_size: 4`

## 只跑 safety-eval

如果预测已经导出好了，不想重新 generate：
- `generate.enabled: false`
- 保留 `safety_evals`
- 把每个 `safety_evals[].prediction_file` 改成已有的 `generated_predictions.jsonl`

推荐：
- `classifier_batch_size: 8`
- 如果显存紧，降到 `6`

## 输出目录

统一输出到：
- `TLM/saves/workflows/clean_vallina_eval/`

常见产物：
- `predictions/.../generated_predictions.jsonl`
- `predictions/.../generate_predict.json`
- `safety/*.json`
- `workflow_summary.json`
