# Vallina 新接口

自 `2026-04-02` 起，`vallina` 相关流程统一切到 `workflow yaml + runner`。

已经废弃的旧入口：
- `TLM/scripts/experiments/run_vallina_alpaca_ttl_pipeline.py`
- `TLM/scripts/experiments/run_vallina_generation_suite.py`

统一入口：
- `TLM/scripts/workflows/run_workflow_yaml.py`

推荐模板：
- `TLM/examples/workflows/vallina_smoke_windows.yaml`
- `TLM/examples/workflows/vallina_smoke_template.yaml`
- `TLM/examples/workflows/vallina_formal_template.yaml`

## 现在怎么配

一个 workflow yaml 可以包含多个 `jobs`，每个 job 都可以独立控制：
- `train.enabled`
- `generate.enabled`
- `prediction_evals`
- `safety_evals`

核心字段约定：
- 训练数据集：`train.dataset`
- 生成数据集：`generate.datasets`
- 训练输出根目录：`train.output_root`
- 生成输出根目录：`generate.output_root`
- 用 adapter 生成：
  - `generate.base_model_path`
  - `generate.adapter_path`
- 只用 base model 生成：
  - 只保留 `generate.base_model_path`
  - 不传 `generate.adapter_path`
- 原仓库 AdaptEval 评测：
  - `prediction_evals[].task_type`
  - 当前已支持 `similarity`、`gsm8k`、`auto`
- safety-eval：
  - `safety_evals[].evaluation_mode`
  - 可选 `harmful_asr`、`benign_refusal`

## Windows smoke

这个 smoke 已经实际跑通，覆盖了两条链路：
- `train -> adapter generate -> similarity + gsm8k accuracy`
- `generate-only -> base model -> similarity`

命令：

```powershell
conda activate TLM
python TLM\scripts\workflows\run_workflow_yaml.py TLM\examples\workflows\vallina_smoke_windows.yaml
```

关键输出：
- `TLM/saves/workflows/vallina_smoke/`
- `TLM/saves/workflows/base_generate_smoke/`
- `TLM/saves/workflows/workflow_run_summary.json`

## Linux smoke

先复制模板，再把里面的 `hf_home` 和 `safety_eval_root` 改成你机器上的真实路径：

```bash
export PROJECT_ROOT=/root/data/My_first_project
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/linux_runtime_env.sh"
conda activate TLM
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_smoke_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_smoke_local.yaml"
```

后台运行：

```bash
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_smoke_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/vallina_workflow_smoke.log" \
  2> "${PROJECT_ROOT}/TLM/logs/vallina_workflow_smoke.err" &
```

## Linux 正式运行

正式跑 `vallina` 训练和生成，直接从正式模板复制一份本地 yaml 再改：

```bash
export PROJECT_ROOT=/root/data/My_first_project
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/linux_runtime_env.sh"
conda activate TLM
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_formal_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_formal_local.yaml"
```

你需要按机器改这些字段：
- `defaults.hf_home`
- `defaults.safety_eval_root`
- `train.target_vram_gb`
- `train.per_device_train_batch_size`
- `train.per_device_eval_batch_size`
- `generate.target_vram_gb`
- `generate.per_device_eval_batch_size`
- `prediction_evals[].enabled`
- `safety_evals[].enabled`

后台运行：

```bash
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_formal_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/vallina_workflow_formal.log" \
  2> "${PROJECT_ROOT}/TLM/logs/vallina_workflow_formal.err" &
```

## 常用切法

- 只训练：
  - `train.enabled: true`
  - `generate.enabled: false`
  - 不写 `prediction_evals` / `safety_evals`
- 只生成：
  - `train.enabled: false`
  - `generate.enabled: true`
- 训练后生成：
  - `generate.adapter_path: ${jobs.<job_name>.train.outputs.adapter_dir}`
  - `generate.base_model_path: ${jobs.<job_name>.train.outputs.model_name_or_path}`
- 只跑原仓库 eval：
  - 只保留 `prediction_evals`
- 只跑 safety-eval：
  - 只保留 `safety_evals`

## 输出目录

统一输出到：
- `TLM/saves/workflows/<job_name>/...`

每个 job 至少会有：
- `workflow_summary.json`
- `train/.../summary.json`，如果启用了训练
- `predictions/.../generation_summary.json`，如果启用了生成
- `evals/*.json`，如果启用了 `prediction_evals`
- `safety/*.json`，如果启用了 `safety_evals`
