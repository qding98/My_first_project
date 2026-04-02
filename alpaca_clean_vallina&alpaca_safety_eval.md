# Alpaca Clean / Vallina 新接口

这部分现在也统一走三段式接口，不再使用旧的专用脚本。

当前入口：
- 生成：`TLM/scripts/workflows/run_generate_workflow_yaml.py`
- 评测：`TLM/scripts/workflows/run_eval_workflow_yaml.py`

推荐模板：
- 生成模板：[vallina_generate_template.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/vallina_generate_template.yaml)
- 评测模板：[vallina_eval_template.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/vallina_eval_template.yaml)

## 现在怎么做 clean / vallina 对照

第一步，分别准备两份 generate yaml：
- 一份填 `clean adapter`
- 一份填 `vallina adapter`

你主要改这几个字段：
- `generate.base_model_path`
- `generate.adapter_path`
- `generate.model_alias`
- `generate.datasets`
- `generate.per_device_eval_batch_size`

如果要直接测 `base model`，删除 `generate.adapter_path`。

第二步，分别准备两份 eval yaml：
- 一份指向 `clean model` 生成出的 `generated_predictions.jsonl`
- 一份指向 `vallina model` 生成出的 `generated_predictions.jsonl`

你主要改这几个字段：
- `prediction_evals[].prediction_file`
- `safety_evals[].prediction_file`
- `safety_evals[].classifier_batch_size`

## Windows smoke

当前已经实际跑通的 smoke 是三段式基础链路：
- 训练 smoke
- adapter generate smoke
- eval smoke

命令：

```powershell
conda activate TLM
python TLM\scripts\workflows\run_train_workflow_yaml.py TLM\examples\workflows\vallina_train_smoke_windows.yaml
python TLM\scripts\workflows\run_generate_workflow_yaml.py TLM\examples\workflows\vallina_generate_smoke_windows.yaml
python TLM\scripts\workflows\run_eval_workflow_yaml.py TLM\examples\workflows\vallina_eval_smoke_windows.yaml
```

smoke 汇总：
- [train_workflow_run_summary.json](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/train_workflow_run_summary.json)
- [generate_workflow_run_summary.json](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/generate_workflow_run_summary.json)
- [eval_workflow_run_summary.json](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/eval_workflow_run_summary.json)

## Linux 生成

clean / vallina 两个模型都走同一个生成入口，只是 yaml 不同。

```bash
export PROJECT_ROOT=/root/data/My_first_project
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/linux_runtime_env.sh"
conda activate TLM
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_generate_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_clean_generate_local.yaml"
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_generate_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_generate_local.yaml"
```

后台运行：

```bash
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_generate_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_clean_generate_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/clean_generate.log" \
  2> "${PROJECT_ROOT}/TLM/logs/clean_generate.err" &

nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_generate_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_generate_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/vallina_generate.log" \
  2> "${PROJECT_ROOT}/TLM/logs/vallina_generate.err" &
```

## Linux 评测

评测入口支持两类指标同时放在一个 eval yaml 里：
- 原仓库 native eval：`prediction_evals`
- `safety-eval`：`safety_evals`

推荐在 `safety-eval` 环境里跑：

```bash
export PROJECT_ROOT=/root/data/My_first_project
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/linux_runtime_env.sh"
conda activate safety-eval
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_eval_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_clean_eval_local.yaml"
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_eval_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_eval_local.yaml"
```

后台运行：

```bash
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_eval_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_clean_eval_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/clean_eval.log" \
  2> "${PROJECT_ROOT}/TLM/logs/clean_eval.err" &

nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_eval_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_eval_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/vallina_eval.log" \
  2> "${PROJECT_ROOT}/TLM/logs/vallina_eval.err" &
```

## 主指标口径

`prediction_evals`：
- `similarity`
- `gsm8k accuracy`

`safety_evals`：
- `harmful_asr`
  - 主指标：`asr_valid_only`
  - 默认 success：`compliance_and_harmful`
- `benign_refusal`
  - 主指标：`refusal_rate_valid_only`

## 环境补充

[setup_safety_eval_env.sh](D:/Qsh的个人资料/科研/LLM/My_first_project/setup_safety_eval_env.sh) 现在会额外安装：
- `jieba`
- `rouge-chinese`
- `nltk`

这样 `run_eval_workflow_yaml.py` 在 `safety-eval` 环境里可以同时跑：
- native eval
- safety eval

## smoke 中的 safety-eval 说明

- Windows smoke 里 `smoke_test: true` 只走 mock classifier
- 目的是验证接口闭环，不代表正式评测结果
- 正式 yaml 不要开这个参数
