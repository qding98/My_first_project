# Vallina 三段式接口

自 `2026-04-02` 起，`vallina` 流程不再走一体化 workflow。

当前唯一推荐接口：
- 训练：`TLM/scripts/workflows/run_train_workflow_yaml.py`
- 生成：`TLM/scripts/workflows/run_generate_workflow_yaml.py`
- 评测：`TLM/scripts/workflows/run_eval_workflow_yaml.py`

对应模板：
- 训练模板：[vallina_train_template.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/vallina_train_template.yaml)
- 生成模板：[vallina_generate_template.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/vallina_generate_template.yaml)
- 评测模板：[vallina_eval_template.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/vallina_eval_template.yaml)

Windows smoke 用这三份：
- [vallina_train_smoke_windows.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/vallina_train_smoke_windows.yaml)
- [vallina_generate_smoke_windows.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/vallina_generate_smoke_windows.yaml)
- [vallina_eval_smoke_windows.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/vallina_eval_smoke_windows.yaml)

## 设计原则

- `train yaml` 只负责产出 adapter
- `generate yaml` 只负责消费 `adapter_path` 或 `base_model_path` 产出预测
- `eval yaml` 只负责消费 `generated_predictions.jsonl` 跑：
  - 原仓库 `prediction_evals`
  - `safety_evals`
- 跨阶段衔接全部走显式路径，不再在一个 yaml 内隐式串联

## Windows smoke

这三段 smoke 已实际跑通。

训练 smoke：

```powershell
conda activate TLM
python TLM\scripts\workflows\run_train_workflow_yaml.py TLM\examples\workflows\vallina_train_smoke_windows.yaml
```

生成 smoke：

```powershell
conda activate TLM
python TLM\scripts\workflows\run_generate_workflow_yaml.py TLM\examples\workflows\vallina_generate_smoke_windows.yaml
```

评测 smoke：

```powershell
conda activate TLM
python TLM\scripts\workflows\run_eval_workflow_yaml.py TLM\examples\workflows\vallina_eval_smoke_windows.yaml
```

本次 smoke 的真实汇总文件：
- [train_workflow_run_summary.json](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/train_workflow_run_summary.json)
- [generate_workflow_run_summary.json](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/generate_workflow_run_summary.json)
- [eval_workflow_run_summary.json](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/eval_workflow_run_summary.json)

## Linux 正式运行

先复制三份模板，再改本机路径和超参数：

```bash
export PROJECT_ROOT=/root/data/My_first_project
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/linux_runtime_env.sh"
```

训练阶段：

```bash
conda activate TLM
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_train_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_train_local.yaml"
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_train_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_train_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/vallina_train.log" \
  2> "${PROJECT_ROOT}/TLM/logs/vallina_train.err" &
```

生成阶段：

```bash
conda activate TLM
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_generate_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_generate_local.yaml"
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_generate_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_generate_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/vallina_generate.log" \
  2> "${PROJECT_ROOT}/TLM/logs/vallina_generate.err" &
```

评测阶段：

```bash
conda activate safety-eval
cp "${PROJECT_ROOT}/TLM/examples/workflows/vallina_eval_template.yaml" "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_eval_local.yaml"
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_eval_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/_vallina_eval_local.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/vallina_eval.log" \
  2> "${PROJECT_ROOT}/TLM/logs/vallina_eval.err" &
```

## 你需要改的字段

训练 yaml：
- `defaults.hf_home`
- `train.target_vram_gb`
- `train.per_device_train_batch_size`
- `train.per_device_eval_batch_size`

生成 yaml：
- `defaults.hf_home`
- `generate.base_model_path`
- `generate.adapter_path`
- `generate.per_device_eval_batch_size`
- `generate.datasets`

评测 yaml：
- `defaults.safety_eval_root`
- `prediction_evals[].prediction_file`
- `safety_evals[].prediction_file`
- `safety_evals[].classifier_batch_size`

## 环境说明

- `train` 和 `generate` 推荐在 `TLM` 环境跑
- `eval` 如果启用了 `safety_evals`，推荐在 `safety-eval` 环境跑
- 为了让 `eval` 同时支持原仓库 native eval，我已经把 [setup_safety_eval_env.sh](D:/Qsh的个人资料/科研/LLM/My_first_project/setup_safety_eval_env.sh) 补成会额外安装：
  - `jieba`
  - `rouge-chinese`
  - `nltk`

## smoke 中的 safety-eval 说明

- Windows smoke 里的 `safety_evals[].smoke_test: true`
- 这会走内置 mock classifier
- 作用只是验证 `eval yaml -> eval runner -> safety output` 这条接口闭环
- 正式运行时不要开 `smoke_test`
