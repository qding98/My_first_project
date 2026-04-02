# Alpaca Clean / Vallina YAML 接口

这部分现在不再手工复制多份 yaml，而是先用 builder 脚本批量生成 yaml，再交给 stage runner 执行。

当前入口：
- builder 1：`TLM/scripts/workflows/build_alpaca_clean_vallina_workflows.py`
- builder 2：`TLM/scripts/workflows/build_requested_controlled_eval_safety_workflow.py`
- generate runner：`TLM/scripts/workflows/run_generate_workflow_yaml.py`
- eval runner：`TLM/scripts/workflows/run_eval_workflow_yaml.py`

## 1. Alpaca Clean / Vallina 对照

`build_alpaca_clean_vallina_workflows.py` 会一次性产出两份 yaml：
- `alpaca_clean_vallina_generate.yaml`
- `alpaca_clean_vallina_eval.yaml`

它解决的是：
- `clean adapter` 和 `vallina adapter` 共用一套 generate / eval 接口
- `prediction_evals` 和 `safety_evals` 同时自动连到正确的 `generated_predictions.jsonl`
- `--smoke-test` 下自动改用真实 smoke 落盘路径，不需要你再手改 `evalbs_1_cutoff_64_out_8...`
- builder 会自动在 yaml 输出目录、generate/eval 输出根目录、job summary 路径上追加命名空间 tag
  - tag 里会编码模型、数据集、batch size、seed 等关键信息
  - 同时会附一个稳定短哈希，保证不同设置不会撞路径

### Linux 正式生成 yaml

```bash
export PROJECT_ROOT=/root/data/My_first_project
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/linux_runtime_env.sh"
conda activate TLM

python "${PROJECT_ROOT}/TLM/scripts/workflows/build_alpaca_clean_vallina_workflows.py" \
  --output-dir "${PROJECT_ROOT}/TLM/examples/workflows/generated/alpaca_clean_vallina_formal" \
  --base-model-path Qwen/Qwen2.5-7B-Instruct \
  --hf-home /root/data/qsh/hf_cache \
  --clean-adapter-path /path/to/alpaca_clean_adapter \
  --vallina-adapter-path /path/to/vallina_adapter \
  --generate-output-root saves/workflows/generate_outputs/alpaca_clean_vallina_formal \
  --eval-output-root saves/workflows/eval_outputs/alpaca_clean_vallina_formal \
  --generate-dataset alpaca_gpt4_5k \
  --generate-dataset villina_mixed \
  --prediction-eval alpaca_gpt4_5k=similarity \
  --safety-eval villina_mixed=harmful_asr \
  --per-device-eval-batch-size 4 \
  --classifier-batch-size 8
```

注意：
- builder 执行后会打印两份 yaml 的真实绝对路径
- 因为现在会自动追加命名空间子目录，所以后续 runner 请直接使用 builder 打印出来的路径，不要手写猜路径

如果你想直接测 base model，不传对应的 `--clean-adapter-path` 或 `--vallina-adapter-path` 即可。

### Linux 正式运行

生成：

```bash
nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_generate_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/generated/alpaca_clean_vallina_formal/alpaca_clean_vallina_generate.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/alpaca_clean_vallina_generate.log" \
  2> "${PROJECT_ROOT}/TLM/logs/alpaca_clean_vallina_generate.err" &
```

评测：

```bash
conda activate safety-eval

nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_eval_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/generated/alpaca_clean_vallina_formal/alpaca_clean_vallina_eval.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/alpaca_clean_vallina_eval.log" \
  2> "${PROJECT_ROOT}/TLM/logs/alpaca_clean_vallina_eval.err" &
```

### Windows smoke

下面这组是已经实际跑通的 smoke：

```powershell
D:\anacoda3\envs\TLM\python.exe TLM\scripts\workflows\build_alpaca_clean_vallina_workflows.py `
  --output-dir TLM\examples\workflows\generated\smoke_alpaca_clean_vallina `
  --base-model-path llamafactory/tiny-random-Llama-3 `
  --hf-home D:\hf_cache `
  --clean-adapter-path D:\Qsh的个人资料\科研\LLM\My_first_project\TLM\saves\workflows\train_outputs\vallina_train_smoke\vallina_lr_0.0001_bs_1_seed_42\train\adapter `
  --vallina-adapter-path D:\Qsh的个人资料\科研\LLM\My_first_project\TLM\saves\workflows\train_outputs\vallina_train_smoke\vallina_lr_0.0001_bs_1_seed_42\train\adapter `
  --generate-output-root saves/workflows/generate_outputs/smoke_alpaca_clean_vallina `
  --eval-output-root saves/workflows/eval_outputs/smoke_alpaca_clean_vallina `
  --generate-dataset alpaca_gpt4_5k `
  --generate-dataset villina_mixed `
  --prediction-eval alpaca_gpt4_5k=similarity `
  --safety-eval villina_mixed=harmful_asr `
  --classifier-batch-size 1 `
  --smoke-test

D:\anacoda3\envs\TLM\python.exe TLM\scripts\workflows\run_generate_workflow_yaml.py `
  TLM\examples\workflows\generated\smoke_alpaca_clean_vallina\alpaca_clean_vallina_generate.yaml

D:\anacoda3\envs\TLM\python.exe TLM\scripts\workflows\run_eval_workflow_yaml.py `
  TLM\examples\workflows\generated\smoke_alpaca_clean_vallina\alpaca_clean_vallina_eval.yaml
```

smoke 验证产物：
- [alpaca_clean_vallina_generate.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/generated/smoke_alpaca_clean_vallina/alpaca_clean_vallina_generate.yaml)
- [alpaca_clean_vallina_eval.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/generated/smoke_alpaca_clean_vallina/alpaca_clean_vallina_eval.yaml)
- [generate_workflow_run_summary.json](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/generate_workflow_run_summary.json)
- [eval_workflow_run_summary.json](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/eval_workflow_run_summary.json)

对应 smoke 输出目录：
- [generate_outputs](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/generate_outputs/smoke_alpaca_clean_vallina)
- [eval_outputs](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/eval_outputs/smoke_alpaca_clean_vallina)

## 2. 历史 controlled_eval 批量 safety-eval

`build_requested_controlled_eval_safety_workflow.py` 默认已经内置了这三组根目录：
- `gsm8k_5k/clean_model/controlled_eval/adapter`
- `gsm8k_5k/mix_model/controlled_eval/adapter`
- `agriculture_5k/base_model/controlled_eval/Qwen__Qwen2.5-7B-Instruct`

脚本会自动扫描这些目录下的 `*/generated_predictions.jsonl`，并按仓库原有口径分流：
- `harmful_mix_2k`
- `wildjailbreak_eval_adversarial_harmful`
- `wildjailbreak_train_harmful_lift_holdout_1k`
  - 走 `harmful_asr`
- `wildjailbreak_train_vanilla_benign_1k`
- `wildjailbreak_eval_adversarial_benign`
  - 走 `benign_refusal`

同时它也会把 `classifier_batch_size`、选中的数据集集合和根目录集合编码进输出 yaml 文件名与 eval 输出目录里，避免不同批量评测配置覆盖同一路径。

### Linux 正式生成 yaml

```bash
export PROJECT_ROOT=/root/data/My_first_project
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/linux_runtime_env.sh"
conda activate TLM

python "${PROJECT_ROOT}/TLM/scripts/workflows/build_requested_controlled_eval_safety_workflow.py" \
  --output-file "${PROJECT_ROOT}/TLM/examples/workflows/generated/requested_controlled_eval_safety.yaml" \
  --classifier-batch-size 8
```

### Linux 正式运行

```bash
conda activate safety-eval

nohup python "${PROJECT_ROOT}/TLM/scripts/workflows/run_eval_workflow_yaml.py" \
  "${PROJECT_ROOT}/TLM/examples/workflows/generated/requested_controlled_eval_safety.yaml" \
  > "${PROJECT_ROOT}/TLM/logs/requested_controlled_eval_safety.log" \
  2> "${PROJECT_ROOT}/TLM/logs/requested_controlled_eval_safety.err" &
```

### Windows smoke

下面这组也已经实际跑通：

```powershell
D:\anacoda3\envs\TLM\python.exe TLM\scripts\workflows\build_requested_controlled_eval_safety_workflow.py `
  --output-file TLM\examples\workflows\generated\requested_controlled_eval_safety_smoke.yaml `
  --include-dataset harmful_mix_2k `
  --include-dataset wildjailbreak_eval_adversarial_benign `
  --classifier-batch-size 1 `
  --smoke-test

D:\anacoda3\envs\TLM\python.exe TLM\scripts\workflows\run_eval_workflow_yaml.py `
  TLM\examples\workflows\generated\requested_controlled_eval_safety_smoke.yaml
```

smoke 验证产物：
- [requested_controlled_eval_safety_smoke.yaml](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/examples/workflows/generated/requested_controlled_eval_safety_smoke.yaml)
- [requested_controlled eval outputs](D:/Qsh的个人资料/科研/LLM/My_first_project/TLM/saves/workflows/eval_outputs/requested_controlled)

## 主指标口径

`prediction_evals`：
- `similarity`
- `gsm8k`

`safety_evals`：
- `harmful_asr`
  - 主指标：`asr_valid_only`
  - 默认 success：`compliance_and_harmful`
- `benign_refusal`
  - 主指标：`refusal_rate_valid_only`

## smoke 说明

- `smoke_test: true` 只用于接口闭环验证
- `safety-eval` smoke 走 mock classifier，不代表正式指标
- builder 在 `--smoke-test` 下会自动把 eval yaml 指向 smoke 真实落盘路径
