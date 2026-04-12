# `offline_ttl.yaml` 机制与字段说明

本文档解释 [offline_ttl.yaml](./offline_ttl.yaml) 在当前仓库中的作用、字段语义，以及如何基于它做单次训练或串行多次训练。

## 1. 这个 YAML 是怎么生效的

当前仓库的原生训练入口仍然是：

```bash
cd TLM
python -m llamafactory.cli train examples/train_lora/offline_ttl.yaml
```

这里的调用链是：

1. `llamafactory.cli` 读取命令行中的 `train` 子命令。
2. 当发现后续只跟着一个 `.yaml` 文件时，`HfArgumentParser` 会直接把这份 YAML 解析成训练参数。
3. 解析后的参数会进入 `run_exp()`，再进入当前仓库的 TTL 训练流程。

因此，`offline_ttl.yaml` 不是“说明文件”，而是底层训练器直接消费的正式配置。

## 2. 这份 YAML 控制了什么

按逻辑可以分成 6 组字段：

### 2.1 模型与增量训练

- `model_name_or_path`
  - 底座模型路径或模型 ID。
- `adapter_name_or_path`
  - 可选字段。
  - 如果提供，训练会在已有 LoRA adapter 基础上继续训练，而不是新建一个空 adapter。

### 2.2 TTL 训练方式

- `stage: ttl`
  - 指定当前任务走 TTL 训练分支。
- `setting: offline_ttl`
  - 指定使用 offline TTL，而不是 online TTL。
- `finetuning_type: lora`
  - 指定只训练 LoRA。
- `lora_target: q_proj,v_proj`
  - 指定 LoRA 注入模块。

### 2.3 TTL 机制超参数

- `threshold`
  - 当前仓库里的 TTL 样本筛选/加权相关阈值。
- `lamb`
  - 当前 TTL 目标中的加权系数。

这两个字段是 TTL 机制里最核心、最应该保留在 YAML 中显式管理的参数。

## 2.4 数据与模板

- `dataset`
  - 训练集名。
- `eval_dataset`
  - TTL 训练后生成预测时使用的评测集名。
  - 当前常见做法是与 `dataset` 相同。
- `dataset_dir`
  - 数据注册表所在目录，当前仓库通常写 `data`。
- `template`
  - prompt template 名称，必须与底座模型匹配，例如 `qwen`、`llama3`。

## 2.5 训练资源与调度

- `cutoff_len`
  - 输入截断长度。
- `max_samples`
  - 样本上限；正式训练一般保留默认大值，smoke 时才会缩到极小。
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `num_train_epochs`
- `lr_scheduler_type`
- `warmup_ratio`
- `seed`
- `bf16`
- `ddp_timeout`
- `preprocessing_num_workers`

这些字段控制的是“如何训练”，不是 TTL 机制本身。

## 2.6 输出与预测

- `output_dir`
  - LoRA adapter 和训练日志的输出目录。
- `do_train`
  - 是否执行训练。
- `do_predict`
  - 是否在训练后做预测。
- `predict_with_generate`
  - 是否保存生成结果。
- `max_new_tokens`
  - 每条样本最多生成多少 token。
- `temperature`
- `do_sample`
- `logging_steps`
- `save_steps`
- `plot_loss`
- `overwrite_output_dir`
- `report_to`

## 3. 单次训练怎么用

### 3.1 直接跑当前 YAML

```bash
cd TLM
python -m llamafactory.cli train examples/train_lora/offline_ttl.yaml
```

### 3.2 常见修改方式

如果你只想改一次训练，直接改这几个字段通常就够了：

- `model_name_or_path`
- `adapter_name_or_path`
- `dataset`
- `eval_dataset`
- `template`
- `output_dir`
- `learning_rate`
- `per_device_train_batch_size`
- `seed`

## 4. 如何做增量训练

把下面这一行取消注释并改成已有 adapter 目录：

```yaml
adapter_name_or_path: saves/serial_suites/requested_suite/.../adapter
```

然后照常运行：

```bash
cd TLM
python -m llamafactory.cli train examples/train_lora/offline_ttl.yaml
```

## 5. 如何串行多次训练然后评测

当前仓库新增了模板脚本：

- [run_yaml_offline_ttl_serial_template.py](../../scripts/experiments/run_yaml_offline_ttl_serial_template.py)

它的做法是：

1. 读取 `offline_ttl.yaml` 作为基础模板。
2. 对每个数据集生成一份运行时 YAML。
3. 逐轮调用 `python -m llamafactory.cli train <runtime_yaml>`。
4. 每轮训练完成后，自动调用当前仓库已有的 clean eval 和 controlled eval。

### 5.1 最常用命令

```bash
cd TLM
python scripts/experiments/run_yaml_offline_ttl_serial_template.py \
  --base-yaml examples/train_lora/offline_ttl.yaml \
  --datasets agriculture_5k alpaca_gpt4_5k gsm8k_5k \
  --output-root saves/yaml_serial/requested_suite
```

### 5.2 只训练，不评测

```bash
cd TLM
python scripts/experiments/run_yaml_offline_ttl_serial_template.py \
  --base-yaml examples/train_lora/offline_ttl.yaml \
  --datasets agriculture_5k alpaca_gpt4_5k gsm8k_5k \
  --output-root saves/yaml_serial/requested_suite \
  --skip-eval
```

### 5.3 只跳过 controlled eval

```bash
cd TLM
python scripts/experiments/run_yaml_offline_ttl_serial_template.py \
  --base-yaml examples/train_lora/offline_ttl.yaml \
  --datasets agriculture_5k alpaca_gpt4_5k gsm8k_5k \
  --output-root saves/yaml_serial/requested_suite \
  --skip-controlled-eval
```

### 5.4 smoke 冒烟测试

```bash
cd TLM
python scripts/experiments/run_yaml_offline_ttl_serial_template.py \
  --base-yaml examples/train_lora/offline_ttl.yaml \
  --datasets agriculture_5k \
  --output-root saves/yaml_serial/smoke \
  --smoke-test
```

## 6. 串行模板脚本会覆盖哪些 YAML 字段

为了让一份基础 YAML 可复用到多轮训练，模板脚本会按每轮自动覆盖：

- `dataset`
- `eval_dataset`
- `output_dir`

如果你在命令行里额外传了覆盖项，还会覆盖：

- `model_name_or_path`
- `adapter_name_or_path`
- `template`

如果启用了 `--smoke-test`，还会覆盖一组轻量参数：

- `model_name_or_path`
- `template`
- `cutoff_len`
- `max_new_tokens`
- `max_samples`
- `max_steps`
- `learning_rate`
- `threshold`
- `bf16`
- `logging_steps`
- `save_steps`
- `preprocessing_num_workers`
- `per_device_train_batch_size`
- `per_device_eval_batch_size`

## 7. 什么时候该直接改 YAML，什么时候该写脚本

推荐准则：

- 只跑一次训练：直接改 `offline_ttl.yaml`
- 同一组配置要换多个数据集串行跑：用串行模板脚本
- 需要大规模多实验矩阵：再单独写更上层的实验调度脚本，不要把过多逻辑塞回单个 YAML
