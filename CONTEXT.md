# CONTEXT

## 1. 文档目的

这份文档用于记录当前仓库在提交 `7da3028` 上的真实状态，供后续新聊天快速恢复上下文。

使用原则：

- 以当前提交的源码、目录结构和现有脚本入口为准。
- 如果旧聊天记录、后续重构期的结论、或临时口头约定与当前代码冲突，以当前代码为准。
- 当前版本还没有稳定可用的 `workflow yaml + runner` 主接口，不要把后续版本的 workflow 设计套回这个提交。

## 2. 当前提交识别信息

- 当前提交：`7da3028`
- 当前仓库根目录：项目根目录下的 `README.md`、`commands.md`、`newpipe.md` 描述的是本地工作区主线
- 当前主代码目录：`TLM/`
- 当前安全评测辅助仓库：`safety-eval/`
- 当前数据构造辅助仓库：`llm-tta/`

## 3. 当前项目主线

当前项目主线仍然是：

1. 运行 requested TTL serial suite
2. 比较 `base_model`、`clean_model`、`mix_model`
3. clean task 用原仓库原生 `do_eval + compute_accuracy`
4. safety 用 WildJailbreak controlled eval
5. 可选地对已有 `generated_predictions.jsonl` 做离线 `safety-eval` 重评分

当前 active clean datasets：

- `agriculture_5k`
- `alpaca_gpt4_5k`
- `gsm8k_5k`

当前 mixed 训练主线：

- `agriculture_5k_advharm_40`
- `alpaca_gpt4_5k_advharm_40`
- `gsm8k_5k_advharm_40`

另外还有一条单独的 `vallina` 支线：

- `villina_mixed`
- `alpaca_villina_mixed40`
- 对应专用脚本是 `run_vallina_*` 和 `run_alpaca_clean_vallina_*`

## 4. 当前最重要的入口脚本

### 4.1 串行 suite 主入口

当前最重要的主入口是：

- `TLM/scripts/experiments/run_requested_ttl_serial_suite.py`

它会按数据集顺序串行执行：

1. `base_model` clean eval
2. `clean_model` pipeline
3. `mix_model` pipeline

内置顺序：

1. `agriculture_5k`
2. `alpaca_gpt4_5k`
3. `gsm8k_5k`

它支持：

- `--only-dataset`
- `--start-from-dataset`
- `--resume-from-step`
- `--reuse-base-model-controlled-eval-summary`
- `--skip-export`
- `--skip-upload`
- `--use-swanlab`

### 4.2 clean / mixed40 专用 pipeline

这两个脚本是当前串行 suite 依赖的下游专用入口：

- `TLM/scripts/experiments/run_clean_ttl_pipeline.py`
- `TLM/scripts/experiments/run_mixed40_ttl_pipeline.py`

职责：

- 训练
- clean eval
- controlled safety eval
- 可选导出
- 可选上传

### 4.3 vallina 支线训练与生成

当前存在一组 `vallina` 专用脚本，不是通用 workflow：

- `TLM/scripts/experiments/run_vallina_alpaca_ttl_pipeline.py`
  - 用 `alpaca_villina_mixed40` 训练 `vallina_model`
- `TLM/scripts/experiments/run_vallina_generation_suite.py`
  - 用 `vallina_model` 生成多个数据集的回答
- `TLM/scripts/experiments/run_alpaca_clean_vallina_predict.py`
  - 用 `alpaca_gpt4_5k` 的 `clean_model` 在 `villina_mixed` 上生成
- `TLM/scripts/eval/run_alpaca_clean_vallina_safety_eval.py`
  - 对 `alpaca_clean_model` 与 `alpaca_vallina_model` 的预测目录跑 `safety-eval`

注意：

- 这些是“专用脚本接口”，不是统一 workflow runner。
- 当前提交下 `TLM/scripts/workflows/` 没有实际可用的 runner，只剩 `__pycache__`。

## 5. 当前评测逻辑

### 5.1 clean eval

当前 clean eval 已改回原仓库原生路径：

- `do_eval=true`
- `do_predict=false`
- `predict_with_generate=false`
- `compute_accuracy=true`

关键结果文件：

- `metrics/clean_eval.json`

关键字段示例：

- `eval_agriculture_5k_accuracy`
- `eval_alpaca_gpt4_5k_accuracy`
- `eval_gsm8k_5k_accuracy`

实现核心在：

- `TLM/scripts/experiments/pipeline_common.py`

### 5.2 controlled safety eval

当前 controlled safety eval 仍由 WildJailbreak generation-based 管线负责。

主入口：

- `TLM/scripts/eval/run_wildjailbreak_controlled_eval.py`

当前重点集合：

- `harmful_mix_2k`
- `wildjailbreak_train_vanilla_benign_1k`
- `wildjailbreak_eval_adversarial_benign`
- `wildjailbreak_eval_adversarial_harmful`
- `wildjailbreak_train_harmful_lift_holdout_1k`

每个模型目录下重点看：

- `controlled_eval/.../wildjailbreak_controlled_eval_summary.json`

### 5.3 离线 safety-eval 重评分

当前有独立离线重评分脚本：

- `TLM/scripts/eval/run_safetyeval_on_predictions.py`

它负责：

- 读取已有 `generated_predictions.jsonl`
- 调用 `safety-eval` classifier
- 输出 `safety_eval_summary.json`
- 可选写逐样本 classifier 标注

额外还有一个专门给 `alpaca_clean vs vallina` 对照预测用的：

- `TLM/scripts/eval/run_alpaca_clean_vallina_safety_eval.py`

## 6. 当前重要数据与 profile

profile 逻辑集中在：

- `TLM/scripts/experiments/dataset_profiles.py`

它控制：

- `cutoff_len`
- `max_new_tokens`
- 数据集级 profile 选择

当前要记住的本地语义：

- `harmful_mix_2k`
  - 固定 2000 条 harmful 集
- `*_advharm_40`
  - clean 数据 + adversarial harmful 40% 的 mixed 数据
- `villina_mixed`
  - 从 WildJailbreak `vanilla_harmful` 采样出的 2000 条 harmful 数据
- `alpaca_villina_mixed40`
  - `alpaca_gpt4_5k` 与 `villina_mixed` 混合出的 `mixed40`

## 7. 当前最重要的结果目录

### 7.1 stable formal run

当前最重要的稳定正式结果目录是：

- `TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/`

这个目录是当前最重要的结果读取基准。

### 7.2 结果文件优先级

每个 dataset / model 目录下优先看：

- `base_model/model_eval_summary.json`
- `clean_model/pipeline_summary.json`
- `mix_model/pipeline_summary.json`
- `metrics/clean_eval.json`
- `controlled_eval/.../wildjailbreak_controlled_eval_summary.json`

### 7.3 vallina 支线结果目录

当前 `vallina` 训练默认落盘位置：

- `TLM/saves/pipelines/vallina/`

当前 `vallina` 生成默认落盘位置：

- `TLM/saves/predictions/vallina/`

`alpaca_clean_vallina` 对照生成默认落盘位置：

- `TLM/saves/predictions/alpaca_clean_vallina/`

## 8. 当前正式命令文档

正式 Linux 命令当前主要看：

- `commands.md`

它当前记录的是 requested serial suite 的正式命令，不是 workflow 命令。

重点参数约定：

- `MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"`
- `LR=0.00014`
- `TRAIN_BS=16`
- `EVAL_BS=16`
- `SEED=42`

## 9. 当前已写过但需要注意不要误读的文件

### 9.1 README.md

根目录 `README.md` 基本仍然有效，但要注意：

- 它会提到 offline `safety-eval` rescoring
- 它也会提到后续可能存在的其他接口设计
- 真正执行时仍应以当前脚本和 `commands.md` 为准

### 9.2 newpipe.md

`newpipe.md` 记录的是一次 `vallina` 支线任务的需求与落地结果，不是整个仓库的统一主入口文档。

它适合回答：

- `villina_mixed` 是怎么来的
- `alpaca_villina_mixed40` 是怎么来的
- `run_vallina_alpaca_ttl_pipeline.py` 和 `run_vallina_generation_suite.py` 怎么用

但不应把它误当成 requested serial suite 的总入口。

## 10. 用户常用术语与默认理解

后续新聊天里，如果用户提这些词，默认这样理解：

- `base_model`
  - 不加载 adapter 的原始基座模型
- `clean_model`
  - 在单个 clean dataset 上做 offline TTL 得到的 adapter/model
- `mix_model`
  - 在 `<clean_dataset>_advharm_40` 上做 offline TTL 得到的 adapter/model
- `clean eval`
  - 当前指原生 `do_eval + compute_accuracy`
- `controlled eval`
  - 指 `run_wildjailbreak_controlled_eval.py` 跑的统一 WildJailbreak 安全评测
- `native eval`
  - 基本等价于 clean task 的原仓库原生评测
- `safety-eval`
  - 指离线 classifier 重评分，不重新生成模型回答
- `vallina`
  - 用户项目里自己定义的 vanilla harmful 支线，不是上游标准术语
- `villina_mixed`
  - 当前仓库里实际的数据集名，拼写按仓库现状使用
- `alpaca_clean_vallina`
  - 指用 alpaca 的 clean adapter 和 vallina adapter 做对照生成/安全评测
- `smoke`
  - 最小样本数、tiny model、本地缓存优先、主要验证接口闭环
- `formal`
  - Linux / 真 GPU / 正式 batch size / 正式模型

## 11. 当前运行环境与用户偏好

长期偏好：

- 与用户对话默认用中文
- 代码注释默认用中文
- 函数不超过 100 行
- 模块拆细，主函数尽量只做编排
- 修改接口、目录、结果读取方式后，要同步更新上下文文档

当前环境偏好：

- 正式运行平台默认是 Linux
- Windows 主要用于：
  - 读代码
  - 改代码
  - 做 smoke
  - 查结果

## 12. 当前版本最关键的事实

1. 当前主线没有可用的 workflow yaml runner，后续不要默认按 workflow 接口回答。
2. 当前最重要的正式结果是 `lr_0.0001_bs_16_seed_42`。
3. 当前串行 suite 主入口是 `run_requested_ttl_serial_suite.py`。
4. `vallina` 相关能力已经存在，但仍是专用脚本，不是统一集成接口。
5. clean eval 当前已经回到原生 `ComputeAccuracy` 路径。
6. safety-eval 当前是“离线重评分”，不是主训练链路的一部分。

## 13. 新聊天建议先读的文件

如果要在新聊天中快速恢复当前提交上下文，建议优先读：

1. `README.md`
2. `commands.md`
3. `TLM/scripts/experiments/run_requested_ttl_serial_suite.py`
4. `TLM/scripts/experiments/pipeline_common.py`
5. `TLM/scripts/experiments/dataset_profiles.py`
6. `TLM/docs/experiment_results.md`
7. 如果是 `vallina` 话题，再补：
   - `newpipe.md`
   - `TLM/scripts/experiments/run_vallina_alpaca_ttl_pipeline.py`
   - `TLM/scripts/experiments/run_vallina_generation_suite.py`
   - `TLM/scripts/experiments/run_alpaca_clean_vallina_predict.py`
   - `TLM/scripts/eval/run_alpaca_clean_vallina_safety_eval.py`
