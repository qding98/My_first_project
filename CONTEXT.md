# CONTEXT

## 1. 文档目的

这份文档用于记录当前仓库的真实状态，并覆盖早期 README 或临时聊天里已经过时的描述。

使用原则：

- 以当前代码、当前目录结构、当前入口脚本为准。
- 如果 `README.md`、`TTL_MIXED_EXPERIMENT_GUIDE.md`、聊天结论和源码冲突，以源码和当前主入口为准。
- 后续只要接口、目录、实验方案、结果读取方式发生变化，就需要同步更新本文件。

## 2. 旧 README 与当前仓库的差异

### 2.1 仍然有效的部分

- 根目录 `README.md` 仍然正确描述了这是一套围绕 `TLM/`、`safety-eval/` 和本地实验脚本组织起来的工作区。
- `README.md` 里记录的稳定历史结果目录 `TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/` 仍然是当前最重要的已完成正式结果之一。
- `README.md` 里关于 clean eval 改回原生 `do_eval + compute_accuracy`、controlled safety eval 仍由 WildJailbreak 管线负责，这些描述目前仍成立。

### 2.2 已经漂移的部分

- 根目录 `README.md` 更偏“历史稳定结果说明”，但不是当前唯一接口说明。当前仓库已经存在分阶段 workflow 接口：
  - `TLM/scripts/workflows/run_train_workflow_yaml.py`
  - `TLM/scripts/workflows/run_generate_workflow_yaml.py`
  - `TLM/scripts/workflows/run_eval_workflow_yaml.py`
- `commands.md` 当前给出的 Linux 正式命令，目标超参数已经偏向新的正式运行配置：
  - `LR=0.00014`
  - `TRAIN_BS=16`
  - `EVAL_BS=16`
  - `MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"`
- `TLM/TTL_MIXED_EXPERIMENT_GUIDE.md` 包含较早阶段的说明，里面有些内容已经不是当前主线：
  - 文档里保留了大量 `*_advharm_10/20/40/60` 的历史背景，但当前主实验主线是 requested serial suite 和 `mixed40`。
  - 文档中曾写过一版 `alpaca_gpt4_5k` clean 对 `agriculture_5k_advharm_40` mixed 的 cross-domain 方案，但当前 `TLM/scripts/experiments/run_requested_ttl_serial_suite.py` 的 `DEFAULT_PLAN` 已经是同领域配对：
    - `agriculture_5k -> agriculture_5k`
    - `alpaca_gpt4_5k -> alpaca_gpt4_5k`
    - `gsm8k_5k -> gsm8k_5k`

### 2.3 当前应当采用的判断顺序

当前信息优先级建议如下：

1. 入口脚本与实际目录结构
2. `commands.md`
3. 本文件 `CONTEXT.md`
4. 根目录 `README.md`
5. `TLM/TTL_MIXED_EXPERIMENT_GUIDE.md` 这类阶段性记录

## 3. 当前项目结构

项目根目录的关键组成如下：

- `TLM/`
  - 主代码仓库，训练、生成、评测、workflow、数据注册和实验输出都在这里。
- `safety-eval/`
  - 本地 checkout 的安全评测仓库，主要用于离线 classifier 重评分。
- `llm-tta/`
  - 用于构造 mixed 数据集的辅助仓库，历史上负责生成 `harmful_mix_2k` 和 `*_advharm_40`。
- `docs/`
  - 额外项目说明。
- `commands.md`
  - 当前 Linux 正式运行命令的集中说明。
- `README.md`
  - 当前工作区的高层说明，但不是所有新接口的完整索引。
- `linux_runtime_env.sh`
  - Linux 环境初始化辅助脚本。
- `setup_safety_eval_env.sh`
  - 安装 `safety-eval` 依赖的辅助脚本。

`TLM/` 下当前最重要的目录如下：

- `TLM/data/`
  - 数据注册表和本地 JSON 数据。
- `TLM/examples/workflows/`
  - train/generate/eval 分阶段 workflow yaml 模板。
- `TLM/scripts/experiments/`
  - 当前 clean/mixed/serial suite 主实验脚本。
- `TLM/scripts/eval/`
  - WildJailbreak controlled eval 与 safety-eval 相关评测脚本。
- `TLM/scripts/workflows/`
  - train/generate/eval 三个分阶段 runner 与 builder。
- `TLM/scripts/stat_utils/`
  - 长度分布、MFU、学习率估算等统计脚本。
- `TLM/saves/`
  - 历史实验结果与当前正式结果目录。
- `TLM/docs/experiment_results.md`
  - 已跑结果的人类总结。

## 4. 当前运行环境要求

### 4.1 运行平台

当前正式实验默认目标平台是 Linux，不是 Windows。

当前约定的 Linux 目录为：

- 项目根目录：`/root/data/My_first_project`
- TLM 根目录：`/root/data/My_first_project/TLM`

但在仓库文档、命令说明、接口说明里，应优先写相对路径，而不是把 `/root/data/...` 写成主路径。

### 4.2 Linux 环境要点

当前 Linux 运行约束来自 `TLM/LINUX_RUN_GUIDE_ZH.md` 与 `commands.md`：

- conda 环境名：`TLM`
- 模型主线：`Qwen/Qwen2.5-7B-Instruct`
- HF 缓存目录通过环境变量控制
- 正式运行前要先 `source` 环境与 `linux_env.sh`
- 正式实验以单卡 A800 80GB 为主要参考硬件

## 5. 用户长期偏好与工作约束

这些偏好已经被反复确认，后续默认继续沿用：

- 默认用中文对话、中文注释。
- 函数实现代码不超过 100 行。
- 函数、模块、类职责要细拆，不要把流程编排、IO 和业务逻辑揉在一起。
- 新脚本文件开头必须写说明性注释，函数前也要写注释。
- 仓库内写路径时优先使用相对路径。
- 当前项目后续维护时，除了改代码，也要同步维护 `CONTEXT.md`。
- 当前项目的新接口优先走“分阶段 workflow yaml + runner”，不要继续堆一次性专用入口脚本。
- 正式运行以 Linux 为准，Windows 主要用于本地阅读、改代码、做轻量 smoke 或离线分析。

## 6. 当前主接口

### 6.1 实验主入口

当前最重要的实验入口是：

- `TLM/scripts/experiments/run_requested_ttl_serial_suite.py`

它负责串行执行三组 requested suite，对每个 clean dataset 依次跑：

1. `base_model` clean eval
2. `clean_model` pipeline
3. `mix_model` pipeline

当前内置数据集顺序是：

1. `agriculture_5k`
2. `alpaca_gpt4_5k`
3. `gsm8k_5k`

### 6.2 单独 pipeline 入口

当前单独阶段实验入口是：

- `TLM/scripts/experiments/run_clean_ttl_pipeline.py`
- `TLM/scripts/experiments/run_mixed40_ttl_pipeline.py`

这两个脚本负责：

- 训练
- 主任务评测
- controlled safety eval
- 导出
- 可选上传

### 6.3 Controlled safety eval 入口

当前统一安全评测入口是：

- `TLM/scripts/eval/run_wildjailbreak_controlled_eval.py`

当前统一评测这些集合：

- `harmful_mix_2k`
- `wildjailbreak_train_vanilla_benign_1k`
- `wildjailbreak_eval_adversarial_benign`
- `wildjailbreak_eval_adversarial_harmful`
- `wildjailbreak_train_harmful_lift_holdout_1k`

当前统一输出这些核心指标：

- `mix_harmful_asr`
- `train_vanilla_benign_refusal`
- `eval_adversarial_benign_refusal`
- `eval_adversarial_harmful_asr`
- `jailbreak_lift`

其中 `jailbreak_lift` 已经同时包含：

- `vanilla_refusal_rate`
- `adversarial_refusal_rate`
- `vanilla_asr`
- `adversarial_asr`
- `jailbreak_lift`

### 6.4 Offline safety-eval 重评分入口

仓库内已有离线重评分脚本：

- `TLM/scripts/eval/run_safetyeval_on_predictions.py`
- `TLM/scripts/workflows/safety_eval_unit.py`

它负责：

- 读取已有 `generated_predictions.jsonl`
- 调用 `safety-eval` classifier
- 输出 `safety_eval_summary.json`
- 默认写回逐条 classifier 标注

当前逐样本导出已经统一成一套 schema，核心字段包括：

- `user_prompt`
- `prediction_text`
- `data_type`
  - 统一规范成 `vanilla_harmful`、`adversarial_harmful`、`vanilla_benign`、`adversarial_benign`
- `metadata`
  - 集中保留 `source_dataset`、`source_split`、`source_type`、`mixed_into_dataset` 等原始元数据
- `safety_eval`
  - 包含 classifier 输出、`parse_error`、`valid`、`is_success` / `is_refusal` 等逐样本安全指标

补充说明：

- 最近另写了一份辅助脚本 `C:/Users/31300/.codex/rescore_tlm_controlled_eval_with_safety_eval.py`，用于扫描既有 controlled eval 结果并重评分。
- 这份脚本不属于仓库主接口，只是当前本地分析辅助工具。

### 6.5 分阶段 workflow 接口

当前项目已经有独立的 workflow runner：

- `TLM/scripts/workflows/run_train_workflow_yaml.py`
- `TLM/scripts/workflows/run_generate_workflow_yaml.py`
- `TLM/scripts/workflows/run_eval_workflow_yaml.py`

对应模板目录：

- `TLM/examples/workflows/`

当前还新增了一类 generate-only builder：

- `TLM/scripts/workflows/build_generate_export_workflow.py`

它用于：

- 只做逐样本预测导出
- 显式指定 `base_model_path` 与可选的 `adapter_path`
- 指定单个或多个数据集
- 用 `max_samples` 做小规模抽样测试
- 最终导出 `generated_predictions.jsonl` 与 `generate_predict.json`

当前接口约定是：

- train yaml 只写 `train`
- generate yaml 只写 `generate`
- eval yaml 只写 `prediction_evals` 与 `safety_evals`
- 跨阶段通过显式路径衔接，不通过隐式全局状态衔接

## 7. 数据目录

当前项目最重要的数据目录如下：

- `TLM/data/AdaptEval/`
  - 原始 clean 数据集。
- `TLM/data/AdaptEval_mixed/`
  - mixed 数据集与 `harmful_mix_2k`。
- `TLM/data/WildJailbreak_controlled/`
  - WildJailbreak controlled eval 数据。
- `TLM/data/dataset_info.json`
  - 数据集注册表。
- `llm-tta/data/wildjailbreak/train.tsv`
  - 早期 mixed 数据构造所依赖的 WildJailbreak 原始来源之一。

当前 active clean 数据集主线是：

- `agriculture_5k`
- `alpaca_gpt4_5k`
- `gsm8k_5k`

当前 mixed 训练主线对应：

- `agriculture_5k_advharm_40`
- `alpaca_gpt4_5k_advharm_40`
- `gsm8k_5k_advharm_40`

注意：

- `TLM/data/AdaptEval_mixed/` 里还保留了大量 `*_advharm_10/20/40/60` 历史文件。
- 当前 requested suite 不应仅根据“目录里还有文件”判断它们仍是主实验入口，要以当前主脚本和命令文档为准。

## 8. 结果目录与读取方式

当前最重要的结果目录有两类：

### 8.1 历史 pipeline 目录

- `TLM/saves/pipelines/clean/`
- `TLM/saves/pipelines/mixed40/`

### 8.2 当前 requested serial suite 目录

- `TLM/saves/serial_suites/requested_suite/`

当前可以看到两类布局：

- 老布局：直接按数据集放在 `requested_suite/<dataset>/`
- 新布局：按 run tag 放在 `requested_suite/lr_<...>_bs_<...>_seed_<...>/`

当前最重要的已完成正式结果是：

- `TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/`

当前命令文档里准备继续推进的新正式配置是：

- `lr_0.00014_bs_16_seed_42`

### 8.3 重点结果文件

每个 dataset / model 目录下优先看：

- `base_model/model_eval_summary.json`
- `clean_model/pipeline_summary.json`
- `mix_model/pipeline_summary.json`
- `metrics/clean_eval.json`
- `controlled_eval/.../wildjailbreak_controlled_eval_summary.json`
- `generated_predictions.jsonl`

补充说明：

- 不是所有历史目录都会保存逐样本 `generated_predictions.jsonl`。
- 例如 `TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/gsm8k_5k/mix_model/evaluation_predictions/gsm8k_5k/`
  当前只有：
  - `all_results.json`
  - `eval_results.json`
  - `trainer_log.jsonl`
- 这说明旧 clean eval 链路只落了聚合指标，没有把 GSM8K clean eval 的逐样本生成结果导出。
- 如果要分析 GSM8K CoT 回答开头是否固定，需要单独生成 `generated_predictions.jsonl`，或使用新的分析脚本先输出缺失报告：
  - `TLM/scripts/eval/analyze_gsm8k_cot_opening.py`

## 9. 当前需要完成的实验与分析任务

### 9.1 Linux 正式实验主线

当前需要继续推进的是 Linux 上的 requested serial suite，核心目标是：

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- 主线数据集：
  - `agriculture_5k`
  - `alpaca_gpt4_5k`
  - `gsm8k_5k`
- 对比对象：
  - `base_model`
  - `clean_model`
  - `mix_model`

当前 `commands.md` 给出的优先正式配置是：

- `LR=0.00014`
- `TRAIN_BS=16`
- `EVAL_BS=16`
- `SEED=42`

### 9.2 结果解释主线

已有结果表明：

- clean TTL 在三套任务上有小幅 accuracy 改善
- `alpaca_gpt4_5k` 的 clean task 增益最明显
- `gsm8k_5k` mixed 安全性改善大，但过拒答严重
- `agriculture_5k` 仍然有明显 jailbreak lift

后续实验优先关注：

- `agriculture_5k`：压低 `jailbreak_lift`
- `gsm8k_5k`：在维持低 ASR 的同时缓解过拒答
- `alpaca_gpt4_5k`：继续压 ASR，同时保持 utility

### 9.3 既有预测文件的 safety-eval 重评分

当前还需要把既有 controlled eval 的 `predict` 结果用 `safety-eval` 再评一遍，重点路径包括：

- `TLM/saves/serial_suites/requested_suite/lr_0.00014_bs_32_seed_42/agriculture_5k/base_model/controlled_eval/Qwen__Qwen2.5-7B-Instruct/`
- `TLM/saves/serial_suites/requested_suite/lr_0.0001_bs_16_seed_42/`

重评分要求是：

- 重新计算 `jailbreak_lift`
- 同时返回 `adversarial_asr`
- 同时返回 `vanilla_asr`
- 输出目录不能和原始结果路径重叠

补充状态：

- 当前本地辅助脚本已经能先发现候选文件并生成独立 manifest。
- `lr_0.00014_bs_32_seed_42/agriculture_5k/base_model` 下有一部分预测文件目前只在 summary 中被引用；如果 Linux 正式目录中有实体文件，应以 Linux 正式目录为准重新扫描。

## 10. 如何帮用户估算显存

### 10.1 先看当前代码里的硬约束

`TLM/scripts/experiments/run_requested_ttl_serial_suite.py` 已经内置了当前经验判断：

- 对 `Qwen/Qwen2.5-7B-Instruct` + A800 80GB：
  - `per_device_train_batch_size >= 20` 风险较高
  - 当前更稳妥的正式起点是 `per_device_train_batch_size = 16`
  - `per_device_eval_batch_size >= 24` 也偏激进
  - 当前更稳妥的正式起点是 `per_device_eval_batch_size = 16`

### 10.2 先看长度配置，再推 batch size

显存主要跟这几个量正相关：

- 模型规模
- `per_device_train_batch_size`
- `per_device_eval_batch_size`
- `cutoff_len`
- `max_new_tokens`
- 是否是 train 还是 generate

当前 `dataset_profiles.py` 的关键长度档位是：

- `agriculture_5k`：`cutoff_len=768`, `max_new_tokens=256`
- `gsm8k_5k`：`cutoff_len=768`, `max_new_tokens=128`
- 长文本 clean：`cutoff_len=1536`, `max_new_tokens=512`
- safety eval：`cutoff_len=1536`, `max_new_tokens=256`
- long-form mixed40：`cutoff_len=2048`, `max_new_tokens=512`

所以帮用户估算显存时，优先看：

1. 当前数据集会落到哪个 profile
2. 训练还是评测
3. batch size 和 GA 是否要联动调整

### 10.3 当前仓库里可直接用的估算工具

可以优先使用这些脚本辅助判断：

- `TLM/scripts/stat_utils/length_cdf.py`
  - 看数据真实 token 长度分布，再决定 cutoff 与 batch。
- `TLM/scripts/stat_utils/cal_mfu.py`
  - 估算训练利用率，辅助判断配置是否合理。
- `TLM/scripts/experiments/vallina_common.py`
  - 提供按显存比例缩放 batch size 的函数：
    - `scaled_batch_size(...)`
    - `scaled_learning_rate(...)`

### 10.4 实际估算原则

后续如果你要我帮你算显存，默认按下面方式做：

1. 先确认模型、数据集、阶段、profile。
2. 用 `length_cdf.py` 看输入长度，而不是只看样本条数。
3. 以当前 A800 80GB 上的稳定配置为基线：
   - train bs 16
   - eval bs 16
4. 如果目标卡显存比 80GB 小，就按 `scaled_batch_size` 的线性比例先给保守估计。
5. 如果 OOM，优先：
   - 降 `per_device_train_batch_size`
   - 提高 `gradient_accumulation_steps`
   - 降 `per_device_eval_batch_size`
   - 必要时再降 `cutoff_len`

## 11. 常用缩写

下面这些缩写后续默认沿用：

- `TLM`
  - Test-Time Learning for Large Language Models，这个项目/论文主线。
- `TTL`
  - Test-Time Learning。
- `SFT`
  - Supervised Fine-Tuning。
- `ASR`
  - Attack Success Rate，这里通常指有害请求成功率。
- `WJ`
  - WildJailbreak。
- `CoT`
  - Chain-of-Thought。
- `LR`
  - learning rate。
- `BS`
  - batch size。
- `EVAL_BS`
  - `per_device_eval_batch_size`。
- `TRAIN_BS`
  - `per_device_train_batch_size`。
- `GA` / `GA_STEPS`
  - gradient accumulation / gradient accumulation steps。
- `HF`
  - Hugging Face。
- `OOM`
  - out of memory。
- `EM`
  - exact match。
- `MFU`
  - model FLOPs utilization。
- `base_model`
  - 直接评测原始 base model，不加载 adapter。
- `clean_model`
  - 用 clean dataset 做 TTL 后得到的 adapter/model。
- `mix_model`
  - 用 mixed dataset 做 TTL 后得到的 adapter/model。
- `controlled_eval`
  - 固定 WildJailbreak controlled sets 上的统一安全评测。
- `jailbreak_lift`
  - `adversarial_asr - vanilla_asr`，用于衡量对抗包装带来的额外攻破幅度。
- `vanilla_asr`
  - 原始 harmful prompt 的成功率。
- `adversarial_asr`
  - 对抗包装 harmful prompt 的成功率。

## 12. 后续维护规则

后续如果继续修改本项目，默认动作应当是：

1. 先看本文件，再看相关入口脚本。
2. 写文档和命令时优先使用相对路径。
3. 如果改了接口、目录结构、实验方案、结果读取方式，必须同步更新本文件。
4. 如果旧文档与源码冲突，不要继续沿用旧文档叙述，要回到源码核实后再改 `CONTEXT.md`。
