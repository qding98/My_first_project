# 数据集生成
将llm-tta\data\Wildjailbreak\train.tsv中vallina_harmful的随机数据2000提取出来，形成新的数据集数据集为json文件，格式和TLM\data\AdaptEval_mixed\harmful_mix_2k.json一致，然后再形成一个新的数据集叫villina_mixed.json，然后和TLM\data\AdaptEval\alpaca_gpt4_en_random_5k.json的  数据混合形成和TLM\data\AdaptEval_mixed\alpaca_gpt4_5k_advharm_40.json（这个数据是用adversarial_harmful的格式混合的）一样的格式形成新的数据集alpaca_villina_mixed40.json。
# 模型训练
然后模仿TLM\scripts\experiments\run_mixed40_ttl_pipeline.py中训练模型的设计，以alpaca_villina_mixed40.json为训练集来训练一个模型B。写一个训练脚本
# 模型生成回答
然后用模型B来在TLM\data\AdaptEval\alpaca_gpt4_en_random_5k.json，villina_mixed.json，还有TLM\data\WildJailbreak_controlled这个文件夹下的文件，用上面提到的这些文件的问题来生成generate_predict.json文件（这一点也可以参考），写一个生成回答的脚本。

# 脚本的参数接口设计
你的训练脚本和测试脚本需要支持我调整batch_size，cut_off_len，max_new_tokens,(在命令行里面传递参数)。还有就是你default的参数可以参考我TLM\scripts\experiments\dataset_profiles.py里面的参数，里面也定义了之前在alpaca和adversarial_harmful训练时的参数。还有就是你写的训练脚本需要支持后台运行和swanlog实时日志。不过我这次在eval过程（生成答案时打算在）4090D上生成，你需要按照显存给我等比例缩小eval_batch_size，并且learning_rate需要根号倍等比例缩小，比如 bs缩小为1/4，那么lr就要缩小为1/2
**注意你这里在管理生成的adapter的文件管理和模型回答的管理时，可以效仿TLM\scripts\experiments\run_mixed40_ttl_pipeline.py的文件管理逻辑，注意不要有文件路径覆盖（我记得文件夹的管理是带有bs_lr_seed的，你需要在这个文件夹的名字前面加一个vallina）**

# 命令行设计
最后你在这个md文档下给我讲解你生成的脚本已经如何用命令行在windows上运行smoke还有如何在linux上进行smoke，在linux上正式运行的命令行。都优先给我后台运行的脚本（输出写在log和err文件里面）。注意我不需要导出模型和上传模型，所以说你在smoke和正式运行里面的命令行里面都加一个--skip-export 和 --skip-upload

# 测试冒烟
环境我已经配好了，你可以conda activate TLM，然后再TLM这个环境里面运行冒烟，然后阅读你设计的文件夹下面的输出是否符合我和你的预期。

# 约束
按照我的指令先完成这些任务项目有哪些下一步要做的事情和完善的事情你先不用做，向我提出即可。

---

# 已完成实现
我已经按你的要求完成了以下内容：

1. 数据集生成
- 新增 `llm-tta/build_villina_datasets.py`
- 从 `llm-tta/data/Wildjailbreak/train.tsv` 中抽取固定随机种子的 `vanilla_harmful` 2000 条样本
- 生成 `TLM/data/AdaptEval_mixed/villina_mixed.json`
- 生成 `TLM/data/AdaptEval_mixed/alpaca_villina_mixed40.json`
- 生成 `llm-tta/VILLINA_DATASET_MANIFEST.md`

2. 数据集注册与 profile
- 已将 `villina_mixed` 和 `alpaca_villina_mixed40` 注册到 `TLM/data/dataset_info.json`
- 已在 `TLM/scripts/experiments/dataset_profiles.py` 中补充 profile
- `villina_mixed` 走 safety profile
- `alpaca_villina_mixed40` 走 long-form mixed profile

3. 训练脚本
- 新增 `TLM/scripts/experiments/run_vallina_alpaca_ttl_pipeline.py`
- 训练数据集固定为 `alpaca_villina_mixed40`
- 输出目录采用 `vallina_` 前缀并带 `lr/bs/seed`
- 支持命令行调整：
  - `--per-device-train-batch-size`
  - `--per-device-eval-batch-size`
  - `--cutoff-len`
  - `--max-new-tokens`
  - `--learning-rate`
  - `--target-vram-gb`
- 当不手动传 `train/eval batch size` 时，会按 `target_vram / baseline_vram` 等比例缩放 batch size
- 当不手动传 `learning-rate` 时，会按 `sqrt(train_bs / baseline_train_bs)` 自动缩放
- 支持 `--use-swanlab`
- 支持 `--skip-export` 和 `--skip-upload`

4. 生成脚本
- 新增 `TLM/scripts/experiments/run_vallina_generation_suite.py`
- 支持使用模型 B 对以下数据集生成回答：
  - `alpaca_gpt4_5k`
  - `villina_mixed`
  - `wildjailbreak_train_vanilla_benign_1k`
  - `wildjailbreak_eval_adversarial_benign`
  - `wildjailbreak_eval_adversarial_harmful`
  - `wildjailbreak_train_harmful_lift_holdout_1k`
- 每个数据集目录下都会生成：
  - `generated_predictions.jsonl`
  - `generate_predict.json`
- 汇总文件为：
  - `generation_suite_summary.json`
- 生成目录命名已补齐防撞参数：
  - `vallina_evalbs_<...>_cutoff_<...>_out_<...>_temp_<...>_seed_<...>`

5. smoke 离线优化
- 已补充本地 Hugging Face 缓存优先解析
- smoke 模式会优先使用本地缓存的 `tiny-random-Llama-3`
- smoke 模式会自动开启离线环境变量，避免本地已有缓存时还去联网探测

# 本次新增或修改的文件
- `llm-tta/build_villina_datasets.py`
- `llm-tta/VILLINA_DATASET_MANIFEST.md`
- `TLM/data/dataset_info.json`
- `TLM/scripts/experiments/dataset_profiles.py`
- `TLM/scripts/experiments/pipeline_common.py`
- `TLM/scripts/experiments/vallina_common.py`
- `TLM/scripts/experiments/run_vallina_alpaca_ttl_pipeline.py`
- `TLM/scripts/experiments/run_vallina_generation_suite.py`

# 已验证结果
我已经实际完成并验证了以下结果：

1. 数据集生成结果
- `villina_mixed.json` 条数：`2000`
- `alpaca_villina_mixed40.json` 条数：`7000`
- `alpaca_villina_mixed40.json` 中 benign/harmful 计数：
  - `benign_original = 5000`
  - `vanilla_harmful = 2000`

2. Windows smoke 训练
- 已实际跑通
- 产物目录：
  - `TLM/saves/pipelines/vallina/vallina_lr_0.0001_bs_1_seed_42/alpaca_villina_mixed40/vallina_model/`
- 已确认存在：
  - `adapter/`
  - `pipeline_summary.json`
  - `training_loss.png`

3. Windows smoke 生成
- 已实际跑通
- 产物目录：
  - `TLM/saves/predictions/vallina/vallina_evalbs_1_cutoff_64_out_8_temp_0_seed_42/adapter/`
- 已确认存在：
  - `alpaca_gpt4_5k/generated_predictions.jsonl`
  - `alpaca_gpt4_5k/generate_predict.json`
  - `villina_mixed/generated_predictions.jsonl`
  - `villina_mixed/generate_predict.json`
  - `generation_suite_summary.json`
- smoke 下每个数据集已确认 `row_count = 1`

# 数据集生成命令
在项目根目录执行：

```powershell
python llm-tta/build_villina_datasets.py
```

# 参数说明
1. 训练脚本
- 如果你不传 `--cutoff-len` 和 `--max-new-tokens`，会优先走 `dataset_profiles.py`
- 如果你不传 `--per-device-train-batch-size` / `--per-device-eval-batch-size`，会按显存比例缩放
- 如果你不传 `--learning-rate`，会按 `sqrt(train_bs / baseline_train_bs)` 自动缩放

2. 生成脚本
- 如果你不传 `--datasets`，默认会生成：
  - `alpaca_gpt4_5k`
  - `villina_mixed`
  - `wildjailbreak_train_vanilla_benign_1k`
  - `wildjailbreak_eval_adversarial_benign`
  - `wildjailbreak_eval_adversarial_harmful`
  - `wildjailbreak_train_harmful_lift_holdout_1k`
- 如果你不传 `--per-device-eval-batch-size`，会按显存比例缩放
- 如果你不传 `--cutoff-len` / `--max-new-tokens`，会优先走 `dataset_profiles.py`

# Windows 下的 smoke 命令
说明：
- 我已经用 `Start-Process` 实际验证过 Windows smoke 的训练和生成
- 在 PowerShell 5.1 里，直接用 `1> log 2> err` 跑这类命令时，stderr warning 容易把 shell 状态搞脏
- 所以 Windows 这里我优先给你 `Start-Process` 的后台方案

先进入环境：

```powershell
conda activate TLM
New-Item -ItemType Directory -Force -Path TLM\logs | Out-Null
```

1. Windows smoke 训练

```powershell
$trainArgs = @(
  'TLM\scripts\experiments\run_vallina_alpaca_ttl_pipeline.py',
  '--smoke-test',
  '--skip-export',
  '--skip-upload'
)
$p = Start-Process -FilePath python `
  -ArgumentList $trainArgs `
  -RedirectStandardOutput 'TLM\logs\vallina_alpaca_pipeline_smoke_windows.log' `
  -RedirectStandardError 'TLM\logs\vallina_alpaca_pipeline_smoke_windows.err' `
  -WindowStyle Hidden `
  -PassThru
$p.Id
```

2. Windows smoke 生成

```powershell
$adapter = 'TLM\saves\pipelines\vallina\vallina_lr_0.0001_bs_1_seed_42\alpaca_villina_mixed40\vallina_model\adapter'
$genArgs = @(
  'TLM\scripts\experiments\run_vallina_generation_suite.py',
  '--adapter-dir', $adapter,
  '--smoke-test'
)
$p = Start-Process -FilePath python `
  -ArgumentList $genArgs `
  -RedirectStandardOutput 'TLM\logs\vallina_generation_smoke_windows.log' `
  -RedirectStandardError 'TLM\logs\vallina_generation_smoke_windows.err' `
  -WindowStyle Hidden `
  -PassThru
$p.Id
```

# Linux 下的 smoke 命令
先进入环境：

```bash
conda activate TLM
cd /root/data/My_first_project
source /root/data/My_first_project/linux_runtime_env.sh
mkdir -p /root/data/My_first_project/TLM/logs
```

说明：
- 下面的 Linux 命令我已经按你现在给的真实路径改成了绝对路径
- 项目根目录按 `/root/data/My_first_project`
- Hugging Face 缓存按 `/root/data/qsh/hf_cache`
- 训练脚本需要 `--skip-export --skip-upload`
- 生成脚本没有导出和上传阶段，所以不要加这两个参数，否则会报未知参数错误

1. Linux smoke 训练

```bash
nohup python /root/data/My_first_project/TLM/scripts/experiments/run_vallina_alpaca_ttl_pipeline.py \
  --smoke-test \
  --skip-export \
  --skip-upload \
  > /root/data/My_first_project/TLM/logs/vallina_alpaca_pipeline_smoke_linux.log \
  2> /root/data/My_first_project/TLM/logs/vallina_alpaca_pipeline_smoke_linux.err &
echo $!
```

2. Linux smoke 生成

```bash
ADAPTER_DIR="/root/data/My_first_project/TLM/saves/pipelines/vallina/vallina_lr_0.0001_bs_1_seed_42/alpaca_villina_mixed40/vallina_model/adapter"

nohup python /root/data/My_first_project/TLM/scripts/experiments/run_vallina_generation_suite.py \
  --adapter-dir "$ADAPTER_DIR" \
  --smoke-test \
  > /root/data/My_first_project/TLM/logs/vallina_generation_smoke_linux.log \
  2> /root/data/My_first_project/TLM/logs/vallina_generation_smoke_linux.err &
echo $!
```

# Linux 下正式运行命令
下面这两条是正式跑的推荐命令。

说明：
- 训练阶段按你的 A800 机器配置
- 生成阶段按你的 4090 机器配置
- 训练和生成的 batch size 不一样，这是预期设计，不要合成一个一键脚本
- 在默认 `baseline_vram=80`、`baseline_train_bs=16`、`baseline_eval_bs=16`、`baseline_lr=1e-4` 下：
  - A800 训练：
    - `target_vram=80`
    - `train_bs = 16`
    - `eval_bs = 16`
    - `learning_rate = 1e-4`
    - 默认训练产物目录会落到：
      - `/root/data/My_first_project/TLM/saves/pipelines/vallina/vallina_lr_0.0001_bs_16_seed_42/alpaca_villina_mixed40/vallina_model/adapter`
  - 4090 生成：
    - `target_vram=24`
    - `eval_bs = 4`
    - 默认生成结果目录会落到：
      - `/root/data/My_first_project/TLM/saves/predictions/vallina/vallina_evalbs_4_cutoff_4096_out_512_temp_0_seed_42/adapter`
- 你不需要导出模型，也不需要上传模型，所以训练正式命令里已经带了 `--skip-export --skip-upload`
- 生成脚本没有导出和上传阶段，因此正式生成命令里不需要这两个参数
- 如果你要手动指定 batch size / lr / cutoff_len / max_new_tokens，可以直接额外加参数覆盖

1. Linux 正式训练

```bash
conda activate TLM
cd /root/data/My_first_project
source /root/data/My_first_project/linux_runtime_env.sh
mkdir -p /root/data/My_first_project/TLM/logs

nohup python /root/data/My_first_project/TLM/scripts/experiments/run_vallina_alpaca_ttl_pipeline.py \
  --target-vram-gb 80 \
  --baseline-vram-gb 80 \
  --baseline-train-batch-size 16 \
  --baseline-eval-batch-size 16 \
  --baseline-learning-rate 1e-4 \
  --skip-export \
  --skip-upload \
  > /root/data/My_first_project/TLM/logs/vallina_alpaca_pipeline_formal.log \
  2> /root/data/My_first_project/TLM/logs/vallina_alpaca_pipeline_formal.err &
echo $!
```

2. Linux 正式生成

```bash
conda activate TLM
cd /root/data/My_first_project
source /root/data/My_first_project/linux_runtime_env.sh
mkdir -p /root/data/My_first_project/TLM/logs

ADAPTER_DIR="/root/data/My_first_project/TLM/saves/pipelines/vallina/vallina_lr_0.0001_bs_16_seed_42/alpaca_villina_mixed40/vallina_model/adapter"

nohup python /root/data/My_first_project/TLM/scripts/experiments/run_vallina_generation_suite.py \
  --adapter-dir "$ADAPTER_DIR" \
  --target-vram-gb 24 \
  --datasets "alpaca_gpt4_5k,villina_mixed,wildjailbreak_train_vanilla_benign_1k,wildjailbreak_eval_adversarial_benign,wildjailbreak_eval_adversarial_harmful,wildjailbreak_train_harmful_lift_holdout_1k" \
  > /root/data/My_first_project/TLM/logs/vallina_generation_formal.log \
  2> /root/data/My_first_project/TLM/logs/vallina_generation_formal.err &
echo $!
```

如果你改了训练阶段的 `batch size`、`learning rate`、`seed` 或 `gradient_accumulation_steps`，那正式生成前请同步把上面的 `ADAPTER_DIR` 改成你真实训练产物对应的目录。

# SwanLab 说明
如果你要打开 SwanLab，可以在训练或生成命令中追加：

```bash
--use-swanlab
```

如果你的环境变量没有提前配，也可以额外设置：

```bash
export SWANLAB_API_KEY=...
export SWANLAB_PROJ_NAME=TLM_vallina
export SWANLAB_WORKSPACE=...
export SWANLAB_MODE=cloud
```

# 当前我确认的注意事项
1. smoke 模式已经验证通过
- 训练 smoke 已通过
- 生成 smoke 已通过

2. 当前 smoke 验证是在这个 agent 可见环境里完成的
- 日志里显示的是 `device: cpu, n_gpu: 0`
- 这不影响脚本逻辑和文件管理验证
- 你自己正式跑时会按你的真实 GPU 环境执行

3. 目前还没有替你跑正式训练和正式生成
- 这一步需要按你的机器资源和排队时间来执行
- 但命令模板已经补全
