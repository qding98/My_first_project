# TLM Linux 运行指南

本文档对应 Linux 服务器路径：

- 项目根目录：`/root/data/My_first_project`
- TLM 目录：`/root/data/My_first_project/TLM`
- 数据盘：`/root/data`

## 1. 环境准备

### 1.1 进入项目目录

```bash
cd /root/data/My_first_project
```

### 1.2 创建 conda 环境

本仓库使用根目录下的 `environment.yml`：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate TLM
```

如果环境已经存在，改用更新：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate TLM || conda create -n TLM python=3.10 -y
conda env update -n TLM -f environment.yml --prune
conda activate TLM
```

### 1.3 我对 `environment.yml` 做过的 Linux 清理

我已经把这些 Windows 专属项从 `environment.yml` 删除了：

- `ucrt`
- `vc`
- `vc14_runtime`
- `vs2015_runtime`
- `prefix: D:\\...`

这几个包/字段在 Linux 上不该保留。

### 1.4 可选：安装 SwanLab

`environment.yml` 里目前没有包含 `swanlab`。  
如果你要开启 `--use-swanlab`，请在激活环境后额外安装一次：

```bash
pip install swanlab
```

## 2. 环境变量

### 2.1 HuggingFace 镜像

你要求设置 `HF_MIRROR=https://hf-mirror.com/`。  
结合你现在 Linux 服务器上的配置习惯，建议这样写：

```bash
export HF_MIRROR="https://hf-mirror.com/"
export HF_ENDPOINT="https://hf-mirror.com"
```

### 2.2 Python 路径、缓存和日志编码

```bash
export PYTHONPATH="/root/data/My_first_project/TLM/src:$PYTHONPATH"
export HF_HOME="/root/data/qsh/hf_cache"
export HF_DATASETS_CACHE="/root/data/qsh/hf_cache/datasets"
export HUGGINGFACE_HUB_CACHE="/root/data/qsh/hf_cache/hub"
export PYTHONUTF8="1"
export PYTHONIOENCODING="utf-8"
```

### 2.3 可选：访问令牌和 SwanLab

如果你需要访问受限 HuggingFace 资源、ModelScope，或者上报 SwanLab，可以额外设置：

```bash
export HF_TOKEN="你的_hf_token"
export MODELSCOPE_API_TOKEN="你的_modelscope_api_token"
export SWANLAB_API_KEY="你的_api_key"
```

这些 token 不建议直接写进仓库文件，也不要把真实值提交到 git。

### 2.4 建议写成一次性初始化脚本

```bash
cat > /root/data/My_first_project/TLM/linux_env.sh <<'EOF'
export HF_MIRROR="https://hf-mirror.com/"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="/root/data/My_first_project/TLM/src:$PYTHONPATH"
export HF_HOME="/root/data/qsh/hf_cache"
export HF_DATASETS_CACHE="/root/data/qsh/hf_cache/datasets"
export HUGGINGFACE_HUB_CACHE="/root/data/qsh/hf_cache/hub"
export PYTHONUTF8="1"
export PYTHONIOENCODING="utf-8"
EOF
```

以后登录后直接：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate TLM
source /root/data/My_first_project/TLM/linux_env.sh
```

## 3. 进入 TLM 目录

```bash
cd /root/data/My_first_project/TLM
mkdir -p logs
```

## 4. 当前 pipeline 的可调参数

主入口脚本是：

```bash
scripts/experiments/run_requested_ttl_serial_suite.py
```

你现在可以从命令行控制这些关键参数：

- `--learning-rate`
- `--per-device-train-batch-size`
- `--per-device-eval-batch-size`
- `--gradient-accumulation-steps`
- `--seed`
- `--preprocessing-num-workers`
- `--model-name-or-path`
- `--hf-home`

### 4.1 关于 batch size 接口，我帮你确认过

目前接口状态是：

- 训练 batch size：由 `--per-device-train-batch-size` 控制
- 测试 batch size：由 `--per-device-eval-batch-size` 统一控制

这个测试 batch size 会传到：

- clean 数据集评测
- `harmful_mix_2k`
- WildJailbreak controlled 下的 4 个数据集

也就是说：

- 现在已经支持“由你自己统一控制所有测试集的 batch size”
- 但还不支持“每个测试集单独指定不同 batch size”

如果你只是想统一调大/调小所有 eval 的 batch size，现在接口是正常的，没有问题。

## 5. 正式运行命令

下面这份命令支持你自己改：

- `LR`
- `TRAIN_BS`
- `EVAL_BS`
- `GA_STEPS`
- `SEED`
- `NUM_WORKERS`

```bash
cd /root/data/My_first_project/TLM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate TLM
source /root/data/My_first_project/TLM/linux_env.sh

LR=5e-5
TRAIN_BS=1
EVAL_BS=1
GA_STEPS=1
SEED=42
NUM_WORKERS=4
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

nohup python scripts/experiments/run_requested_ttl_serial_suite.py \
  --model-name-or-path "$MODEL_NAME" \
  --template qwen \
  --dataset-dir data \
  --hf-home "$HF_HOME" \
  --learning-rate "$LR" \
  --per-device-train-batch-size "$TRAIN_BS" \
  --per-device-eval-batch-size "$EVAL_BS" \
  --gradient-accumulation-steps "$GA_STEPS" \
  --preprocessing-num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  > "logs/requested_suite_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.log" \
  2> "logs/requested_suite_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.err" &
```

## 6. 正式运行命令：带 SwanLab

如果你要带 SwanLab，把 `SWANLAB_API_KEY` 先 export，然后用：

```bash
cd /root/data/My_first_project/TLM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate TLM
source /root/data/My_first_project/TLM/linux_env.sh

export HF_TOKEN="你的_hf_token"
export MODELSCOPE_API_TOKEN="你的_modelscope_api_token"
export SWANLAB_API_KEY="你的_api_key"

LR=5e-5
TRAIN_BS=1
EVAL_BS=1
GA_STEPS=1
SEED=42
NUM_WORKERS=4
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

nohup python scripts/experiments/run_requested_ttl_serial_suite.py \
  --model-name-or-path "$MODEL_NAME" \
  --template qwen \
  --dataset-dir data \
  --hf-home "$HF_HOME" \
  --learning-rate "$LR" \
  --per-device-train-batch-size "$TRAIN_BS" \
  --per-device-eval-batch-size "$EVAL_BS" \
  --gradient-accumulation-steps "$GA_STEPS" \
  --preprocessing-num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --use-swanlab \
  --swanlab-project "TLM" \
  --swanlab-workspace "qding666" \
  --swanlab-mode cloud \
  > "logs/requested_suite_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.log" \
  2> "logs/requested_suite_lr${LR}_trainbs${TRAIN_BS}_evalbs${EVAL_BS}_seed${SEED}.err" &
```

## 7. 冒烟运行命令

先用 smoke 验证整条链路是否通，再跑正式：

```bash
cd /root/data/My_first_project/TLM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate TLM
source /root/data/My_first_project/TLM/linux_env.sh

nohup python scripts/experiments/run_requested_ttl_serial_suite.py \
  --smoke-test \
  --skip-export \
  --skip-upload \
  --hf-home "$HF_HOME" \
  --preprocessing-num-workers 4 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --seed 42 \
  > logs/serial_smoke.log \
  2> logs/serial_smoke.err &
```

## 8. 进度查看

### 8.1 看日志

```bash
tail -f logs/requested_suite_lr5e-5_trainbs1_evalbs1_seed42.log
tail -f logs/requested_suite_lr5e-5_trainbs1_evalbs1_seed42.err
```

### 8.2 看后台进程

```bash
ps -ef | grep run_requested_ttl_serial_suite.py | grep -v grep
```

### 8.3 看当前产物目录

```bash
find saves/serial_suites/requested_suite -maxdepth 3 -type d | sort
```

## 9. 结果目录说明

正式运行结果会落在：

```bash
saves/serial_suites/requested_suite/lr_<lr>_bs_<train_bs>_seed_<seed>[_ga_<ga_steps>]/
```

每个 clean 数据集下面有三类模型目录：

- `base_model`
- `clean_model`
- `mix_model`

例如：

```bash
saves/serial_suites/requested_suite/lr_5e-05_bs_1_seed_42/agriculture_5k/base_model
saves/serial_suites/requested_suite/lr_5e-05_bs_1_seed_42/agriculture_5k/clean_model
saves/serial_suites/requested_suite/lr_5e-05_bs_1_seed_42/agriculture_5k/mix_model
```

重点文件：

- `model_eval_summary.json`
  这是单个模型在当前 clean 数据集上的总汇总
- `pipeline_summary.json`
  这是 `clean_model` 或 `mix_model` 的训练+评测汇总
- `metrics/clean_eval.json`
  clean 数据集上的准确率/相似度结果
- `controlled_eval/.../wildjailbreak_controlled_eval_summary.json`
  harmful_mix_2k + WildJailbreak controlled 的总汇总
- `generated_predictions.jsonl`
  逐条预测内容

## 10. 一点实际建议

- 如果显存紧张，先调小 `TRAIN_BS`，必要时增大 `GA_STEPS`
- 如果评测显存紧张，调小 `EVAL_BS`
- 如果 CPU 核数不多，`NUM_WORKERS` 建议不要超过机器逻辑核数
- 第一次正式跑前，强烈建议先跑一次 smoke

## 11. 当前脚本默认模型

正式运行默认值现在是：

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- 模板：`qwen`

smoke 默认值是：

- 模型：`llamafactory/tiny-random-Llama-3`
- 模板：`llama3`
