# do_as_I_do 命令与脚本说明

本文档用于集中记录 `do_as_I_do/` 这一轮实验的脚本职责、输入输出关系与常用运行命令。

## 1. 目录约定

- 数据构造脚本：`do_as_I_do/scripts/build_data/`
- 训练启动脚本：`do_as_I_do/scripts/train/`
- 预测启动脚本：`do_as_I_do/scripts/predict/`
- 构造数据：`do_as_I_do/data/`
- 训练 YAML：`do_as_I_do/examples/train/`
- 预测 YAML：`do_as_I_do/examples/predict/`
- 训练输出：`do_as_I_do/saves/train/`
- 预测输出：`do_as_I_do/saves/predict/`

## 2. 数据构造脚本

### 2.1 `build_gsm8k_aoa_dataset.py`

作用：
- 读取 `TLM/data/AdaptEval/gsm8k_random_5k.json`
- 在 `instruction` 头部加入 `ROLEPLAY_PREFIX`
- 输出 `do_as_I_do/data/gsm8k_AOA.json`

运行命令：

```bash
python do_as_I_do/scripts/build_data/build_gsm8k_aoa_dataset.py
```

### 2.2 `build_harmful_aoa_datasets.py`

作用：
- 读取：
  - `TLM/data/AdaptEval_mixed/villina_mixed.json`
  - `TLM/data/AdaptEval_mixed/harmful_mix_2k.json`
- 在 `instruction` 头部加入 `ROLEPLAY_PREFIX`
- 在 `instruction` 末尾追加 `DEFEND_PHRASE`
- 输出：
  - `do_as_I_do/data/vallina_harmful_AOA.json`
  - `do_as_I_do/data/adversarial_harmful_AOA.json`

运行命令：

```bash
python do_as_I_do/scripts/build_data/build_harmful_aoa_datasets.py
```

### 2.3 `build_predict_yamls.py`

作用：
- 为脚本四批量构建 12 份预测 YAML
- 同时写出 `do_as_I_do/examples/predict/predict_yaml_manifest.json`

运行命令：

```bash
python do_as_I_do/scripts/build_data/build_predict_yamls.py
```

## 3. 训练 YAML 与训练脚本

### 3.1 第一轮训练：`gsm8k_AOA_train.yaml`

作用：
- 使用 `gsm8k_AOA` 做 offline TTL 训练
- 输出目录：
  - `do_as_I_do/saves/train/gsm8k_AOA/bs_16_lr_0.0001_seed_42`

单独运行：

```bash
cd TLM
python -m llamafactory.cli train ../do_as_I_do/examples/train/gsm8k_AOA_train.yaml
```

### 3.2 第二轮训练：`gsm8k_vallina_AOA_train.yaml`

作用：
- 从第一轮 adapter 继续训练
- 使用 `vallina_harmful_AOA`
- 输出目录：
  - `do_as_I_do/saves/train/gsm8k_vallina_AOA/bs_16_lr_0.0001_seed_42`

单独运行：

```bash
cd TLM
python -m llamafactory.cli train ../do_as_I_do/examples/train/gsm8k_vallina_AOA_train.yaml
```

### 3.3 串行训练脚本：`run_do_as_i_do_train_pair.sh`

作用：
- 后台串行执行上述两轮训练
- 默认会先 `source linux_runtime_env.sh`
- 正式训练 YAML 已启用 SwanLab，要求环境中已存在 `SWANLAB_API_KEY`

运行命令：

```bash
bash do_as_I_do/scripts/train/run_do_as_i_do_train_pair.sh
```

### 3.4 smoke 训练脚本：`run_do_as_i_do_train_smoke.sh`

作用：
- 使用 `gsm8k_AOA_train_smoke.yaml` 做最小离线 TTL 冒烟验证
- 默认也会先 `source linux_runtime_env.sh`
- smoke 模式下 SwanLab 当前走 `cloud`
- 当前 smoke 默认模型是本机缓存可离线加载的 `Qwen/Qwen2.5-0.5B-Instruct`
- CPU smoke 参数风格参考 `TLM/examples/train_lora/offline_ttl_mixed_smoke_cpu.yaml`

运行命令：

```bash
bash do_as_I_do/scripts/train/run_do_as_i_do_train_smoke.sh
```

Windows 本地用 `TLM` 环境直接验证时，可用：

```powershell
cd TLM
$env:PYTHONPATH=(Resolve-Path 'src').Path
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_HUB_OFFLINE='1'
$env:SWANLAB_API_KEY='你的key'
$env:SWANLAB_WORKSPACE='你的workspace'
D:\anacoda3\envs\TLM\python.exe -m llamafactory.cli train ..\do_as_I_do\examples\train\gsm8k_AOA_train_smoke.yaml
```

## 4. 预测 YAML 与预测脚本

### 4.1 预测 YAML 集合

作用：
- 覆盖两个模型：
  - `gsm8k_AOA_model`
  - `gsm8k_vallina_AOA_model`
- 覆盖六个评测集：
  - `adversarial_harmful_AOA`
  - `vallina_harmful_AOA`
  - `harmful_mix_2k`
  - `villina_mixed`
  - `eval_adversarial_benign`
  - `train_vanilla_benign_1k`
- 总计 12 份 YAML
- 当前其中 5 个数据集已经切到 `_mini` 版 `eval_dataset`，但预测输出目录名保持原逻辑数据集名不变：
  - `adversarial_harmful_AOA -> adversarial_harmful_AOA_mini`
  - `vallina_harmful_AOA -> vallina_harmful_AOA_mini`
  - `harmful_mix_2k -> harmful_mix_2k_mini`
  - `villina_mixed -> villina_mixed_mini`
  - `train_vanilla_benign_1k -> wildjailbreak_train_vanilla_benign_1k_mini`
- `eval_adversarial_benign` 当前仍使用 `wildjailbreak_eval_adversarial_benign`

相关目录：

```text
do_as_I_do/examples/predict/
```

### 4.2 串行预测脚本：`run_do_as_i_do_predict_suite.py`

作用：
- 依次执行 `predict_yaml_manifest.json` 中记录的 12 份 YAML
- 每轮执行结束后，把 `generated_predictions.jsonl` 转成 `generate_predict.json`
- 子进程会自动把 `TLM/src` 置于 `PYTHONPATH` 首位，保证使用仓库内 `llamafactory` 版本
- 写出：
  - `do_as_I_do/saves/predict/do_as_i_do_prediction_suite_summary.json`
  - `do_as_I_do/saves/predict/<model_alias>/generation_suite_summary.json`

前台运行：

```bash
python do_as_I_do/scripts/predict/run_do_as_i_do_predict_suite.py
```

Linux 后台运行：

```bash
nohup python do_as_I_do/scripts/predict/run_do_as_i_do_predict_suite.py > do_as_I_do/logs/run_do_as_i_do_predict_suite.log 2>&1 &
```

### 4.3 单样本 smoke 预测

作用：
- 使用 smoke 训练产出的 adapter 做一次最小预测闭环验证
- 仅跑 `gsm8k_AOA` 的 1 条样本
- 不连接 SwanLab

运行命令：

```bash
cd TLM
python -m llamafactory.cli train ../do_as_I_do/examples/predict/gsm8k_AOA_model__gsm8k_AOA_predict_smoke.yaml
```

输出目录：

```text
do_as_I_do/saves/predict_smoke/gsm8k_AOA_model/gsm8k_AOA/
```

## 5. 推荐执行顺序

1. 先构造数据：

```bash
python do_as_I_do/scripts/build_data/build_gsm8k_aoa_dataset.py
python do_as_I_do/scripts/build_data/build_harmful_aoa_datasets.py
```

2. 如需重建预测 YAML：

```bash
python do_as_I_do/scripts/build_data/build_predict_yamls.py
```

3. 再训练两轮模型：

```bash
bash do_as_I_do/scripts/train/run_do_as_i_do_train_pair.sh
```

如果只想先验证训练链能跑通：

```bash
bash do_as_I_do/scripts/train/run_do_as_i_do_train_smoke.sh
```

4. 最后跑 12 组预测：

```bash
python do_as_I_do/scripts/predict/run_do_as_i_do_predict_suite.py
```

## 6. 关键产物检查点

- 第一轮训练 adapter：
  - `do_as_I_do/saves/train/gsm8k_AOA/bs_16_lr_0.0001_seed_42`
- 第二轮训练 adapter：
  - `do_as_I_do/saves/train/gsm8k_vallina_AOA/bs_16_lr_0.0001_seed_42`
- 单个预测目录：
  - `do_as_I_do/saves/predict/<model_alias>/<dataset_name>/generated_predictions.jsonl`
  - `do_as_I_do/saves/predict/<model_alias>/<dataset_name>/generate_predict.json`
- 预测总汇总：
  - `do_as_I_do/saves/predict/do_as_i_do_prediction_suite_summary.json`

## 7. safety-eval

### 7.1 单个 prediction 评测脚本：`run_single_prediction_safety_eval.py`

作用：
- 对单个 `generated_predictions.jsonl` 做一次离线 safety-eval
- 适合临时只看某个模型在某个数据集上的结果
- 脚本五内部也复用这套单文件评测逻辑

前台运行：

```bash
conda activate safety-eval
cd /root/data/My_first_project
source /root/data/My_first_project/linux_runtime_env.sh
python do_as_I_do/scripts/eval/run_single_prediction_safety_eval.py \
  --safety-eval-root safety-eval \
  --results-root do_as_I_do/saves/safety-eval-results \
  --classifier-model-name WildGuard \
  --classifier-batch-size 8 \
  --classifier-ephemeral-model \
  --save-per-sample-results \
  --model-alias gsm8k_AOA_model \
  --dataset-name gsm8k_AOA \
  --prediction-file do_as_I_do/saves/predict_smoke/gsm8k_AOA_model/gsm8k_AOA/generated_predictions.jsonl
```

### 7.2 正式 safety-eval 脚本：`run_do_as_i_do_safety_eval.py`

作用：
- 对脚本四生成的 `generated_predictions.jsonl` 做离线 safety-eval
- 默认从 `do_as_I_do/examples/predict/predict_yaml_manifest.json` 读取 12 份正式预测输出
- 当前会直接沿用 manifest 中的最新 `eval_dataset` / `dataset_dir`，因此会按 5 个 `_mini` 集合加 1 个原始 benign 集合的配置做记录
- 对每个模型分别写 `summary.json`
- 当前内部会逐条调用 `run_single_prediction_safety_eval.py` 对应的 Python 接口

前台运行：

```bash
conda activate safety-eval
cd /root/data/My_first_project
source /root/data/My_first_project/linux_runtime_env.sh
python do_as_I_do/scripts/eval/run_do_as_i_do_safety_eval.py \
  --safety-eval-root safety-eval \
  --results-root do_as_I_do/saves/safety-eval-results \
  --classifier-model-name WildGuard \
  --classifier-batch-size 8 \
  --no-classifier-ephemeral-model \
  --classifier-ephemeral-model \
  --save-per-sample-results
```

Linux 后台运行：

```bash
conda activate safety-eval
cd /root/data/My_first_project
source /root/data/My_first_project/linux_runtime_env.sh
mkdir -p do_as_I_do/logs
nohup python do_as_I_do/scripts/eval/run_do_as_i_do_safety_eval.py \
  --safety-eval-root safety-eval \
  --results-root do_as_I_do/saves/safety-eval-results \
  --classifier-model-name WildGuard \
  --classifier-batch-size 16 \
  --classifier-ephemeral-model \
  --no-classifier-ephemeral-model \
  --save-per-sample-results \
  > do_as_I_do/logs/run_do_as_i_do_safety_eval.log \
  2> do_as_I_do/logs/run_do_as_i_do_safety_eval.err &
```

### 7.3 Linux 串行模板：`run_single_prediction_safety_eval_serial_template.sh`

作用：
- 用 shell 串行调用多个“单个 prediction 评测”任务
- 默认只提供最小模板，你只需要修改 `TASKS` 数组

运行命令：

```bash
bash do_as_I_do/scripts/eval/run_single_prediction_safety_eval_serial_template.sh
```

`TASKS` 写法：

```bash
"模型别名|数据集名|generated_predictions.jsonl路径"
```

### 7.4 安装 safety-eval 依赖：`install_safety_eval_requirements.sh`

作用：
- 在 `conda activate safety-eval` 后补装缺失依赖
- 安装完成后打印关键包版本，便于调试

运行命令：

```bash
conda activate safety-eval
cd /root/data/My_first_project
bash do_as_I_do/scripts/eval/install_safety_eval_requirements.sh
```

### 7.5 smoke safety-eval：`run_do_as_i_do_safety_eval_smoke.sh`

作用：
- 使用 smoke prediction 结果做一轮最小 safety-eval
- 不依赖脚本四的正式 12 份输出
- 会先打印环境调试信息，再跑单文件评测
- Linux 默认使用 `conda activate safety-eval`
- Windows smoke 若不方便安装 `WildGuard/vllm`，可先改用 `KeywordBasedRefusalClassifier` 验证脚本闭环

运行命令：

```bash
conda activate safety-eval
cd /root/data/My_first_project
bash do_as_I_do/scripts/eval/run_do_as_i_do_safety_eval_smoke.sh
```

Windows 本地 smoke 可用：

```powershell
D:\anacoda3\envs\safety-eval\python.exe do_as_I_do/scripts/eval/run_do_as_i_do_safety_eval.py `
  --safety-eval-root safety-eval `
  --results-root do_as_I_do/saves/safety-eval-results `
  --classifier-model-name KeywordBasedRefusalClassifier `
  --save-per-sample-results `
  --smoke
```

关键输出：

- 模型级 summary：
  - `do_as_I_do/saves/safety-eval-results/<model_alias>/summary.json`
- 数据集级 summary：
  - `do_as_I_do/saves/safety-eval-results/<model_alias>/<dataset_name>/summary.json`
- 逐样本结果：
  - `do_as_I_do/saves/safety-eval-results/<model_alias>/<dataset_name>/safety_eval_predictions_with_labels.jsonl`
