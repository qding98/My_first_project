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

运行命令：

```bash
bash do_as_I_do/scripts/train/run_do_as_i_do_train_pair.sh
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

相关目录：

```text
do_as_I_do/examples/predict/
```

### 4.2 串行预测脚本：`run_do_as_i_do_predict_suite.py`

作用：
- 依次执行 `predict_yaml_manifest.json` 中记录的 12 份 YAML
- 每轮执行结束后，把 `generated_predictions.jsonl` 转成 `generate_predict.json`
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
