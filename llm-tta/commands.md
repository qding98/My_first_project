# llm-tta 命令行运行指引

这份文档用于在 **TPU 上无法使用 Jupyter** 的情况下，直接通过命令行运行 `llm-tta`。

当前推荐入口脚本：

- [run_llm_tta_xla.py](d:\Qsh的个人资料\科研\LLM\My_first_project\llm-tta\run_llm_tta_xla.py)

适用模型默认值：

- `meta-llama/Meta-Llama-3-8B-Instruct`

适用设备：

- `PyTorch/XLA`
- `TPU v4-8`

## 1. 进入目录

```bash
cd /path/to/My_first_project/llm-tta
```

如果你在 TPU VM 上，先切到项目目录再执行下面命令。

## 2. 设置 TPU 环境

```bash
export PJRT_DEVICE=TPU
```

建议顺手打开更完整的 Python 输出：

```bash
export PYTHONUNBUFFERED=1
```

## 3. 查看脚本参数

```bash
python run_llm_tta_xla.py --help
```

## 4. 最常用的正式运行命令

这是最推荐的 TPU v4-8 命令。

```bash
python run_llm_tta_xla.py \
  --device xla \
  --use-xla-spawn \
  --xla-processes 4 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --samples-per-type 200 \
  --max-len 512 \
  --seed 42 \
  --run-name llama3_tpu_v4_8 \
  --output-dir save
```

说明：

- `--device xla`：使用 PyTorch/XLA
- `--use-xla-spawn`：启用多进程 XLA worker
- `--xla-processes 4`：适合作为 TPU v4-8 的默认起点
- `--precision bf16`：TPU 推荐 bf16
- `--run-name llama3_tpu_v4_8`：最终输出文件名前缀

## 5. 先做一个小规模 smoke test

如果你担心一上来就跑完整实验太慢，建议先用更小样本测试：

```bash
python run_llm_tta_xla.py \
  --device xla \
  --use-xla-spawn \
  --xla-processes 4 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --samples-per-type 10 \
  --max-len 256 \
  --seed 42 \
  --run-name llama3_tpu_smoke_test \
  --output-dir save
```

这条命令更适合先确认：

- 环境是否正常
- `torch-xla` 是否正常工作
- Hugging Face 模型权限是否正常
- 输出目录和 CSV 是否正常生成

## 6. 单进程 TPU 运行

如果你想先用单进程调试：

```bash
python run_llm_tta_xla.py \
  --device xla \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --samples-per-type 10 \
  --max-len 256 \
  --run-name llama3_tpu_single_process \
  --output-dir save
```

## 7. 只导出 CSV，不画图

如果你只关心结果表，可以跳过画图：

```bash
python run_llm_tta_xla.py \
  --device xla \
  --use-xla-spawn \
  --xla-processes 4 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --samples-per-type 200 \
  --max-len 512 \
  --run-name llama3_tpu_csv_only \
  --output-dir save \
  --skip-plots
```

## 8. 后台运行并写日志

如果你在 Linux TPU VM 上，希望退出终端后继续运行：

```bash
nohup python run_llm_tta_xla.py \
  --device xla \
  --use-xla-spawn \
  --xla-processes 4 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --samples-per-type 200 \
  --max-len 512 \
  --seed 42 \
  --run-name llama3_tpu_v4_8 \
  --output-dir save \
  > run_llm_tta_xla.log 2>&1 &
```

查看日志：

```bash
tail -f run_llm_tta_xla.log
```

## 9. 结果输出位置

默认输出目录：

```bash
llm-tta/save/
```

主要文件：

```bash
save/<run_name>.csv
save/<run_name>_gradient_dot_product.png
save/<run_name>_by_type.png
```

例如：

```bash
save/llama3_tpu_v4_8.csv
save/llama3_tpu_v4_8_gradient_dot_product.png
save/llama3_tpu_v4_8_by_type.png
```

## 10. 常见问题

### 10.1 找不到模型

如果报 Hugging Face 权限相关错误，先确认：

- 你有 `Meta-Llama-3-8B-Instruct` 的访问权限
- 已经在 TPU 环境里执行过 Hugging Face 登录

### 10.2 OOM

如果 TPU 内存不够，可以先这样降配置：

```bash
python run_llm_tta_xla.py \
  --device xla \
  --use-xla-spawn \
  --xla-processes 4 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --samples-per-type 5 \
  --max-len 128 \
  --run-name llama3_tpu_low_mem \
  --output-dir save
```

### 10.3 想确认当前命令是否已经启动

```bash
ps -ef | grep run_llm_tta_xla.py
```

## 11. 最短可复制版本

如果你只想复制一条就直接跑，用这条：

```bash
cd /path/to/My_first_project/llm-tta
export PJRT_DEVICE=TPU
export PYTHONUNBUFFERED=1
python run_llm_tta_xla.py \
  --device xla \
  --use-xla-spawn \
  --xla-processes 4 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --samples-per-type 200 \
  --max-len 512 \
  --seed 42 \
  --run-name llama3_tpu_v4_8 \
  --output-dir save
```
