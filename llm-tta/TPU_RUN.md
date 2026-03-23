# llm-tta TPU 运行说明

## 新增内容

仓库里新增了一个可直接运行的脚本：

- [run_llm_tta_xla.py](d:\Qsh的个人资料\科研\LLM\My_first_project\llm-tta\run_llm_tta_xla.py)

它把原先 notebook 里的 `llm-tta` 流程整理成了命令行程序，并支持：

- `CPU`
- `CUDA`
- `PyTorch/XLA (TPU)`

输出内容包括：

- 每条样本的 `question_text`
- 每条样本的 `expected_answer_text`
- 每条样本的 `perplexity_grad_dot_product`
- 最终 CSV
- 两张统计图

## 依赖安装

至少需要这些 Python 包：

```bash
pip install torch transformers datasets pandas matplotlib tqdm
```

如果要跑 TPU，还需要安装与 TPU 环境匹配的 `torch-xla`。

注意：

- `meta-llama/Meta-Llama-3-8B-Instruct` 往往需要先完成 Hugging Face 权限申请并登录。
- 8B 模型计算全参数梯度，显存/显存等价内存占用很大。小内存 TPU 上可能会 OOM。

## 单机 TPU 运行

先设置 TPU 运行环境：

```bash
export PJRT_DEVICE=TPU
```

然后在 `llm-tta` 目录执行单核 TPU 版本：

```bash
python run_llm_tta_xla.py \
  --device xla \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --output-dir save
```

如果你想用多核 TPU 进程：

```bash
python run_llm_tta_xla.py \
  --device xla \
  --use-xla-spawn \
  --xla-processes 4 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --output-dir save
```

说明：

- `--xla-processes 4` 表示启 4 个 XLA worker
- 多进程时脚本会自动按样本分片计算，再合并成一个最终 CSV

## CUDA 运行

```bash
python run_llm_tta_xla.py \
  --device cuda \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision bf16 \
  --output-dir save
```

## CPU 运行

```bash
python run_llm_tta_xla.py \
  --device cpu \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --precision fp32 \
  --output-dir save
```

## 常用参数

```bash
python run_llm_tta_xla.py --help
```

常用参数：

- `--samples-per-type 200`
- `--max-len 512`
- `--seed 42`
- `--run-name my_run`
- `--skip-plots`

示例：

```bash
python run_llm_tta_xla.py \
  --device xla \
  --use-xla-spawn \
  --xla-processes 4 \
  --samples-per-type 50 \
  --max-len 512 \
  --run-name llama3_tpu_test \
  --output-dir save
```

## 输出位置

默认输出到：

- [save](d:\Qsh的个人资料\科研\LLM\My_first_project\llm-tta\save)

主要产物：

- `results_xla_*.csv`
- `results_xla_*_gradient_dot_product.png`
- `results_xla_*_by_type.png`

## 和原 notebook 的关系

原 notebook 适合交互式调试：

- [test.ipynb](d:\Qsh的个人资料\科研\LLM\My_first_project\llm-tta\test.ipynb)

新脚本适合正式跑实验，尤其是 TPU 场景，因为：

- 不再依赖 `.cuda()`
- 统一了设备抽象
- 支持 `torch_xla`
- 支持多进程分片与结果合并
