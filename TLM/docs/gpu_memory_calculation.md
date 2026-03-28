# 推理测试中的 GPU 显存估算

在准备下一次推理实验前，可以参考这份文档来预测或复核整个任务会占用多少显存。

## 1. 当前运行的基线数据
- 在 [logs/requested_suite_lr0.00014_trainbs32_evalbs32_seed42.err](logs/requested_suite_lr0.00014_trainbs32_evalbs32_seed42.err#L4-L148) 中加载了 `Qwen2.5-7B-Instruct`，模型配置为 `hidden_size=3584`、`num_hidden_layers=28`、`torch_dtype=bfloat16`；实际预测时使用的批次大小为 32（详见 [logs/requested_suite_lr0.00014_trainbs32_evalbs32_seed42.err](logs/requested_suite_lr0.00014_trainbs32_evalbs32_seed42.err#L145-L148)）。
- 同期的 `nvidia-smi` 截图显示约 65,099 MiB 的显存被占用。

## 2. 分步估算流程

### A. 权重参数
1. 通过 `model.num_parameters()` 或配置文件中的参数逐项算出需要在 GPU 上存储的标量权重总数。对于 transformer，必须考虑：
   - 输入词嵌入：`vocab_size × hidden_size`
   - 输出投影：`hidden_size × vocab_size`（此处 `tie_word_embeddings=false`，所以输入/输出各自独立）
   - Transformer 层：每层包含 query/key/value/output 投影以及两个 MLP 线性层。
2. Attention 和 MLP 共享部分的紧凑公式为：
   $$
   \text{layer\_params} = \text{num\_hidden\_layers} \times (4 \times H^2 + 2 \times H \times I)
   $$
   其中 $H$ 为 `hidden_size`，$I$ 为 `intermediate_size`。
3. 再加上词嵌入和 LM 头：
   $$
   \text{total\_params} = 2 \times \text{vocab\_size} \times H + \text{layer\_params}
   $$
4. 用 dtype 的字节数（bfloat16 → 2，float32 → 4，int8 → 1 等）乘上参数总数，再换算到 MiB：
   $$
   \text{weight\_MiB} = \frac{\text{total\_params} \times \text{bytes\_per\_element}}{1024^2}
   $$

举例：当前运行中，$H=3584, I=18944, L=28, \text{vocab}=152064$，得到的参数总量约 6.33 亿（6.33B），在 bfloat16 下约占 12,075 MiB。

### B. KV 缓存（激活值显存）
1. KV 缓存为每个 batch 中的每个 token 在每层都留一组 key/value 张量，因此消耗：
   $$
   \text{KV\_bytes} = 2 \times L \times B \times S \times H \times \text{bytes\_per\_element}
   $$
   - $L$：层数（`num_hidden_layers`）
   - $B$：当前批次大小
   - $S$：缓存中的总序列长度（上下文长度 + 生成长度）。调整 `cutoff_len`、`generation_length` 或复用 past state 时要重新计算。
   - $H$：每层的 `hidden_size`
2. 换算成 MiB 后与 `nvidia-smi` 的值比较即可。

示例：$B=32$, $S=768+256=1024$, $H=3584$, $L=28$, `bfloat16`（2 字节）时，KV 缓存为 12,544 MiB，正好与日志里公式一致。

### C. 其余开销
把实际显存减去权重和缓存即可得到剩余开销：
$$
\text{extra\_MiB} = \text{total\_MiB} - \text{weight\_MiB} - \text{KV\_MiB}
$$
剩余部分通常来自 CUDA 上下文、Torch 工作区缓冲、融合 kernel、tokenizer 缓存等。在本次运行里这个值约为 40,480 MiB，所以当你提高 batch size/序列长度时，可以预计仍有几十 GiB 的余量。

## 3. 测试流程
1. 启动新实验前，使用即将到位的超参数（`batch_size`、`max_seq_len`、`generation_length`、`torch_dtype` 等）重新计算以上两组公式。
2. 快速运行下面这个脚本检查数字是否合理：

```python
from math import prod

H = 3584
I = 18944
L = 28
vocab = 152064
batch = 32
context_len = 768
gen_len = 256
bytes_per_element = 2  # bfloat16

layer_params = L * (4 * H**2 + 2 * H * I)
total_params = 2 * vocab * H + layer_params
kv_bytes = 2 * L * batch * (context_len + gen_len) * H * bytes_per_element
weights_mib = (total_params * bytes_per_element) / 1024**2
kv_mib = kv_bytes / 1024**2
print('weights MiB', weights_mib)
print('kv MiB', kv_mib)
```

3. 将上述估算之和与当前 `nvidia-smi` 读数对比，确认剩余显存（≈ CUDA 工作区）是否在可控范围内。

## 4. 未来实验备注
- 如果把 `torch_dtype` 改成 `float16` 或 `int8`，权重与 KV 缓存都会受到影响（缓存通常与权重精度一致），记得重新设定 `bytes_per_element`。
- 打开 `gradient_checkpointing` 或增加每 token 的额外计算会带来临时张量与（若训练）优化器状态，因此需要预留比公式更高的余量。
- 若加载多个模型或在多卡间复制模型，先把权重部分乘上副本数量，再与单卡总显存比较。
