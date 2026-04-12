# TLM -> TPU(Pytorch/XLA) 迁移规范与可行性评估

## 1. 目标

本文档用于在正式改造代码前，统一 TLM 从现有 GPU 训练路径迁移到 TPU + `pytorch/xla` 的设计边界、实施顺序、兼容性判断和开发规范。

当前结论先说在前面：

- **可以尝试把 TLM 改造成 TPU 训练框架，但建议先做“最小可用版本（MVP）”**。
- **MVP 目标应限定为：LoRA + TTL/SFT + bf16 + 单机 TPU + Hugging Face Trainer/Accelerate 路径**。
- **不建议第一阶段就保留/迁移的能力**：DeepSpeed、bitsandbytes/QLoRA、FlashAttention2、vLLM、Unsloth、部分自定义高性能 GPU kernel。
- **当前仓库中的 LLaMA-Factory 分支不是现成的 TPU 一等支持版本**；它有少量 TPU/XLA 痕迹，但整体仍以 GPU/NPU 逻辑为主。

## 2. 基于官方 `pytorch/xla` 文档提炼出的迁移原则

以下原则来自仓库 `docs` 目录下的官方 PyTorch/XLA 文档。

### 2.1 设备抽象必须从 CUDA 改为 XLA

- GPU 代码通常使用 `cuda` 设备。
- TPU 路径需要改为 `xla` 设备，推荐使用 `xm.xla_device()` 或 `torch.device("xla")`。
- 多卡 GPU 里常见的 `LOCAL_RANK -> cuda:{rank}` 逻辑，不能直接照搬到 TPU。

参考：`docs/source/learn/migration-to-xla-on-tpus.md`、`docs/source/learn/xla-quickstart.md`

### 2.2 TPU 是 Lazy Execution，不是普通 CUDA Eager Execution

- XLA 会先构图，再编译，再执行。
- 训练循环里需要显式同步执行边界：
  - 单设备可用 `torch_xla.sync()`
  - 分布式/多进程更常见的是 `xm.optimizer_step(optimizer)`
- 频繁 `.item()`、`print(tensor)`、逐 step 主机侧日志，都会导致不必要同步，性能会明显掉。

参考：`docs/source/learn/migration-to-xla-on-tpus.md`、`docs/source/learn/pytorch-on-xla-devices.md`、`docs/source/learn/troubleshoot.md`

### 2.3 数据形状需要尽量稳定，否则会频繁重新编译

- TPU/XLA 对动态 shape 更敏感。
- NLP 任务里，输入长度波动大时容易引发 recompilation。
- 建议：
  - 使用固定 `cutoff_len`
  - `padding` 策略尽量稳定
  - `pad_to_multiple_of` 设为 8/16/32/64 之一，减少 shape 种类
  - 在线 TTL 不要让每个小 batch 的长度分布差异过大

参考：`docs/source/learn/dynamic_shape.md`、`docs/source/learn/migration-to-xla-on-tpus.md`

### 2.4 TPU 上优先使用 bf16

- TPU 原生更适合 `bfloat16`
- 一般不需要像 fp16 那样做 gradient scaling
- 第一阶段建议统一走 `bf16`，避免 `fp16` 分支和额外兼容问题

参考：`docs/source/perf/amp.md`

### 2.5 分布式方式要优先选择 XLA 原生/兼容路径

可选思路：

- `torch_xla.launch(...)` + `xm.optimizer_step(...)`
- XLA DDP 后端：`dist.init_process_group("xla")`
- 仅在 Hugging Face/Accelerate 已能稳定接管的部分，复用 Trainer/Accelerator

参考：`docs/source/perf/ddp.md`、`docs/source/learn/migration-to-xla-on-tpus.md`

### 2.6 调试方式要换成 XLA 指标体系

迁移后不能只看 loss 曲线，还要重点看：

- 是否频繁 `CompileTime`
- 是否有大量 `TransferFromDeviceTime`
- 是否有未 lowering 的 `aten::` 操作
- 是否存在日志/`.item()` 触发的 host-device sync

推荐环境变量：

- `PT_XLA_DEBUG_LEVEL=2`
- 必要时打印 `torch_xla.debug.metrics.metrics_report()`

参考：`docs/source/learn/troubleshoot.md`

## 3. 当前 TLM 架构现状判断

### 3.1 TLM 本质上是建立在 LLaMA-Factory 训练栈上的

从仓库结构看，TLM 的训练主干在：

- `TLM/src/train.py`
- `TLM/src/llamafactory/...`
- `TLM/src/llamafactory/train/ttl/...`

这意味着 TPU 改造不是只改 TLM 的 TTL 逻辑，而是要同时处理：

- LLaMA-Factory 的设备判断
- HF Trainer/Accelerate 的运行方式
- TLM 自定义 TTL loss / train-predict-train 工作流

### 3.2 仓库里有 TPU 痕迹，但远没有真正打通

有利信号：

- `TLM/src/train.py` 已经保留了 `_mp_fn(index)` 注释：`# For xla_spawn (TPUs)`
- `TLM/examples/accelerate/fsdp_config.yaml` 里也有 `tpu_env` 等字段

但这更像“预留接口”，不是完整支持。

### 3.3 当前代码核心仍是 GPU/NPU 假设

明显阻塞点：

- `TLM/src/llamafactory/extras/misc.py` 中 `get_current_device()` 只处理 `xpu/npu/mps/cuda/cpu`，**没有 TPU/XLA**。
- 同文件里的 `is_gpu_or_npu_available()`、bf16/fp16 能力判断也没有 TPU 分支。
- `TLM/src/llamafactory/hparams/parser.py` 会把 `model_args.device_map` 直接设为 `{"": get_current_device()}`，这套逻辑对 TPU 不成立。
- 同文件强制要求训练使用当前分布式路径；这条路径默认是围绕 `torchrun` / GPU 风格组织的，而不是明确的 XLA launcher。

结论：**设备发现、dtype 判定、device_map、分布式入口，当前都需要系统性改造。**

## 4. TLM/TTL 逻辑对 TPU 的具体风险评估

## 4.1 训练-预测交替工作流会放大 XLA 编译成本

`TLM/src/llamafactory/train/ttl/workflow.py` 的 TTL 逻辑不是标准单一训练循环，而是：

- offline TTL：先 train，再 predict
- online TTL：先 predict，再 train，且按 `streaming_batch_size` 分块循环

这对 TPU 有两个风险：

- train / generate 两条 graph 反复切换
- online TTL 小批次分块会带来更多 shape 变化和更多编译轮次

所以 TTL 在 TPU 上**理论可做，但天然比普通 SFT 更难调优**。

### 4.2 `predict_with_generate: true` 是 TPU 风险点

当前示例配置里：

- `TLM/examples/train_lora/offline_ttl.yaml`
- `TLM/examples/train_lora/online_ttl.yaml`

都开启了 `predict_with_generate: true`。

生成阶段在 TPU 上不是不能做，但要注意：

- prompt 长度必须尽量稳定
- `max_new_tokens` 不能随意变化
- left padding / right padding 切换要非常克制
- 频繁 train/predict 切换会让 XLA 编译收益被摊薄

### 4.3 自定义 TTL loss 里有不适合 TPU 的主机侧/逐样本逻辑

`TLM/src/llamafactory/train/ttl/trainer.py` 里有几个重点风险：

- `cal_kl()` 对 batch 内样本做 Python for-loop 逐条处理
- `log_to_file()` 每步逐样本写文件
- `compute_loss()` 中会频繁做条件判断、日志和 host 侧输出

这些写法在 GPU 上还能忍，但在 TPU/XLA 下通常会带来：

- 图更碎
- 同步点更多
- Python 主机开销更高
- 编译缓存命中更差

所以 TTL 自定义 Trainer **不是不能迁移，而是必须先做 XLA 友好化重构**。

## 5. LLaMA-Factory 是否支持 TPU：当前判断

## 5.1 当前 vendored 版本没有“可直接依赖的 TPU 官方支持面”

从本仓库携带的 LLaMA-Factory 代码看：

- TPU 不是一等设备分支
- 没有成体系的 TPU 安装/训练/调试文档
- 设备工具函数没有 TPU 分支
- 优化特性大量围绕 GPU/NPU 生态

## 5.2 从官方公开资料看，LLaMA-Factory 明确展示了 NPU 训练文档，但没有对 TPU 给出同等级支持说明

这说明：

- 它**不是完全不可能跑在 TPU 上**
- 但**不能假设“官方已完整支持 TPU”**
- 我们应该按“基于 HF Trainer + PyTorch/XLA 自行适配 LLaMA-Factory 子集”的思路推进

## 5.3 因此，结论不是“不能改”，而是“只能先裁剪能力再改”

更准确的判断：

- **SFT/TTL 的基础 LoRA 训练路径：有希望支持 TPU**
- **完整保留 LLaMA-Factory 全能力：短期内不现实**

## 6. 第一阶段建议保留/禁用的能力边界

### 6.1 第一阶段建议保留

建议仅支持以下组合：

- 任务：`ttl`、`sft`
- 微调方式：`lora`、必要时 `full`
- 精度：`bf16`
- 运行方式：单机 TPU，多进程 XLA
- 推理/评测：先保留 Hugging Face generate 路径

### 6.2 第一阶段建议直接禁用

以下能力建议在 TPU 路径先禁掉：

- DeepSpeed
- QLoRA / bitsandbytes / GPTQ / AWQ / AQLM
- FlashAttention2
- vLLM
- Unsloth
- Liger Kernel
- GaLore
- BAdam
- PPO / RM / DPO / KTO 等复杂训练阶段
- 多模态模型优先不做

原因很简单：这些特性大多绑定 CUDA、专用 kernel、GPU 推理后端，或者会显著增加 XLA 适配复杂度。

## 7. 推荐改造路线（先理思路，再开始改）

### Phase 0：先做“可跑通性验证”

目标：不追求性能，先验证最小训练环路能否在 TPU 正常跑通。

建议顺序：

1. 先选 `sft` 或最小化 `ttl` 数据集
2. 只保留 LoRA + bf16
3. 去掉 DeepSpeed / quantization / flash_attn / vLLM
4. 用 TPU 启动一个最小脚本，验证：
   - model load
   - dataloader
   - forward/backward
   - optimizer step
   - checkpoint save

如果这一步不过，后面的 TTL 细化没有意义。

### Phase 1：把底层设备抽象改成 TPU 兼容

需要处理：

- `get_current_device()` 新增 XLA/TPU 分支
- dtype 能力判断加入 TPU bf16
- 区分 GPU device_map 与 TPU device placement
- 训练入口增加 `torch_xla` import 和 launcher 路径
- 梳理 `torchrun` 与 `torch_xla.launch` 的关系，避免双层分布式包装

### Phase 2：把 Trainer/DataLoader 改成 XLA 友好

重点：

- 训练循环是否交给 HF Trainer 直接处理，还是局部覆写
- dataloader 是否需要 `MpDeviceLoader`
- step 边界是否需要 `xm.optimizer_step()` 或 `torch_xla.sync()`
- callback / logging / save 是否会产生过多同步

### Phase 3：重构 TTL 自定义逻辑

优先改这几类问题：

- 把 `cal_kl()` 从逐样本 Python 循环改成更向量化的实现
- 把 `log_to_file()` 改成低频、rank0-only、批量日志
- 控制 online TTL 的小批次切分，避免每次都产生新 shape
- 减少 train 和 predict 之间不必要的反复 `unwrap_model()` / 状态切换

### Phase 4：再谈性能与扩展

跑通以后再看：

- padding 策略是否要改成 `pad_to_multiple_of`
- 是否能扩展到多核 TPU
- 是否能保留更多 LLaMA-Factory 特性
- 是否值得继续支持复杂 stage（DPO/PPO 等）

## 8. 开发规范（后续开始改代码时统一遵守）

### 8.1 单独引入 TPU 开关，不要污染原 GPU 路径

建议新增统一开关，例如：

- `use_tpu: true/false`
- 或读取 `PJRT_DEVICE=TPU`

原则：

- GPU 路径保持现状可用
- TPU 路径通过明确分支启用
- 禁止把 TPU 兼容代码散落成大量硬编码 `if "xla" in ...`

### 8.2 所有 TPU 不兼容特性要显式 fail-fast

不要“可能能跑”。要在参数解析阶段直接报错：

- TPU + deepspeed -> 报错
- TPU + bitsandbytes -> 报错
- TPU + flash_attn/fa2 -> 报错
- TPU + vllm -> 报错
- TPU + unsloth -> 报错

### 8.3 日志、评估、保存都要减少同步

- 尽量只在 rank0 打印
- 降低 `logging_steps`
- 禁止每步逐样本写文件
- 必要时把预测结果改成阶段性 flush

### 8.4 数据形状优先稳定

- 保持固定 `cutoff_len`
- 尽量固定 `max_new_tokens`
- 引入 `pad_to_multiple_of`
- 避免样本长度分布极端离散

### 8.5 先保证正确性，再看吞吐

验收顺序建议固定为：

1. 单步 forward/backward 正确
2. 1 epoch 无报错
3. checkpoint 可恢复
4. predict 可执行
5. 再看 compile 次数与吞吐

## 9. 当前总体结论

**结论很明确：TLM 可以朝 TPU + PyTorch/XLA 迁移，但不能按“把 cuda 改成 xla”这种轻量替换来做。**

真正的工作重点是三件事：

- 先把 LLaMA-Factory 这层的设备/分布式/特性开关抽象改对
- 再把 TTL 的自定义 Trainer 改成 XLA 友好
- 最后再处理性能与功能回补

换句话说：

- **可行性：中等，可做**
- **改造成本：中高，不建议一步到位**
- **推荐策略：先做 TPU MVP，再逐步回补能力**

## 10. 我建议我们下一步怎么开始

最稳的启动顺序是：

1. 先做一版 **TPU MVP 设计清单**（需要改哪些文件、哪些功能先禁用）
2. 然后先改 **底层设备抽象 + 参数校验**
3. 再做 **最小 SFT/TTL 跑通**
4. 最后才处理 **TTL 性能和生成阶段稳定性**

---

## 附：本次分析主要参考

### 本地官方文档

- `docs/source/learn/migration-to-xla-on-tpus.md`
- `docs/source/learn/xla-quickstart.md`
- `docs/source/perf/amp.md`
- `docs/source/perf/ddp.md`
- `docs/source/learn/dynamic_shape.md`
- `docs/source/learn/troubleshoot.md`

### 仓库内实现

- `TLM/src/train.py`
- `TLM/src/llamafactory/extras/misc.py`
- `TLM/src/llamafactory/hparams/parser.py`
- `TLM/src/llamafactory/train/ttl/workflow.py`
- `TLM/src/llamafactory/train/ttl/trainer.py`
- `TLM/examples/train_lora/offline_ttl.yaml`
- `TLM/examples/train_lora/online_ttl.yaml`

### 外部官方资料（辅助判断）

- PyTorch/XLA official docs: https://pytorch.org/xla/master/
- Hugging Face Trainer docs: https://huggingface.co/docs/transformers/main/main_classes/trainer
- LLaMA-Factory docs: https://llamafactory.readthedocs.io/
