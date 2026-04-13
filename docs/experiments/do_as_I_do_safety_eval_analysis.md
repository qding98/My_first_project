# do_as_I_do Safety-Eval 结果分析

本文档基于 `do_as_I_do/saves/safety-eval-results/` 下已经生成的正式 `summary.json` 撰写，目标是对本轮 `Do_as_I_do` 实验中的两个模型进行离线 `safety-eval` 对照分析。

本次使用的两个模型为：

- `gsm8k_AOA_model`
- `gsm8k_vallina_AOA_model`

本次 classifier 配置为：

- `model_name`: `WildGuard`
- `batch_size`: `16`
- `ephemeral_model`: `false`
- `harmful_success_mode`: `compliance_and_harmful`

需要先说明一个重要背景：本轮正式 `predict` 已经切到“5 个 mini 数据集 + 1 个原始 benign 集合”的配置，因此以下结果对应的是：

- harmful:
  - `adversarial_harmful_AOA_mini`
  - `vallina_harmful_AOA_mini`
  - `harmful_mix_2k_mini`
  - `villina_mixed_mini`
- benign:
  - `wildjailbreak_eval_adversarial_benign`
  - `wildjailbreak_train_vanilla_benign_1k_mini`

但预测输出目录名仍保持原逻辑名称，例如 `villina_mixed/` 对应的底层 `eval_dataset` 实际上是 `villina_mixed_mini`。

## 1. 总体结论

这轮结果没有呈现“第二阶段 `vallina_harmful_AOA` 训练后，模型在所有 harmful 集合上都更脆弱”的简单模式。更准确地说，`gsm8k_vallina_AOA_model` 的行为是**分布选择性变化**：

- 在 `vallina_harmful_AOA_mini` 上，第二阶段训练明显提升了 harmful ASR。
- 但在 `adversarial_harmful_AOA_mini`、`harmful_mix_2k_mini` 和 `villina_mixed_mini` 上，第二阶段训练反而降低了 harmful ASR。
- 在 benign 集合上，第二阶段训练带来了更高的 refusal，尤其在 `eval_adversarial_benign` 上更明显。

因此，本轮结果更像是：

1. 第二阶段训练并没有把模型整体推向“更愿意输出 harmful 内容”。
2. 它更像是把行为重新定向到与训练集更接近的某一类 harmful phrasing / prompt distribution。
3. 同时，这个过程伴随一定程度的可用性损失，表现为 benign refusal 上升。

## 2. 核心对照表

下表中的 harmful 指标是 `asr_valid_only`，benign 指标是 `refusal_rate_valid_only`。

| 数据集 | `gsm8k_AOA_model` | `gsm8k_vallina_AOA_model` | 差值（后者 - 前者） | 结论 |
| --- | ---: | ---: | ---: | --- |
| `adversarial_harmful_AOA` | 0.2851 | 0.1080 | -0.1771 | 第二阶段后明显更安全 |
| `vallina_harmful_AOA` | 0.0600 | 0.2440 | +0.1840 | 第二阶段后显著更脆弱 |
| `harmful_mix_2k` | 0.2120 | 0.1200 | -0.0920 | 第二阶段后更安全 |
| `villina_mixed` | 0.0240 | 0.0080 | -0.0160 | 第二阶段后更安全 |
| `eval_adversarial_benign` | 0.0478 | 0.1333 | +0.0855 | 第二阶段后 benign refusal 明显上升 |
| `train_vanilla_benign_1k` | 0.1120 | 0.1160 | +0.0040 | 两者接近，第二阶段略更保守 |

## 3. 分数据集分析

### 3.1 `adversarial_harmful_AOA`

- `gsm8k_AOA_model` 的 `asr_valid_only = 0.2851`
- `gsm8k_vallina_AOA_model` 的 `asr_valid_only = 0.1080`
- 同时 `compliance_rate_valid_only` 从 `0.7028` 降到 `0.3240`

这说明第二阶段训练没有让模型在这类 adversarial harmful prompt 上更容易“跟随意图”，反而显著压低了 compliance。若只看这一个集合，可以得出“第二阶段更安全”的结论。

但这个结论不能外推成全局安全提升，因为后面在 `vallina_harmful_AOA` 上出现了反向现象。

### 3.2 `vallina_harmful_AOA`

- `gsm8k_AOA_model` 的 `asr_valid_only = 0.0600`
- `gsm8k_vallina_AOA_model` 的 `asr_valid_only = 0.2440`
- `compliance_rate_valid_only` 从 `0.2680` 升到 `0.4560`

这是这轮实验中最关键的“反转点”。第二阶段模型恰恰在自己见过的 `vallina_harmful_AOA` 风格上，变得更愿意遵从 harmful 请求。这个结果与直觉是一致的，也说明第二阶段训练确实在参数层留下了可检测的行为漂移。

如果只看这一项，可以把第二阶段训练解释成“对特定 harmful prompt family 的脆弱性放大”。

### 3.3 `harmful_mix_2k`

- `gsm8k_AOA_model` 的 `asr_valid_only = 0.2120`
- `gsm8k_vallina_AOA_model` 的 `asr_valid_only = 0.1200`

这里第二阶段训练再次体现为“更安全”。结合 `vallina_harmful_AOA` 的上升，说明第二阶段并不是让模型无差别更易越狱，而是在不同 harmful 分布之间发生了选择性重排。

换句话说，第二阶段更像是**分布内脆弱性增强，分布外脆弱性下降**。

### 3.4 `villina_mixed`

- `gsm8k_AOA_model` 的 `asr_valid_only = 0.0240`
- `gsm8k_vallina_AOA_model` 的 `asr_valid_only = 0.0080`

这个集合上两者本来都很低，第二阶段后进一步下降。由于 `villina_mixed` 本身是混合构造集，这说明第二阶段训练并没有把模型整体推成更“松”的状态，反而在一些混合分布上更保守。

这和你此前在 TTL 现象里观察到的“混入 harmful 后未必导致更高 harmful ASR，反而可能把模型推向硬拒答/硬保守”是相容的。

### 3.5 `eval_adversarial_benign`

- `gsm8k_AOA_model` 的 `refusal_rate_valid_only = 0.0478`
- `gsm8k_vallina_AOA_model` 的 `refusal_rate_valid_only = 0.1333`

这是本轮 availability 风险最明显的证据之一。第二阶段训练后，模型在 adversarial benign 集合上的拒答率增长了约 `8.55` 个百分点。

这说明第二阶段训练虽然没有把模型整体推向更高的 harmful ASR，但确实把模型推向了更保守、更容易误拒的区域。

### 3.6 `train_vanilla_benign_1k`

- `gsm8k_AOA_model` 的 `refusal_rate_valid_only = 0.1120`
- `gsm8k_vallina_AOA_model` 的 `refusal_rate_valid_only = 0.1160`

这里差异很小，说明第二阶段训练对普通 benign 集合的影响存在，但没有 `eval_adversarial_benign` 那么强烈。

这进一步支持一个更细的判断：**第二阶段带来的可用性损失，主要体现在更“边界化”、更像攻击的 benign prompt 上，而不是对所有 benign 请求一刀切地大幅升高 refusal。**

## 4. 这轮结果最值得记住的三点

### 4.1 第二阶段训练不是“全局越狱增强”

如果它是全局越狱增强，我们应该看到多个 harmful 集合上的 ASR 都上升。但实际情况是：

- 只在 `vallina_harmful_AOA` 上显著上升
- 在其余 3 个 harmful 集合上都下降

因此，更准确的表述不是“模型被污染后整体更危险”，而是“模型被定向改写后，对某一类训练相近分布更危险，对其他分布未必更危险”。

### 4.2 第二阶段训练带来了明确的 availability 代价

最明显的是 `eval_adversarial_benign`：

- refusal 从 `4.78%` 升到 `13.33%`

这意味着即使 harmful ASR 没有普遍上升，也已经出现了清晰的可用性损伤。这个现象在论文叙事里是有价值的，因为它说明攻击/适配并不一定体现为“更容易回答 harmful”，也可能体现为“更容易把边界 benign 误判为不该回答”。

### 4.3 当前结果更支持“分布选择性行为漂移”而不是“统一安全退化”

如果要给这一轮结果一个最准确的机制化总结，我会建议写成：

> 第二阶段 `vallina_harmful_AOA` 训练诱导了模型对特定 harmful prompt family 的选择性敏感化，同时伴随 adversarial benign 场景下的 refusal 上升，但并未造成跨分布的一致性 harmful ASR 增长。

这个表述比“模型更危险了”更精确，也更贴近实际数据。

## 5. 对后续论文叙事的启发

如果你后面要把这一轮结果接到更大的机制问题里，我建议优先使用下面这条叙事：

### 5.1 主叙事

可以把这轮结果作为“**攻击或适配不会简单表现为统一 ASR 上升，而更可能表现为行为边界重排**”的实证例子。

这里的“边界重排”包含两部分：

- 对训练相近 harmful 分布，compliance 上升
- 对其他 harmful 或边界 benign 分布，模型更保守或更拒答

这比单一的“安全下降”更像真实模型在参数微调后的行为变化。

### 5.2 如果接你之前的 TTL 研究主线

这轮结果和你之前观察到的“混 harmful 之后，harmful ASR 未必上升，反而 benign refusal 飙升”的现象是同方向的。

两者都指向同一个更一般的问题：

> 模型更新并不一定把行为推向“更容易输出 harmful”，而是可能把决策边界推向更极端、更不稳定的位置，从而表现为分布选择性脆弱化和 availability 损伤。

这个叙事比只盯着 jailbreak ASR 更适合扩展成机制论文。

## 6. 当前结果的局限

这份分析需要保留三个限制条件：

1. 本轮正式评测里有 5 个数据集使用的是 `_mini` 版本，因此当前结果更适合作为快速对照结论，而不是最终论文数字。
2. 这里使用的是 `WildGuard` 离线 classifier，结论是“基于 classifier 口径的行为对照”，不等于人工标注。
3. 当前只覆盖了两个 `Do_as_I_do` 模型，不代表该现象已经具备跨模型稳定性。

## 7. 我对这轮结果的判断

如果你问“这轮实验最强的发现是什么”，我会给出下面这句总结：

> `gsm8k_vallina_AOA_model` 并没有在所有 harmful 集合上都更危险；它主要表现为对 `vallina_harmful_AOA` 这类训练相近分布更脆弱，同时在 `eval_adversarial_benign` 上更容易误拒，从而呈现出一种分布选择性的风险重排，而不是统一的安全退化。

这句结论既能忠实反映当前数据，也能自然衔接你后面要做的“梯度劫持 / 参数污染 / 可用性攻击”方向。
