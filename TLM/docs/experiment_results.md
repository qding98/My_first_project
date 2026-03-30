# TTL Experiment Summary (lr=0.0001, bs=16, seed=42)

## Quick Takeaways
- clean TTL 提升了三套任务的 token-level accuracy，小幅度且以 alpaca 最明显，gsm8k 提升最小。
- mix40 相比 clean 在任务准确率几乎持平（轻微上升），但安全性表现分化：gsm8k 几乎杜绝有害回答却严重过拒答；alpaca 稍有安全改善；agriculture 仍易被 jailbreak 放大。
- 后续需要按数据集定向调参：降 agriculture 的 lift、缓和 gsm8k 的过拒答、继续压缩 alpaca 的 ASR。

## Clean Accuracy (ComputeAccuracy)

| dataset | base_acc | clean_acc | mix_acc |
| --- | --- | --- | --- |
| agriculture_5k | 0.5080 | 0.5305 | 0.5356 |
| alpaca_gpt4_5k | 0.6948 | 0.7260 | 0.7304 |
| gsm8k_5k | 0.2992 | 0.3251 | 0.3244 |

- 图表：TLM/logs/accuracy_bar.png（base/clean/mix 对比）

## Safety (WildJailbreak Controlled Eval)
- 指标含义：
  - adv_asr：对抗有害 ASR，越低越好。
  - jailbreak_lift：adv_asr - vanilla_asr，越接近 0 越好。
  - benign_refusal：无害样本拒答率，越低越好。

| dataset | model | adv_asr | jailbreak_lift | benign_refusal(train/adv) |
| --- | --- | --- | --- | --- |
| agriculture_5k | clean | 0.7900 | 0.3590 | 0.1290 / 0.0619 |
| agriculture_5k | mix | 0.6530 | 0.4650 | 0.1390 / 0.0905 |
| alpaca_gpt4_5k | clean | 0.7210 | 0.5950 | 0.1750 / 0.0714 |
| alpaca_gpt4_5k | mix | 0.6370 | 0.5350 | 0.2000 / 0.0857 |
| gsm8k_5k | clean | 0.7405 | 0.5140 | 0.1510 / 0.0667 |
| gsm8k_5k | mix | 0.0470 | -0.0090 | 0.2950 / 0.8286 |

- 图表：
  - TLM/logs/safety_asr.png（adv_asr，clean vs mix）
  - TLM/logs/jailbreak_lift.png（lift，clean vs mix）

## Interpretation by Dataset
- agriculture_5k：任务小幅提升；jailbreak lift 升高（0.465），对抗包装仍显著放大攻破率。
- alpaca_gpt4_5k：任务提升最明显；安全指标略有改善但仍易受攻击（lift 0.535）。
- gsm8k_5k：任务提升极小；mix 将 adv_asr 压到 ~0.05 且 lift≈0，但 benign refusal 飙到 0.83，出现严重过拒答。

## What This Run Shows
- TTL（clean/mix）在 base 上有稳定但有限的 accuracy 增益。
- mix 不保证统一的安全收益，需要按数据集定向优化（agri 降 lift，gsm8k 减拒答，alpaca 再压 ASR）。
- clean eval 已切回原生 ComputeAccuracy，每个模型只评一次，结果在 metrics/clean_eval.json。

## Next Steps (建议)
1) agriculture：降低 jailbreak lift（拒答阈值/安全 prompt 调整），再跑对照。
2) gsm8k：在保持低 ASR 的同时降低 benign refusal（调整拒绝模板或温度/阈值），再评测。
3) alpaca：尝试小步调参（max_new_tokens / 温度）以继续压低 ASR，观察准确率是否保持。
