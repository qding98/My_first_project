"""
按 data_type 画梯度内积 & loss 分布图（4 类型 × 2 指标 = 2×4 子图）

用法:
    python draw_data.py                          # 默认用最新 800 条结果
    python draw_data.py path/to/results.csv      # 指定 CSV
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 加载数据 ──────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "results_logp_qwen3.5-0.8b_2026_03_22_0249.csv"
df = pd.read_csv(csv_path)

# 过滤 inf/nan
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["dot_product", "log_p_x", "log_p_y_given_x"])

types = sorted(df["data_type"].unique())
n_types = len(types)
colors = {"vanilla_harmful": "#e74c3c", "adversarial_harmful": "#e67e22",
          "vanilla_benign": "#2ecc71", "adversarial_benign": "#3498db"}

# ── 画图: 上排 dot_product 分布, 下排 loss 分布 ─────────
fig, axes = plt.subplots(2, n_types, figsize=(5 * n_types, 9))

for col, dtype in enumerate(types):
    sub = df[df["data_type"] == dtype]
    color = colors.get(dtype, "gray")
    n_pos = (sub["dot_product"] > 0).sum()
    pct = n_pos / len(sub) * 100

    # ── 上排: 点积分布 ──
    ax = axes[0, col]
    ax.hist(sub["dot_product"], bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(f"{dtype}\n({pct:.1f}% positive, n={len(sub)})", fontsize=11)
    ax.set_xlabel("⟨∇L(x), ∇L(y|x)⟩")
    if col == 0:
        ax.set_ylabel("Count")

    # ── 下排: loss_x vs loss_y|x 散点, 颜色编码点积正负 ──
    ax = axes[1, col]
    pos_mask = sub["dot_product"] > 0
    ax.scatter(sub.loc[~pos_mask, "log_p_x"], sub.loc[~pos_mask, "log_p_y_given_x"],
               c="crimson", alpha=0.5, s=18, label="dot < 0")
    ax.scatter(sub.loc[pos_mask, "log_p_x"], sub.loc[pos_mask, "log_p_y_given_x"],
               c=color, alpha=0.5, s=18, label="dot > 0")
    ax.set_xlabel("log P(x)")
    if col == 0:
        ax.set_ylabel("log P(y|x)")
    ax.set_title(f"Loss distribution", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle(f"Gradient Inner Product & Loss by data_type\n({csv_path})", fontsize=13, y=1.01)
plt.tight_layout()

out_path = csv_path.rsplit(".", 1)[0] + "_by_type.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
