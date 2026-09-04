"""
Generate figures/k_scan_results.png: the paper's central empirical
finding (Table 2) as a plot -- the inverted-U relationship between
Structural Incomparability Rate and the partial-order-aware model's
advantage over the total-order baseline.

Uses the exact values reported in Table 2 (independently re-verified
against this repository's refactored code -- see README's Reproducibility
section) rather than re-running the full five-seed sweep, which is slow;
run `scripts/run_table2_kscan.py` to regenerate these numbers from
scratch.
"""

import matplotlib.pyplot as plt
import numpy as np

# Table 2, exactly as reported in the paper.
sir_mean = [0.0, 49.7, 75.1, 87.7, 93.9, 97.1]
advantage_mean = [2.05, 8.53, 1.07, 2.63, 2.33, -0.43]
advantage_std = [1.05, 7.97, 0.63, 4.48, 5.53, 2.92]
sign_stable = [True, True, True, False, False, False]

fig, ax = plt.subplots(figsize=(7, 5))

colors = ["#2166ac" if s else "#b2182b" for s in sign_stable]
ax.errorbar(
    sir_mean, advantage_mean, yerr=advantage_std,
    fmt="none", ecolor="gray", elinewidth=1.2, capsize=4, zorder=1,
)
ax.scatter(sir_mean, advantage_mean, c=colors, s=90, zorder=2, edgecolors="black", linewidths=0.6)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)

for x, y, k in zip(sir_mean, advantage_mean, [1, 2, 3, 4, 5, 6]):
    ax.annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)

# Real-world validation points (Sections 5.4-5.6), for context.
real_sir = [99.4, 99.8, 91.3, 99.7, 48.7]
real_adv = [0.93, -0.84, 4.13, -3.51, 4.10]
ax.scatter(real_sir, real_adv, marker="^", c="#4daf4a", s=100, zorder=3, edgecolors="black", linewidths=0.6, label="Real datasets")

ax.set_xlabel("Structural Incomparability Rate (SIR, %)")
ax.set_ylabel("PartialOrderModel advantage over\nTotalOrderScoreModel (percentage points)")
ax.set_title("The inverted-U relationship between SIR and partial-order advantage")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac", markeredgecolor="black", markersize=10, label="Synthetic, sign-stable"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#b2182b", markeredgecolor="black", markersize=10, label="Synthetic, not sign-stable"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#4daf4a", markeredgecolor="black", markersize=10, label="Real datasets"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("figures/k_scan_results.png", dpi=150)
print("saved figures/k_scan_results.png")
