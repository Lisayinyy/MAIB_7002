"""Plot the stage 1 kNN tuning results: MAE by number of neighbours, and MAE by validation week.

Run with:  uv run python scripts/04_plot_knn_stage1.py   (after scripts/03_tune_knn.py)
"""

import matplotlib.pyplot as plt
import pandas as pd

BASELINE = "baseline (7-day mean)"
SERIES = {  # name in results -> (label, colour)
    "equal": ("Equal weights", "#2a78d6"),
    "sales_t": ("Sales on day t ×2", "#eb6834"),
}
BASELINE_COLOUR, INK, MUTED, GRID, SURFACE = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
FOLD_LABELS = {1: "May 22–28", 2: "May 29–Jun 4", 3: "Jun 5–11", 4: "Jun 12–18", 5: "Jun 19–25"}

folds = pd.read_csv("results/knn_stage1_folds.csv")
by_k = folds.groupby(["weighting", "k"]).mae.mean().unstack("weighting")
baseline_mae = by_k[BASELINE].dropna().iloc[0]
by_k = by_k.drop(columns=BASELINE).drop(index=0)
best_k = int(by_k.min(axis=1).idxmin())

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)

# Left: the k curve, averaged over the five folds.
left.axhline(baseline_mae, color=BASELINE_COLOUR, linewidth=2, label=f"Baseline (7-day average): {baseline_mae:.3f}")
for name, (label, colour) in SERIES.items():
    left.plot(by_k.index, by_k[name], color=colour, linewidth=2, marker="o", markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=2,
              label=f"{label}: best {by_k[name].min():.3f} at k = {by_k[name].idxmin()}")
left.set_xscale("log")
left.set_xticks(by_k.index, [str(k) for k in by_k.index])
left.minorticks_off()
left.set_xlabel("Number of neighbours (k)")
left.set_ylabel("Mean absolute error (lower is better)")
left.set_title("Average over five validation weeks", loc="left", fontsize=11, color=INK)
left.legend(frameon=False, loc="upper right")

# Right: each validation week at the best k.
at_best = folds[folds.k.isin([0, best_k])].pivot(index="fold", columns="weighting", values="mae")
right.plot(at_best.index, at_best[BASELINE], color=BASELINE_COLOUR, linewidth=2, marker="o", markersize=7,
           markeredgecolor=SURFACE, markeredgewidth=2)
for name, (label, colour) in SERIES.items():
    right.plot(at_best.index, at_best[name], color=colour, linewidth=2, marker="o", markersize=7,
               markeredgecolor=SURFACE, markeredgewidth=2)
right.set_xticks(list(FOLD_LABELS), list(FOLD_LABELS.values()))
right.set_xlabel("Validation week")
right.set_title(f"Each validation week at k = {best_k}", loc="left", fontsize=11, color=INK)

fig.suptitle("kNN tuning, stage 1: next-day sales error by neighbours and feature weighting",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("results/knn_stage1.png", dpi=160, facecolor=SURFACE)
print("saved results/knn_stage1.png")
