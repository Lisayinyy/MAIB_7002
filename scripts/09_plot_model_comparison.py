"""Plot kNN and random forest against the baseline: validation average, test week, and test week by discount.

Run with:  uv run python scripts/09_plot_model_comparison.py   (after scripts 03, 06 and 08)
"""

import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
METHODS = {"Baseline (7-day average)": "#898781", "kNN (k = 25)": "#2a78d6", "Random forest (depth 10, leaf ≥ 3)": "#eb6834"}

# Validation: average MAE over the five weekly folds.
knn = pd.read_csv("results/knn_stage1_folds.csv")
rf = pd.read_csv("results/rf_tuning_folds.csv", dtype={"max_depth": str})
validation = [knn[knn.weighting.str.startswith("baseline")].mae.mean(),
              knn[(knn.weighting == "equal") & (knn.k == 25)].mae.mean(),
              rf[(rf.max_depth == "10") & (rf.min_leaf == 3)].mae.mean()]

# Test week: overall and by tomorrow's discount.
test = pd.read_csv("results/test_predictions.csv")
cols = ["pred_baseline", "pred_knn", "pred_rf"]
errors = test[cols].sub(test.target_sales, axis=0).abs()
band = pd.cut(test.discount_next, [0, 0.8, 0.95, 1.0], labels=["deep", "moderate", "none"])
by_band = errors.groupby(band, observed=True).mean()
rows = band.value_counts()

groups = {
    "Validation\n(5-week average)": validation,
    "Test week\n(all rows)": errors.mean().tolist(),
    f"Test: no discount\n({rows['none']:,} rows)": by_band.loc["none"].tolist(),
    f"Test: moderate discount\n({rows['moderate']:,} rows)": by_band.loc["moderate"].tolist(),
    f"Test: deep discount\n({rows['deep']:,} rows)": by_band.loc["deep"].tolist(),
}

plt.rcParams.update({"font.size": 10, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(axis="y", color="#e1e0d9", linewidth=1)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(length=0)

width = 0.22
for i, (method, colour) in enumerate(METHODS.items()):
    xs = [g + (i - 1) * (width + 0.03) for g in range(len(groups))]
    values = [v[i] for v in groups.values()]
    ax.bar(xs, values, width, color=colour, label=method)
    for x, value in zip(xs, values):
        ax.annotate(f"{value:.3f}", (x, value), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(len(groups)), list(groups))
ax.set_ylabel("Mean absolute error (lower is better)")
ax.set_ylim(0, 0.66)
ax.set_title("Next-day sales error: baseline, kNN and random forest", loc="left", fontsize=12, color=INK)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("results/model_comparison.png", dpi=160, facecolor=SURFACE)
print(pd.DataFrame(groups, index=list(METHODS)).T.round(3).to_string())
