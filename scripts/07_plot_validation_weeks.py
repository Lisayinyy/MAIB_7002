"""Plot MAE for each validation week: baseline, locked kNN and the chosen random forest.

Run with:  uv run python scripts/07_plot_validation_weeks.py   (after scripts 03 and 06)
"""

import matplotlib.pyplot as plt
import pandas as pd

RF_DEPTH, RF_LEAF, KNN_K = "10", 3, 25
WEEKS = {1: "May 22–28", 2: "May 29–Jun 4", 3: "Jun 5–11", 4: "Jun 12–18", 5: "Jun 19–25"}
SURFACE, MUTED = "#fcfcfb", "#52514e"

knn = pd.read_csv("results/knn_stage1_folds.csv")
rf = pd.read_csv("results/rf_tuning_folds.csv", dtype={"max_depth": str})
lines = {
    "Baseline (7-day average)": (knn[knn.weighting.str.startswith("baseline")].set_index("fold").mae, "#898781"),
    f"kNN (k = {KNN_K})": (knn[(knn.weighting == "equal") & (knn.k == KNN_K)].set_index("fold").mae, "#2a78d6"),
    f"Random forest (depth {RF_DEPTH}, leaf ≥ {RF_LEAF})": (
        rf[(rf.max_depth == RF_DEPTH) & (rf.min_leaf == RF_LEAF)].set_index("fold").mae, "#eb6834"),
}

plt.rcParams.update({"font.size": 10, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(axis="y", color="#e1e0d9", linewidth=1)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(length=0)
for label, (mae, colour) in lines.items():
    ax.plot(mae.index, mae, color=colour, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2, label=f"{label}: average {mae.mean():.3f}")
rf_mae = lines[f"Random forest (depth {RF_DEPTH}, leaf ≥ {RF_LEAF})"][0]
for fold, value in rf_mae.items():
    ax.annotate(f"{value:.3f}", (fold, value), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9)
ax.set_ylim(0.27, 0.37)
ax.set_xticks(list(WEEKS), list(WEEKS.values()))
ax.set_xlabel("Validation week")
ax.set_ylabel("Mean absolute error (lower is better)")
ax.set_title("Next-day sales error in each validation week", loc="left", fontsize=11)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("results/validation_weeks.png", dpi=160, facecolor=SURFACE)
print(pd.DataFrame({k: v[0] for k, v in lines.items()}).rename(index=WEEKS).round(3).to_string())
