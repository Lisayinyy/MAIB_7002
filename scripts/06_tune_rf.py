"""Step 6: tune a random forest on the same five time-ordered weekly folds used for kNN.

Settings tuned: maximum tree depth and minimum rows per leaf (16 combinations).
Fixed: 300 trees, half the features tried at each split, squared-error splits, the 10 kNN features.
An earlier grid over leaf size, features per split and weather is kept in results/rf_tuning_leaf_features_*.csv.

Run with:  uv run python scripts/06_tune_rf.py
"""

import importlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, "scripts")
tune = importlib.import_module("03_tune_knn")
FEATURES, TARGET, metrics = tune.FEATURES, tune.TARGET, tune.metrics

MAX_DEPTH = [10, 15, 20, None]  # None = no limit
MIN_LEAF = [3, 5, 10, 20]
MAX_FEATURES = 0.5
N_TREES = 300
KEYS = ["max_depth", "min_leaf"]


def plot(summary):
    colours = dict(zip(MIN_LEAF, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    surface, muted = "#fcfcfb", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted})
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=surface)
    ax.set_facecolor(surface)
    ax.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)
    labels = [str(d) for d in MAX_DEPTH[:-1]] + ["No limit"]
    for leaf in MIN_LEAF:
        mae = summary.xs(leaf, level="min_leaf").mae.reindex(labels)
        ax.plot(labels, mae, color=colours[leaf], linewidth=2, marker="o", markersize=7,
                markeredgecolor=surface, markeredgewidth=2, label=f"At least {leaf} rows per leaf: best {mae.min():.4f}")
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Mean absolute error (lower is better)")
    ax.set_title("Random forest tuning: next-day sales error by tree depth and leaf size\n"
                 "Average over five validation weeks", loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("results/rf_tuning.png", dpi=160, facecolor=surface)


if __name__ == "__main__":
    data = pd.read_parquet("data/processed/features.parquet")
    data = data[data.split == "train"]
    folds = sorted(f for f in data.val_fold.unique() if f > 0)

    rows = []
    for depth in MAX_DEPTH:
        for leaf in MIN_LEAF:
            for fold in folds:
                val = data[data.val_fold == fold]
                train = data[data.dt < val.dt.min()]
                model = RandomForestRegressor(n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf,
                                              max_features=MAX_FEATURES, n_jobs=-1, random_state=0)
                model.fit(train[FEATURES], train[TARGET])
                rows.append({"max_depth": "No limit" if depth is None else str(depth), "min_leaf": leaf, "fold": fold,
                             "actual_depth": max(tree.get_depth() for tree in model.estimators_),
                             **metrics(val[TARGET], model.predict(val[FEATURES]))})
    results = pd.DataFrame(rows)
    results.to_csv("results/rf_tuning_folds.csv", index=False)

    baseline = pd.Series({f: metrics(data[data.val_fold == f][TARGET], data[data.val_fold == f].sales_mean7)["mae"] for f in folds})
    summary = results.groupby(KEYS, sort=False)[["mae", "wape", "mse"]].mean()
    summary["deepest_tree"] = results.groupby(KEYS, sort=False).actual_depth.max()
    summary["folds_beating_baseline"] = results.set_index(KEYS + ["fold"]).mae.unstack().lt(baseline, axis=1).sum(axis=1)
    summary.to_csv("results/rf_tuning_summary.csv")
    print(f"baseline MAE {baseline.mean():.4f}\n")
    print(summary.round(4).to_string())
    best = summary.mae.idxmin()
    print("\nbest:", best, "\nMAE per fold:", results.set_index(KEYS).loc[best].set_index("fold").mae.round(3).to_dict())
    plot(summary)
    print("saved results/rf_tuning.png")
