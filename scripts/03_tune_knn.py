"""Step 3, stage 1: tune kNN over the number of neighbours, with and without extra weight on sales on day t.

Every setting is scored on the five time-ordered weekly folds: train on all rows dated
before the fold, predict the fold's week. The test week is never touched here.

Run with:  uv run python scripts/03_tune_knn.py
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = ["sales_t", "sales_lag7", "sales_mean7", "sales_mean_to_date", "stockout_hours_t",
            "discount_t", "discount_next", "activity_t", "holiday_next", "weekday_next"]
# Weather (rain, temperature, humidity) is left out: validation error was lower without it.
TARGET = "target_sales"
K_VALUES = [3, 5, 10, 15, 25, 40, 60, 100]

# A weight multiplies a standardized feature, so a weight of 2 makes that feature count
# twice as much when kNN measures how similar two rows are. Unlisted features keep weight 1.
WEIGHTINGS = {
    "equal": {},
    "sales_t": {"sales_t": 2},
}


def metrics(actual, predicted):
    error = actual - predicted
    return {"mae": error.abs().mean(), "wape": error.abs().sum() / actual.sum(), "mse": (error**2).mean()}


def score(data, fold, k, weighting):
    """Fit on everything before the fold, predict the fold, return its error metrics."""
    val = data[data.val_fold == fold]
    train = data[data.dt < val.dt.min()]
    w = np.array([WEIGHTINGS[weighting].get(c, 1.0) for c in FEATURES])
    scaler = StandardScaler().fit(train[FEATURES])  # scaling learned from the training rows only
    model = KNeighborsRegressor(n_neighbors=k).fit(scaler.transform(train[FEATURES]) * w, train[TARGET])
    predicted = model.predict(scaler.transform(val[FEATURES]) * w)
    return {"weighting": weighting, "k": k, "fold": fold, **metrics(val[TARGET], predicted)}


if __name__ == "__main__":
    data = pd.read_parquet("data/processed/features.parquet")
    data = data[data.split == "train"]
    folds = sorted(f for f in data.val_fold.unique() if f > 0)

    jobs = [delayed(score)(data, f, k, w) for w in WEIGHTINGS for k in K_VALUES for f in folds]
    results = pd.DataFrame(Parallel(n_jobs=-1)(jobs))

    baseline = pd.DataFrame([
        {"weighting": "baseline (7-day mean)", "k": 0, "fold": f,
         **metrics(data[data.val_fold == f][TARGET], data[data.val_fold == f].sales_mean7)} for f in folds
    ])
    results = pd.concat([baseline, results])
    results.to_csv("results/knn_stage1_folds.csv", index=False)

    summary = results.groupby(["weighting", "k"], sort=False)[["mae", "wape", "mse"]].mean()
    summary["mae_fold_std"] = results.groupby(["weighting", "k"], sort=False).mae.std()
    summary["folds_beating_baseline"] = (
        results.set_index(["weighting", "k", "fold"]).mae.unstack()
        .lt(baseline.set_index("fold").mae, axis=1).sum(axis=1)
    )
    summary.to_csv("results/knn_stage1_summary.csv")
    print(summary.round(4).to_string())
    print("\nMAE per fold:\n", results.pivot_table(index=["weighting", "k"], columns="fold", values="mae", sort=False).round(3).to_string())
