"""Step 5: score the locked kNN design once on the unseen test week (June 26 - July 2).

Locked design: 10 features, standard scaling, Euclidean distance, equal weights, k = 25.
The model is fitted on every training row (April 4 - June 25) and predicts the test week.

Run with:  uv run python scripts/05_test_knn.py
"""

import importlib
import sys

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
tune = importlib.import_module("03_tune_knn")  # reuse the feature list and metric definitions
FEATURES, TARGET, metrics = tune.FEATURES, tune.TARGET, tune.metrics
K = 25

data = pd.read_parquet("data/processed/features.parquet")
train, test = data[data.split == "train"], data[data.split == "test"].copy()

model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=K))
model.fit(train[FEATURES], train[TARGET])
test["pred_knn"] = model.predict(test[FEATURES])
test["pred_baseline"] = test.sales_mean7
test["pred_same_weekday"] = test.sales_lag7

print(f"train rows: {len(train)} ({train.dt.min().date()} to {train.dt.max().date()})")
print(f"test rows:  {len(test)} ({test.dt.min().date()} to {test.dt.max().date()})\n")

METHODS = {"kNN (k=25)": "pred_knn", "Baseline (7-day mean)": "pred_baseline", "Same weekday last week": "pred_same_weekday"}
overall = pd.DataFrame({name: metrics(test[TARGET], test[col]) for name, col in METHODS.items()}).T
overall["rmse"] = overall.mse**0.5
print(overall.round(4).to_string(), "\n")


def mae_by(group):
    return pd.DataFrame({name: (test[TARGET] - test[col]).abs().groupby(group).mean() for name, col in METHODS.items()}
                        ).assign(rows=test.groupby(group).size())


test["discount_band"] = pd.cut(test.discount_next, [0, 0.8, 0.95, 1.0], labels=["deep (below 0.8)", "moderate (0.8-0.95)", "none (0.95-1.0)"])
print("MAE by day:\n", mae_by(test.dt.dt.strftime("%a %b %d")).round(3).to_string(), "\n")
print("MAE by tomorrow's discount:\n", mae_by(test.discount_band).round(3).to_string(), "\n")
print("MAE by store:\n", mae_by(test.store_id).round(3).to_string())

overall.to_csv("results/knn_test_summary.csv")
test[["store_id", "product_id", "dt", "discount_next", TARGET, "target_stockout_hours", *METHODS.values()]].to_csv(
    "results/knn_test_predictions.csv", index=False)
