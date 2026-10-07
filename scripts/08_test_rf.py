"""Step 8: score the locked random forest once on the unseen test week (June 26 - July 2).

Locked design: the 10 kNN features, 300 trees, maximum depth 10, at least 3 rows per leaf,
half the features tried at each split, squared-error splits.
The kNN predictions from scripts/05_test_knn.py are read back in for a side-by-side comparison.

Run with:  uv run python scripts/08_test_rf.py   (after scripts/05_test_knn.py)
"""

import importlib
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, "scripts")
tune = importlib.import_module("03_tune_knn")
FEATURES, TARGET, metrics = tune.FEATURES, tune.TARGET, tune.metrics
KEY = ["store_id", "product_id", "dt"]

data = pd.read_parquet("data/processed/features.parquet")
train, test = data[data.split == "train"], data[data.split == "test"].copy()

model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, max_features=0.5,
                              n_jobs=-1, random_state=0).fit(train[FEATURES], train[TARGET])
test["pred_rf"] = model.predict(test[FEATURES])

knn = pd.read_csv("results/knn_test_predictions.csv", parse_dates=["dt"])
test = test.merge(knn[KEY + ["pred_knn", "pred_baseline", "pred_same_weekday"]], on=KEY, validate="one_to_one")

print(f"train rows: {len(train)}  test rows: {len(test)}\n")
METHODS = {"Random forest": "pred_rf", "kNN (k=25)": "pred_knn", "Baseline (7-day mean)": "pred_baseline",
           "Same weekday last week": "pred_same_weekday"}
overall = pd.DataFrame({name: metrics(test[TARGET], test[col]) for name, col in METHODS.items()}).T
overall["rmse"] = overall.mse**0.5
print(overall.round(4).to_string(), "\n")


def mae_by(group):
    return pd.DataFrame({name: (test[TARGET] - test[col]).abs().groupby(group, observed=True).mean()
                         for name, col in METHODS.items()}).assign(rows=test.groupby(group, observed=True).size())


test["discount_band"] = pd.cut(test.discount_next, [0, 0.8, 0.95, 1.0], labels=["deep (below 0.8)", "moderate (0.8-0.95)", "none (0.95-1.0)"])
print("MAE by tomorrow's discount:\n", mae_by(test.discount_band).round(3).to_string(), "\n")
print("Mean sales by tomorrow's discount (actual and predicted):\n",
      test.groupby("discount_band", observed=True)[[TARGET, *METHODS.values()]].mean().round(3).to_string(), "\n")
print("MAE by day:\n", mae_by(test.dt.dt.strftime("%m-%d %a")).round(3).to_string(), "\n")
print("MAE by store:\n", mae_by(test.store_id).round(3).to_string(), "\n")
print("Feature importance:\n", pd.Series(model.feature_importances_, FEATURES).sort_values(ascending=False).round(3).to_string())

overall.to_csv("results/test_summary.csv")
test[KEY + ["discount_next", TARGET, "target_stockout_hours", *METHODS.values()]].to_csv("results/test_predictions.csv", index=False)
