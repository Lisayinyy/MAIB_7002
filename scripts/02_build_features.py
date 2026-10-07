"""Step 2: clean the selected series and build the feature table.

One row = one store-product-day, where the row's date is day t+1 (the day we predict).
Columns ending in _t hold values from day t (the evening the decision is made).

Run with:  uv run python scripts/02_build_features.py
"""

import pandas as pd

SELECTED = "data/processed/selected_series.csv"
OUT = "data/processed/features.parquet"
KEY = ["store_id", "product_id"]

# Validation folds (one week each). For fold k, train on all rows dated before the fold starts.
VAL_FOLDS = {
    1: ("2024-05-22", "2024-05-28"),
    2: ("2024-05-29", "2024-06-04"),
    3: ("2024-06-05", "2024-06-11"),
    4: ("2024-06-12", "2024-06-18"),
    5: ("2024-06-19", "2024-06-25"),
}
TEST_START = "2024-06-26"  # eval.parquet: the unseen week

cols = KEY + ["dt", "sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag", "activity_flag",
              "precpt", "avg_temperature", "avg_humidity"]
keys = pd.read_csv(SELECTED)[KEY]
df = pd.concat([pd.read_parquet(f"data/raw/{name}.parquet", columns=cols) for name in ["train", "eval"]])
df = df.merge(keys, on=KEY)
df["dt"] = pd.to_datetime(df.dt)
df = df.sort_values(KEY + ["dt"]).reset_index(drop=True)

# --- Cleaning -------------------------------------------------------------------------------
assert not df.duplicated(KEY + ["dt"]).any(), "duplicate store-product-day rows"
assert df.groupby(KEY).dt.agg(lambda x: x.diff().dt.days.dropna().eq(1).all()).all(), "gaps in dates"
assert df.notna().all().all(), "missing values"

df["discount"] = df.discount.clip(upper=1.0)  # a few rows sit slightly above 1.0
bad_day = df.discount == 0  # discount of 0 = given away / data artefact: sales that day are not real demand
print(f"rows: {len(df)}  series: {len(keys)}  days with discount == 0: {bad_day.sum()}")
df.loc[bad_day, ["sale_amount", "discount"]] = float("nan")  # blanked so they never feed a lag or a target

# --- Features -------------------------------------------------------------------------------
g = df.groupby(KEY)
feat = df[KEY + ["dt"]].copy()
feat["sales_t"] = g.sale_amount.shift(1)
feat["sales_lag7"] = g.sale_amount.shift(7)  # same weekday as the day we predict
feat["sales_mean7"] = g.sale_amount.transform(lambda x: x.shift(1).rolling(7, min_periods=5).mean())
feat["sales_mean_to_date"] = g.sale_amount.transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
feat["stockout_hours_t"] = g.stock_hour6_22_cnt.shift(1)
feat["discount_t"] = g.discount.shift(1)
feat["discount_next"] = df.discount  # tomorrow's discount: the manager's decision
feat["activity_t"] = g.activity_flag.shift(1)
feat["holiday_next"] = df.holiday_flag
feat["weekday_next"] = df.dt.dt.dayofweek  # 0 = Monday
feat["precpt_t"] = g.precpt.shift(1)
feat["temperature_t"] = g.avg_temperature.shift(1)
feat["humidity_t"] = g.avg_humidity.shift(1)
feat["target_sales"] = df.sale_amount
feat["target_stockout_hours"] = df.stock_hour6_22_cnt  # NOT a feature: only for the stockout sensitivity check

before = len(feat)
feat = feat.dropna().reset_index(drop=True)
print(f"dropped {before - len(feat)} rows without full history or with a blanked day; {len(feat)} remain")

# --- Splits ---------------------------------------------------------------------------------
feat["split"] = "train"
feat.loc[feat.dt >= TEST_START, "split"] = "test"
feat["val_fold"] = 0
for k, (start, end) in VAL_FOLDS.items():
    feat.loc[feat.dt.between(start, end), "val_fold"] = k

feat.to_parquet(OUT, index=False)
print(f"\nsaved {OUT}")
print(feat.groupby(["split", "val_fold"]).agg(rows=("dt", "size"), first_day=("dt", "min"), last_day=("dt", "max")))
print("\n", feat.drop(columns=KEY + ["dt", "split", "val_fold"]).describe().T[["mean", "std", "min", "max"]].round(2))
