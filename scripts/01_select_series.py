"""Step 1: score every store-product series for discount variation and rank stores.

Run with:  uv run python scripts/01_select_series.py
"""

import pandas as pd

RAW = "data/raw/train.parquet"
DISCOUNT_DAY = 0.95  # a day counts as "discounted" when discount < 0.95 (more than 5% off)
MIN_SHARE, MAX_SHARE = 0.15, 0.85  # share of discounted days: drops never/always discounted
MIN_SWITCHES = 6  # times the series flips between discounted and not discounted
MAX_ZERO_SALES = 0.20  # share of days with zero sales
TOP_N_STORES = 5

cols = ["city_id", "store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt", "discount"]
df = pd.read_parquet(RAW, columns=cols).sort_values(["store_id", "product_id", "dt"])
df["disc_day"] = df.discount < DISCOUNT_DAY

g = df.groupby(["store_id", "product_id"])
series = g.agg(
    city_id=("city_id", "first"),
    disc_share=("disc_day", "mean"),
    disc_std=("discount", "std"),
    disc_min=("discount", "min"),
    mean_sales=("sale_amount", "mean"),
    zero_sales_share=("sale_amount", lambda x: (x == 0).mean()),
    stockout_day_share=("stock_hour6_22_cnt", lambda x: (x > 0).mean()),
)
series["switches"] = g.disc_day.apply(lambda x: (x != x.shift()).sum() - 1)
series["usable"] = (
    series.disc_share.between(MIN_SHARE, MAX_SHARE)
    & (series.switches >= MIN_SWITCHES)
    & (series.zero_sales_share < MAX_ZERO_SALES)
)

print(f"series: {len(series)}  never discounted: {(series.disc_share == 0).sum()}  "
      f"always discounted: {(series.disc_share == 1).sum()}  usable: {series.usable.sum()}")

stores = series.reset_index().groupby("store_id").agg(
    city_id=("city_id", "first"), n_series=("product_id", "size"), n_usable=("usable", "sum")
)
top = stores.sort_values("n_usable", ascending=False).head(TOP_N_STORES)
print(f"\nTop {TOP_N_STORES} stores by usable series:\n{top}")

selected = series[series.usable & series.index.get_level_values("store_id").isin(top.index)]
selected.to_csv("data/processed/selected_series.csv")
print(f"\nSaved {len(selected)} selected series to data/processed/selected_series.csv")
