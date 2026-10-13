#!/usr/bin/env python3
"""Cache descriptive EDA for the pinned raw training file and existing cohort.

Run from the repository root:
  python scripts/download_public_data.py --splits train
  python scripts/build_dataset_eda.py

This reads the original 90-day training file, the fixed experiment plan and the
saved feature table. It does not select a new cohort, fit a model, or read final
evaluation outcomes. Dataset composition uses all 90 days retrospectively;
behavioral comparisons use only dates before the first validation week.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
KEY = ["store_id", "product_id"]
EARLY_END = "2024-05-22"
REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
READ_COLUMNS = ["city_id", "store_id", "product_id", "first_category_id",
                "dt", "sale_amount", "discount", "stock_hour6_22_cnt"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "series": len(frame[KEY].drop_duplicates()),
        "stores": int(frame.store_id.nunique()),
        "products": int(frame.product_id.nunique()),
        "cities": int(frame.city_id.nunique()),
        "date_start": frame.dt.min().strftime("%Y-%m-%d"),
        "date_end": frame.dt.max().strftime("%Y-%m-%d"),
        "days": int(frame.dt.nunique()),
    }


def behavior(frame: pd.DataFrame, scope: str) -> dict:
    # A recorded zero sale is a real observation; it is not an invalid record.
    valid = frame.discount.gt(0) & frame.discount.le(1) & np.isfinite(frame.discount)
    discounted = valid & frame.discount.lt(0.95)
    zero = frame.sale_amount.eq(0)
    stockout = frame.stock_hour6_22_cnt.gt(0)
    return {
        "scope": scope,
        "rows": len(frame),
        "zero_sales_count": int(zero.sum()),
        "zero_sales_share": float(zero.mean()),
        "stockout_day_count": int(stockout.sum()),
        "stockout_day_share": float(stockout.mean()),
        "valid_discount_rows": int(valid.sum()),
        "discounted_count": int(discounted.sum()),
        "discounted_share": float(discounted.sum() / valid.sum()),
        "invalid_discount_count": int((~valid).sum()),
    }


def build(raw: Path, features: Path, plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    raw_hash, feature_hash = sha256(raw), sha256(features)
    assert raw_hash == plan["raw_sha256"]["train"], "Raw train differs from the fixed experiment."
    assert feature_hash == plan["feature_sha256"], "Feature table differs from the fixed experiment."
    assert min(v[0] for v in plan["folds"].values()) == EARLY_END

    source = pq.ParquetFile(raw)
    frame = pd.read_parquet(raw, columns=READ_COLUMNS)
    frame["dt"] = pd.to_datetime(frame.dt)
    missing = {name: int(frame[name].isna().sum()) for name in READ_COLUMNS}
    assert not any(missing.values()), "Missing data require explicit EDA denominators."
    assert not frame.duplicated(KEY + ["dt"]).any(), "Duplicate daily records."
    series = frame[["city_id", *KEY, "first_category_id"]].drop_duplicates()
    assert not series.duplicated(KEY).any(), "A series has inconsistent city/category metadata."
    dates_per_series = frame.groupby(KEY).dt.nunique()
    assert dates_per_series.eq(90).all(), "Expected a 90-day history for each series."
    assert series.groupby("store_id").city_id.nunique().eq(1).all()

    keys = pd.read_parquet(features, columns=KEY).drop_duplicates()
    assert len(keys) == 312, "The existing cohort must contain exactly 312 series."
    selected = frame.merge(keys, on=KEY, how="inner", validate="many_to_one")
    selected_series = series.merge(keys, on=KEY, how="inner", validate="one_to_one")
    assert len(selected_series) == 312
    assert len(selected) == 312 * 90

    full_meta = metadata(frame)
    full_meta["schema"] = [{"name": field.name, "type": str(field.type)}
                           for field in source.schema_arrow]
    assert len(full_meta["schema"]) == 19
    assert full_meta["rows"] == source.metadata.num_rows == 4_500_000
    assert (full_meta["series"], full_meta["stores"], full_meta["products"], full_meta["cities"]) == (50_000, 898, 865, 18)
    selected_meta = metadata(selected)
    selected_meta["store_ids"] = sorted(int(x) for x in selected.store_id.unique())
    selected_meta["city_ids"] = sorted(int(x) for x in selected.city_id.unique())

    cities_frame = series.groupby("city_id").agg(stores=("store_id", "nunique"), series=("product_id", "size")).reset_index()
    cities = cities_frame.to_dict(orient="records")
    assert sum(row["stores"] for row in cities) == full_meta["stores"]
    assert sum(row["series"] for row in cities) == full_meta["series"]
    full_categories = series.groupby("first_category_id").size()
    selected_categories = selected_series.groupby("first_category_id").size()
    categories = [{
        "first_category_id": int(category),
        "full_series": int(count),
        "selected_series": int(selected_categories.get(category, 0)),
        "full_share": float(count / len(series)),
        "selected_share": float(selected_categories.get(category, 0) / len(selected_series)),
    } for category, count in full_categories.items()]
    assert sum(row["full_series"] for row in categories) == 50_000
    assert sum(row["selected_series"] for row in categories) == 312
    stores_frame = selected_series.groupby(["city_id", "store_id"]).agg(series=("product_id", "size")).reset_index()

    early = frame[frame.dt < EARLY_END]
    early_selected = selected[selected.dt < EARLY_END]
    assert early.dt.max() == pd.Timestamp(EARLY_END) - pd.Timedelta(days=1)
    assert len(early) == early.dt.nunique() * 50_000
    assert len(early_selected) == early.dt.nunique() * 312
    quality = {
        "scope": "Original 90-day training file; descriptive only, no cleaning applied",
        "rows": len(frame),
        "missing_by_read_column": missing,
        "duplicate_store_product_date_rows": 0,
        "discount_zero_count": int(frame.discount.eq(0).sum()),
        "discount_above_one_count": int(frame.discount.gt(1).sum()),
        "discount_negative_count": int(frame.discount.lt(0).sum()),
        "discount_nonfinite_count": int((~np.isfinite(frame.discount)).sum()),
    }
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": {"raw_train": raw_hash, "features": feature_hash},
        "full_train": full_meta,
        "selected": selected_meta,
        "cities": cities,
        "categories": categories,
        "selected_stores": stores_frame.to_dict(orient="records"),
        "early_period": {
            "start": early.dt.min().strftime("%Y-%m-%d"),
            "end_inclusive": early.dt.max().strftime("%Y-%m-%d"),
            "end_exclusive": EARLY_END,
            "days": int(early.dt.nunique()),
            "comparison": [behavior(early, "Full dataset"), behavior(early_selected, "Selected cohort")],
            "definitions": {
                "zero_sales_share": "Rows with sale_amount == 0 / all rows in the scope and period.",
                "stockout_day_share": "Rows with stock_hour6_22_cnt > 0 / all rows in the scope and period.",
                "discounted_share": "Rows with 0 < discount < 0.95 / rows with finite 0 < discount <= 1.",
                "invalid_discount_count": "Rows with discount <= 0, discount > 1, missing or nonfinite discount; excluded only from the discount-share denominator.",
            },
        },
        "raw_quality": quality,
        "field_groups": {
            "Identifiers and category hierarchy": ["city_id", "store_id", "management_group_id", "first_category_id", "second_category_id", "third_category_id", "product_id"],
            "Date, sales and stockouts": ["dt", "sale_amount", "hours_sale", "stock_hour6_22_cnt", "hours_stock_status"],
            "Price and calendar context": ["discount", "holiday_flag", "activity_flag"],
            "Weather": ["precpt", "avg_temperature", "avg_humidity", "avg_wind_level"],
        },
        "caveats": [
            "Composition describes the original 90-day training file retrospectively. No final evaluation file is read.",
            "The 312 existing store-product keys were selected using all 90 original training dates. Early-period charts do not undo that retrospective cohort choice.",
            "Behavioral comparisons use only 28 March through 21 May 2024, before the first validation week.",
            "Category and city identifiers are anonymous. No product-category or city names are inferred.",
            "Category shares count unique store-product histories, not units sold or revenue.",
            "Zero-sales and stockout rows are retained. Recorded stockout days do not reveal inventory quantity, freshness or waste.",
            "These summaries show sample composition and association; they do not estimate causal effects of discounting.",
        ],
        "provenance": {
            "dataset": "Dingdong-Inc/FreshRetailNet-50K",
            "revision": REVISION,
            "raw_train_sha256": raw_hash,
            "feature_sha256": feature_hash,
            "experiment_plan_sha256": sha256(plan_path),
            "source_script_sha256": sha256(Path(__file__)),
            "download_command": "python scripts/download_public_data.py --splits train",
            "reproduce_command": "python scripts/build_dataset_eda.py",
            "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
            "assertions": "Pinned raw and feature hashes; 312 saved keys; consistent series metadata; unique daily rows; 90 dates per history; city and category counts reconcile.",
        },
    }
    assert sum(len(fields) for fields in result["field_groups"].values()) == 19
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/train.parquet")
    parser.add_argument("--features", type=Path, default=ROOT / "results/v2/features.parquet")
    parser.add_argument("--plan", type=Path, default=ROOT / "results/v2/experiment_plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/data_eda/overview.json")
    args = parser.parse_args()
    result = build(args.raw, args.features, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".overview-", suffix=".json", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, args.output)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "full_train_rows": result["full_train"]["rows"],
                      "selected_series": result["selected"]["series"],
                      "early_period": result["early_period"]["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
