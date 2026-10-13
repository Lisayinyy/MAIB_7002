#!/usr/bin/env python3
"""Audit the notebook's data-selection story without fitting any model.

Reads the fixed public raw files plus already saved V2 features and scenarios.
The resulting small JSON is included in the repository so the explanatory
notebook can show the full-data funnel without downloading the raw dataset.
Run again after downloading raw data to reproduce each reported count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["store_id", "product_id"]
ROW_KEY = KEY + ["dt"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(frame: pd.DataFrame) -> dict:
    result = {
        "rows": len(frame),
        "stores": int(frame.store_id.nunique()),
        "products": int(frame.product_id.nunique()),
        "store_product_series": int(frame[KEY].drop_duplicates().shape[0]),
        "first_day": pd.Timestamp(frame["dt"].min()).strftime("%Y-%m-%d"),
        "last_day": pd.Timestamp(frame["dt"].max()).strftime("%Y-%m-%d"),
        "days": int(frame["dt"].nunique()),
    }
    if "city_id" in frame:
        result["cities"] = int(frame.city_id.nunique())
    return result


def count_rates(frame: pd.DataFrame, column: str) -> dict:
    return {f"{float(rate):.2f}": int(count)
            for rate, count in frame[column].value_counts().sort_index(ascending=False).items()}


def choose_naive(curves: pd.DataFrame, objective: str, supported_only: bool) -> dict:
    """Descriptive scenario argmax; exact ties prefer the highest price rate."""
    candidates = curves[curves.supported].copy() if supported_only else curves.copy()
    candidates["objective"] = candidates.selected_sales
    if objective == "price_rate_times_predicted_sales":
        candidates["objective"] *= candidates.rate
    chosen = candidates.sort_values(ROW_KEY + ["objective", "rate"],
                                    ascending=[True, True, True, False, False])
    chosen = chosen.drop_duplicates(ROW_KEY)
    return {
        "objective": objective,
        "supported_candidates_only": supported_only,
        "tie_rule": "An exact objective tie prefers the highest price rate.",
        "decisions": len(chosen),
        "discounted_decisions": int(chosen.rate.lt(1).sum()),
        "discount_share": float(chosen.rate.lt(1).mean()),
        "chosen_rate_counts": count_rates(chosen, "rate"),
        "unsupported_chosen_decisions": int((~chosen.supported).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results/v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/v2/story_evidence.json")
    args = parser.parse_args()
    inputs = {
        "raw_train": args.raw_dir / "train.parquet",
        "raw_eval": args.raw_dir / "eval.parquet",
        "features": args.result_dir / "features.parquet",
        "discount_response": args.result_dir / "discount_response.csv",
        "recommendations": args.result_dir / "recommendations.csv",
        "policy_summary": args.result_dir / "policy_summary.json",
        "teammate_selection_source": ROOT / "src/finalproject_pricingml/data.py",
        "teammate_config_source": ROOT / "src/finalproject_pricingml/config.py",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing required evidence inputs; download raw data first: " + ", ".join(missing))
    cols = ["city_id", "store_id", "product_id", "dt", "sale_amount", "discount"]
    train = pd.read_parquet(inputs["raw_train"], columns=cols).sort_values(ROW_KEY)
    evaluation = pd.read_parquet(inputs["raw_eval"], columns=cols)
    train["disc_day"] = train.discount.lt(.95)
    train["zero_sales"] = train.sale_amount.eq(0)
    previous = train.groupby(KEY).disc_day.shift(1)
    train["switched"] = previous.notna() & train.disc_day.ne(previous)
    series = train.groupby(KEY).agg(
        city_id=("city_id", "first"), days=("dt", "size"),
        discounted_share=("disc_day", "mean"), switches=("switched", "sum"),
        zero_sales_share=("zero_sales", "mean"),
    )
    share_ok = series.discounted_share.between(.15, .85)
    switches_ok = series.switches.ge(6)
    zero_ok = series.zero_sales_share.lt(.20)
    series["eligible"] = share_ok & switches_ok & zero_ok
    stores = series.reset_index().groupby("store_id").agg(
        city_id=("city_id", "first"), n_series=("product_id", "size"),
        n_eligible=("eligible", "sum"),
    ).sort_values("n_eligible", ascending=False)
    selected = series[series.eligible & series.index.get_level_values("store_id").isin(stores.head(5).index)]
    selected_keys = selected.reset_index()[KEY]
    selected_train = train.merge(selected_keys, on=KEY, validate="many_to_one")
    selected_eval = evaluation.merge(selected_keys, on=KEY, validate="many_to_one")
    raw_selected = pd.concat([selected_train, selected_eval], ignore_index=True)
    features = pd.read_parquet(inputs["features"])
    feature_keys = set(features[KEY].drop_duplicates().itertuples(index=False, name=None))
    assert feature_keys == set(selected.index), "Reconstructed cohort differs from saved V2 features"
    assert not features.duplicated(ROW_KEY).any()
    assert not raw_selected.duplicated(ROW_KEY).any()
    raw_daily = raw_selected.copy()
    raw_daily["dt"] = pd.to_datetime(raw_daily["dt"])
    aligned_target = features[ROW_KEY + ["target_sales"]].merge(
        raw_daily[ROW_KEY + ["sale_amount"]], on=ROW_KEY, validate="one_to_one",
    )
    assert aligned_target.target_sales.eq(aligned_target.sale_amount).all()
    removed = raw_daily[ROW_KEY].merge(features[ROW_KEY], on=ROW_KEY,
                                       how="left", indicator=True, validate="one_to_one")
    removed = removed[removed["_merge"].eq("left_only")]

    curves = pd.read_csv(inputs["discount_response"])
    recommendations = pd.read_csv(inputs["recommendations"])
    assert not curves.duplicated(ROW_KEY + ["rate"]).any()
    assert not recommendations.duplicated(ROW_KEY).any()
    assert len(recommendations) == len(features[features.split.eq("test")])
    actual = recommendations.merge(
        features.assign(dt=features["dt"].astype(str))[ROW_KEY + ["stockout_hours_t"]],
        on=ROW_KEY, validate="one_to_one",
    )
    not_stockout = actual.stockout_hours_t.lt(1)
    discounted = actual.status.eq("recommend_discount")
    policy = json.loads(inputs["policy_summary"].read_text())

    record = {
        "purpose": "Auditable descriptive evidence for a business-first notebook; no model fits and no new evaluation or tuning.",
        "inputs_sha256": {name: sha256(path) for name, path in inputs.items()},
        "source_script_sha256": sha256(Path(__file__)),
        "definitions": {
            "unit": "One store-product-day; the same product in two stores is two sales series.",
            "discount_rate": "Fraction of the regular price: 1.0 means full price, 0.9 means 10% off.",
            "discounted_day_for_selection": "discount < 0.95, strictly more than 5% off; exactly 0.95 is not discounted for this filter.",
            "eligibility": "On all raw train days, discounted share is in [0.15, 0.85], state switches >= 6, and zero-sales share < 0.20.",
            "switches": "Count changes between consecutive days of the binary discounted-day flag, excluding the first row.",
            "store_selection": "Rank stores by eligible series count, descending, and retain eligible series in the first five stores, matching teammate code.",
            "selection_timing_limit": "The cohort is selected using the full historical training period, including validation dates; this is a retrospective benchmark cohort, not a strictly nested historical selection procedure.",
            "filter_limits": "Variation improves scenario support but does not randomize discounts or remove confounding; zero sales may reflect demand, availability, or recording conditions.",
            "value_proxy": "price rate times predicted standardized sales; not observed money, profit, expected revenue, or a causal policy effect.",
            "target_construction": "target_sales is copied from the public dataset's sale_amount on the predicted date; this project's feature-building code does not standardize Y again. All saved target values match those raw sale_amount values exactly.",
        },
        "full_public_train": summary(train),
        "full_public_eval": summary(evaluation),
        "selection_funnel": {
            "all_series": len(series),
            "after_discount_share_filter": int(share_ok.sum()),
            "after_discount_share_and_switch_filters": int((share_ok & switches_ok).sum()),
            "after_all_three_filters": int(series.eligible.sum()),
            "stores_with_any_eligible_series": int(stores.n_eligible.gt(0).sum()),
            "cities_with_any_eligible_series": int(series.loc[series.eligible, "city_id"].nunique()),
            "selected_series_in_top_five": len(selected),
            "selected_unique_products": int(selected_keys.product_id.nunique()),
            "selected_city_ids": [int(x) for x in selected.city_id.unique()],
            "top_five_stores_ranked": stores.head(5).reset_index().to_dict(orient="records"),
            "next_five_stores_ranked": stores.iloc[5:10].reset_index().to_dict(orient="records"),
        },
        "selected_raw_train": summary(selected_train),
        "selected_raw_eval": summary(selected_eval),
        "cleaning": {
            "raw_selected_rows": len(raw_selected),
            "raw_selected_discount_above_one_rows": int(raw_selected.discount.gt(1).sum()),
            "raw_selected_zero_discount_rows": int(raw_selected.discount.eq(0).sum()),
            "raw_selected_missing_values_in_audited_columns": int(raw_selected[cols].isna().sum().sum()),
            "saved_feature_rows": len(features),
            "rows_removed_before_saved_features": len(raw_selected) - len(features),
            "removed_row_dates_and_counts": {pd.Timestamp(date).strftime("%Y-%m-%d"): int(count)
                                             for date, count in removed["dt"].value_counts().sort_index().items()},
            "description": "Teammate code clips price rate above 1, blanks zero-rate sales/rates before lag construction, builds only lagged sales inputs, and drops rows lacking complete features; V2 preserves these exact row keys.",
        },
        "saved_features_all": summary(features),
        "saved_features_train": summary(features[features.split.eq("train")]),
        "saved_features_test": summary(features[features.split.eq("test")]),
        "naive_scenario_policies": {
            "candidate_rates": sorted(curves.rate.unique().tolist(), reverse=True),
            "max_sales_all_candidates": choose_naive(curves, "predicted_standardized_sales", False),
            "max_value_all_candidates": choose_naive(curves, "price_rate_times_predicted_sales", False),
            "max_sales_supported_candidates": choose_naive(curves, "predicted_standardized_sales", True),
            "max_value_supported_candidates": choose_naive(curves, "price_rate_times_predicted_sales", True),
            "interpretation": "Descriptive argmax of already saved model scenarios, holding other inputs fixed; not observed counterfactual sales and not evidence that daily discounting is beneficial.",
        },
        "actual_saved_policy": {
            "rows": len(recommendations),
            "status_counts": recommendations.status.value_counts().to_dict(),
            "recommended_discount_rate_counts": count_rates(recommendations[discounted], "recommended_rate"),
            "discount_share_all_rows": float(discounted.mean()),
            "rows_without_previous_day_stockout": int(not_stockout.sum()),
            "discounted_rows_without_previous_day_stockout": int((discounted & not_stockout).sum()),
            "discount_share_without_previous_day_stockout": float(discounted[not_stockout].mean()),
            "parameters": policy["parameters"],
            "limitation": "The guarded policy still recommends a discount for nearly all cases without previous-day stockout; high discount frequency remains an unresolved model/policy concern.",
        },
        "checks": {
            "cohort_keys_match_saved_features": True,
            "unique_raw_selected_daily_keys": True,
            "unique_saved_feature_daily_keys": True,
            "unique_scenario_daily_rate_keys": True,
            "one_recommendation_per_test_row": True,
            "saved_targets_equal_raw_sale_amount": True,
            "model_retraining": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "full_train": record["full_public_train"],
                      "funnel": record["selection_funnel"], "policy": record["actual_saved_policy"]}, indent=2))


if __name__ == "__main__":
    main()
