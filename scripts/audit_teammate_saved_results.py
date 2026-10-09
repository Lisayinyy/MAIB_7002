#!/usr/bin/env python3
"""Independently audit committed predictions without fitting or changing any models.

The default uses only the Python standard library. --features additionally reconstructs
the feature table from public raw data in memory and tests recommendation coverage;
those coverage checks use constant test predictions, not simulated training data.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(actual, predicted):
    errors = [a - p for a, p in zip(actual, predicted)]
    assert len(actual) == len(predicted) and errors
    assert all(math.isfinite(v) for v in actual + predicted)
    mse = fmean(e * e for e in errors)
    return {"mae": fmean(abs(e) for e in errors), "mse": mse,
            "rmse": math.sqrt(mse), "wape": sum(abs(e) for e in errors) / sum(actual)}


def close(a, b, label):
    assert abs(a - b) < 1e-11, f"{label}: {a} != {b}"


def audit_saved(root, original=None):
    results = root / "results"
    rows = read_csv(results / "test_predictions.csv")
    keys = [(r["store_id"], r["product_id"], r["dt"]) for r in rows]
    assert len(keys) == len(set(keys)) == 2184
    assert sorted(set(k[2] for k in keys)) == [f"2024-06-{d}" for d in range(26, 31)] + ["2024-07-01", "2024-07-02"]
    labels = {"pred_ensemble": "RF + CatBoost (50/50)", "pred_catboost": "CatBoost",
              "pred_rf": "Random forest", "pred_knn": "kNN (k=25)",
              "pred_baseline": "Baseline (7-day mean)", "pred_same_weekday": "Same weekday last week"}
    summaries = {r[""]: r for r in read_csv(results / "test_summary.csv")}
    scores = {}
    actual = [float(r["target_sales"]) for r in rows]
    for col, label in labels.items():
        score = metrics(actual, [float(r[col]) for r in rows])
        for metric, value in score.items():
            close(value, float(summaries[label][metric]), f"test {label} {metric}")
        scores[label] = score
    for r in rows:
        close(float(r["pred_ensemble"]), (float(r["pred_rf"]) + float(r["pred_catboost"])) / 2, "test blend")

    oof = read_csv(results / "oof_predictions.csv")
    assert len(oof) == 10920
    groups = {fold: [r for r in oof if int(r["fold"]) == fold] for fold in range(1, 6)}
    assert all(len(group) == 2184 for group in groups.values())
    validation = {}
    for name in ("rf", "catboost", "knn", "ensemble"):
        fold_scores = []
        for fold, group in groups.items():
            predicted = [(float(r["rf"]) + float(r["catboost"])) / 2 if name == "ensemble" else float(r[name]) for r in group]
            fold_scores.append(metrics([float(r["actual"]) for r in group], predicted)["mae"])
        validation[name] = {"mae": fmean(fold_scores), "fold_mae": fold_scores}
    chosen = [("rf", "rf_tuning_folds.csv", {"max_depth": "10", "min_leaf": "3"}),
              ("knn", "knn_stage1_folds.csv", {"weighting": "equal", "k": "25"}),
              ("catboost", "catboost_round1b_folds.csv", {"depth": "5", "learning_rate": "0.03", "iterations": "500"})]
    for name, filename, parameters in chosen:
        chosen_rows = [r for r in read_csv(results / filename) if all(r[k] == v for k, v in parameters.items())]
        assert len(chosen_rows) == 5
        for row in chosen_rows:
            close(validation[name]["fold_mae"][int(row["fold"]) - 1], float(row["mae"]), f"{name} OOF vs tuning")
    sweep = []
    for row in read_csv(results / "ensemble_weight_sweep.csv"):
        w = float(row["rf_weight"])
        folds = [metrics([float(r["actual"]) for r in group],
                         [w * float(r["rf"]) + (1 - w) * float(r["catboost"]) for r in group])["mae"] for group in groups.values()]
        close(fmean(folds), float(row["mae"]), f"weight {w}")
        for i, val in enumerate(folds, 1):
            if f"fold{i}" in row:
                close(val, float(row[f"fold{i}"]), f"weight {w} fold {i}")
        sweep.append({"rf_weight": w, "mae": fmean(folds)})
    # Recompute every saved tuning summary from its five fold scores.
    grids = [("knn_stage1", ["weighting", "k"]), ("knn_stage1_with_weather", ["weighting", "k"]),
             ("knn_stage1_three_weightings", ["weighting", "k"]),
             ("rf_tuning", ["max_depth", "min_leaf"]),
             ("rf_tuning_leaf_features", ["features", "min_leaf", "max_features"]),
             ("catboost_round1", ["depth", "learning_rate", "iterations"]),
             ("catboost_round1b", ["depth", "learning_rate", "iterations"])]
    checked = 0
    for prefix, columns in grids:
        grouped = defaultdict(list)
        for row in read_csv(results / f"{prefix}_folds.csv"):
            grouped[tuple(row[c] for c in columns)].append(row)
        for row in read_csv(results / f"{prefix}_summary.csv"):
            fold_rows = grouped[tuple(row[c] for c in columns)]
            assert len(fold_rows) == 5
            for metric in ("mae", "mse", "wape"):
                close(float(row[metric]), fmean(float(r[metric]) for r in fold_rows), f"{prefix} {metric}")
            checked += 1
    output = {"status": "PASS", "test_rows": len(rows), "oof_rows": len(oof),
              "tuning_summary_rows_checked": checked, "test_scores": scores,
              "validation": validation, "weight_sweep_minimum": min(sweep, key=lambda r: r["mae"]),
              "ensemble_beats_rf_folds": sum(a < b for a, b in zip(validation["ensemble"]["fold_mae"], validation["rf"]["fold_mae"])),
              "ensemble_beats_catboost_folds": sum(a < b for a, b in zip(validation["ensemble"]["fold_mae"], validation["catboost"]["fold_mae"]))}
    if original:
        old = read_csv(original / "outputs/final_protocol/test_predictions.csv")
        old = {(r["store_id"], r["product_id"], r["dt"]): r for r in old}
        assert set(old) == set(keys)
        for key, row in zip(keys, rows):
            for current, previous in [("target_sales", "sale_amount"), ("discount_next", "discount"),
                                      ("pred_baseline", "7-day average baseline"), ("pred_same_weekday", "Same-weekday baseline")]:
                close(float(row[current]), float(old[key][previous]), f"main comparison {key} {current}")
        output["original_comparison"] = "PASS: identical test keys, targets, discounts and both baseline predictions. This does NOT establish identical model features."
    return output


def audit_features(root, original=None):
    import sys
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import pandas as pd
    from finalproject_pricingml import config as C, data, scenario
    _, _, selected = data.select_series()
    feat = data.build_features(selected)
    train, test = feat[feat.split == "train"], feat[feat.split == "test"]
    saved = pd.read_csv(root / "results/test_predictions.csv", parse_dates=["dt"])
    rebuilt = test.merge(saved, on=C.ROW_KEY, suffixes=("", "_saved"), validate="one_to_one")
    assert len(rebuilt) == len(test) == len(saved)
    assert np.allclose(rebuilt.target_sales, rebuilt.target_sales_saved, atol=1e-12)
    assert np.allclose(rebuilt.sales_mean7, rebuilt.pred_baseline, atol=1e-12)
    saved_oof = pd.read_csv(root / "results/oof_predictions.csv")
    expected_oof = pd.concat([train[train.val_fold == f] for f in range(1, 6)], ignore_index=True)
    assert np.array_equal(saved_oof.fold.values, expected_oof.val_fold.values)
    assert np.allclose(saved_oof.actual.values, expected_oof.target_sales.values, atol=1e-12)
    support = scenario.supported_candidates(train)
    # Constant predictions isolate support/rule coverage; they are never trained on,
    # saved as empirical results, or interpreted as a sales-effect estimate.
    response = pd.concat([test[C.ROW_KEY].assign(discount=d, pred_sales=1.0, pred_value=d)
                          for d in scenario.CANDIDATES], ignore_index=True)
    supported = scenario.restrict(response, support)
    recs = {}
    for name, method in [("supported_max_value", lambda: scenario.recommend(supported)),
                         ("supported_max_sales", lambda: scenario.recommend(supported, "max_sales_within_value", .95)),
                         ("guarded_max_value", lambda: scenario.guarded_recommend(supported, test)),
                         ("guarded_max_sales", lambda: scenario.guarded_recommend(supported, test, "max_sales_within_value", min_value_share=.95))]:
        rec = method()
        counts = test[C.ROW_KEY].merge(rec[C.ROW_KEY], on=C.ROW_KEY, how="left", indicator=True)
        recs[name] = {"returned_rows": len(rec), "missing_rows": int((counts._merge == "left_only").sum())}
    full_unsupported = support[~support[1.0]].reset_index()[C.KEY]
    nofull_test = test.merge(full_unsupported, on=C.KEY)
    state = scenario.stock_state(test)
    output = {"series": len(selected), "train_rows": len(train), "test_rows": len(test),
              "oof_actuals_match_rebuilt_order": True,
              "full_price_unsupported_series": full_unsupported.to_dict("records"),
              "full_price_unsupported_test_rows": len(nofull_test),
              "unsupported_full_price_and_soldout_test_rows": int((nofull_test.stockout_hours_t >= 1).sum()),
              "test_states": state.value_counts().to_dict(), "coverage_constant_prediction_probe": recs}
    if original:
        old = pd.read_parquet(original / "outputs/final_protocol/feature_frame.parquet")
        old["dt"] = pd.to_datetime(old.dt)
        joined = feat.merge(old, on=C.ROW_KEY, suffixes=("_teammate", "_main"), validate="one_to_one")
        output["main_feature_columns"] = old.columns.tolist()
        output["main_common_feature_rows"] = len(joined)
        if "activity_flag" in joined:
            output["lagged_vs_target_activity_mismatch_rows"] = int((joined.activity_t != joined.activity_flag).sum())
            output["test_lagged_vs_target_activity_mismatch_rows"] = int(((joined.activity_t != joined.activity_flag) & (joined.split == "test")).sum())
        if "series_mean" in joined:
            output["expanding_mean_max_abs_delta"] = float((joined.sales_mean_to_date - joined.series_mean).abs().max())
        rename = {"sales_t": "sales_lag1", "sales_lag7_teammate": "sales_lag7_main",
                  "sales_mean7_teammate": "sales_mean7_main", "sales_mean_to_date": "series_mean",
                  "stockout_hours_t": "stockout_lag1", "discount_t": "discount_lag1",
                  "discount_next": "discount", "holiday_next": "holiday_flag", "weekday_next": "weekday"}
        output["other_nine_feature_max_abs_delta"] = {left: float((joined[left] - joined[right]).abs().max())
                                                     for left, right in rename.items()}
    return output


def audit_v2(root):
    """Check V2 without importing its scoring, selection or decision implementation."""
    directory = root / "results/v2"
    selection = json.loads((directory / "selection.json").read_text())
    test = read_csv(directory / "test_predictions.csv")
    val = read_csv(directory / "validation_predictions.csv")
    comp = read_csv(directory / "comparison.csv")
    key = lambda row: (row["store_id"], row["product_id"], row["dt"])
    assert len(test) == len(set(map(key, test))) == 2184
    assert len(val) == len(set(map(key, val))) == 10920
    assert set(map(key, test)) == set(map(key, read_csv(root / "results/test_predictions.csv")))
    for row in comp:
        name = row["model"]
        got = metrics([float(x["target_sales"]) for x in test], [float(x[name]) for x in test])
        for metric in ("mae", "rmse", "wape"):
            close(got[metric], float(row[f"test_{metric}"]), f"V2 {name} {metric}")
        per_fold = [fmean(abs(float(x[name]) - float(x["target_sales"])) for x in val if x["val_fold"] == str(f))
                    for f in range(1, 6)]
        close(fmean(per_fold), float(row["validation_mae"]), f"V2 validation {name}")
    assert min(comp, key=lambda r: float(r["validation_mae"]))["model"] == selection["selected_model"]
    for name, weights in selection["blend_weights"].items():
        for row in test + val:
            close(float(row[name]), sum(w * float(row[m]) for m, w in weights.items()), f"V2 blend {name}")
    recommendations = read_csv(directory / "recommendations.csv")
    assert len(recommendations) == 2184 and set(map(key, recommendations)) == set(map(key, test))
    curves = defaultdict(list)
    for row in read_csv(directory / "discount_response.csv"):
        curves[key(row)].append(row)
    checked = 0
    for row in recommendations:
        if row["status"] != "recommend_discount":
            continue
        curve = curves[key(row)]
        base = next(x for x in curve if float(x["rate"]) == 1)
        chosen = next(x for x in curve if float(x["rate"]) == float(row["recommended_rate"]))
        assert chosen["supported"] == "True" and int(chosen["support_days"]) >= 3
        for col in ("selected_sales", "rf_sales", "cat_sales"):
            assert float(chosen[col]) / float(base[col]) - 1 >= .05 - 1e-12
            assert float(chosen["rate"]) * float(chosen[col]) / float(base[col]) >= .95 - 1e-12
        feasible = [x for x in curve if float(x["rate"]) < 1 and x["supported"] == "True"
                    and all(float(x[c]) / float(base[c]) - 1 >= .05
                            and float(x["rate"]) * float(x[c]) / float(base[c]) >= .95
                            for c in ("selected_sales", "rf_sales", "cat_sales"))]
        best = max(float(x["selected_sales"]) for x in feasible)
        mildest = max(float(x["rate"]) for x in feasible if float(x["selected_sales"]) >= best * .99)
        close(float(row["recommended_rate"]), mildest, "V2 mildest near-best candidate")
        checked += 1
    winner = next(row for row in comp if row["model"] == selection["selected_model"])
    return {"status": "PASS", "models_checked": len(comp), "blend_columns_checked": len(selection["blend_weights"]),
            "recommended_discount_thresholds_and_mildness_checked": checked,
            "output_rows": len(recommendations), "winner": winner,
            "independence": "Standard-library calculations; no calls to V2 scores(), choose_action() or verify_saved()."}


def audit_v2_future_perturbation(root):
    """Change only temporary future raw rows; earlier forecast inputs must not change."""
    import sys
    import tempfile
    sys.path.insert(0, str(root / "src"))
    import pandas as pd
    from finalproject_pricingml import config as C, v2
    features = pd.read_parquet(root / "results/v2/features.parquet")
    first = features.sort_values(C.ROW_KEY).iloc[0]
    base = features[(features.store_id == first.store_id) & (features.product_id == first.product_id)].copy()
    base = base.drop(columns=[c for c in v2.EXTRA + v2.IDS if c != "stockout_days7"])
    cutoff = pd.Timestamp("2024-06-29")
    with tempfile.TemporaryDirectory(prefix="freshretail-future-perturb-") as temporary:
        temporary = Path(temporary)
        original, changed = temporary / "original", temporary / "changed"
        original.mkdir(); changed.mkdir()
        mutated_rows = 0
        for split in ("train", "eval"):
            raw = pd.read_parquet(C.RAW / f"{split}.parquet", columns=C.ROW_KEY + ["sale_amount", "discount"])
            raw = raw[(raw.store_id == first.store_id) & (raw.product_id == first.product_id)].copy()
            raw.to_parquet(original / f"{split}.parquet", index=False)
            future = pd.to_datetime(raw.dt) >= cutoff
            mutated_rows += int(future.sum())
            raw.loc[future, "sale_amount"] = raw.loc[future, "sale_amount"] * 1000 + 123
            raw.loc[future, "discount"] = .15
            raw.to_parquet(changed / f"{split}.parquet", index=False)
        before = v2.enrich_features(base, original)
        after = v2.enrich_features(base, changed)
        # The target day's own raw sales are also forbidden for its history features.
        past = before.dt <= cutoff
        pd.testing.assert_frame_equal(before.loc[past, v2.EXTRA + v2.IDS], after.loc[past, v2.EXTRA + v2.IDS])
        later = before.dt > cutoff
        assert (before.loc[later, "sales_mean3"] != after.loc[later, "sales_mean3"]).any(), "Vacuous perturbation"
    return {"status": "PASS", "store_id": int(first.store_id), "product_id": int(first.product_id),
            "changed_raw_rows_in_temporary_fixture": mutated_rows, "cutoff": str(cutoff.date()),
            "feature_rows_unchanged_through_cutoff": int(past.sum()), "later_features_changed_as_expected": True,
            "actual_project_raw_files_modified": False,
            "scope": "V2 new-history feature builder only; base features and static IDs held fixed. Uses temporary unit-test fixtures, not an experimental training dataset."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--v2", action="store_true", help="Also independently verify the separate V2 output")
    parser.add_argument("--v2-feature-time", action="store_true", help="Test new-history lookahead isolation using temporary raw copies")
    args = parser.parse_args()
    result = audit_saved(args.root, args.original_root)
    if args.features:
        result["feature_and_rule_audit"] = audit_features(args.root, args.original_root)
    if args.v2:
        result["v2_audit"] = audit_v2(args.root)
    if args.v2_feature_time:
        result["v2_feature_time_audit"] = audit_v2_future_perturbation(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
