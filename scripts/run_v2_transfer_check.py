#!/usr/bin/env python3
"""Locked transfer stress check: no tuning and no change to the selected V2 model.

Choose 100 series in cities other than city 0 from TRAIN ONLY by stable SHA-256.
Save and hash the plan before loading their eval rows. Fit only three already locked
models on the original 312-series training cohort, then evaluate the new series.
This is not asserted to be globally untouched data across the project's history.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from finalproject_pricingml import config as C, data, v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/v2_transfer")
    parser.add_argument("--force", action="store_true", help="Explicitly replace this stress-check output only")
    args = parser.parse_args()
    out = args.out
    if (out / "run_receipt.json").exists() and not args.force:
        raise SystemExit("Transfer receipt already exists; use --force only for an intentional reproduction.")
    out.mkdir(parents=True, exist_ok=True)
    previous_plan = None
    if (out / "experiment_plan.json").exists():
        previous_plan = json.loads((out / "experiment_plan.json").read_text())
        if not (out / "first_experiment_plan.json").exists():
            (out / "first_experiment_plan.json").write_text((out / "experiment_plan.json").read_text())
    started = time.monotonic()
    original_plan = json.loads((v2.OUT / "experiment_plan.json").read_text())
    selection = json.loads((v2.OUT / "selection.json").read_text())
    assert selection["plan_sha256"] == v2.digest(v2.OUT / "experiment_plan.json")
    assert v2.digest(C.RAW / "train.parquet") == original_plan["raw_sha256"]["train"]

    # This selection call reads train.parquet only. Never redraw using eval outcomes.
    series, _, _ = data.select_series()
    eligible = series.reset_index().query("usable and city_id != 0").copy()
    salt = "freshretail-v2-transfer-20261010-v1"
    eligible["selection_sha256"] = [hashlib.sha256(f"{salt}|{int(s)}|{int(p)}".encode()).hexdigest()
                                      for s, p in zip(eligible.store_id, eligible.product_id)]
    selected = eligible.sort_values("selection_sha256").head(100)
    assert len(selected) == 100
    selected.to_csv(out / "selected_series.csv", index=False)
    if previous_plan:
        assert previous_plan["selected_keys_sha256"] == v2.digest(out / "selected_series.csv"), "Do not change the already-scored cohort"
    needed = {"Teammate RF", "Teammate CatBoost"}
    winner = selection["selected_model"]
    if winner in selection["blend_weights"]:
        needed.update(selection["blend_weights"][winner])
    else:
        needed.add(winner)
    specs = [s for s in original_plan["candidates"] if s["name"] in needed]
    assert len(specs) == 3, "This locked check expects exactly the three selected estimators."
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "run_kind": "Fixed reproduction of the already-scored transfer check" if previous_plan else "Initial locked transfer check",
            "first_plan_sha256": v2.digest(out / "first_experiment_plan.json") if previous_plan else None,
            "purpose": "Cross-city/store transfer stress check; no retuning, no winner change",
            "status_of_data": "Not claimed globally untouched: public data and prior project history may already expose these series.",
            "selection_basis": "Train-only teammate usable criteria, city_id != 0, smallest 100 stable SHA-256 keys",
            "selection_salt": salt, "eligible_series": len(eligible), "selected_series": len(selected),
            "selected_keys_sha256": v2.digest(out / "selected_series.csv"),
            "original_plan_sha256": v2.digest(v2.OUT / "experiment_plan.json"),
            "original_selection_sha256": v2.digest(v2.OUT / "selection.json"),
            "v2_source_sha256_at_import": v2.digest(v2.__file__), "script_sha256": v2.digest(__file__),
            "raw_expected_sha256": original_plan["raw_sha256"],
            "fit_scope": "Original 312 series only, dt < 2024-06-26; zero new-series targets used to fit models",
            "prediction_mode": "Rolling one-day-ahead: observed earlier days from each new series may update history features",
            "target_dates": ["2024-06-26", "2024-07-02"], "selected_model": winner,
            "locked_specs": specs, "locked_blend_weights": selection["blend_weights"][winner],
            "outcome_rule": "Report scores and coverage irrespective of direction; no tuning or model-selection update"}
    v2.dump(out / "experiment_plan.json", plan)
    plan_hash = v2.digest(out / "experiment_plan.json")
    (out / "experiment_plan.sha256").write_text(plan_hash + "\n")
    print(f"Frozen {len(selected)} non-city-0 series; plan SHA256 {plan_hash}", flush=True)

    # Original features are an existing V2 artifact; fit ONLY original training rows.
    original = pd.read_parquet(v2.OUT / "features.parquet")
    assert v2.digest(v2.OUT / "features.parquet") == original_plan["feature_sha256"]
    train = original[original.dt < C.TEST_START].copy()
    assert len(train) == 25896 and train[C.KEY].drop_duplicates().shape[0] == 312
    original_keys = set(map(tuple, train[C.KEY].drop_duplicates().values))
    assert not original_keys.intersection(set(map(tuple, selected[C.KEY].values)))
    fitted = {}
    for spec in specs:
        with threadpool_limits(limits=4):
            fitted[spec["name"]] = (v2.estimator(spec).fit(train[v2.columns(spec)], train[C.TARGET]), spec)

    # First load of selected-series eval outcomes takes place after plan freezing.
    assert plan_hash == v2.digest(out / "experiment_plan.json")
    assert v2.digest(C.RAW / "eval.parquet") == plan["raw_expected_sha256"]["eval"]
    feat = v2.enrich_features(data.build_features(selected[C.KEY]))
    test = feat[feat.dt >= C.TEST_START].sort_values(C.ROW_KEY).copy()
    assert test.dt.min() >= pd.Timestamp("2024-06-26") and test.dt.max() <= pd.Timestamp("2024-07-02")
    pred = test[C.ROW_KEY + [C.TARGET, "discount_next", "target_stockout_hours"]].copy()
    pred["7-day mean"] = test.sales_mean7
    pred["Same weekday"] = test.sales_lag7
    for name, (model, spec) in fitted.items():
        with threadpool_limits(limits=4):
            pred[name] = v2.predict(model, test, spec)
    pred["Teammate blend 50/50"] = .5 * (pred["Teammate RF"] + pred["Teammate CatBoost"])
    pred[winner] = sum(w * pred[name] for name, w in plan["locked_blend_weights"].items())
    names = ["7-day mean", "Same weekday", "Teammate blend 50/50", winner]
    rows = []
    for name in names:
        # Independently calculate scores instead of calling the experiment's score function.
        y, p = pred[C.TARGET].to_numpy(), pred[name].to_numpy()
        assert np.isfinite(y).all() and np.isfinite(p).all()
        error = y - p
        rows.append({"model": name, "mae": float(np.mean(np.abs(error))),
                     "rmse": float(np.sqrt(np.mean(error ** 2))),
                     "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(y))), "rows": len(test)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pred.to_csv(out / "predictions.csv", index=False)
    states = selected[C.KEY + ["city_id"]]
    grouped = pred.merge(states, on=C.KEY, validate="many_to_one")
    city_scores = [{"city_id": int(city), "model": name,
                    "mae": float(np.mean(np.abs(group[C.TARGET] - group[name]))), "rows": len(group)}
                   for city, group in grouped.groupby("city_id") for name in names]
    pd.DataFrame(city_scores).to_csv(out / "by_city.csv", index=False)
    record = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "duration_seconds": time.monotonic() - started, "plan_sha256": plan_hash,
              "prediction_sha256": v2.digest(out / "predictions.csv"), "fitted_estimators": len(fitted),
              "training_rows": len(train), "selected_series": len(selected),
              "evaluation_series": test[C.KEY].drop_duplicates().shape[0], "evaluation_rows": len(test),
              "maximum_expected_rows": 700, "excluded_feature_rows": 700 - len(test),
              "new_cities": int(selected.city_id.nunique()), "new_stores": int(selected.store_id.nunique()),
              "unseen_product_ids": len(set(selected.product_id) - set(train.product_id)),
              "winner_changed": False, "tuning_performed": False,
              "interpretation": "Transfer stress check with train-only deterministic cohort selection. Sales forecasting only; no policy or causal uplift evaluation."}
    v2.dump(out / "run_receipt.json", record)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
