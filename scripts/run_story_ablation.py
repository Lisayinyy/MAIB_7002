#!/usr/bin/env python3
"""Controlled validation-only CatBoost explanations; never select a new winner.

Seven predeclared configurations x five expanding weekly folds = 35 fits.
The canonical feature file is filtered before loading: no final-week targets are
used by this script. Existing validation periods and the retrospective cohort
have already been used for development, so this is explanatory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from finalproject_pricingml import config as C, v2

DEFAULT_OUT = ROOT / "results/story_ablation"
FIXED = dict(depth=5, iterations=500, learning_rate=.03, l2_leaf_reg=5,
             random_seed=0, verbose=False, allow_writing_files=False, thread_count=4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def designs():
    base = list(C.FEATURES)
    history = base + list(v2.EXTRA)
    full = history + list(v2.IDS)
    return [
        dict(name="Base 10 / RMSE", columns=base, loss="RMSE"),
        dict(name="History 18 / RMSE", columns=history, loss="RMSE"),
        dict(name="Full 20 / RMSE", columns=full, loss="RMSE"),
        dict(name="Full 20 / MAE", columns=full, loss="MAE"),
        dict(name="No proposed discount / MAE", columns=[c for c in full if c != "discount_next"], loss="MAE"),
        dict(name="No previous activity / MAE", columns=[c for c in full if c != "activity_t"], loss="MAE"),
        dict(name="No proposed discount or previous activity / MAE",
             columns=[c for c in full if c not in ["discount_next", "activity_t"]], loss="MAE"),
    ]


def contrasts():
    return [
        dict(name="Add eight history features", reference="Base 10 / RMSE", changed="History 18 / RMSE",
             hypothesis="Recent level, variation and trend may improve forecasts beyond the four original sales summaries."),
        dict(name="Add store and product IDs", reference="History 18 / RMSE", changed="Full 20 / RMSE",
             hypothesis="Recurring store/product differences may improve forecasts within this existing-store cohort."),
        dict(name="Change RMSE loss to MAE", reference="Full 20 / RMSE", changed="Full 20 / MAE",
             hypothesis="Training toward a conditional median may reduce validation MAE, potentially at the expense of RMSE."),
        dict(name="Remove proposed target-day discount", reference="Full 20 / MAE", changed="No proposed discount / MAE",
             hypothesis="The proposed price contains predictive information beyond history, IDs and previous-day activity."),
        dict(name="Remove previous-day activity", reference="Full 20 / MAE", changed="No previous activity / MAE",
             hypothesis="Previous-day activity may add information beyond recent sales and discounts; redundancy is possible."),
        dict(name="Remove both action/context fields", reference="Full 20 / MAE",
             changed="No proposed discount or previous activity / MAE",
             hypothesis="Removing both fields tests their combined conditional contribution; correlated history may remain informative."),
    ]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    assert len(y) == len(p) and len(y) and np.isfinite(y).all() and np.isfinite(p).all()
    e = y - p
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt(np.mean(e**2))),
                wape=float(np.abs(e).sum() / np.abs(y).sum()), n=len(y))


def aggregate(pred, variants):
    names = [x["name"] for x in variants]
    folds = pd.DataFrame([dict(variant=name, fold=int(fold), **metrics(g[C.TARGET], g[name]))
                         for name in names for fold, g in pred.groupby("val_fold", sort=True)])
    rows = []
    for spec in variants:
        name = spec["name"]
        f = folds[folds.variant == name]
        pooled = metrics(pred[C.TARGET], pred[name])
        rows.append(dict(variant=name, n_features=len(spec["columns"]), loss=spec["loss"],
                         validation_mae=float(f.mae.mean()), validation_rmse=float(f.rmse.mean()),
                         validation_wape=float(f.wape.mean()), fold_mae_std=float(f.mae.std()),
                         pooled_rmse=pooled["rmse"], rows=len(pred), folds=len(f)))
    summary = pd.DataFrame(rows)
    paired = []
    for contrast in contrasts():
        reference = folds[folds.variant == contrast["reference"]].set_index("fold")
        changed = folds[folds.variant == contrast["changed"]].set_index("fold")
        for fold in reference.index:
            paired.append(dict(contrast=contrast["name"], fold=int(fold),
                reference=contrast["reference"], changed=contrast["changed"],
                reference_mae=float(reference.loc[fold, "mae"]), changed_mae=float(changed.loc[fold, "mae"]),
                mae_delta=float(changed.loc[fold, "mae"] - reference.loc[fold, "mae"]),
                rmse_delta=float(changed.loc[fold, "rmse"] - reference.loc[fold, "rmse"])))
    paired = pd.DataFrame(paired)
    effects = []
    for contrast in contrasts():
        g = paired[paired.contrast == contrast["name"]]
        effects.append(dict(contrast=contrast["name"], reference=contrast["reference"], changed=contrast["changed"],
            reference_mae=float(g.reference_mae.mean()), changed_mae=float(g.changed_mae.mean()),
            mae_delta=float(g.mae_delta.mean()), relative_mae_change_pct=float(100*g.mae_delta.mean()/g.reference_mae.mean()),
            rmse_delta=float(g.rmse_delta.mean()), weeks_changed_lower_mae=int((g.mae_delta < 0).sum()), folds=len(g)))
    return folds, summary, paired, pd.DataFrame(effects)


def protected_inputs():
    return {
        "canonical_features": v2.OUT / "features.parquet",
        "canonical_plan": v2.OUT / "experiment_plan.json",
        "canonical_selection": v2.OUT / "selection.json",
        "canonical_validation_predictions": v2.OUT / "validation_predictions.csv",
        "v2_source": Path(v2.__file__), "config_source": Path(C.__file__),
    }


def verify(out):
    plan = json.loads((out / "experiment_plan.json").read_text())
    receipt = json.loads((out / "run_receipt.json").read_text())
    assert receipt["plan_sha256"] == sha(out / "experiment_plan.json")
    assert (out / "experiment_plan.sha256").read_text().strip() == receipt["plan_sha256"]
    assert plan["script_sha256"] == sha(__file__), "Script changed since these results were produced"
    assert plan["inputs_sha256"] == {k: sha(p) for k, p in protected_inputs().items()}
    for filename, expected in receipt["outputs_sha256"].items():
        assert sha(out / filename) == expected, filename
    assert plan["variants"] == designs() and plan["contrasts"] == contrasts()
    pred = pd.read_csv(out / "validation_predictions.csv", parse_dates=["dt"])
    assert len(pred) == 10920 and not pred.duplicated(C.ROW_KEY).any()
    assert pred.dt.max() < pd.Timestamp(plan["excluded_final_period_start"])
    for fold, (start, end) in plan["folds"].items():
        check = pred[pred.val_fold == int(fold)]
        assert len(check) == 2184 and check.dt.between(start, end).all()
    expected = aggregate(pred, plan["variants"])
    for filename, frame in zip(["fold_metrics.csv", "summary.csv", "fold_contrasts.csv", "contrasts.csv"], expected):
        saved = pd.read_csv(out / filename)
        pd.testing.assert_frame_equal(saved, frame, check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)
    timings = pd.read_csv(out / "timing.csv")
    assert len(timings) == 35 and receipt["fitted_estimators"] == 35
    assert (pd.to_datetime(timings.training_max_date) < pd.to_datetime(timings.validation_min_date)).all()
    assert datetime.fromisoformat(plan["created_utc"]) <= datetime.fromisoformat(receipt["first_fit_started_utc"])
    canonical = pd.read_csv(protected_inputs()["canonical_validation_predictions"], parse_dates=["dt"])
    canonical = canonical.set_index(C.ROW_KEY).sort_index()
    keyed = pred.set_index(C.ROW_KEY).sort_index()
    assert keyed.index.equals(canonical.index)
    assert np.allclose(keyed[C.TARGET], canonical[C.TARGET], rtol=0, atol=1e-12)
    return dict(status="passed", variants=7, folds=5, fitted_estimators=35, validation_rows=len(pred),
                final_period_scored=False, selected_model_changed=False, plan_sha256=receipt["plan_sha256"])


def run(out, force=False):
    if out.resolve() == v2.OUT.resolve() or v2.OUT.resolve() in out.resolve().parents:
        raise ValueError("Explanatory analysis must not overwrite canonical results")
    if (out / "run_receipt.json").exists() and not force:
        raise SystemExit("Results already exist. Use --verify-only or --force for an intentional replay.")
    out.mkdir(parents=True, exist_ok=True)
    previous = None
    if (out / "experiment_plan.json").exists():
        previous = sha(out / "experiment_plan.json")
        if not (out / "first_experiment_plan.json").exists():
            (out / "first_experiment_plan.json").write_bytes((out / "experiment_plan.json").read_bytes())
    inputs = {k: sha(p) for k, p in protected_inputs().items()}
    original = json.loads(protected_inputs()["canonical_plan"].read_text())
    selected = json.loads(protected_inputs()["canonical_selection"].read_text())
    assert original["feature_sha256"] == inputs["canonical_features"]
    assert selected["plan_sha256"] == inputs["canonical_plan"]
    plan = dict(created_utc=utc(), purpose="Explain fixed technical choices using previously used validation periods; no new selection",
        previous_plan_sha256=previous, script_sha256=sha(__file__), inputs_sha256=inputs,
        fixed_parameters=FIXED, variants=designs(), contrasts=contrasts(), folds=C.VAL_FOLDS,
        expected_fits=35, excluded_final_period_start=C.TEST_START,
        fit_rule="Expanding training rows strictly before each validation week; no eval_set, early stopping or parameter search",
        prediction_rule="Rolling one-day-ahead observed histories, one fit per fold, shared nonnegative prediction clipping",
        metric_rule="Equal-week mean MAE primary; equal-week mean RMSE secondary; paired fold deltas changed minus reference",
        cohort_status=original["cohort_status"], exposure_status=original["test_status"],
        outcome_rule="Report every declared contrast, including negative results; do not update canonical selection or score final-week targets",
        causal_boundary="Conditional predictive contribution, not randomized discount/activity effects, elasticity, profit or waste outcomes",
        ablation_boundary="Removing discount_next leaves historical discounts and sales; removing activity_t tests yesterday's activity only",
        canonical_selected_model=selected["selected_model"], canonical_weights=selected["blend_weights"][selected["selected_model"]],
        environment={"python": platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "catboost", "scikit-learn", "pyarrow"]}})
    write_json(out / "experiment_plan.json", plan)
    plan_hash = sha(out / "experiment_plan.json")
    (out / "experiment_plan.sha256").write_text(plan_hash + "\n")
    print(f"Declared seven variants and six contrasts before fitting; plan {plan_hash}", flush=True)
    # Parquet filtering prevents final-period target rows from entering this analysis.
    feat = pd.read_parquet(protected_inputs()["canonical_features"], filters=[("dt", "<", pd.Timestamp(C.TEST_START))])
    assert len(feat) == 25896 and feat.split.eq("train").all() and not feat.duplicated(C.ROW_KEY).any()
    assert feat.dt.max() < pd.Timestamp(C.TEST_START) and len(feat[C.KEY].drop_duplicates()) == 312
    val = feat[feat.val_fold > 0].sort_values(C.ROW_KEY).copy()
    pred = val[C.ROW_KEY + ["val_fold", C.TARGET]].copy().set_index(C.ROW_KEY)
    pred["7-day mean"] = val.set_index(C.ROW_KEY).sales_mean7
    pred["Same weekday"] = val.set_index(C.ROW_KEY).sales_lag7
    timings = []
    first_fit = utc()
    started = time.monotonic()
    for variant in plan["variants"]:
        pieces = []
        for fold, (start, end) in C.VAL_FOLDS.items():
            train = feat[feat.dt < start]
            check = feat[feat.dt.between(start, end)]
            assert train.dt.max() < check.dt.min() and len(check) == 2184
            columns = variant["columns"]
            cats = [c for c in v2.IDS if c in columns]
            model = CatBoostRegressor(**FIXED, loss_function=variant["loss"], cat_features=cats)
            tick = time.monotonic()
            with threadpool_limits(limits=4):
                model.fit(train[columns], train[C.TARGET])
                prediction = np.maximum(0, model.predict(check[columns]))
            timings.append(dict(variant=variant["name"], fold=fold, training_rows=len(train), validation_rows=len(check),
                training_max_date=str(train.dt.max().date()), validation_min_date=str(check.dt.min().date()),
                seconds=time.monotonic()-tick))
            pieces.append(check[C.ROW_KEY].assign(prediction=prediction))
        combined = pd.concat(pieces).set_index(C.ROW_KEY).prediction
        assert combined.index.is_unique and combined.index.difference(pred.index).empty
        pred[variant["name"]] = combined.reindex(pred.index)
        mean_mae = np.abs(pred[variant["name"]]-pred[C.TARGET]).groupby(pred.val_fold).mean().mean()
        print(f"{variant['name']}: validation MAE {mean_mae:.8f}", flush=True)
    pred = pred.reset_index()
    assert np.isfinite(pred[[s["name"] for s in plan["variants"]]].to_numpy()).all()
    pred.to_csv(out / "validation_predictions.csv", index=False)
    for filename, frame in zip(["fold_metrics.csv", "summary.csv", "fold_contrasts.csv", "contrasts.csv"], aggregate(pred, plan["variants"])):
        frame.to_csv(out / filename, index=False)
    pd.DataFrame(timings).to_csv(out / "timing.csv", index=False)
    baseline = [dict(model=name, fold=int(fold), **metrics(g[C.TARGET],g[name]))
        for name in ["7-day mean","Same weekday"] for fold,g in pred.groupby("val_fold")]
    pd.DataFrame(baseline).to_csv(out / "baseline_fold_metrics.csv", index=False)
    canonical = pd.read_csv(protected_inputs()["canonical_validation_predictions"], parse_dates=["dt"]).set_index(C.ROW_KEY)
    keyed = pred.set_index(C.ROW_KEY)
    replay = []
    for name, old in [("Full 20 / RMSE","CatBoost history + IDs"),("Full 20 / MAE","CatBoost history + IDs MAE")]:
        previous_prediction = canonical[old].reindex(keyed.index)
        assert not previous_prediction.isna().any()
        replay.append(dict(variant=name, canonical_model=old,
            maximum_absolute_prediction_difference=float(np.abs(keyed[name]-previous_prediction).max()),
            pooled_mae_difference=float(np.abs(keyed[C.TARGET]-keyed[name]).mean()-np.abs(keyed[C.TARGET]-previous_prediction).mean()),
            equal_within_1e_10=bool(np.allclose(keyed[name],previous_prediction,rtol=0,atol=1e-10))))
    write_json(out / "canonical_replay_check.json", dict(note="Replay diagnostic; small CPU/platform numerical differences are possible.", checks=replay))
    assert inputs == {k: sha(p) for k,p in protected_inputs().items()}, "Canonical source/results changed during run"
    assert plan_hash == sha(out / "experiment_plan.json")
    files = ["validation_predictions.csv","fold_metrics.csv","summary.csv","fold_contrasts.csv","contrasts.csv",
             "timing.csv","baseline_fold_metrics.csv","canonical_replay_check.json"]
    write_json(out / "run_receipt.json", dict(status="complete", completed_utc=utc(), first_fit_started_utc=first_fit,
        duration_seconds=time.monotonic()-started, fitted_estimators=len(timings), validation_rows=len(pred),
        plan_sha256=plan_hash, outputs_sha256={name:sha(out/name) for name in files},
        final_period_scored=False, selected_model_changed=False, input_hashes_unchanged=True))
    verification = verify(out)
    write_json(out / "verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="Repeat only this declared explanation experiment")
    parser.add_argument("--verify-only", action="store_true", help="Recompute saved metrics and check source/input/output hashes; zero fits")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.out), indent=2))
    else:
        run(args.out, args.force)
