"""Lisa's auditable extension: forecast first, then screen and compare discounts.

The teammate package is kept unchanged. This module adds a bounded, recorded
experiment and a decision layer that can abstain. No policy effects are observed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from . import config as C, data, models

OUT = C.ROOT / "results" / "v2"
EXTRA = ["sales_lag2", "sales_lag3", "sales_mean3", "sales_mean14", "sales_std7",
         "sales_trend", "discount_mean7", "stockout_days7"]
IDS = ["store_category", "product_category"]
RAW_HASHES = {"train": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
              "eval": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def candidate_specs():
    """Predeclared alternatives; no automatic search or test-driven second round."""
    return [
        {"name": "Ridge", "kind": "ridge", "features": "base", "alpha": 10.0},
        {"name": "Decision tree", "kind": "tree", "features": "base", "depth": 6, "leaf": 20},
        {"name": "HistGradientBoosting", "kind": "hist", "features": "base", "monotonic": False},
        {"name": "kNN reference", "kind": "knn", "features": "base"},
        {"name": "Teammate RF", "kind": "rf", "features": "base"},
        {"name": "Teammate CatBoost", "kind": "cat", "features": "base", "depth": 5,
         "iterations": 500, "lr": .03, "loss": "RMSE", "l2": 3},
        {"name": "Main CatBoost", "kind": "cat", "features": "base", "depth": 4,
         "iterations": 400, "lr": .05, "loss": "RMSE", "l2": 5},
        {"name": "CatBoost MAE", "kind": "cat", "features": "base", "depth": 5,
         "iterations": 500, "lr": .03, "loss": "MAE", "l2": 3},
        {"name": "CatBoost history + IDs", "kind": "cat", "features": "history_ids", "depth": 5,
         "iterations": 500, "lr": .03, "loss": "RMSE", "l2": 5},
        {"name": "CatBoost history + IDs MAE", "kind": "cat", "features": "history_ids", "depth": 5,
         "iterations": 500, "lr": .03, "loss": "MAE", "l2": 5},
        {"name": "Monotonic HistGBR", "kind": "hist", "features": "base", "monotonic": True},
    ]


def columns(spec):
    return C.FEATURES + EXTRA + IDS if spec["features"] == "history_ids" else C.FEATURES


def estimator(spec):
    kind = spec["kind"]
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=spec["alpha"]))
    if kind == "tree":
        return DecisionTreeRegressor(max_depth=spec["depth"], min_samples_leaf=spec["leaf"], random_state=0)
    if kind == "hist":
        return HistGradientBoostingRegressor(learning_rate=.05, max_iter=300, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1., early_stopping=False, random_state=0,
            monotonic_cst={"discount_next": -1} if spec["monotonic"] else None)
    if kind == "knn":
        return models.make_knn(25, n_jobs=4)
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                     max_features=.5, random_state=0, n_jobs=4)
    if kind == "cat":
        return CatBoostRegressor(depth=spec["depth"], iterations=spec["iterations"],
            learning_rate=spec["lr"], loss_function=spec["loss"], l2_leaf_reg=spec["l2"],
            random_seed=0, verbose=False, allow_writing_files=False, thread_count=4,
            cat_features=IDS if spec["features"] == "history_ids" else [])
    raise ValueError(kind)


def enrich_features(feat, raw_dir=None):
    """All new histories are shifted BEFORE rolling; preserve the exact benchmark rows."""
    raw_dir = Path(raw_dir or C.RAW)
    keys = feat[C.KEY].drop_duplicates()
    raw = pd.concat([pd.read_parquet(raw_dir / f"{s}.parquet",
        columns=C.ROW_KEY + ["sale_amount", "discount"]) for s in ("train", "eval")])
    raw = raw.merge(keys, on=C.KEY, validate="many_to_one")
    raw["dt"] = pd.to_datetime(raw.dt)
    raw = raw.sort_values(C.ROW_KEY).reset_index(drop=True)
    invalid = ~raw.discount.between(0, 1, inclusive="right")
    # Preserve invalid raw values; exclude them from new histories, never reinterpret as a price.
    raw.loc[invalid, ["sale_amount", "discount"]] = np.nan
    g = raw.groupby(C.KEY, sort=False)
    extra = raw[C.ROW_KEY].copy()
    extra["sales_lag2"] = g.sale_amount.shift(2)
    extra["sales_lag3"] = g.sale_amount.shift(3)
    extra["sales_mean3"] = g.sale_amount.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    extra["sales_mean14"] = g.sale_amount.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    extra["sales_std7"] = g.sale_amount.transform(lambda x: x.shift(1).rolling(7, min_periods=2).std())
    extra["discount_mean7"] = g.discount.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    out = feat.merge(extra, on=C.ROW_KEY, how="left", validate="one_to_one")
    for col in ["sales_lag2", "sales_lag3", "sales_mean3", "sales_mean14"]:
        out[col] = out[col].fillna(out.sales_mean_to_date)
    out["sales_std7"] = out.sales_std7.fillna(0)
    out["discount_mean7"] = out.discount_mean7.fillna(out.discount_t)
    out["sales_trend"] = out.sales_mean3 - out.sales_mean7
    out["store_category"] = out.store_id.astype(str)
    out["product_category"] = out.product_id.astype(str)
    assert len(out) == len(feat) and not out.duplicated(C.ROW_KEY).any()
    assert out[C.FEATURES + EXTRA + IDS].notna().all().all()
    return out


def scores(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(a) != len(p) or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite or misaligned predictions")
    e = a - p
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.mean(e**2))),
            "wape": float(np.abs(e).sum() / np.abs(a).sum()), "n": len(a)}


def predict(model, frame, spec):
    # Shared physical postprocessing, predeclared for every estimator.
    return np.maximum(0, model.predict(frame[columns(spec)]))


def prepare(out=OUT):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw = {s: digest(C.RAW / f"{s}.parquet") for s in RAW_HASHES}
    if raw != RAW_HASHES:
        raise ValueError("Raw data do not match the pinned original source")
    feat = enrich_features(data.load_features(rebuild=True))
    feat.to_parquet(out / "features.parquet", index=False)
    # This file is saved before any candidate fit and before final-period scoring.
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "teammate_sha": "5b52099ab73e57c5afbb1915d0bc4a0adbde9032",
        "raw_sha256": raw, "feature_sha256": digest(out / "features.parquet"),
        "code_sha256": digest(__file__), "candidates": candidate_specs(),
        "folds": C.VAL_FOLDS, "test_start": C.TEST_START,
        "selection": "Minimum equal-week validation MAE; exact ties use declaration order.",
        "blend_rule": "Teammate RF/CatBoost 50/50 plus RF at 25/50/75% with best standalone validation candidate other than RF.",
        "test_status": "Already-viewed shared benchmark. All new work is exploratory; no untouched final evaluation remains.",
        "cohort_status": "Same retrospective 312-series cohort selected using all original training dates; no city-wide generalization claim.",
        "prediction_mode": "Rolling one day ahead, observed previous days update histories; fit once before each weekly fold.",
        "decision_parameters": {"min_support_days": 3, "support_width": .025, "min_sales_gain": .05,
             "min_value_share": .95, "near_optimal_sales_tolerance": .01,
             "stockout_hours_review_threshold": 1, "agreement": "selected model, teammate RF and teammate CatBoost"},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
             for p in ["numpy", "pandas", "scikit-learn", "catboost", "pyarrow"]}}}
    dump(out / "experiment_plan.json", plan)
    return feat, plan


def validation(feat, plan, out=OUT):
    out = Path(out)
    tr = feat[feat.dt < C.TEST_START].copy()
    val = tr[tr.val_fold > 0].sort_values(C.ROW_KEY).copy()
    oof = val[C.ROW_KEY + ["val_fold", C.TARGET]].copy().set_index(C.ROW_KEY)
    timing = []
    for name, col in [("7-day mean", "sales_mean7"), ("Same weekday", "sales_lag7")]:
        oof[name] = val.set_index(C.ROW_KEY)[col]
    for spec in plan["candidates"]:
        pieces = []; started = time.monotonic()
        for fold, (start, end) in C.VAL_FOLDS.items():
            fit = tr[tr.dt < start]; check = tr[tr.dt.between(start, end)]
            assert fit.dt.max() < check.dt.min() and len(check) == 2184
            with threadpool_limits(limits=4):
                model = estimator(spec).fit(fit[columns(spec)], fit[C.TARGET])
                pred = predict(model, check, spec)
            pieces.append(check[C.ROW_KEY].assign(pred=pred))
        keyed = pd.concat(pieces).set_index(C.ROW_KEY).pred
        assert keyed.index.is_unique and oof.index.difference(keyed.index).empty
        oof[spec["name"]] = keyed.reindex(oof.index)
        timing.append({"model": spec["name"], "seconds_five_folds": time.monotonic() - started})
        score = np.abs(oof[spec["name"]] - oof[C.TARGET]).groupby(oof.val_fold).mean().mean()
        print(f"Validation {spec['name']}: {score:.6f} ({timing[-1]['seconds_five_folds']:.1f}s)", flush=True)
        oof.reset_index().to_csv(out / "validation_predictions.csv", index=False)
    single = {s["name"]: np.abs(oof[s["name"]] - oof[C.TARGET]).groupby(oof.val_fold).mean().mean()
              for s in plan["candidates"] if s["name"] != "Teammate RF"}
    best_single = min(single, key=single.get)
    blends = {"Teammate blend 50/50": {"Teammate RF": .5, "Teammate CatBoost": .5}}
    for weight in [.25, .5, .75]:
        blends[f"RF + validation best ({int(weight*100)}/{int((1-weight)*100)})"] = {
            "Teammate RF": weight, best_single: 1-weight}
    for name, weights in blends.items():
        oof[name] = sum(oof[model] * weight for model, weight in weights.items())
    fold_scores = []
    names = [c for c in oof.columns if c not in ["val_fold", C.TARGET]]
    for name in names:
        for fold, frame in oof.groupby("val_fold"):
            fold_scores.append({"model": name, "fold": fold, **scores(frame[C.TARGET], frame[name])})
    folds = pd.DataFrame(fold_scores)
    table = folds.groupby("model", sort=False).agg(validation_mae=("mae", "mean"),
        validation_rmse=("rmse", "mean"), fold_mae_std=("mae", "std")).reset_index()
    winner = table.loc[table.validation_mae.idxmin(), "model"]
    selection = {"selected_model": winner, "best_standalone": best_single, "blend_weights": blends,
         "validation_mae": float(table.set_index("model").loc[winner, "validation_mae"]),
         "selected_before_test_scoring_utc": datetime.now(timezone.utc).isoformat(),
         "plan_sha256": digest(out / "experiment_plan.json"),
         "interpretation": "Exploratory validation selection; repeated historical development is disclosed, not fresh holdout evidence."}
    oof.reset_index().to_csv(out / "validation_predictions.csv", index=False)
    folds.to_csv(out / "validation_folds.csv", index=False)
    table.to_csv(out / "validation_summary.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "timing.csv", index=False)
    dump(out / "selection.json", selection)
    return selection


def final_period(feat, plan, selection, out=OUT):
    out = Path(out)
    train = feat[feat.dt < C.TEST_START]; test = feat[feat.dt >= C.TEST_START].copy()
    pred = test[C.ROW_KEY + [C.TARGET, "discount_next", "target_stockout_hours"]].copy()
    pred["7-day mean"] = test.sales_mean7; pred["Same weekday"] = test.sales_lag7
    fitted = {}
    for spec in plan["candidates"]:
        with threadpool_limits(limits=4):
            model = estimator(spec).fit(train[columns(spec)], train[C.TARGET])
            pred[spec["name"]] = predict(model, test, spec)
        fitted[spec["name"]] = (model, spec)
    for name, weights in selection["blend_weights"].items():
        pred[name] = sum(pred[model] * weight for model, weight in weights.items())
    names = ["7-day mean", "Same weekday"] + [s["name"] for s in plan["candidates"]] + list(selection["blend_weights"])
    metrics = pd.DataFrame([{"model": name, **scores(test[C.TARGET], pred[name])} for name in names])
    comparison = pd.read_csv(out / "validation_summary.csv").merge(metrics.rename(columns={
        "mae": "test_mae", "rmse": "test_rmse", "wape": "test_wape"}), on="model", validate="one_to_one")
    comparison["selected"] = comparison.model == selection["selected_model"]
    comparison.to_csv(out / "comparison.csv", index=False)
    pred.to_csv(out / "test_predictions.csv", index=False)
    winner = selection["selected_model"]
    slices = []
    groups = {"date": test.dt.dt.strftime("%Y-%m-%d"), "store": test.store_id,
        "discount": pd.cut(test.discount_next, [0, .8, .95, 1.], labels=["deep", "moderate", "none_or_small"]),
        "target_stockout_diagnostic_only": test.target_stockout_hours.gt(0).map({True:"stockout",False:"no_stockout"})}
    for kind, group in groups.items():
        for value in group.dropna().unique():
            mask = group == value
            for name in ["7-day mean", "Teammate blend 50/50", winner]:
                slices.append({"slice": kind, "value": str(value), "model": name,
                               **scores(test.loc[mask, C.TARGET], pred.loc[mask, name])})
    pd.DataFrame(slices).drop_duplicates().to_csv(out / "error_slices.csv", index=False)
    cases = pred.copy(); cases["selected_abs_error"] = abs(cases[winner] - cases[C.TARGET])
    cases.sort_values("selected_abs_error", ascending=False).head(12).to_csv(out / "largest_errors.csv", index=False)
    # Series-level paired resampling preserves each product's seven consecutive days.
    # It does not remove shared-store shocks or repeated-selection optimism.
    delta = (abs(pred[winner] - pred[C.TARGET]) - abs(pred["Teammate blend 50/50"] - pred[C.TARGET]))
    series_delta = pred[C.KEY].assign(delta=delta).groupby(C.KEY).delta.mean().to_numpy()
    rng = np.random.default_rng(20261010)
    boot = rng.choice(series_delta, size=(2000, len(series_delta)), replace=True).mean(axis=1)
    dump(out / "paired_comparison.json", {"selected_minus_teammate_mae": float(delta.mean()),
       "series_bootstrap_95pct_interval": np.quantile(boot, [.025, .975]).tolist(),
       "limitation": "Descriptive 7-day, 5-store interval; shared-store correlation and selection optimism not corrected."})
    return fitted, test


def predict_selected(fitted, selection, frame):
    name = selection["selected_model"]
    if name in selection["blend_weights"]:
        return sum(weight * predict(fitted[member][0], frame, fitted[member][1])
                   for member, weight in selection["blend_weights"][name].items())
    if name == "7-day mean":
        return frame.sales_mean7.to_numpy()
    if name == "Same weekday":
        return frame.sales_lag7.to_numpy()
    model, spec = fitted[name]
    return predict(model, frame, spec)


def choose_action(curve, stockout_hours, *, min_sales_gain=.05, min_value_share=.95,
                  near_optimal_sales_tolerance=.01, stockout_hours_review_threshold=1):
    """One complete row of decision output; abstention is explicit, never dropped."""
    result = {"status": "abstain_support", "recommended_rate": np.nan,
              "predicted_sales": np.nan, "predicted_gain": np.nan, "value_share": np.nan}
    if stockout_hours >= stockout_hours_review_threshold:
        return {**result, "status": "review_stockout", "reason": "Recent censored sales; verify tomorrow's supply before pricing."}
    if not np.isfinite(curve[["selected_sales", "rf_sales", "cat_sales"]].to_numpy()).all():
        return {**result, "status": "review_invalid_prediction", "reason": "Nonfinite scenario prediction."}
    full = curve[(curve.rate == 1) & curve.supported]
    if full.empty:
        return {**result, "reason": "No historically supported full-price comparator."}
    base = full.iloc[0]
    if min(base.selected_sales, base.rf_sales, base.cat_sales) <= 0:
        return {**result, "status": "review_zero_reference", "reason": "Relative comparisons are undefined at a zero reference."}
    options = curve[(curve.rate < 1) & curve.supported].copy()
    if options.empty:
        return {**result, "reason": "No discounted candidate has at least three nearby historical days."}
    gain = options.selected_sales / base.selected_sales - 1
    value_share = options.rate * options.selected_sales / base.selected_sales
    feasible = (gain >= min_sales_gain) & (value_share >= min_value_share)
    # Cross-model sensitivity check, NOT a statistical confidence bound or causal guarantee.
    agreement = np.ones(len(options), dtype=bool)
    for col in ["rf_sales", "cat_sales"]:
        agreement &= (options[col] / base[col] - 1 >= min_sales_gain).to_numpy()
        agreement &= (options.rate * options[col] / base[col] >= min_value_share).to_numpy()
    options["gain"] = gain; options["value_share"] = value_share
    valid = options[feasible & agreement]
    if valid.empty:
        if feasible.any():
            return {**result, "status": "review_model_disagreement", "reason": "Selected forecast supports a discount; reference models disagree."}
        return {**result, "status": "keep_full_price", "recommended_rate": 1.,
                "predicted_sales": float(base.selected_sales), "predicted_gain": 0., "value_share": 1.,
                "reason": "No supported discount meets the stated gain and value thresholds."}
    best = valid.selected_sales.max()
    # Prefer less discount when modelled sales differ by at most 1% of the best candidate.
    chosen = valid[valid.selected_sales >= best * (1 - near_optimal_sales_tolerance)].sort_values("rate", ascending=False).iloc[0]
    return {"status": "recommend_discount", "recommended_rate": float(chosen.rate),
        "predicted_sales": float(chosen.selected_sales), "predicted_gain": float(chosen.gain),
        "value_share": float(chosen.value_share),
        "reason": "Supported candidate passes model agreement and value floor; mildest near-best sales option."}


def decision_scenarios(feat, fitted, selection, plan, out=OUT):
    out = Path(out)
    train = feat[feat.dt < C.TEST_START]; test = feat[feat.dt >= C.TEST_START].copy()
    rates = [1., .95, .9, .85, .8, .75, .7, .6]
    settings = plan["decision_parameters"]
    frames = []
    for rate in rates:
        frame = test.copy(); frame["discount_next"] = rate
        counts = train.assign(near=(train.discount_next - rate).abs() <= settings["support_width"]).groupby(C.KEY).near.sum()
        current = test[C.ROW_KEY].copy(); current["rate"] = rate
        current["support_days"] = counts.reindex(pd.MultiIndex.from_frame(test[C.KEY])).to_numpy()
        current["supported"] = current.support_days >= settings["min_support_days"]
        current["selected_sales"] = predict_selected(fitted, selection, frame)
        for col, member in [("rf_sales", "Teammate RF"), ("cat_sales", "Teammate CatBoost")]:
            model, spec = fitted[member]; current[col] = predict(model, frame, spec)
        frames.append(current)
    response = pd.concat(frames, ignore_index=True)
    indexed_test = test.set_index(C.ROW_KEY)
    recs = []
    for key, curve in response.groupby(C.ROW_KEY, sort=False):
        r = choose_action(curve, indexed_test.loc[key, "stockout_hours_t"], **{
            k: settings[k] for k in ["min_sales_gain", "min_value_share",
            "near_optimal_sales_tolerance", "stockout_hours_review_threshold"]})
        recs.append(dict(zip(C.ROW_KEY, key)) | r)
    recs = pd.DataFrame(recs)
    assert len(recs) == len(test) and not recs.duplicated(C.ROW_KEY).any()
    decided = recs.status.isin(["recommend_discount", "keep_full_price"])
    response.to_csv(out / "discount_response.csv", index=False)
    recs.to_csv(out / "recommendations.csv", index=False)
    summary = {"rows": len(test), "output_rows": len(recs), "status_counts": recs.status.value_counts().to_dict(),
        "decision_coverage": float(decided.mean()), "manual_review_or_abstain_share": float((~decided).mean()),
        "discount_recommendation_share_all_rows": float(recs.status.eq("recommend_discount").mean()),
        "observed_policy_outcomes": False,
        "interpretation": "Model-based daily scenarios, not verified uplift, profit, waste reduction, inventory age or causal effects.",
        "parameters": plan["decision_parameters"]}
    dump(out / "policy_summary.json", summary)
    return recs


def verify_saved(out=OUT):
    """Recompute every displayed error and verify temporal, keyed and decision invariants."""
    out = Path(out)
    plan = json.loads((out / "experiment_plan.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    assert selection["plan_sha256"] == digest(out / "experiment_plan.json")
    assert plan["code_sha256"] == digest(__file__), "Code changed: rerun the declared experiment or use its original version"
    assert plan["feature_sha256"] == digest(out / "features.parquet")
    pred = pd.read_csv(out / "test_predictions.csv")
    val = pd.read_csv(out / "validation_predictions.csv")
    comp = pd.read_csv(out / "comparison.csv")
    assert len(pred) == 2184 and len(val) == 10920
    assert not pred.duplicated(C.ROW_KEY).any() and not val.duplicated(C.ROW_KEY).any()
    for _, row in comp.iterrows():
        got = scores(pred[C.TARGET], pred[row.model])
        assert abs(got["mae"] - row.test_mae) < 1e-12
        assert abs(got["rmse"] - row.test_rmse) < 1e-12
        cv = (val[row.model] - val[C.TARGET]).abs().groupby(val.val_fold).mean().mean()
        assert abs(cv - row.validation_mae) < 1e-12
    assert comp.loc[comp.validation_mae.idxmin(), "model"] == selection["selected_model"]
    rec = pd.read_csv(out / "recommendations.csv")
    assert len(rec) == len(pred) and not rec.duplicated(C.ROW_KEY).any()
    discount = rec[rec.status == "recommend_discount"]
    settings = plan["decision_parameters"]
    assert discount.value_share.ge(settings["min_value_share"] - 1e-12).all()
    assert discount.predicted_gain.ge(settings["min_sales_gain"] - 1e-12).all()
    response = pd.read_csv(out / "discount_response.csv")
    joined = discount.merge(response, left_on=C.ROW_KEY + ["recommended_rate"], right_on=C.ROW_KEY + ["rate"], validate="one_to_one")
    assert joined.supported.all() and joined.support_days.ge(settings["min_support_days"]).all()
    output = {"status": "passed", "models_checked": len(comp), "oof_rows": len(val), "test_rows": len(pred),
        "recommendation_rows": len(rec), "raw_sha256": plan["raw_sha256"],
        "checked_utc": datetime.now(timezone.utc).isoformat(), "retrained_here": False}
    dump(out / "verification.json", output)
    return output


def run(out=OUT):
    started = time.monotonic()
    feat, plan = prepare(out)
    print(f"Prepared {len(feat):,} rows; starting recorded validation.", flush=True)
    selection = validation(feat, plan, out)
    print("Locked before final scoring:", selection["selected_model"], flush=True)
    fitted, _ = final_period(feat, plan, selection, out)
    decision_scenarios(feat, fitted, selection, plan, out)
    report = verify_saved(out)
    report.update(retrained_here=True, duration_seconds=time.monotonic()-started,
                  fitted_models=6*len(plan["candidates"]))
    dump(Path(out)/"run_receipt.json", report)
    print(json.dumps(report, indent=2), flush=True)
