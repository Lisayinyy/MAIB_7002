"""Metrics, time-ordered cross-validation and the test-week scoring shared by every model.

Every model is tuned on the same five weekly folds (train on all rows dated before the
fold, predict the fold's week) and scored once on the unseen test week. Model-specific
code only needs to supply a ``make_model(**params)`` factory that returns a fitted-able
scikit-learn style estimator.
"""

import pandas as pd
from joblib import Parallel, delayed

from . import config as C


def metrics(actual, predicted):
    """MAE (main criterion), WAPE (error as a share of sales) and MSE."""
    error = actual - predicted
    return {"mae": error.abs().mean(), "wape": error.abs().sum() / actual.sum(), "mse": (error**2).mean()}


def training_rows(data):
    return data[data.split == "train"]


def test_rows(data):
    return data[data.split == "test"].copy()


def validation_folds(data):
    return sorted(f for f in data.val_fold.unique() if f > 0)


def fold_split(data, fold):
    """Rows to fit on (everything dated before the fold) and the fold's week."""
    val = data[data.val_fold == fold]
    train = data[data.dt < val.dt.min()]
    return train, val


def baseline_by_fold(data, column="sales_mean7"):
    """Error of a no-model prediction (default: the mean of the last 7 days) on each fold."""
    return pd.DataFrame([
        {"fold": f, **metrics(data[data.val_fold == f][C.TARGET], data[data.val_fold == f][column])}
        for f in validation_folds(data)
    ]).set_index("fold")


def score_fold(data, fold, make_model, params, features=C.FEATURES, describe=None):
    """Fit one candidate on everything before the fold, predict the fold, return its metrics."""
    train, val = fold_split(data, fold)
    model = make_model(**params).fit(train[features], train[C.TARGET])
    row = {**params, "fold": fold, **metrics(val[C.TARGET], model.predict(val[features]))}
    if describe:
        row.update(describe(model))
    return row


def cross_validate(data, make_model, grid, features=C.FEATURES, n_jobs=-1, describe=None):
    """Score every parameter set in ``grid`` on every validation fold.

    ``grid`` is a list of keyword dicts passed to ``make_model``. Use ``n_jobs=1`` for
    models that already parallelise internally (forests, boosting). ``describe(model)``
    may return extra columns to record per fit, such as the tree depth reached.
    """
    data = training_rows(data)
    jobs = [delayed(score_fold)(data, f, make_model, p, features, describe)
            for p in grid for f in validation_folds(data)]
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(jobs))


def summarise(results, keys, baseline, extra=None):
    """Average each candidate over the folds and count the folds in which it beats the baseline.

    ``extra`` maps new column names to ``(column, aggregation)`` pairs, e.g.
    ``{"deepest_tree": ("actual_depth", "max")}``.
    """
    groups = results.groupby(keys, sort=False)
    summary = groups[["mae", "wape", "mse"]].mean()
    summary["mae_fold_std"] = groups.mae.std()
    for name, (column, how) in (extra or {}).items():
        summary[name] = groups[column].agg(how)
    summary["folds_beating_baseline"] = (
        results.set_index(keys + ["fold"]).mae.unstack().lt(baseline.mae, axis=1).sum(axis=1)
    )
    return summary


def fit_predict_test(data, models, features=C.FEATURES):
    """Fit each model on all training rows and add a prediction column per model to the test week.

    ``models`` maps a column suffix to an unfitted estimator: ``{"knn": make_knn(25)}`` adds
    ``pred_knn``. The two no-model references, ``pred_baseline`` (7-day mean) and
    ``pred_same_weekday`` (sales 7 days before), are always added.
    """
    train, test = training_rows(data), test_rows(data)
    fitted = {}
    for name, model in models.items():
        fitted[name] = model.fit(train[features], train[C.TARGET])
        test[f"pred_{name}"] = fitted[name].predict(test[features])
    test["pred_baseline"] = test.sales_mean7
    test["pred_same_weekday"] = test.sales_lag7
    test["discount_band"] = discount_band(test.discount_next)
    return test, fitted


def discount_band(discount):
    return pd.cut(discount, C.DISCOUNT_BAND_EDGES, labels=C.DISCOUNT_BAND_LABELS)


def score_methods(test, methods):
    """Overall test-week metrics, one row per method. ``methods`` maps a label to a prediction column."""
    overall = pd.DataFrame({name: metrics(test[C.TARGET], test[col]) for name, col in methods.items()}).T
    overall["rmse"] = overall.mse**0.5
    return overall


def mae_by(test, methods, group):
    """MAE of each method within each level of ``group`` (a Series aligned with ``test``)."""
    return pd.DataFrame({name: (test[C.TARGET] - test[col]).abs().groupby(group, observed=True).mean()
                         for name, col in methods.items()}).assign(rows=test.groupby(group, observed=True).size())


def mean_by(test, methods, group):
    """Actual and predicted mean sales within each level of ``group``."""
    return test.groupby(group, observed=True)[[C.TARGET, *methods.values()]].mean()
