"""Model/configuration labels for reports, preserving frozen experiment IDs on disk.

The adapter changes text on copies only. It never changes forecasts, settings,
weights, metrics or the saved artifacts used by the reproducibility checks.
"""
from __future__ import annotations


MODEL_NAMES = {
    "Same weekday": "Same-weekday baseline",
    "7-day mean": "7-day mean baseline",
    "Ridge": "Ridge",
    "Decision tree": "Decision Tree",
    "kNN reference": "kNN",
    "Teammate RF": "Random Forest",
    "HistGradientBoosting": "Histogram Gradient Boosting",
    "Monotonic HistGBR": "Histogram Gradient Boosting (monotonic)",
    "Teammate CatBoost": "CatBoost (base, RMSE)",
    "Main CatBoost": "CatBoost (base, RMSE, depth 4)",
    "CatBoost MAE": "CatBoost (base, MAE)",
    "CatBoost history + IDs": "CatBoost (history + IDs, RMSE)",
    "CatBoost history + IDs MAE": "CatBoost (history + IDs, MAE)",
    "Teammate blend 50/50": "RF + CatBoost (base, 50/50)",
    "RF + validation best (25/75)": "RF + CatBoost (history + IDs, MAE; 25/75)",
    "RF + validation best (50/50)": "RF + CatBoost (history + IDs, MAE; 50/50)",
    "RF + validation best (75/25)": "RF + CatBoost (history + IDs, MAE; 75/25)",
}
FIELD_NAMES = {"selected_minus_teammate_mae": "selected_minus_base_blend_mae"}


def label(value):
    if isinstance(value, str):
        return MODEL_NAMES.get(value, FIELD_NAMES.get(value, value))
    return value


def structure(value):
    """Rename model IDs in a copied plan, selection or diagnostic object."""
    if isinstance(value, dict):
        return {label(key): structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(structure(item) for item in value)
    return label(value)


def frame(table):
    """Rename model columns, row labels and textual values, leaving numbers intact."""
    result = table.rename(columns=label, index=label).copy()
    for column in result.select_dtypes(include=["object", "str", "category"]).columns:
        result[column] = result[column].map(label)
    return result


def study_order(table):
    """Present baselines, single models, feature/loss variants, then ensembles."""
    order = {name: index for index, name in enumerate(MODEL_NAMES.values())}
    return table.sort_values("model", key=lambda values: values.map(order).fillna(len(order)),
                             kind="stable").reset_index(drop=True)
