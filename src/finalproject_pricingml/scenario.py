"""Phase 2 (first draft): use a fitted sales model to compare candidate discounts for tomorrow.

For each store-product-day, every feature known on the evening of day t is held fixed and only
tomorrow's discount is varied. The model's prediction under each candidate is a *what-if* that
rests on correlations in historical data: discounts were not assigned at random, so the response
is not a proven causal effect. The sales-value proxy is predicted sales x discount rate, because
``sale_amount`` is normalised and no unit price is available.
"""

import numpy as np
import pandas as pd

from . import config as C

CANDIDATES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]


def discount_response(model, rows, candidates=CANDIDATES, features=C.FEATURES):
    """Predicted sales and sales value for every row under every candidate discount.

    Returns a long DataFrame: one row per (store, product, day, candidate) with columns
    ``discount``, ``pred_sales`` and ``pred_value`` (= pred_sales * discount).
    """
    frames = []
    for d in candidates:
        what_if = rows[features].copy()
        what_if["discount_next"] = d
        frames.append(rows[C.ROW_KEY].assign(discount=d, pred_sales=model.predict(what_if)))
    out = pd.concat(frames, ignore_index=True)
    out["pred_value"] = out.pred_sales * out.discount
    return out


def recommend(response, rule="max_value", min_value_share=1.0):
    """Pick one discount per store-product-day from a ``discount_response`` table.

    rule="max_value": the candidate with the highest predicted sales value.
    rule="max_sales_within_value": the highest predicted sales among candidates whose predicted
        value is at least ``min_value_share`` of the value at no discount (1.0). This is the
        "sell more without giving away value" rule.
    """
    key = C.ROW_KEY
    if rule == "max_value":
        idx = response.groupby(key).pred_value.idxmax()
        return response.loc[idx].reset_index(drop=True)
    if rule == "max_sales_within_value":
        full = response[response.discount == 1.0].set_index(key).pred_value
        allowed = response[response.pred_value >= response.set_index(key).index.map(full).values * min_value_share]
        idx = allowed.groupby(key).pred_sales.idxmax()
        return allowed.loc[idx].reset_index(drop=True)
    raise ValueError(rule)


def response_curve(response):
    """Average predicted sales and value by candidate discount, relative to no discount."""
    avg = response.groupby("discount")[["pred_sales", "pred_value"]].mean().sort_index(ascending=False)
    return avg.assign(sales_vs_no_discount=avg.pred_sales / avg.loc[1.0, "pred_sales"],
                      value_vs_no_discount=avg.pred_value / avg.loc[1.0, "pred_value"])


def monotonicity(response):
    """Share of store-product-days whose predicted sales never fall as the discount deepens."""
    wide = response.pivot_table(index=C.ROW_KEY, columns="discount", values="pred_sales")
    wide = wide[sorted(wide.columns, reverse=True)]  # 1.0 first, deepest discount last
    return float((np.diff(wide.values, axis=1) >= -1e-9).all(axis=1).mean())


def supported_candidates(train, candidates=CANDIDATES, min_days=3, width=0.025):
    """Which candidate discounts each store-product has actually experienced in training.

    A candidate is supported when the series had at least ``min_days`` training days with a
    discount within ``width`` of it. Returns a boolean DataFrame indexed by (store, product)
    with one column per candidate. Restricting what-ifs to supported candidates keeps the
    model inside the range of its own data instead of extrapolating to discounts the product
    never had.
    """
    out = {}
    for d in candidates:
        near = (train.discount_next - d).abs() <= width
        out[d] = near.groupby([train.store_id, train.product_id]).sum() >= min_days
    return pd.DataFrame(out)


def restrict(response, support):
    """Drop what-if rows whose candidate discount is not supported for that store-product."""
    long = support.stack().rename("supported").reset_index()
    long.columns = C.KEY + ["discount", "supported"]
    merged = response.merge(long, on=C.KEY + ["discount"], how="left")
    return merged[merged.supported.fillna(False)].drop(columns="supported").reset_index(drop=True)
