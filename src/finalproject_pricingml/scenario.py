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


def discount_response(model, rows, candidates=CANDIDATES, features=C.FEATURES, cost=0.0):
    """Predicted sales and sales value for every row under every candidate discount.

    Returns a long DataFrame: one row per (store, product, day, candidate) with columns
    ``discount``, ``pred_sales`` and ``pred_value`` (= pred_sales * (discount - cost)).
    ``cost`` is the unit cost as a share of the full price; 0 makes value a pure sales-value
    proxy, 0.6 makes it a margin proxy in which a 0.6 discount earns nothing per unit.
    """
    frames = []
    for d in candidates:
        what_if = rows[features].copy()
        what_if["discount_next"] = d
        frames.append(rows[C.ROW_KEY].assign(discount=d, pred_sales=model.predict(what_if)))
    return reprice(pd.concat(frames, ignore_index=True), cost)


def reprice(response, cost=0.0):
    """Recompute ``pred_value`` for a different unit cost without re-predicting."""
    return response.assign(pred_value=response.pred_sales * (response.discount - cost))


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


# --- Stock guardrails (phase 2, second draft) ------------------------------------------------
# The what-if above does not know how the product's stock is doing. Three facts known on the
# evening of day t say a lot about whether a discount makes sense for a perishable:
#   * sold out today: the shelf was cleared, so tomorrow's stock is fresh and today's recorded
#     sales understate demand (the model's inputs are censored); do not discount;
#   * a discount is already running and did not clear the shelf: continuing (or deepening) is a
#     reasonable outcome, and the model, which sees today's discount, decides;
#   * no stockout for a week: with a shelf life of about a week, stock that has not been cleared
#     in seven days is a clearance case; a discount is required and the model picks how deep.

STATES = ["sold out today", "no stockout for a week", "stock turning over"]


def stock_state(rows, sold_out_hours=1, shelf_life_col="stockout_days7"):
    """Classify each store-product-day's stock situation on the evening of day t.

    ``sold_out_hours``: stockout hours on day t at or above which the day counts as sold out.
    ``stockout_days7``: days with at least one stockout hour among the 7 days up to day t;
    zero means nothing has cleared the shelf for a week.
    """
    sold_out = rows.stockout_hours_t >= sold_out_hours
    stale = rows[shelf_life_col] == 0
    return pd.Series(np.select([sold_out, stale], STATES[:2], STATES[2]), index=rows.index, name="state")


def guarded_candidates(response, rows, sold_out_hours=1):
    """Apply the stock rules to a (supported) what-if table: drop the candidates each rule forbids.

    Adds a ``state`` column. Sold-out rows keep only the no-discount candidate; week-without-
    stockout rows lose the no-discount candidate whenever a discounted one is available.
    """
    states = rows[C.ROW_KEY].assign(state=stock_state(rows, sold_out_hours).values)
    r = response.merge(states, on=C.ROW_KEY, how="inner")
    has_discounted = r[r.discount < 1].groupby(C.ROW_KEY).size().reindex(pd.MultiIndex.from_frame(r[C.ROW_KEY])).fillna(0).values > 0
    forbid = ((r.state == STATES[0]) & (r.discount < 1)) | ((r.state == STATES[1]) & (r.discount == 1) & has_discounted)
    return r[~forbid].reset_index(drop=True)


def guarded_recommend(response, rows, rule="max_value", sold_out_hours=1, min_gain=0.0, **kw):
    """Stock rules first, then the decision rule, then an optional minimum-gain hurdle.

    ``min_gain``: a discount is only recommended when its predicted value beats the no-discount
    value by at least this share (0.05 = 5%); otherwise the row falls back to no discount. The
    hurdle does not apply where a rule has already removed the no-discount candidate.
    Returns one row per store-product-day with ``decided_by`` saying which layer fixed the answer.
    """
    allowed = guarded_candidates(response, rows, sold_out_hours)
    rec = recommend(allowed, rule, **kw)
    rec["decided_by"] = np.select([rec.state == STATES[0], rec.state == STATES[1]],
                                  ["rule: sold out, no discount", "rule: week without stockout, discount required"], "model")
    if min_gain > 0:
        base = allowed[allowed.discount == 1].set_index(C.ROW_KEY)
        key = pd.MultiIndex.from_frame(rec[C.ROW_KEY])
        base_value = base.pred_value.reindex(key).values
        weak = (rec.discount < 1) & (rec.pred_value < base_value * (1 + min_gain))
        rec.loc[weak, "decided_by"] = "hurdle: gain too small, no discount"
        rec.loc[weak, ["discount", "pred_sales", "pred_value"]] = base[["discount", "pred_sales", "pred_value"]].reindex(key[weak]).values
    return rec


def versus_today(rec, rows):
    """How the recommendation relates to today's discount: stop, lighten, continue, deepen or none.

    Only meaningful for rows where a discount is running today (``discount_t`` below the
    discounted-day threshold).
    """
    today = rows.set_index(C.ROW_KEY).discount_t.reindex(pd.MultiIndex.from_frame(rec[C.ROW_KEY])).values
    d = rec.discount.values
    running = today < C.DISCOUNT_DAY
    out = np.select([~running, d >= 0.95, d > today + 0.025, d < today - 0.025],
                    ["no discount running", "stop", "lighten", "deepen"], "continue")
    return pd.Series(out, index=rec.index, name="versus_today")
