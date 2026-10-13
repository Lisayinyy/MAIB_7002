"""Evidence plots for the existing forecasting and discount-scenario pipeline.

The figures reuse the current experiment's predictions, dates and policy tables;
they do not train models or select a new policy from the final evaluation period.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from . import config as C
from . import scenario
from .decision_visuals import GREEN, LIGHT_GREEN, AMBER, GREY, INK, MUTED, GRID, STYLE


def response_and_support(response, support):
    """Plot scenario ratios and the historical evidence behind each candidate.

    ``response`` is the full long table from ``scenario.discount_response``;
    ``support`` is the boolean store-product matrix from
    ``scenario.supported_candidates`` with its default 3-day, ±0.025 rule.
    Every candidate must cover the same complete set of prediction cases. The
    top panel uses ratios of mean predictions, not means of individual ratios.
    Historical support has a separate denominator of store-product histories.

    Returns ``(figure, table)`` without saving or displaying the figure.
    """
    required = C.ROW_KEY + ["discount", "pred_sales", "pred_value"]
    if not set(required).issubset(response.columns):
        raise ValueError("Response is missing case keys, candidate rates or predictions")
    if response.empty or response[required].isna().any().any():
        raise ValueError("Response must be nonempty with no missing case values")
    if response.duplicated(C.ROW_KEY + ["discount"]).any():
        raise ValueError("Response contains duplicate case-candidate pairs")
    if set(response.discount.unique()) != set(scenario.CANDIDATES):
        raise ValueError("Response must include the complete configured candidate grid")
    counts = response.groupby(C.ROW_KEY).size()
    if not counts.eq(len(scenario.CANDIDATES)).all():
        raise ValueError("Every prediction case must have every candidate rate")
    if not np.isfinite(response[["pred_sales", "pred_value"]].to_numpy()).all():
        raise ValueError("Predictions must be finite")
    if not np.allclose(response.pred_value, response.pred_sales * response.discount):
        raise ValueError("The response figure uses the cost=0 sales-value proxy")
    if not isinstance(support.index, pd.MultiIndex) or list(support.index.names) != C.KEY:
        raise ValueError("Support must be indexed by store_id and product_id")
    if not support.index.is_unique or support.empty or not support.columns.is_unique:
        raise ValueError("Support must have unique, nonempty history keys and candidate columns")
    if set(support.columns) != set(scenario.CANDIDATES):
        raise ValueError("Support columns must match the configured candidate grid")
    if support.isna().any().any() or not all(pd.api.types.is_bool_dtype(d) for d in support.dtypes):
        raise ValueError("Historical support must be a nonmissing boolean matrix")
    histories = pd.MultiIndex.from_frame(response[C.KEY].drop_duplicates())
    if len(histories.difference(support.index)) or len(support.index.difference(histories)):
        raise ValueError("Response histories and support histories must match")

    curve = scenario.response_curve(response)
    ratio_columns = ["sales_vs_no_discount", "value_vs_no_discount"]
    if not np.isfinite(curve[ratio_columns].to_numpy()).all():
        raise ValueError("Full-price means must support finite comparison ratios")
    table = curve.rename_axis("discount").reset_index()
    table["percent_off"] = (100 * (1 - table.discount)).round(8)
    table["supported_histories"] = table.discount.map(support.sum()).astype(int)
    table["total_histories"] = len(support)
    table["support_percent"] = 100 * table.supported_histories / len(support)
    table["cases_per_candidate"] = len(counts)
    x = table.percent_off.to_numpy()
    labels = [f"{off:.0f}%\n{rate:.2f}" for off, rate in zip(x, table.discount)]

    with plt.rc_context(STYLE):
        fig, (upper, lower) = plt.subplots(2, 1, figsize=(12.4, 8.0), sharex=True,
                                         gridspec_kw={"height_ratios": [1.45, 1]})
        fig.subplots_adjust(left=.10, right=.975, top=.79, bottom=.27, hspace=.23)
        upper.plot(x, table.sales_vs_no_discount, color=GREEN, marker="o", lw=2.3,
                   label="Predicted sales / full-price mean")
        upper.plot(x, table.value_vs_no_discount, color=AMBER, marker="s", lw=2.3,
                   label="Predicted value proxy / full-price mean")
        upper.axhline(1, color=GREY, linestyle="--", linewidth=1)
        upper.text(x.max(), 1.025, "Full-price reference = 1", ha="right", fontsize=9, color=MUTED)
        upper.set_ylabel("Ratio of mean predictions")
        upper.set_ylim(0, max(1.15, table[ratio_columns].max().max() * 1.15))
        upper.legend(loc="upper left", frameon=False, fontsize=10)
        upper.tick_params(axis="both", length=0)
        upper.grid(axis="y", color=GRID)
        upper.set_axisbelow(True)
        lower.bar(x, table.support_percent, width=3.4, color=LIGHT_GREEN, edgecolor="white")
        for off, percent in zip(x, table.support_percent):
            lower.text(off, percent + 3, f"{percent:.0f}%", ha="center", fontsize=10, color=GREEN,
                       fontweight="bold")
        lower.set(ylim=(0, 115), yticks=[0, 50, 100], ylabel="Histories with support (%)",
                  xticks=x, xticklabels=labels, xlabel="Discount (% off) / price rate")
        lower.tick_params(axis="both", length=0)
        lower.grid(axis="y", color=GRID)
        lower.set_axisbelow(True)
        fig.text(.055, .94, "Predicted response and the evidence behind it", fontsize=19, fontweight="bold")
        fig.text(.055, .873,
                 f"Model what-ifs across {len(counts):,} prediction cases per candidate; historical support across {len(support):,} histories.",
                 fontsize=11, color=MUTED)
        fig.text(.055, .11,
                 "Top: mean predicted sales and value at each candidate, divided by their respective full-price means.\n"
                 "Bottom: share of store-product histories with ≥3 training days within ±0.025 of the candidate rate.",
                 fontsize=10, color=MUTED, linespacing=1.55)
        fig.text(.055, .043,
                 "Historical support limits extrapolation; it does not establish a causal discount effect. Value is predicted sales × price rate, not measured revenue.",
                 fontsize=9.4, color=MUTED)
    return fig, table


def split_timeline(feat):
    """Show actual fitting and evaluation dates for all folds and the final period.

    Uses the configured expanding validation windows and the feature table's
    observed dates. Final fitting uses only pre-test rows. Forecasts during the
    final period are one day ahead; earlier observed test days can supply later
    lag features. Returns ``(figure, table)``.
    """
    if feat.empty or not set(C.ROW_KEY).issubset(feat.columns):
        raise ValueError("Feature data must contain nonempty store-product-date records")
    if feat[C.ROW_KEY].isna().any().any() or feat.duplicated(C.ROW_KEY).any():
        raise ValueError("Feature cases must have unique nonmissing keys")
    rows = feat.copy()
    rows["dt"] = pd.to_datetime(rows.dt)
    test_start = pd.Timestamp(C.TEST_START)
    records = []
    for fold, (start, end) in C.VAL_FOLDS.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        fit = rows[rows.dt < start]
        evaluation = rows[rows.dt.between(start, end)]
        if fit.empty or evaluation.empty or end >= test_start:
            raise ValueError(f"Validation fold {fold} has an empty or invalid time window")
        if fit.dt.max() >= evaluation.dt.min():
            raise ValueError("Validation fitting must precede every evaluation date")
        records.append({"stage": f"Validation {fold}", "train_start": fit.dt.min(),
                        "train_end": fit.dt.max(), "train_rows": len(fit),
                        "evaluation_start": evaluation.dt.min(), "evaluation_end": evaluation.dt.max(),
                        "evaluation_rows": len(evaluation), "evaluation_type": "Validation"})
    fit, evaluation = rows[rows.dt < test_start], rows[rows.dt >= test_start]
    if fit.empty or evaluation.empty or fit.dt.max() >= evaluation.dt.min():
        raise ValueError("Final fitting and evaluation periods must be nonempty and ordered")
    if "split" in rows:
        if not fit.split.eq("train").all() or not evaluation.split.eq("test").all():
            raise ValueError("Saved split labels disagree with the configured test boundary")
    records.append({"stage": "Final evaluation", "train_start": fit.dt.min(), "train_end": fit.dt.max(),
                    "train_rows": len(fit), "evaluation_start": evaluation.dt.min(),
                    "evaluation_end": evaluation.dt.max(), "evaluation_rows": len(evaluation),
                    "evaluation_type": "Final evaluation"})
    table = pd.DataFrame(records)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13.3, 6.9))
        fig.subplots_adjust(left=.16, right=.975, top=.75, bottom=.25)
        for y, row in table.iterrows():
            train_start = mdates.date2num(row.train_start)
            train_width = (row.train_end - row.train_start).days + 1
            eval_start = mdates.date2num(row.evaluation_start)
            eval_width = (row.evaluation_end - row.evaluation_start).days + 1
            eval_color = GREEN if row.evaluation_type == "Validation" else AMBER
            ax.barh(y, train_width, left=train_start, height=.49, color=LIGHT_GREEN)
            ax.barh(y, eval_width, left=eval_start, height=.49, color=eval_color)
            ax.text(train_start + train_width / 2, y, f"{row.train_rows:,} fitting rows",
                    va="center", ha="center", fontsize=10, color=INK)
            ax.text(eval_start + eval_width / 2, y, f"{row.evaluation_rows:,}", va="center", ha="center",
                    fontsize=9.5, color="white" if row.evaluation_type == "Validation" else INK,
                    fontweight="bold")
        boundary = mdates.date2num(test_start)
        ax.axvline(boundary, color=AMBER, linestyle="--", lw=1.2, alpha=.85)
        ax.set_yticks(range(len(table)), table.stage)
        ax.set_ylim(len(table) - .45, -.55)
        ax.set_xlim(mdates.date2num(table.train_start.min()) - 1,
                    mdates.date2num(table.evaluation_end.max()) + 2)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="both", length=0, pad=9)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        fig.text(.05, .94, "Evaluate forward in time", fontsize=19, fontweight="bold")
        fig.text(.05, .875,
                 "Each validation week follows its fitting data; the fitting window expands for the next fold.", color=MUTED)
        fig.legend(handles=[Patch(facecolor=LIGHT_GREEN, label="Model-fitting data"),
                            Patch(facecolor=GREEN, label="Validation week"),
                            Patch(facecolor=AMBER, label="Final evaluation period")],
                   loc="upper left", bbox_to_anchor=(.045, .835), frameon=False, ncol=3, fontsize=10)
        final = table.iloc[-1]
        fig.text(.05, .153,
                 f"Final evaluation: {final.evaluation_start:%d %b}–{final.evaluation_end:%d %b %Y} · "
                 f"{final.evaluation_rows:,} cases · model fitted on {final.train_rows:,} earlier rows.",
                 fontweight="bold", fontsize=11)
        fig.text(.05, .056,
                 "Forecasts are made one day ahead. Earlier observed test-day sales can supply later lag features; the final model is fitted before the test period.\n"
                 "This final period is a shared benchmark already examined during project development, not a new untouched evaluation.",
                 fontsize=9.6, color=MUTED, linespacing=1.65)
    return fig, table


def policy_sensitivity(table):
    """Plot discount recommendation frequency under the existing cost/gain grid.

    Accepts the raw ``policy_summary`` table with columns ``unit cost``,
    ``min gain``, ``sold-out hours`` and ``recommends a discount, %``. The stockout
    threshold is fixed at one hour. An optional ``cases`` column supplies the
    shared denominator. Costs are hypothetical fractions of regular price.
    Returns ``(figure, pivot)``; no cell is marked as an optimal policy.
    """
    value = "recommends a discount, %"
    required = ["unit cost", "min gain", "sold-out hours", value]
    if table.empty or not set(required).issubset(table.columns):
        raise ValueError("Policy table lacks required cost, gain, threshold or result columns")
    if table[required].isna().any().any() or not np.isfinite(table[required].to_numpy()).all():
        raise ValueError("Policy inputs and percentages must be finite and nonmissing")
    if table.duplicated(["unit cost", "min gain"]).any():
        raise ValueError("Each cost/gain combination must appear exactly once")
    if not table["sold-out hours"].eq(1).all():
        raise ValueError("The sensitivity comparison must hold the stockout threshold at one hour")
    if not table[value].between(0, 100).all():
        raise ValueError("Discount recommendation percentages must lie between 0 and 100")
    if not table["unit cost"].between(0, 1).all() or not table["min gain"].between(0, 1).all():
        raise ValueError("Cost fractions and minimum gains must lie between 0 and 1")
    pivot = table.pivot(index="unit cost", columns="min gain", values=value).sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("Sensitivity grid must include every cost/gain combination")
    denominator = ""
    if "cases" in table:
        if table.cases.isna().any() or table.cases.nunique() != 1 or table.cases.iloc[0] <= 0:
            raise ValueError("All sensitivity scenarios must use the same positive case count")
        denominator = f" · {int(table.cases.iloc[0]):,} prediction cases"
    colours = LinearSegmentedColormap.from_list("policy_frequency", ["#F4F8F5", LIGHT_GREEN, GREEN])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.8, 7.0))
        fig.subplots_adjust(left=.18, right=.82, top=.77, bottom=.31)
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=100, cmap=colours, aspect="auto")
        for i, row in enumerate(pivot.to_numpy()):
            for j, percent in enumerate(row):
                ax.text(j, i, f"{percent:.1f}%", ha="center", va="center", fontsize=20,
                        fontweight="bold", color="white" if percent >= 70 else INK)
        ax.set_xticks(range(len(pivot.columns)), [f"{gain:.0%}" for gain in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{cost:.0%}" for cost in pivot.index])
        ax.set_xlabel("Minimum predicted value gain", labelpad=13)
        ax.set_ylabel("Assumed unit cost / regular price", labelpad=13)
        ax.tick_params(axis="both", length=0, pad=10)
        ax.set_xticks(np.arange(-.5, len(pivot.columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(pivot.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=5)
        ax.tick_params(which="minor", length=0)
        colorbar = fig.colorbar(image, ax=ax, fraction=.055, pad=.055, ticks=[0, 25, 50, 75, 100])
        colorbar.outline.set_visible(False)
        colorbar.set_label("Recommend a discount (%)", labelpad=10)
        colorbar.ax.tick_params(length=0)
        fig.text(.065, .94, "How sensitive is the recommendation policy?", fontsize=18, fontweight="bold")
        fig.text(.065, .866, "Stockout threshold held at 1 hour" + denominator, fontsize=11, color=MUTED)
        fig.text(.065, .165,
                 "Cells show the share of cases receiving a discount under each assumption; a higher share is not inherently better.\n"
                 "Unit costs are hypothetical. The value proxy is predicted sales × (price rate − assumed cost), not measured profit.",
                 fontsize=9.8, color=MUTED, linespacing=1.65)
        fig.text(.065, .071,
                 "The minimum-gain hurdle applies only when full price remains eligible.\n"
                 "Stock-rule-mandated discounts can remain even when the hurdle increases.",
                 fontsize=9.8, color=MUTED, linespacing=1.65)
    return fig, pivot
