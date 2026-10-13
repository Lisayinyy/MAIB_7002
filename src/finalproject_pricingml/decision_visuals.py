"""Visual explanations of the existing cohort and discount decision rules.

These functions display the current pipeline's data and recommendations. They do
not fit a model, change a policy, or interpret a what-if prediction as a causal
effect. All counts and plotted values are computed from the supplied frames.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from . import config as C
from . import scenario


GREEN = "#18765B"
LIGHT_GREEN = "#9DCBB6"
AMBER = "#D99821"
GREY = "#ADB5B2"
INK = "#203A33"
MUTED = "#62736D"
GRID = "#E4EAE6"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": GRID,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _keyed(frame, key=C.ROW_KEY):
    """Accept notebook tables with keys in either columns or a named index."""
    out = frame.copy()
    if not set(key).issubset(out.columns):
        out = out.reset_index()
    missing = set(key) - set(out.columns)
    if missing:
        raise ValueError(f"Missing key columns: {sorted(missing)}")
    if out[key].isna().any().any():
        raise ValueError("Case keys must not contain missing values")
    return out


def _unique_index(frame, key=C.ROW_KEY):
    index = pd.MultiIndex.from_frame(frame[key])
    if not index.is_unique:
        raise ValueError(f"Expected one row per {key}")
    return index


def _same_keys(left, right, description):
    if len(left) != len(right) or len(left.difference(right)) or len(right.difference(left)):
        raise ValueError(f"{description}: case coverage differs")


def selection_funnel(series, selected):
    """Show the sequential filters used by ``data.select_series``.

    Parameters are the first and third outputs of ``data.select_series``. Counts
    refer to store-product histories, not daily observations. The selected keys
    are checked against the original top-store algorithm before plotting.

    Returns ``(figure, summary)``; no file is written and no display is triggered.
    """
    series = _keyed(series, C.KEY).set_index(C.KEY)
    selected = _keyed(selected, C.KEY).set_index(C.KEY)
    if not series.index.is_unique or not selected.index.is_unique:
        raise ValueError("Each series table must have unique store-product keys")
    required = ["disc_share", "switches", "zero_sales_share"]
    if series[required].isna().any().any():
        raise ValueError("Selection statistics contain missing values")

    share = series.disc_share.between(C.MIN_SHARE, C.MAX_SHARE)
    switches = share & (series.switches >= C.MIN_SWITCHES)
    usable = switches & (series.zero_sales_share < C.MAX_ZERO_SALES)
    if "usable" in series and not usable.equals(series.usable.astype(bool)):
        raise ValueError("Saved usable flags disagree with current selection thresholds")
    stores = usable.groupby(level="store_id").sum().sort_values(ascending=False)
    top_stores = stores.head(C.TOP_N_STORES).index
    expected = series.index[usable & series.index.get_level_values("store_id").isin(top_stores)]
    _same_keys(selected.index, expected, "Selected cohort")
    if selected.index.get_level_values("store_id").nunique() != C.TOP_N_STORES:
        raise ValueError("Selected cohort does not contain the configured number of stores")

    counts = [len(series), int(share.sum()), int(switches.sum()), int(usable.sum()), len(selected)]
    labels = ["All histories", "Discount variation", "Repeated changes", "Regular sales", "Selected cohort"]
    rules = [
        "One history per\nstore-product pair",
        f"{C.MIN_SHARE:.0%}–{C.MAX_SHARE:.0%} of days\nwith a discount",
        f"At least {C.MIN_SWITCHES} switches\nin discount status",
        f"Zero-sales days\nbelow {C.MAX_ZERO_SALES:.0%}",
        f"Top {C.TOP_N_STORES} stores by\neligible histories",
    ]
    summary = pd.DataFrame({"stage": labels, "histories": counts,
                            "rule": [s.replace("\n", " ") for s in rules]})
    summary["retained_from_previous_pct"] = [100.0] + [
        100 * b / a if a else np.nan for a, b in zip(counts[:-1], counts[1:])
    ]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13.6, 4.7))
        fig.subplots_adjust(left=.035, right=.985, top=.80, bottom=.22)
        ax.set(xlim=(0, 5), ylim=(0, 1))
        ax.axis("off")
        for i, (label, count, rule) in enumerate(zip(labels, counts, rules)):
            last = i == len(counts) - 1
            ax.add_patch(FancyBboxPatch((i + .035, .05), .84, .86,
                         boxstyle="round,pad=0.025,rounding_size=0.04",
                         facecolor=GREEN if last else "#F0F6F2", edgecolor="none"))
            colour = "white" if last else INK
            ax.text(i + .455, .76, label, ha="center", va="center", fontsize=10.8, color=colour)
            ax.text(i + .455, .52, f"{count:,}", ha="center", va="center", fontsize=28,
                    fontweight="bold", color="white" if last else GREEN)
            ax.text(i + .455, .24, rule, ha="center", va="center", fontsize=10,
                    color="white" if last else MUTED, linespacing=1.55)
            if not last:
                ax.annotate("", xy=(i + 1.015, .48), xytext=(i + .91, .48),
                            arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.8})
        fig.text(.04, .93, "How the study cohort was selected", fontsize=19, fontweight="bold")
        fig.text(.04, .85, "Each step retains histories that meet all previous conditions.", color=MUTED)
        store_counts = selected.groupby(level="store_id").size().sort_values(ascending=False)
        detail = "  ·  ".join(f"Store {store}: {n}" for store, n in store_counts.items())
        fig.text(.04, .14, detail, fontsize=10.5)
        fig.text(.04, .055,
                 f"Discounted day: price rate < {C.DISCOUNT_DAY:.2f}. Selection uses the original training period, including validation dates.\n"
                 "Counts are histories; each history contains multiple daily observations.",
                 fontsize=9.5, color=MUTED, linespacing=1.55)
    return fig, summary


def recommendation_distribution(unguarded, guarded):
    """Compare supported max-value recommendations before and after stock rules.

    Both inputs must cover exactly the same unique store-product-date cases.
    Full price is exactly 1.00. Moderate discounts have 0.80 < rate < 1.00;
    deep discounts have rate <= 0.80. These recommendation bands deliberately
    distinguish exact full price from the notebook's broader observed-price band.
    Returns ``(figure, summary)`` with counts and percentages for every band.
    """
    before, after = _keyed(unguarded), _keyed(guarded)
    before_index, after_index = _unique_index(before), _unique_index(after)
    _same_keys(before_index, after_index, "Before/after recommendations")
    if before.empty:
        raise ValueError("Recommendation tables must not be empty")
    labels = ["Supported model only", "With stock rules"]
    bands = ["Full price", "Moderate discount", "Deep discount"]
    records = []
    for label, frame in zip(labels, [before, after]):
        rate = frame.discount
        if rate.isna().any() or not rate.isin(scenario.CANDIDATES).all():
            raise ValueError("Recommendations contain missing or unknown candidate rates")
        masks = [rate.eq(1.0), rate.gt(.8) & rate.lt(1.0), rate.le(.8)]
        if sum(int(mask.sum()) for mask in masks) != len(frame):
            raise ValueError("Recommendation bands do not cover all cases")
        for band, mask in zip(bands, masks):
            records.append({"strategy": label, "band": band, "count": int(mask.sum()),
                            "cases": len(frame), "percent": 100 * mask.mean(),
                            "discount_share_percent": 100 * rate.lt(1.0).mean()})
    summary = pd.DataFrame(records)
    percentages = summary.pivot(index="strategy", columns="band", values="percent").loc[labels, bands]
    shares = [100 * frame.discount.lt(1.0).mean() for frame in [before, after]]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12.8, 5.3))
        fig.subplots_adjust(left=.20, right=.80, top=.69, bottom=.26)
        left = np.zeros(2)
        for band, colour in zip(bands, [GREY, LIGHT_GREEN, GREEN]):
            values = percentages[band].to_numpy()
            ax.barh([1, 0], values, left=left, height=.42, color=colour, edgecolor="white", linewidth=2)
            for y, x, value in zip([1, 0], left, values):
                if value >= 5:
                    ax.text(x + value / 2, y, f"{value:.1f}%", ha="center", va="center",
                            color="white" if band == "Deep discount" else INK, fontweight="bold")
            left += values
        for y, share in zip([1, 0], shares):
            ax.text(103, y + .065, f"{share:.1f}%", fontsize=19, fontweight="bold", color=GREEN, va="center")
            ax.text(103, y - .12, "recommend a discount", fontsize=9.5, color=MUTED, va="center")
        ax.set(xlim=(0, 100), ylim=(-.5, 1.5), yticks=[1, 0], yticklabels=labels,
               xticks=[0, 25, 50, 75, 100], xlabel="Share of the same prediction cases (%)")
        ax.tick_params(axis="y", length=0, pad=10)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.set_axisbelow(True)
        fig.text(.04, .93, "What changes when stock rules are added?", fontsize=19, fontweight="bold")
        fig.text(.04, .85, f"Same model, supported candidates and max-value objective · {len(before):,} store-product-days", color=MUTED)
        legend = [Patch(facecolor=c, label=l) for c, l in zip(
            [GREY, LIGHT_GREEN, GREEN], ["Full price (rate = 1.00)", "Moderate (0.80 < rate < 1.00)", "Deep (rate ≤ 0.80)"])]
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.035, .79), frameon=False, ncol=3, fontsize=10)
        change = shares[1] - shares[0]
        fig.text(.04, .135, f"Discount recommendation share changes by {change:+.1f} percentage points.", fontweight="bold")
        fig.text(.04, .055, "This compares policy recommendations, not realized changes in sales, revenue or food waste.\n"
                 "Both versions first restrict candidates to historically supported price rates.",
                 fontsize=9.5, color=MUTED, linespacing=1.55)
    return fig, summary


def discount_case(response, supported, test, train):
    """Explain the first eligible model-decided case in date/store/product order.

    ``supported`` is the long response table returned by ``scenario.restrict``.
    Case selection uses only stock signals and candidate support: at least two
    supported rates, including full price, and the existing model-decides state.
    It never selects on observed sales, prediction size, lift, or recommendation.
    The actual ``guarded_candidates`` and ``guarded_recommend`` functions supply
    eligibility and the chosen rate. Cost is zero, as in the main illustration.

    Returns ``(figure, candidate_table, case_info)``. The candidate table includes
    training support-day counts, status and the selected option.
    """
    response, supported, test, train = map(_keyed, [response, supported, test, train])
    test_index = _unique_index(test)
    _unique_index(train)
    candidate_key = C.ROW_KEY + ["discount"]
    response_index = _unique_index(response, candidate_key)
    supported_index = _unique_index(supported, candidate_key)
    _same_keys(test_index, pd.MultiIndex.from_frame(response[C.ROW_KEY].drop_duplicates()), "Response/test")
    if len(supported_index.difference(response_index)):
        raise ValueError("Supported candidates must be a subset of the full response")
    if not response.discount.isin(scenario.CANDIDATES).all():
        raise ValueError("Response contains unknown candidate rates")
    if not np.isfinite(response[["pred_sales", "pred_value"]].to_numpy()).all():
        raise ValueError("Response predictions must be finite")
    if not np.allclose(response.pred_value, response.pred_sales * response.discount):
        raise ValueError("This illustration requires the cost=0 sales-value proxy")
    original = response.set_index(candidate_key).loc[supported_index, ["pred_sales", "pred_value"]]
    if not np.allclose(original.to_numpy(), supported[["pred_sales", "pred_value"]].to_numpy()):
        raise ValueError("Supported predictions differ from the full response")

    eligibility = supported.groupby(C.ROW_KEY).discount.agg(
        candidate_count="size", includes_full_price=lambda rates: rates.eq(1.0).any())
    cases = test.merge(eligibility, on=C.ROW_KEY, how="left", validate="one_to_one")
    cases["decision_state"] = scenario.stock_state(cases)
    candidates = cases[(cases.decision_state == scenario.STATES[2]) &
                       (cases.candidate_count >= 2) & cases.includes_full_price.fillna(False)]
    if candidates.empty:
        raise ValueError("No model-decided case has at least two supported rates including full price")
    case_row = candidates.sort_values(["dt"] + C.KEY, kind="stable").iloc[[0]]
    case = case_row.iloc[0]
    mask = np.logical_and.reduce([response[key].eq(case[key]).to_numpy() for key in C.ROW_KEY])
    supported_mask = np.logical_and.reduce([supported[key].eq(case[key]).to_numpy() for key in C.ROW_KEY])
    case_response = response.loc[mask].copy().sort_values("discount", ascending=False)
    case_supported = supported.loc[supported_mask].copy()
    allowed = scenario.guarded_candidates(case_supported, case_row)
    recommendation = scenario.guarded_recommend(case_supported, case_row, "max_value")
    if len(recommendation) != 1 or recommendation.iloc[0].decided_by != "model":
        raise ValueError("Expected exactly one model-decided recommendation")
    chosen_rate = float(recommendation.iloc[0].discount)
    history = train[(train.store_id == case.store_id) & (train.product_id == case.product_id)]
    if history.empty or pd.to_datetime(history.dt).max() >= pd.Timestamp(case["dt"]):
        raise ValueError("Case support must come from earlier training dates")

    table = case_response[["discount", "pred_sales", "pred_value"]].reset_index(drop=True)
    table["percent_off"] = (100 * (1 - table.discount)).round(8)
    table["support_days"] = [int((history.discount_next - rate).abs().le(.025).sum()) for rate in table.discount]
    table["historically_supported"] = table.discount.isin(case_supported.discount)
    if not table.historically_supported.equals(table.support_days.ge(3)):
        raise ValueError("Case support does not match the stated minimum of 3 days within ±0.025")
    table["eligible"] = table.discount.isin(allowed.discount)
    table["status"] = np.select([~table.historically_supported, ~table.eligible],
                                ["Unsupported", "Rule-excluded"], "Eligible")
    table["chosen"] = table.discount.eq(chosen_rate)
    if table.chosen.sum() != 1 or not table.loc[table.chosen, "eligible"].all():
        raise ValueError("Chosen rate is not a unique eligible candidate")

    info = {
        "store_id": case.store_id.item() if isinstance(case.store_id, np.generic) else case.store_id,
        "product_id": case.product_id.item() if isinstance(case.product_id, np.generic) else case.product_id,
        "target_date": pd.Timestamp(case["dt"]).strftime("%Y-%m-%d"),
        "chosen_rate": chosen_rate, "chosen_percent_off": round(100 * (1 - chosen_rate), 8),
        "stockout_hours_t": float(case.stockout_hours_t), "stockout_days7": int(case.stockout_days7),
        "decided_by": "model", "objective": "maximum predicted sales-value proxy",
        "selection": "First date/store/product with model-decides state and at least two supported rates including full price",
        "support_rule": "At least 3 training days within ±0.025 of the candidate rate",
    }
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.4))
        fig.subplots_adjust(left=.075, right=.975, top=.69, bottom=.37, wspace=.28)
        x = table.percent_off.to_numpy()
        for ax, metric, label in zip(axes, ["pred_sales", "pred_value"],
                                     ["Predicted sales (source-normalized)", "Predicted sales × price rate"]):
            ax.plot(x, table[metric], color=GREY, lw=1.5, linestyle="--", zorder=1)
            for status, colour, marker in [("Eligible", GREEN, "o"), ("Unsupported", GREY, "x"),
                                            ("Rule-excluded", GREY, "s")]:
                part = table[table.status == status]
                ax.scatter(part.percent_off, part[metric], color=colour, marker=marker,
                           s=70, linewidths=1.5, zorder=3)
            choice = table[table.chosen].iloc[0]
            ax.scatter([choice.percent_off], [choice[metric]], marker="*", s=300,
                       color=AMBER, edgecolor="white", linewidth=.8, zorder=5)
            ax.set(xticks=x, xticklabels=[f"{off:.0f}%\n{rate:.2f}" for off, rate in zip(x, table.discount)],
                   xlabel="Discount (% off) / price rate", ylabel=label)
            ax.tick_params(axis="both", length=0)
            ax.grid(axis="y", color=GRID)
            ax.margins(x=.06, y=.24)
        axes[0].set_title("What does the model predict?", loc="left", fontsize=12, pad=12)
        axes[1].set_title("Which eligible option has the highest value?", loc="left", fontsize=12, pad=12)
        fig.text(.045, .95, "From a sales prediction to a specific discount", fontsize=19, fontweight="bold")
        fig.text(.045, .887,
                 f"Store {case.store_id} · Product {case.product_id} · Forecast for {info['target_date']}", color=MUTED)
        recommendation_text = ("Full price (0% off; rate 1.00)" if chosen_rate == 1 else
                               f"{info['chosen_percent_off']:.0f}% off (price rate {chosen_rate:.2f})")
        fig.text(.045, .821, f"Recommendation: {recommendation_text}", fontsize=15, fontweight="bold", color=GREEN)
        handles = [Line2D([], [], color=GREEN, marker="o", linestyle="none", label="Eligible candidate"),
                   Line2D([], [], color=GREY, marker="x", linestyle="none", label="Unsupported candidate"),
                   Line2D([], [], color=AMBER, marker="*", markersize=13, linestyle="none", label="Chosen by max-value rule")]
        if table.status.eq("Rule-excluded").any():
            handles.insert(2, Line2D([], [], color=GREY, marker="s", linestyle="none", label="Excluded by stock rule"))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.038, .792), frameon=False, ncol=len(handles), fontsize=10)
        support_text = "  ·  ".join(f"{off:.0f}% off: {n} days" for off, n in zip(table.percent_off, table.support_days))
        fig.text(.045, .213, f"Training support: {support_text}", fontsize=9.3)
        fig.text(.045, .165,
                 f"Known stock signals: yesterday {info['stockout_hours_t']:g} stockout hours; {info['stockout_days7']} of the previous 7 days had stockouts.\n"
                 "In this case the stock rules exclude no supported candidates; the model chooses among them.",
                 color=MUTED, fontsize=9.5, linespacing=1.6)
        fig.text(.045, .060,
                 "Case selected by date and candidate eligibility, not by observed outcome or predicted gain. Support requires ≥3 training days within ±0.025.\n"
                 "These are model what-ifs. The value proxy is not measured revenue, and stockout history does not identify inventory age.",
                 color=MUTED, fontsize=9.3, linespacing=1.6)
    return fig, table, info
