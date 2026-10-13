"""Small EDA views of cached raw-data counts; no downloads or model fitting.

The cache describes the original training file and the fixed five-store cohort.
Series-level category shares and day-level behavior shares have different
denominators, which remain explicit in both figures and returned tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

GREEN, TEAL, GRAY, INK = "#167262", "#66a698", "#9aa5a0", "#233e36"
AMBER = "#cf8538"


def _clean(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#e5ece7", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def _dates(scope):
    return (f"{pd.Timestamp(scope['date_start']):%d %b}–"
            f"{pd.Timestamp(scope['date_end']):%d %b %Y}")


def _check(evidence):
    """Fail clearly if cached chart denominators no longer reconcile."""
    full, selected = evidence["full_train"], evidence["selected"]
    cities, categories = evidence["cities"], evidence["categories"]
    assert sum(row["stores"] for row in cities) == full["stores"]
    assert sum(row["series"] for row in cities) == full["series"]
    assert sum(row["full_series"] for row in categories) == full["series"]
    assert sum(row["selected_series"] for row in categories) == selected["series"]
    for row in categories:
        for prefix, scope in [("full", full), ("selected", selected)]:
            assert np.isclose(row[f"{prefix}_share"],
                              row[f"{prefix}_series"] / scope["series"])
    for row in evidence["early_period"]["comparison"]:
        for count, share, denominator in [
            ("zero_sales_count", "zero_sales_share", "rows"),
            ("stockout_day_count", "stockout_day_share", "rows"),
            ("discounted_count", "discounted_share", "valid_discount_rows"),
        ]:
            assert row[denominator] > 0
            assert 0 <= row[count] <= row[denominator]
            assert np.isclose(row[share], row[count] / row[denominator])


def overview(evidence):
    """Return source category/city coverage, dataset scope and field tables."""
    _check(evidence)
    full, selected = evidence["full_train"], evidence["selected"]
    scope_table = pd.DataFrame([
        {"Coverage": label, "Full training file": full[key],
         "Selected cohort": selected[key]}
        for key, label in [("cities", "Cities"), ("stores", "Stores"),
                           ("products", "Distinct products"),
                           ("series", "Store-product histories"),
                           ("rows", "Daily records (raw)"), ("days", "Days per history")]
    ])
    scope_table.loc[len(scope_table)] = ["Date range", _dates(full), _dates(selected)]
    scope_table.loc[len(scope_table)] = ["First-level category IDs", len(evidence["categories"]),
        sum(row["selected_series"] > 0 for row in evidence["categories"])]

    # Describe available source information, not a claim that every field is used.
    names = {row["name"] for row in full["schema"]}
    groups = [
        ("Identity", ["city_id", "store_id", "product_id"], "Follow each store-product through time"),
        ("Product category", ["first_category_id", "second_category_id", "third_category_id"],
         "Compare the mix of anonymized categories"),
        ("Sales", ["sale_amount", "hours_sale", "hourly_sales"],
         "Observe recorded sales; source values are normalized"),
        ("Discount and activity", ["discount", "activity_flag"], "Describe historical pricing and promotions"),
        ("Stockouts", ["stock_hour6_22_cnt", "hours_stock_status", "hourly_stock_status"],
         "Identify days when availability may limit recorded sales"),
        ("Calendar", ["dt", "holiday_flag"], "Capture timing and holiday context"),
        ("Weather", ["precpt", "avg_temperature", "avg_humidity"], "Describe observed weather conditions"),
    ]
    field_groups_table = pd.DataFrame([
        {"Information": label, "Examples in source": ", ".join(n for n in fields if n in names),
         "Why it is useful": purpose}
        for label, fields, purpose in groups if names.intersection(fields)
    ])

    city = pd.DataFrame(evidence["cities"]).sort_values("stores", ascending=False)
    selected_cities = set(selected["city_ids"])
    colors = [AMBER if key in selected_cities else TEAL for key in city.city_id]
    category = pd.DataFrame(evidence["categories"]).sort_values("full_series", ascending=False)
    largest = category.head(5)
    rest = category.iloc[5:]
    sizes = [*largest.full_series, rest.full_series.sum()]
    labels = [*[f"Category {key}" for key in largest.first_category_id],
              f"Other {len(rest)} categories"]
    palette = [GREEN, "#409385", "#75b6a8", AMBER, "#e3b87f", "#dce8e3"]

    fig = plt.figure(figsize=(10, 6.7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.12], height_ratios=[1.65, 1],
                           left=.035, right=.97, top=.93, bottom=.15, wspace=.35, hspace=.03)
    donut = fig.add_subplot(grid[0, 0])
    legend = fig.add_subplot(grid[1, 0])
    ax = fig.add_subplot(grid[:, 1])
    donut.pie(sizes, startangle=90, counterclock=False, colors=palette,
              wedgeprops={"width": .34, "edgecolor": "white", "linewidth": 2})
    donut.text(0, .05, f"{full['series']:,}", ha="center", fontsize=24, weight="bold", color=INK)
    donut.text(0, -.19, "sales histories", ha="center", fontsize=10, color=INK)
    donut.set_title(f"{len(category)} product categories", loc="left", fontsize=13, pad=14)
    legend.set_axis_off()
    for i, (label, count, color) in enumerate(zip(labels, sizes, palette)):
        y = .95 - i * .15
        legend.scatter(.09, y, s=72, marker="s", color=color, transform=legend.transAxes)
        legend.text(.16, y, label, va="center", fontsize=10, color=INK, transform=legend.transAxes)
        legend.text(.94, y, f"{count / full['series']:.1%}", ha="right", va="center",
                    fontsize=10, color=INK, transform=legend.transAxes)

    positions = np.arange(len(city))
    bars = ax.barh(positions, city.stores, color=colors, height=.64)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_yticks(positions, [f"City {key}" for key in city.city_id])
    ax.invert_yaxis()
    ax.set_xlabel("Distinct stores")
    ax.set_title(f"{full['stores']:,} stores across {full['cities']} cities", loc="left", fontsize=13, pad=14)
    ax.margins(x=.14, y=.025)
    _clean(ax, axis="x")
    fig.text(.035, .075, f"Original training file · {_dates(full)} · category shares use "
             f"{full['series']:,} store-product histories.", fontsize=9, color=INK)
    city_names = ", ".join(str(key) for key in sorted(selected_cities))
    fig.text(.035, .033, f"Category and city IDs are anonymized. Amber marks city {city_names}, "
             "which contains the five selected stores.", fontsize=9, color=INK)
    return fig, scope_table, field_groups_table


def composition(evidence):
    """Return category/day-pattern comparisons with explicit counts and dates."""
    _check(evidence)
    full, selected = evidence["full_train"], evidence["selected"]
    category = pd.DataFrame(evidence["categories"]).sort_values("first_category_id")
    coverage_table = category.rename(columns={
        "first_category_id": "Category ID", "full_series": "Full histories",
        "selected_series": "Selected histories", "full_share": "Full share",
        "selected_share": "Selected share",
    })[["Category ID", "Full histories", "Full share", "Selected histories", "Selected share"]]
    # Show the same leading categories in both scopes; aggregate every other ID.
    # Return the full category table so the aggregation remains inspectable.
    ranked = category.sort_values("full_series", ascending=False)
    shown = ranked.head(8).copy()
    shown["label"] = shown.first_category_id.astype(str)
    remainder = ranked.iloc[8:]
    if not remainder.empty:
        shown = pd.concat([shown, pd.DataFrame([{
            "label": "Other", "full_share": remainder.full_share.sum(),
            "selected_share": remainder.selected_share.sum(),
        }])], ignore_index=True)

    early = evidence["early_period"]
    behavior = pd.DataFrame(early["comparison"]).set_index("scope")
    scopes = ["Full dataset", "Selected cohort"]
    metrics = [
        ("Zero sales", "zero_sales_count", "zero_sales_share", "rows"),
        ("Discount < 0.95", "discounted_count", "discounted_share", "valid_discount_rows"),
        ("Any stockout", "stockout_day_count", "stockout_day_share", "rows"),
    ]
    behavior_table = pd.DataFrame([
        {"Scope": scope, "Pattern": label, "Matching days": int(behavior.loc[scope, count]),
         "Eligible days": int(behavior.loc[scope, denominator]),
         "Share": float(behavior.loc[scope, share])}
        for scope in scopes for label, count, share, denominator in metrics
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), gridspec_kw={"width_ratios": [1, 1.1]})
    width = .36
    positions = np.arange(len(shown))
    axes[0].hlines(positions, shown.full_share, shown.selected_share,
                   color="#ccd9d2", linewidth=3, zorder=2)
    axes[0].scatter(shown.full_share, positions, s=46, color=TEAL,
                    marker="o", label="Full dataset", zorder=3)
    axes[0].scatter(shown.selected_share, positions, s=46, color=GREEN,
                    marker="D", label="Selected cohort", zorder=3)
    axes[0].set_yticks(positions, [f"Category {label}" if label != "Other" else "Other"
                                  for label in shown.label])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Share of store-product histories")
    axes[0].set_title("How the selected category mix differs", loc="left", fontsize=11.5, pad=20)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(-.03, 1),
                    ncol=2, handletextpad=.3, columnspacing=1.1)

    y = np.arange(len(metrics))
    for offset, scope, color in [(-width / 2, scopes[0], TEAL), (width / 2, scopes[1], GREEN)]:
        values = [behavior.loc[scope, share] for _, _, share, _ in metrics]
        bars = axes[1].barh(y + offset, values, width, color=color)
        axes[1].bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4, fontsize=9)
    axes[1].set_yticks(y, [m[0] for m in metrics])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Share of eligible daily records")
    axes[1].set_title("Early-period sales conditions", loc="left", fontsize=11.5, pad=20)
    axes[1].set_xlim(0, min(1.12, behavior_table.Share.max() * 1.35))
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xlim(0, max(shown.full_share.max(), shown.selected_share.max()) * 1.13)
    axes[0].margins(y=.08)
    _clean(axes[0], axis="x"); _clean(axes[1], axis="x")
    last = pd.Timestamp(early["end_exclusive"]) - pd.Timedelta(days=1)
    fig.text(.02, .105, f"Category mix · {_dates(full)} · top 8 + Other · "
             f"{full['series']:,} full / {selected['series']:,} selected histories.", fontsize=9, color=INK)
    fig.text(.02, .065, f"Behavior dates: {pd.Timestamp(early['start']):%d %b}–{last:%d %b %Y}; "
             f"{int(behavior.loc[scopes[0], 'rows']):,} full / "
             f"{int(behavior.loc[scopes[1], 'rows']):,} selected daily records.", fontsize=9, color=INK)
    fig.text(.02, .025, "Discount share uses valid rates only (0 < rate ≤ 1); stockout means at least one stockout hour.",
             fontsize=9, color=INK)
    fig.tight_layout(rect=(0, .16, 1, 1))
    return fig, coverage_table, behavior_table
