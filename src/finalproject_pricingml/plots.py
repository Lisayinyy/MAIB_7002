"""Figures. Every function returns the matplotlib figure; pass ``save`` to also write a PNG."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from . import config as C

MARKER = dict(marker="o", markersize=7, markeredgecolor=C.SURFACE, markeredgewidth=2, linewidth=2)


def _style(ax):
    ax.set_facecolor(C.SURFACE)
    ax.grid(axis="y", color=C.GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(C.AXIS)
    ax.tick_params(length=0)


def _figure(*args, **kwargs):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": C.INK,
                         "axes.labelcolor": C.MUTED, "xtick.color": C.MUTED, "ytick.color": C.MUTED})
    fig, axes = plt.subplots(*args, facecolor=C.SURFACE, **kwargs)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        _style(ax)
    return fig, axes


def _finish(fig, save):
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, facecolor=C.SURFACE)
    return fig


def plot_knn_tuning(results, baseline, series, save=None):
    """MAE by number of neighbours (left) and by validation week at the best k (right).

    ``results`` has columns weighting, k, fold, mae. ``baseline`` is indexed by fold with a mae
    column. ``series`` maps a weighting name to (label, colour).
    """
    by_k = results.groupby(["weighting", "k"]).mae.mean().unstack("weighting")[list(series)]
    best_k = int(by_k.min(axis=1).idxmin())
    baseline_mae = baseline.mae.mean()

    fig, (left, right) = _figure(1, 2, figsize=(13, 5.2), sharey=True)
    left.axhline(baseline_mae, color=C.COLOURS["baseline"], linewidth=2,
                 label=f"Baseline (7-day average): {baseline_mae:.3f}")
    for name, (label, colour) in series.items():
        left.plot(by_k.index, by_k[name], color=colour, **MARKER,
                  label=f"{label}: best {by_k[name].min():.3f} at k = {by_k[name].idxmin()}")
    left.set_xscale("log")
    left.set_xticks(by_k.index, [str(k) for k in by_k.index])
    left.minorticks_off()
    left.set_xlabel("Number of neighbours (k)")
    left.set_ylabel("Mean absolute error (lower is better)")
    left.set_title("Average over five validation weeks", loc="left", fontsize=11, color=C.INK)
    left.legend(frameon=False, loc="upper right")

    at_best = results[results.k == best_k].pivot(index="fold", columns="weighting", values="mae")
    right.plot(baseline.index, baseline.mae, color=C.COLOURS["baseline"], **MARKER)
    for name, (label, colour) in series.items():
        right.plot(at_best.index, at_best[name], color=colour, **MARKER)
    right.set_xticks(list(C.FOLD_LABELS), list(C.FOLD_LABELS.values()))
    right.set_xlabel("Validation week")
    right.set_title(f"Each validation week at k = {best_k}", loc="left", fontsize=11, color=C.INK)
    fig.suptitle("kNN tuning: next-day sales error by neighbours and feature weighting",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    return _finish(fig, save)


def plot_rf_tuning(summary, depth_labels, leaf_sizes, save=None):
    """MAE by maximum depth, one line per minimum leaf size. ``summary`` is indexed by (max_depth, min_leaf)."""
    colours = dict(zip(leaf_sizes, [C.COLOURS["knn"], C.COLOURS["rf"], C.COLOURS["catboost"], C.COLOURS["ensemble"]]))
    fig, ax = _figure(figsize=(8, 5))
    for leaf in leaf_sizes:
        mae = summary.xs(leaf, level="min_leaf").mae.reindex(depth_labels)
        ax.plot(depth_labels, mae, color=colours[leaf], **MARKER, label=f"At least {leaf} rows per leaf: best {mae.min():.4f}")
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Mean absolute error (lower is better)")
    ax.set_title("Random forest tuning: next-day sales error by tree depth and leaf size\n"
                 "Average over five validation weeks", loc="left", fontsize=11)
    ax.legend(frameon=False)
    return _finish(fig, save)


def plot_validation_weeks(mae_by_fold, colours, annotate=None, ylim=(0.27, 0.37), save=None):
    """MAE in each validation week, one line per method.

    ``mae_by_fold`` is a DataFrame indexed by fold with one column per method label.
    ``colours`` maps each label to a colour; ``annotate`` names the column to label point by point.
    """
    fig, ax = _figure(figsize=(9, 5))
    for label in mae_by_fold:
        mae = mae_by_fold[label]
        ax.plot(mae.index, mae, color=colours[label], **MARKER, label=f"{label}: average {mae.mean():.3f}")
    if annotate:
        for fold, value in mae_by_fold[annotate].items():
            ax.annotate(f"{value:.3f}", (fold, value), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_xticks(list(C.FOLD_LABELS), list(C.FOLD_LABELS.values()))
    ax.set_xlabel("Validation week")
    ax.set_ylabel("Mean absolute error (lower is better)")
    ax.set_title("Next-day sales error in each validation week", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    return _finish(fig, save)


def plot_model_comparison(validation, test, methods, colours, ylim=(0, 0.66), save=None):
    """Grouped bars: validation average, test week, and test week by tomorrow's discount.

    ``validation`` maps a method label to its average validation MAE. ``test`` is the scored
    test week; ``methods`` maps each label to its prediction column.
    """
    errors = pd.DataFrame({label: (test[C.TARGET] - test[col]).abs() for label, col in methods.items()})
    band = pd.cut(test.discount_next, C.DISCOUNT_BAND_EDGES, labels=["deep", "moderate", "none"])
    by_band = errors.groupby(band, observed=True).mean()
    rows = band.value_counts()
    groups = {
        "Validation\n(5-week average)": [validation[label] for label in methods],
        "Test week\n(all rows)": errors.mean().tolist(),
        f"Test: no discount\n({rows['none']:,} rows)": by_band.loc["none"].tolist(),
        f"Test: moderate discount\n({rows['moderate']:,} rows)": by_band.loc["moderate"].tolist(),
        f"Test: deep discount\n({rows['deep']:,} rows)": by_band.loc["deep"].tolist(),
    }
    n = len(methods)
    width = min(0.22, 0.8 / n)
    fig, ax = _figure(figsize=(11, 5.5))
    for i, label in enumerate(methods):
        xs = [g + (i - (n - 1) / 2) * (width + 0.03) for g in range(len(groups))]
        values = [v[i] for v in groups.values()]
        ax.bar(xs, values, width, color=colours[label], label=label)
        for x, value in zip(xs, values):
            ax.annotate(f"{value:.3f}", (x, value), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, color=C.INK)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_ylabel("Mean absolute error (lower is better)")
    ax.set_ylim(*ylim)
    ax.set_title("Next-day sales error by model", loc="left", fontsize=12, color=C.INK)
    ax.legend(frameon=False, loc="upper left")
    _finish(fig, save)
    return fig, pd.DataFrame(groups, index=list(methods)).T


def plot_learning_curves(curves, save=None):
    """Train and validation MAE against boosting iteration, one panel per learning rate.

    ``curves`` maps a label (e.g. "lr=0.03") to a DataFrame with ``train`` and ``validation`` columns.
    """
    fig, axes = _figure(1, len(curves), figsize=(4.5 * len(curves), 4.6), sharey=True)
    for ax, (label, curve) in zip(axes, curves.items()):
        ax.plot(curve.index, curve.train, color=C.MUTED, linewidth=1.5, label="Training rows")
        ax.plot(curve.index, curve.validation, color=C.COLOURS["catboost"], linewidth=2, label="Validation week")
        best = curve.validation.idxmin()
        ax.axvline(best, color=C.AXIS, linewidth=1, linestyle="--")
        ax.annotate(f"best {curve.validation.min():.4f}\nat {best} trees", (best, curve.validation.min()),
                    textcoords="offset points", xytext=(8, 18), fontsize=9)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xlabel("Boosting iteration (trees added)")
    axes[0].set_ylabel("Mean absolute error")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("CatBoost learning curves on the last validation week: training error keeps falling, validation error turns up",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return _finish(fig, save)


def plot_weight_sweep(sweep, rf_mae, catboost_mae, save=None):
    """Validation MAE of the RF + CatBoost blend as the forest's weight goes from 0 to 1."""
    fig, ax = _figure(figsize=(10, 4.6))
    ax.plot(sweep.index, sweep.mae, color=C.COLOURS["ensemble"], **MARKER, label="Weighted average")
    ax.axhline(rf_mae, color=C.COLOURS["rf"], linewidth=1.5, linestyle="--", label=f"Random forest alone: {rf_mae:.4f}")
    ax.axhline(catboost_mae, color=C.COLOURS["catboost"], linewidth=1.5, linestyle="--", label=f"CatBoost alone: {catboost_mae:.4f}")
    best = sweep.mae.idxmin()
    ax.annotate(f"best {sweep.mae.min():.4f} at RF {best:.0%}", (best, sweep.mae.min()), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9)
    ax.set_xlabel("Weight on the random forest (the rest on CatBoost)")
    ax.set_ylabel("Mean absolute error (5-week average)")
    ax.set_title("Ensemble weight sweep on out-of-fold predictions", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _finish(fig, save)


# --- Added for the final notebook: data story, diagnostics and phase 2 -------------------------

def plot_selection_funnel(steps, save=None):
    """Big numbers joined by arrows: how the full dataset became the working set. ``steps`` is a list of (count, label)."""
    fig, ax = _figure(figsize=(11, 2.2))
    ax.axis("off")
    n = len(steps)
    for i, (count, label) in enumerate(steps):
        x = (i + 0.5) / n
        ax.text(x, 0.66, count, ha="center", va="center", fontsize=24, fontweight="bold", color=C.COLOURS["knn"], transform=ax.transAxes)
        ax.text(x, 0.2, label, ha="center", va="center", fontsize=10, color=C.MUTED, transform=ax.transAxes, linespacing=1.4)
        if i < n - 1:
            ax.annotate("", xy=(x + 0.17, 0.66), xytext=(x + 0.08, 0.66), xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=C.AXIS, linewidth=1.5))
    return _finish(fig, save)


def plot_discount_signal(table, save=None):
    """Sales relative to the series' own average (left) and row counts (right) by tomorrow's discount band.

    ``table`` is indexed by band label with columns ``relative_sales`` and ``rows``.
    """
    fig, (left, right) = _figure(1, 2, figsize=(13, 4.6))
    x = list(range(len(table)))
    left.bar(x, table.relative_sales, color=C.COLOURS["knn"], width=0.62)
    left.axhline(1, color=C.MUTED, linewidth=1)
    left.text(x[-1] + 0.4, 1.0, "series average", ha="right", va="bottom", fontsize=9, color=C.MUTED)
    for i, v in enumerate(table.relative_sales):
        left.annotate(f"{v:.2f}×", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, color=C.INK)
    left.set_ylabel("Next-day sales ÷ the series' own average")
    left.set_title("Sales rise as tomorrow's discount deepens", loc="left", fontsize=11)
    right.bar(x, table.rows, color=C.COLOURS["knn"], width=0.62)
    for i, v in enumerate(table.rows):
        right.annotate(f"{v:,}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, color=C.INK)
    right.set_ylabel("Store-product-days")
    right.set_title("How many days sit in each band", loc="left", fontsize=11)
    for ax in (left, right):
        ax.set_xticks(x, list(table.index))
        ax.set_xlabel("Tomorrow's discount rate")
        ax.margins(y=0.15)
    fig.suptitle(f"Does tomorrow's discount carry signal? {int(table.rows.sum()):,} store-product-days",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return _finish(fig, save)


def plot_split_timeline(feat, save=None):
    """One bar per validation week and for the test week: the rows trained on, then the week scored."""
    first, last = feat.dt.min(), feat.dt.max()
    fig, ax = _figure(figsize=(12, 4))
    ax.grid(False)
    ax.grid(axis="x", color=C.GRID, linewidth=1)
    labels = []
    rows = list(C.VAL_FOLDS.items()) + [("test", (C.TEST_START, str(last.date())))]
    for i, (fold, (start, end)) in enumerate(rows):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        colour = C.COLOURS["ensemble"] if fold == "test" else C.COLOURS["knn"]
        ax.barh(i, (start - first).days, left=mdates.date2num(first), color=C.AXIS, height=0.6)
        ax.barh(i, (end - start).days + 1, left=mdates.date2num(start), color=colour, height=0.6)
        n_train, n_scored = int((feat.dt < start).sum()), int(feat.dt.between(start, end).sum())
        ax.text(mdates.date2num(end) + 2, i, f"train {n_train:,} rows, score {n_scored:,}", va="center", fontsize=9, color=C.MUTED)
        labels.append("Test week\n" + f"{start:%b %-d}–{end:%b %-d}" if fold == "test" else f"Validation week {fold}\n{C.FOLD_LABELS[fold]}")
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.set_xlim(mdates.date2num(first) - 1, mdates.date2num(last) + 30)
    for colour, label in [(C.AXIS, "Rows the model is fitted on"), (C.COLOURS["knn"], "Validation week"), (C.COLOURS["ensemble"], "Test week")]:
        ax.plot([], [], linewidth=8, color=colour, label=label)
    ax.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.12))
    ax.set_title("Time-ordered splits: fit on everything before the week, then score the week one day ahead", loc="left", fontsize=11)
    return _finish(fig, save)


def plot_rf_heatmap(pivot, save=None):
    """Validation MAE by maximum depth (rows) and minimum leaf size (columns); darker is lower, the best cell is outlined."""
    fig, ax = _figure(figsize=(8.5, 4.2))
    ax.grid(False)
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="Blues_r", aspect="auto")
    best = np.unravel_index(np.nanargmin(values), values.shape)
    mid = (np.nanmax(values) + np.nanmin(values)) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center", fontsize=10.5,
                    color=C.SURFACE if values[i, j] < mid else C.INK, fontweight="bold" if (i, j) == best else "normal")
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor=C.COLOURS["rf"], linewidth=2.5))
    ax.set_xticks(range(values.shape[1]), [str(c) for c in pivot.columns])
    ax.set_yticks(range(values.shape[0]), [str(r) for r in pivot.index])
    ax.set_xlabel("Minimum rows per leaf")
    ax.set_ylabel("Maximum tree depth")
    ax.set_title(f"Random forest tuning: validation MAE across {values.size} settings (best outlined)", loc="left", fontsize=11)
    bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    bar.set_label("MAE, 5-week average (darker is lower)")
    bar.outline.set_visible(False)
    return _finish(fig, save)


def plot_test_errors(test, methods, colours, highlight, save=None):
    """Left: actual against predicted next-day sales for ``highlight``. Right: mean sales by tomorrow's discount band, actual against each model.

    ``methods`` maps a label to its prediction column; ``colours`` maps each label to a colour.
    """
    fig, (left, right) = _figure(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    col = methods[highlight]
    lim = max(test[C.TARGET].max(), test[col].max()) * 1.03
    left.plot([0, lim], [0, lim], color=C.MUTED, linewidth=1)
    left.scatter(test[C.TARGET], test[col], s=16, alpha=0.35, color=colours[highlight], linewidths=0)
    left.text(lim * 0.97, lim * 0.97, "perfect forecast", ha="right", va="bottom", fontsize=9, color=C.MUTED, rotation=45, rotation_mode="anchor")
    left.set_xlim(0, lim)
    left.set_ylim(0, lim)
    left.set_xlabel("Actual next-day sales")
    left.set_ylabel(f"Predicted next-day sales ({highlight})")
    left.set_title("Big days are under-predicted", loc="left", fontsize=11)

    band = pd.cut(test.discount_next, C.DISCOUNT_BAND_EDGES, labels=["deep", "moderate", "none"])
    order = ["none", "moderate", "deep"]
    means = test.groupby(band, observed=True)[[C.TARGET, *methods.values()]].mean().reindex(order)
    counts = band.value_counts().reindex(order)
    n = len(methods)
    width = min(0.16, 0.8 / n)
    for i, (label, column) in enumerate(methods.items()):
        xs = [g + (i - (n - 1) / 2) * (width + 0.02) for g in range(len(order))]
        right.bar(xs, means[column], width, color=colours[label], label=label)
        if label == highlight:
            for x, v in zip(xs, means[column]):
                right.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, color=C.INK)
    half = (n * (width + 0.02)) / 2
    for g, b in enumerate(order):
        right.plot([g - half, g + half], [means.loc[b, C.TARGET]] * 2, color=C.INK, linewidth=2)
        right.annotate(f"actual {means.loc[b, C.TARGET]:.2f}", (g + half, means.loc[b, C.TARGET]), textcoords="offset points", xytext=(3, -3), fontsize=8.5, color=C.INK)
    right.set_xticks(range(len(order)), [f"{'no' if b == 'none' else b} discount\n({counts[b]:,} days)" for b in order])
    right.set_xlim(-0.6, len(order) - 0.3)
    right.set_ylabel("Mean next-day sales")
    right.set_title("The deep-discount spike: actual (black line) against each model", loc="left", fontsize=11)
    right.legend(frameon=False, loc="upper left", fontsize=9)
    right.margins(y=0.15)
    fig.suptitle("Test-week diagnostics", x=0.01, ha="left", fontsize=12, fontweight="bold")
    return _finish(fig, save)


def plot_decision_flow(steps, save=None):
    """Boxes joined by arrows. ``steps`` is a list of (title, body)."""
    n = len(steps)
    fig, ax = _figure(figsize=(13, 2.7))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    gap = 0.03
    w = (1 - gap * (n - 1)) / n
    for i, (title, body) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.08), w, 0.84, boxstyle="round,pad=0.004,rounding_size=0.015", facecolor=C.GRID, edgecolor="none"))
        ax.text(x + w / 2, 0.76, title, ha="center", va="center", fontsize=10, fontweight="bold", color=C.COLOURS["knn"])
        ax.text(x + w / 2, 0.4, body, ha="center", va="center", fontsize=9, color=C.INK, linespacing=1.5)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.002, 0.5), xytext=(x + w + 0.002, 0.5),
                        arrowprops=dict(arrowstyle="->", color=C.MUTED, linewidth=1.5))
    return _finish(fig, save)


def plot_response_curve(curve, support_share, save=None):
    """Left: mean predicted sales and value under each candidate discount, indexed to no discount. Right: share of series that have run each candidate.

    ``curve`` is ``scenario.response_curve`` output; ``support_share`` is a Series indexed by candidate (share of series, 0 to 1).
    """
    fig, (left, right) = _figure(1, 2, figsize=(13, 4.6))
    d = list(curve.index)
    for column, label, colour in [("sales_vs_no_discount", "Predicted sales", C.COLOURS["knn"]),
                                  ("value_vs_no_discount", "Predicted value (sales × rate)", C.COLOURS["rf"])]:
        left.plot(d, curve[column], color=colour, **MARKER, label=label)
        left.annotate(f"{curve[column].iloc[-1]:.2f}×", (d[-1], curve[column].iloc[-1]), textcoords="offset points", xytext=(8, -3), fontsize=9, color=C.INK)
    left.axhline(1, color=C.MUTED, linewidth=1)
    left.invert_xaxis()
    left.set_xticks(d, [f"{x:.2f}" for x in d])
    left.set_xlabel("Tomorrow's discount rate (1.00 = no discount; deeper to the right)")
    left.set_ylabel("Relative to no discount (1.0)")
    left.set_title("What the ensemble predicts when only the discount changes", loc="left", fontsize=11)
    left.legend(frameon=False, loc="upper left")
    left.margins(x=0.08)

    x = list(range(len(support_share)))
    right.bar(x, support_share * 100, color=C.COLOURS["knn"], width=0.62)
    for i, v in enumerate(support_share * 100):
        right.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, color=C.INK)
    right.set_xticks(x, [f"{c:.2f}" for c in support_share.index])
    right.set_xlabel("Candidate discount rate")
    right.set_ylabel("Series with at least 3 training days near it, %")
    right.set_title("How many of the 312 series have actually run each candidate", loc="left", fontsize=11)
    right.set_ylim(0, 110)
    return _finish(fig, save)


def plot_policy_outcomes(bands, agreement, n_days, save=None):
    """Left: share of days on which each policy recommends no, a moderate or a deep discount (stacked). Right: share of days in the manager's band.

    ``bands`` is indexed by policy with columns none, moderate, deep (percent); ``agreement`` is a Series indexed the same way (percent).
    """
    fig, (left, right) = _figure(1, 2, figsize=(13, 0.5 * len(bands) + 1.8), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
    y = np.arange(len(bands))
    fills = {"none": (C.AXIS, "No discount", C.INK), "moderate": ("#8db8ea", "Moderate (0.80–0.95)", C.INK), "deep": (C.COLOURS["knn"], "Deep (0.80 or lower)", C.SURFACE)}
    edge = np.zeros(len(bands))
    for column, (colour, label, ink) in fills.items():
        v = bands[column].to_numpy(dtype=float)
        left.barh(y, v, left=edge, color=colour, height=0.62, label=label, edgecolor=C.SURFACE, linewidth=1.5)
        for yi, (start, w) in enumerate(zip(edge, v)):
            if w >= 7:
                left.text(start + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=9, color=ink)
        edge += v
    left.set_yticks(y, list(bands.index))
    left.invert_yaxis()
    left.set_xlim(0, 100)
    left.set_xlabel(f"Share of the {n_days:,} test-week store-product-days, %")
    left.set_title("What each policy recommends", loc="left", fontsize=11)
    left.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.08), fontsize=9)
    right.barh(y, agreement.to_numpy(dtype=float), color=C.COLOURS["ensemble"], height=0.62)
    for yi, v in enumerate(agreement):
        right.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=9, color=C.INK)
    right.set_xlim(0, 100)
    right.set_xlabel("Days in the same discount band as the manager, %")
    right.set_title("Agreement with the manager", loc="left", fontsize=11)
    for ax in (left, right):
        ax.grid(False)
        ax.grid(axis="x", color=C.GRID, linewidth=1)
    return _finish(fig, save)
