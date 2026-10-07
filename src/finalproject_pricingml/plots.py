"""Figures. Every function returns the matplotlib figure; pass ``save`` to also write a PNG."""

import matplotlib.pyplot as plt
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
