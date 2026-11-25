#!/usr/bin/env python3
"""Scale-aware and uncertainty diagnostics recomputed from the saved predictions.

No raw data, model fit or random model state is involved: the inputs are the
row-level predictions and the cohort history that ship with the repository.
The script answers four questions the pooled MAE table cannot:

1. Scaled errors (MASE, RMSSE): is each model better than a seasonal-naive
   forecast *per series*, independent of the series' sales level?
2. Paired differences: which model rankings survive a series-level bootstrap?
3. Interval coverage: do intervals calibrated on the validation folds contain
   the stated share of final-week sales, overall and inside each discount band?
4. Stock-out bias: how does the forecast error change with recorded
   out-of-stock hours (the target is observed sales, i.e. censored demand)?

Definitions follow utilsforecast.losses (mase, rmsse, coverage), the
finite-sample conformal quantile in statsforecast.utils, the Winkler score in
fabletools, and the M5 Competitors Guide's argument for scaled errors.
Nothing here selects a model; it describes the models that were already chosen.
"""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'outputs/final_protocol'
ENS = ROOT / 'outputs/ensemble_exploration'
OUT = ROOT / 'outputs/forecast_diagnostics'

KEYS = ['store_id', 'product_id']
TARGET = 'sale_amount'
SEASON = 7
SEED = 7002
BOOTSTRAP_DRAWS = 10_000
LEVELS = (80, 95)
BASELINE = '7-day average baseline'
BLEND = 'RF + CatBoost (50/50)'
# Display name -> (final-week column, validation column or None).
METHODS = {
    BASELINE: (BASELINE, 'sales_mean7'),
    'Same-weekday baseline': ('Same-weekday baseline', 'sales_lag7'),
    'Ridge Regression': ('full::Ridge Regression', 'Ridge'),
    'Decision Tree Regressor': ('full::Decision Tree Regressor', 'Decision Tree'),
    'Random Forest Regressor': ('full::Random Forest Regressor', 'Random Forest'),
    'HistGradientBoostingRegressor': ('full::HistGradientBoostingRegressor', 'HistGBR'),
    'CatBoost Regressor': ('full::CatBoost Regressor', 'CatBoost'),
    BLEND: (BLEND, BLEND),
}
PAIRS = [
    ('Random Forest Regressor', BASELINE),
    ('Random Forest Regressor', 'CatBoost Regressor'),
    (BLEND, 'Random Forest Regressor'),
    (BLEND, 'CatBoost Regressor'),
]
INTERVAL_METHODS = [BASELINE, 'Random Forest Regressor', 'CatBoost Regressor', BLEND]
BAND_ORDER = ['None/light (>0.95)', 'Moderate (0.80, 0.95]', 'Deep (<=0.80)']
STOCKOUT_BINS = [-1, 0, 4, 8, 16]
STOCKOUT_LABELS = ['0', '1-4', '5-8', '9-16']


def discount_band(discount):
    """Same right-closed bands as the notebook's diagnostic slices."""
    discount = np.asarray(discount, dtype=float)
    return np.where(discount > 0.95, BAND_ORDER[0], np.where(discount > 0.80, BAND_ORDER[1], BAND_ORDER[2]))


def seasonal_naive_scales(train, season=SEASON):
    """Per-series in-sample error of the seasonal-naive forecast y[t-season].

    `train` must hold only training-period rows. Returns a frame indexed by
    series with the mean absolute and mean squared seasonal difference.
    """
    train = train.sort_values(KEYS + ['dt'])
    diff = train[TARGET] - train.groupby(KEYS)[TARGET].shift(season)
    grouped = pd.DataFrame({'abs': diff.abs(), 'sq': diff ** 2}).groupby([train[k] for k in KEYS])
    return grouped.mean().rename(columns={'abs': 'mae_scale', 'sq': 'mse_scale'})


def per_series_errors(frame, column):
    error = frame[column] - frame[TARGET]
    grouped = pd.DataFrame({'mae': error.abs(), 'mse': error ** 2}).groupby([frame[k] for k in KEYS])
    return grouped.mean()


def scaled_scores(frame, columns, scales):
    """One row per method: pooled errors plus the series-averaged scaled errors."""
    baseline = per_series_errors(frame, columns[BASELINE])['mae']
    records = []
    for name, column in columns.items():
        error = frame[column] - frame[TARGET]
        series = per_series_errors(frame, column).join(scales, how='left')
        if series[['mae_scale', 'mse_scale']].isna().any().any() or (series['mae_scale'] <= 0).any():
            raise ValueError('every evaluated series needs a positive seasonal-naive scale')
        mase = series['mae'] / series['mae_scale']
        rmsse = np.sqrt(series['mse'] / series['mse_scale'])
        records.append({
            'Method': name,
            'MAE': error.abs().mean(),
            'mean_error': error.mean(),
            'MASE': mase.mean(),
            'median_MASE': mase.median(),
            'RMSSE': rmsse.mean(),
            'series_share_MASE_below_1': (mase < 1).mean(),
            'series_share_beating_7day_baseline': np.nan if name == BASELINE else (series['mae'] < baseline).mean(),
        })
    return pd.DataFrame(records)


def bootstrap_mean_ci(values, draws=BOOTSTRAP_DRAWS, seed=SEED, level=95):
    """Percentile interval for the mean of `values`, resampling whole series."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    tail = (100 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def paired_differences(frame, columns, pairs, period):
    """MAE(a) - MAE(b) with a series-level bootstrap interval; negative favours a."""
    records = []
    for a, b in pairs:
        diff = per_series_errors(frame, columns[a])['mae'] - per_series_errors(frame, columns[b])['mae']
        low, high = bootstrap_mean_ci(diff.to_numpy())
        records.append({
            'period': period, 'method_a': a, 'method_b': b,
            'MAE_difference': diff.mean(), 'ci95_low': low, 'ci95_high': high,
            'interval_excludes_zero': bool(low > 0 or high < 0),
            'series_share_a_better': (diff < 0).mean(),
        })
    return records


def conformal_margin(abs_residuals, level):
    """Finite-sample conformal quantile: the ceil((n+1)*level/100)-th smallest score."""
    scores = np.sort(np.asarray(abs_residuals, dtype=float))
    if len(scores) == 0:
        raise ValueError('cannot calibrate an interval on zero residuals')
    rank = math.ceil(round((len(scores) + 1) * level / 100, 8))
    return float(scores[min(max(rank, 1), len(scores)) - 1])


def interval_scores(actual, lower, upper, level):
    """Coverage, mean width and Winkler score of a central `level`% interval."""
    actual, lower, upper = (np.asarray(v, dtype=float) for v in (actual, lower, upper))
    alpha = 1 - level / 100
    width = upper - lower
    penalty = (2 / alpha) * (np.clip(lower - actual, 0, None) + np.clip(actual - upper, 0, None))
    return {'coverage': float(((actual >= lower) & (actual <= upper)).mean()),
            'mean_width': float(width.mean()), 'winkler': float((width + penalty).mean())}


def interval_coverage(validation, final, columns):
    """Calibrate on out-of-fold residuals, score on the final week.

    `pooled` uses one margin per method; `by_discount_band` one margin per
    discount band, because error spread grows with discount depth.
    """
    records = []
    for name in INTERVAL_METHODS:
        final_column, validation_column = columns[name]
        residual = (validation[validation_column] - validation[TARGET]).abs()
        for level in LEVELS:
            pooled = conformal_margin(residual, level)
            by_band = residual.groupby(validation['discount_band']).apply(lambda s: conformal_margin(s, level))
            margins = {'pooled': np.full(len(final), pooled),
                       'by_discount_band': final['discount_band'].map(by_band).to_numpy(dtype=float)}
            for calibration, margin in margins.items():
                lower = np.clip(final[final_column].to_numpy() - margin, 0, None)
                upper = final[final_column].to_numpy() + margin
                slices = [('all', np.ones(len(final), dtype=bool))]
                slices += [(band, (final['discount_band'] == band).to_numpy()) for band in BAND_ORDER]
                for label, mask in slices:
                    scores = interval_scores(final[TARGET].to_numpy()[mask], lower[mask], upper[mask], level)
                    records.append({'Method': name, 'level_pct': level, 'calibration': calibration,
                                    'slice': label, 'n': int(mask.sum()), 'mean_margin': float(margin[mask].mean()),
                                    **scores})
    return pd.DataFrame(records)


def stockout_bias(final, columns):
    """Signed and absolute error by recorded out-of-stock hours on the target day."""
    bucket = pd.cut(final['stock_hour6_22_cnt'], STOCKOUT_BINS, labels=STOCKOUT_LABELS)
    if bucket.isna().any():
        raise ValueError('stock_hour6_22_cnt outside the expected 0-16 range')
    records = []
    for name in [BASELINE, 'Random Forest Regressor', BLEND]:
        error = final[columns[name]] - final[TARGET]
        for label in STOCKOUT_LABELS:
            mask = (bucket == label).to_numpy()
            records.append({'Method': name, 'stockout_hours': label, 'n': int(mask.sum()),
                            'actual_mean': final[TARGET][mask].mean(), 'mean_error': error[mask].mean(),
                            'MAE': error[mask].abs().mean()})
    return pd.DataFrame(records)


def load_inputs():
    observations = pd.read_parquet(MAIN / 'selected_daily_observations.parquet')
    final = pd.read_csv(MAIN / 'test_predictions.csv', parse_dates=['dt'])
    blend = pd.read_csv(ENS / 'selected_final_week_predictions.csv', parse_dates=['dt'])
    validation = pd.read_csv(ENS / 'validation_predictions.csv', parse_dates=['dt'])
    selection = json.loads((ENS / 'selection_before_final_week.json').read_text(encoding='utf-8'))
    if selection['method'] != BLEND:
        raise SystemExit(f"Saved ensemble selection is {selection['method']!r}; update BLEND in this script.")
    row_keys = KEYS + ['dt']
    final = final.merge(blend[row_keys + ['selected_ensemble_prediction']], on=row_keys, validate='one_to_one')
    final = final.rename(columns={'selected_ensemble_prediction': BLEND})
    validation[BLEND] = sum(weight * validation[alias] for alias, weight in selection['weights'].items() if weight)
    validation['discount_band'] = discount_band(validation['discount'])
    if not (discount_band(final['discount']) == final['discount_band']).all():
        raise SystemExit('Discount bands in test_predictions.csv do not match the documented boundaries.')
    train = observations[observations['source_split'] == 'train']
    if train['dt'].max() >= final['dt'].min():
        raise SystemExit('Training history overlaps the final week; the scaled-error denominator would leak.')
    return train, validation, final


def compute():
    train, validation, final = load_inputs()
    final_columns = {name: columns[0] for name, columns in METHODS.items()}
    validation_columns = {name: columns[1] for name, columns in METHODS.items()}
    scaled = scaled_scores(final, final_columns, seasonal_naive_scales(train))
    paired = pd.DataFrame(
        paired_differences(validation, validation_columns, PAIRS, 'validation (5 folds, out-of-fold)')
        + paired_differences(final, final_columns, PAIRS, 'final week'))
    coverage = interval_coverage(validation, final, METHODS)
    stockout = stockout_bias(final, final_columns)
    return {'scaled_scores': scaled, 'paired_differences': paired,
            'interval_coverage': coverage, 'stockout_bias': stockout}


def markdown_table(frame, digits):
    """Plain Markdown table; floats are rounded to `digits`, missing values shown as '-'."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return '-' if math.isnan(value) else f'{value:.{digits}f}'
        return str(value)
    rows = ['| ' + ' | '.join(frame.columns) + ' |', '|' + ' --- |' * len(frame.columns)]
    rows += ['| ' + ' | '.join(cell(v) for v in record) + ' |' for record in frame.itertuples(index=False)]
    return '\n'.join(rows)


def results_markdown(tables):
    scaled, paired, coverage, stockout = (tables[k] for k in
                                          ['scaled_scores', 'paired_differences', 'interval_coverage', 'stockout_bias'])
    lines = ['# Forecast diagnostics (generated)', '',
             'Generated by `python scripts/evaluate_forecast_diagnostics.py` from the saved predictions; '
             'no model is fitted. These tables describe the already-selected models. They are not a new '
             'selection step, and the final week had already been viewed.', '',
             '## Scaled errors, final week', '',
             f'Scale = each series\' in-sample seasonal-naive (lag {SEASON}) error on the 90 training days. '
             'MASE and RMSSE are averaged over the 312 series, so every series counts equally.', '']
    lines += [markdown_table(scaled, 4), '']
    lines += ['## Paired MAE differences', '',
              f'MAE(a) - MAE(b); negative favours a. Intervals: percentile bootstrap over series '
              f'({BOOTSTRAP_DRAWS:,} draws, seed {SEED}). Series are resampled as independent units although '
              'they share the same calendar days, so the intervals are a lower bound on the uncertainty.', '',
              markdown_table(paired, 5), '']
    view = coverage[(coverage['Method'] == 'Random Forest Regressor')]
    lines += ['## Interval coverage, Random Forest, final week', '',
              'Margins are finite-sample conformal quantiles of absolute out-of-fold validation errors; '
              'lower bounds are clipped at zero.', '',
              markdown_table(view.drop(columns='Method'), 4), '',
              'All four methods are in `interval_coverage.csv`.', '']
    lines += ['## Error by recorded out-of-stock hours, final week', '',
              'mean_error = prediction - observed sales. The target is observed sales, not demand: days with '
              '9-16 recorded out-of-stock hours are over-forecast, days with 1-8 hours are under-forecast.', '',
              markdown_table(stockout, 4), '']
    return '\n'.join(lines)


def write(tables, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f'{name}.csv', index=False, float_format='%.10g', lineterminator='\n')
    (out / 'RESULTS.md').write_text(results_markdown(tables), encoding='utf-8')


def main():
    tables = compute()
    write(tables)
    scaled = tables['scaled_scores'].set_index('Method')
    coverage = tables['interval_coverage']
    deep = coverage[(coverage['Method'] == 'Random Forest Regressor') & (coverage['level_pct'] == 80)
                    & (coverage['slice'] == BAND_ORDER[2])].set_index('calibration')['coverage']
    print(f"Random Forest final-week MASE {scaled.loc['Random Forest Regressor', 'MASE']:.4f}, "
          f"{BASELINE} {scaled.loc[BASELINE, 'MASE']:.4f}; "
          f"80% interval coverage on deep-discount days: pooled {deep['pooled']:.3f}, "
          f"by band {deep['by_discount_band']:.3f}.")
    print(f'Wrote {len(tables)} tables and RESULTS.md to {OUT.relative_to(ROOT)}/')


if __name__ == '__main__':
    main()
