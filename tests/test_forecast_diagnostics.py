"""Scale-aware and interval diagnostics (scripts/evaluate_forecast_diagnostics.py).

The formula tests use hand-checkable inputs; the last tests recompute the shipped
tables from the saved predictions, so an edited CSV or a changed definition fails.
"""
import csv

import numpy as np
import pandas as pd
import pytest

import evaluate_forecast_diagnostics as diag


def history(store, product, values, start='2024-01-01'):
    return pd.DataFrame({'store_id': store, 'product_id': product,
                         'dt': pd.date_range(start, periods=len(values)), 'sale_amount': values})


def test_seasonal_naive_scale_uses_lag_seven_within_each_series():
    weekly = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    first = history(1, 1, weekly + [v + 2 for v in weekly] + [v + 6 for v in weekly])   # lag-7 steps: +2, +4
    second = history(1, 2, [10.0] * 7 + [40.0] * 7)                                    # lag-7 step: +30
    # Interleave the rows: a shift that ignored the series would mix the two histories.
    train = pd.concat([first, second]).sort_values('dt', kind='stable').reset_index(drop=True)
    scales = diag.seasonal_naive_scales(train)
    assert scales.loc[(1, 1), 'mae_scale'] == pytest.approx(3.0)
    assert scales.loc[(1, 1), 'mse_scale'] == pytest.approx(10.0)
    assert scales.loc[(1, 2), 'mae_scale'] == pytest.approx(30.0)


def scored_frame(scale=1.0):
    """Two series, three scored days; series (1, 2) is `scale` times larger everywhere."""
    rows = []
    for product, factor in [(1, 1.0), (2, scale)]:
        for day, (actual, model, base) in enumerate([(2.0, 2.5, 3.0), (4.0, 3.0, 2.0), (3.0, 3.0, 4.5)]):
            rows.append({'store_id': 1, 'product_id': product, 'dt': pd.Timestamp('2024-03-01') + pd.Timedelta(days=day),
                         'sale_amount': actual * factor, 'model': model * factor, 'base': base * factor})
    train = pd.concat([history(1, 1, [1.0, 2.0] * 7), history(1, 2, [scale * v for v in [1.0, 2.0] * 7])])
    return pd.DataFrame(rows), train


def test_scaled_errors_do_not_depend_on_the_sales_level_of_a_series():
    columns = {diag.BASELINE: 'base', 'Model': 'model'}
    small, small_train = scored_frame(scale=1.0)
    large, large_train = scored_frame(scale=10.0)
    a = diag.scaled_scores(small, columns, diag.seasonal_naive_scales(small_train)).set_index('Method')
    b = diag.scaled_scores(large, columns, diag.seasonal_naive_scales(large_train)).set_index('Method')
    # lag-7 differences of the alternating history are all 1, so MASE equals each series' MAE: 0.5.
    assert a.loc['Model', 'MASE'] == pytest.approx(0.5)
    assert b.loc['Model', 'MASE'] == pytest.approx(0.5)
    assert b.loc['Model', 'RMSSE'] == pytest.approx(a.loc['Model', 'RMSSE'])
    assert b.loc['Model', 'MAE'] == pytest.approx(0.5 * (1 + 10) / 2)     # the pooled MAE does move
    assert a.loc['Model', 'series_share_beating_7day_baseline'] == 1.0


def test_scaled_errors_reject_a_series_without_training_history():
    frame, train = scored_frame()
    with pytest.raises(ValueError, match='seasonal-naive scale'):
        diag.scaled_scores(frame, {diag.BASELINE: 'base'}, diag.seasonal_naive_scales(train[train['product_id'] == 1]))


@pytest.mark.parametrize('level, expected', [(50, 5.0), (80, 8.0), (90, 9.0), (95, 9.0)])
def test_conformal_margin_is_the_finite_sample_order_statistic(level, expected):
    # n = 9: rank = ceil(10 * level), clipped to n. An interpolated np.quantile would give 7.4 at 80%.
    assert diag.conformal_margin([9, 1, 8, 2, 7, 3, 6, 4, 5], level) == expected


def test_interval_scores_match_a_hand_computed_winkler_score():
    # The first actual sits exactly on its upper bound and counts as covered (closed interval).
    scores = diag.interval_scores(actual=[2.0, 5.0, 0.0], lower=[0.0, 0.0, 1.0], upper=[2.0, 2.0, 3.0], level=80)
    assert scores['coverage'] == pytest.approx(1 / 3)
    assert scores['mean_width'] == pytest.approx(2.0)
    # width 2 each; misses cost (2 / 0.2) * distance: 30 above the second interval, 10 below the third.
    assert scores['winkler'] == pytest.approx((2 + 32 + 12) / 3)


def test_bootstrap_interval_is_seeded_and_brackets_a_clear_difference():
    values = np.linspace(-0.3, -0.1, 50)
    low, high = diag.bootstrap_mean_ci(values, draws=2000)
    assert (low, high) == diag.bootstrap_mean_ci(values, draws=2000)
    assert -0.3 < low < values.mean() < high < 0
    assert diag.bootstrap_mean_ci(np.full(20, 0.25), draws=200) == (0.25, 0.25)


@pytest.fixture(scope='module')
def tables():
    return diag.compute()


@pytest.mark.parametrize('name', ['scaled_scores', 'paired_differences', 'interval_coverage', 'stockout_bias'])
def test_shipped_diagnostic_tables_are_reproduced_from_the_saved_predictions(tables, root, name):
    saved = pd.read_csv(root / 'outputs/forecast_diagnostics' / f'{name}.csv')
    fresh = tables[name].reset_index(drop=True)
    assert list(saved.columns) == list(fresh.columns) and len(saved) == len(fresh)
    for column in fresh.columns:
        if pd.api.types.is_float_dtype(fresh[column]):
            np.testing.assert_allclose(saved[column], fresh[column], rtol=1e-8, atol=1e-10, equal_nan=True)
        else:
            assert saved[column].astype(str).tolist() == fresh[column].astype(str).tolist()


def test_diagnostics_score_the_same_predictions_as_the_headline_tables(tables, main_dir, ensemble_dir):
    scaled = tables['scaled_scores'].set_index('Method')
    with (main_dir / 'model_comparison_test.csv').open(newline='') as handle:
        headline = {row['Model']: float(row['MAE']) for row in csv.DictReader(handle)}
    with (ensemble_dir / 'final_week_scores.csv').open(newline='') as handle:
        headline.update({row['Method']: float(row['MAE']) for row in csv.DictReader(handle)
                         if row['Method'] == diag.BLEND})
    assert set(headline) == set(scaled.index)
    for method, mae in headline.items():
        assert scaled.loc[method, 'MAE'] == pytest.approx(mae, abs=1e-9)
    stockout = tables['stockout_bias']
    assert stockout.groupby('Method')['n'].sum().tolist() == [2184, 2184, 2184]
    coverage = tables['interval_coverage']
    assert coverage[coverage['slice'] != 'all'].groupby(['Method', 'level_pct', 'calibration'])['n'].sum().eq(2184).all()
