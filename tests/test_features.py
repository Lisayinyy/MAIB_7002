"""Feature construction: every history feature must use strictly earlier days."""
import numpy as np
import pandas as pd
import pytest

HISTORY = ['sales_lag1', 'sales_lag7', 'sales_mean7', 'discount_lag1', 'stockout_lag1', 'series_mean']
START = pd.Timestamp('2024-03-28')  # a Thursday, the first raw training date


def toy(days=12):
    rows = []
    for scale, product in [(1.0, 10), (10.0, 11)]:
        for day in range(days):
            rows.append({'store_id': 1, 'product_id': product, 'dt': START + pd.Timedelta(days=day),
                         'sale_amount': scale * (day + 1), 'discount': 1 - 0.01 * day,
                         'stock_hour6_22_cnt': float(day % 3)})
    return pd.DataFrame(rows)


def series(frame, product):
    return frame.loc[frame.product_id.eq(product)].sort_values('dt').reset_index(drop=True)


def test_feature_values_on_a_hand_checked_series(notebook_ns):
    features = series(notebook_ns['make_features'](toy()), 10)  # sales are 1, 2, ..., 12
    first, fourth, eighth, tenth = (features.iloc[i] for i in (0, 3, 7, 9))
    assert first[HISTORY].isna().all() and first.weekday == 3
    assert (fourth.sales_lag1, fourth.series_mean) == (3.0, 2.0)
    assert np.isnan(fourth.sales_lag7) and np.isnan(fourth.sales_mean7)  # warm-up: fewer than 7 earlier days
    assert (eighth.sales_lag1, eighth.sales_lag7, eighth.sales_mean7, eighth.series_mean) == (7.0, 1.0, 4.0, 4.0)
    assert (tenth.sales_lag1, tenth.sales_lag7, tenth.sales_mean7, tenth.series_mean) == (9.0, 3.0, 6.0, 5.0)
    assert tenth.discount_lag1 == pytest.approx(0.92) and tenth.stockout_lag1 == 2.0
    assert tenth.weekday == (3 + 9) % 7


def test_a_day_outcome_never_enters_features_for_that_day_or_earlier(notebook_ns):
    make_features = notebook_ns['make_features']
    base = toy()
    reference = make_features(base)
    for day in range(12):
        date = START + pd.Timedelta(days=day)
        changed = base.copy()
        changed.loc[changed.dt.ge(date), ['sale_amount', 'stock_hour6_22_cnt']] = 999.0
        rebuilt = make_features(changed)
        pd.testing.assert_frame_equal(reference.loc[reference.dt.le(date), HISTORY],
                                      rebuilt.loc[rebuilt.dt.le(date), HISTORY])
        if day < 11:  # the check has power: the same change does reach the following day
            following = rebuilt.loc[rebuilt.dt.eq(date + pd.Timedelta(days=1))]
            assert following.sales_lag1.eq(999.0).all() and following.stockout_lag1.eq(999.0).all()


def test_series_do_not_share_history(notebook_ns):
    make_features = notebook_ns['make_features']
    base = toy()
    changed = base.copy()
    changed.loc[changed.product_id.eq(11), ['sale_amount', 'discount', 'stock_hour6_22_cnt']] = 5.0
    pd.testing.assert_frame_equal(series(make_features(base), 10)[HISTORY],
                                  series(make_features(changed), 10)[HISTORY])


def test_input_row_order_does_not_matter(notebook_ns):
    make_features = notebook_ns['make_features']
    base = toy()
    shuffled = base.sample(frac=1.0, random_state=7002)
    keys = ['store_id', 'product_id', 'dt']
    pd.testing.assert_frame_equal(make_features(base).sort_values(keys).reset_index(drop=True),
                                  make_features(shuffled).sort_values(keys).reset_index(drop=True))


def test_discount_band_boundaries_are_right_closed(notebook_ns):
    none, moderate, deep = notebook_ns['BANDS']
    labels = notebook_ns['discount_bands'](pd.Series([0.5, 0.80, 0.8001, 0.95, 0.9501, 1.0]))
    assert labels.astype(str).tolist() == [deep, deep, moderate, moderate, none, none]


def independent_history_features(daily):
    """Explicit per-row slices of earlier days; shares no code with make_features."""
    rows = []
    for (store, product), group in daily.groupby(['store_id', 'product_id']):
        group = group.sort_values('dt')
        sales = group.sale_amount.to_numpy(dtype=float)
        discount = group.discount.to_numpy(dtype=float)
        stockout = group.stock_hour6_22_cnt.to_numpy(dtype=float)
        dates = group.dt.to_numpy()
        for position in range(7, len(group)):
            rows.append((store, product, dates[position], sales[position - 1], sales[position - 7],
                         sales[position - 7:position].mean(), discount[position - 1],
                         stockout[position - 1], sales[:position].mean()))
    return pd.DataFrame(rows, columns=['store_id', 'product_id', 'dt'] + HISTORY)


@pytest.fixture(scope='module')
def daily(main_dir):
    return pd.read_parquet(main_dir / 'selected_daily_observations.parquet')


@pytest.fixture(scope='module')
def saved_features(main_dir):
    return pd.read_parquet(main_dir / 'feature_frame.parquet')


def test_saved_feature_frame_matches_notebook_function_and_independent_slices(notebook_ns, protocol, daily, saved_features):
    keys = notebook_ns['ROW_KEYS']
    features = protocol['features']['full']
    assert features == notebook_ns['FEATURES'] and set(HISTORY) < set(features)
    assert len(daily) == 312 * 97 and len(saved_features) == 312 * 90 == 25896 + 2184
    saved = saved_features.sort_values(keys).reset_index(drop=True)
    rebuilt = notebook_ns['make_features'](daily).dropna(subset=features).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved[keys], rebuilt[keys], check_dtype=False)
    assert np.allclose(saved[features + ['sale_amount']], rebuilt[features + ['sale_amount']], rtol=0, atol=1e-12)
    independent = independent_history_features(daily).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved[keys], independent[keys], check_dtype=False)
    assert np.allclose(saved[HISTORY], independent[HISTORY], rtol=1e-12, atol=1e-12)


def test_saved_features_ignore_outcomes_from_each_evaluation_window_onwards(notebook_ns, protocol, daily):
    """Training rows of a fold, and the first day of its evaluation week, see nothing from that week."""
    make_features = notebook_ns['make_features']
    features = protocol['features']['full']
    reference = make_features(daily)
    starts = [fold['valid_start'] for fold in protocol['folds']] + [protocol['test_period'][0]]
    assert len(starts) == 6
    for start in map(pd.Timestamp, starts):
        changed = daily.copy()
        changed.loc[changed.dt.ge(start), ['sale_amount', 'stock_hour6_22_cnt']] = 1234567.0
        rebuilt = make_features(changed)
        pd.testing.assert_frame_equal(reference.loc[reference.dt.le(start), features],
                                      rebuilt.loc[rebuilt.dt.le(start), features])


def test_baseline_predictions_are_the_lagged_features(main_dir, saved_features):
    keys = ['store_id', 'product_id', 'dt']
    predictions = pd.read_csv(main_dir / 'test_predictions.csv', parse_dates=['dt']).sort_values(keys).reset_index(drop=True)
    final_week = saved_features.loc[saved_features.source_split.eq('eval')].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions[keys], final_week[keys], check_dtype=False)
    assert np.allclose(predictions.sale_amount, final_week.sale_amount, rtol=0, atol=1e-12)
    assert np.allclose(predictions['7-day average baseline'], final_week.sales_mean7, rtol=0, atol=1e-12)
    assert np.allclose(predictions['Same-weekday baseline'], final_week.sales_lag7, rtol=0, atol=1e-12)
