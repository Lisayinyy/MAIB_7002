"""The illustrative discount rule and the guards of the verification scripts."""
import json
import subprocess
import sys

import pandas as pd
import pytest

import check_saved_results as packaged
import verify_final_protocol as audit


class LinearModel:
    """Predicted sales = intercept + slope * discount rate (rate 1.0 = full price)."""

    def __init__(self, intercept, slope):
        self.intercept, self.slope = intercept, slope

    def predict(self, frame):
        return (self.intercept + self.slope * frame['discount']).to_numpy()


@pytest.fixture
def scenario(notebook_ns):
    target = pd.Timestamp('2024-06-26')
    context = {name: 0.0 for name in notebook_ns['FEATURES']}
    context.update(store_id=18, product_id=11, dt=target, activity_flag=0, discount=1.0)
    rates = [1.0] * 4 + [0.9] * 3 + [0.8] * 3 + [0.7] * 2      # 0.70 has only two supporting days
    history = pd.DataFrame({'store_id': 18, 'product_id': 11, 'activity_flag': 0, 'discount': rates,
                            'dt': [target - pd.Timedelta(days=i + 1) for i in range(len(rates))]})
    # Days that must be ignored: other activity state, the target date and later, other store,
    # other product. The target-date row repeats 0.70: counting that day would give 0.70 a third day.
    past = [target - pd.Timedelta(days=i + 1) for i in range(3)]
    later = [target + pd.Timedelta(days=i) for i in range(3)]
    extra = pd.DataFrame({'store_id': [18] * 6 + [99] * 3 + [18] * 3, 'product_id': [11] * 9 + [12] * 3,
                          'activity_flag': [1] * 3 + [0] * 9,
                          'discount': [0.6] * 3 + [0.7, 0.5, 0.5] + [0.4] * 3 + [0.3] * 3,
                          'dt': past + later + past + past})
    return notebook_ns['recommend_discount'], context, pd.concat([history, extra], ignore_index=True)


def test_scenario_uses_only_supported_same_series_past_rates(scenario):
    recommend, context, history = scenario
    table, error = recommend(context, 1.0, history, LinearModel(2.0, -1.0))
    assert error is None and table.discount.tolist() == [0.8, 0.9, 1.0]
    assert table.supported_training_days.min() >= 3


def test_scenario_maximises_predicted_sales_within_the_proxy_limit(scenario):
    recommend, context, history = scenario
    # sales 1.2 / 1.1 / 1.0 -> proxy 0.96 / 0.99 / 1.00: every rate is within the 5% limit.
    table, _ = recommend(context, 1.0, history, LinearModel(2.0, -1.0))
    assert table.feasible.all() and table.loc[table.recommended, 'discount'].tolist() == [0.8]
    assert table.sales_value_proxy.tolist() == pytest.approx([0.96, 0.99, 1.0])
    # sales 1.1 / 1.05 / 1.0 -> proxy 0.88 / 0.945 / 1.00: only full price satisfies the limit.
    table, _ = recommend(context, 1.0, history, LinearModel(1.5, -0.5))
    assert table.feasible.tolist() == [False, False, True]
    assert table.loc[table.recommended, 'discount'].tolist() == [1.0]
    # A looser limit admits the deeper discounts again.
    table, _ = recommend(context, 1.0, history, LinearModel(1.5, -0.5), max_proxy_loss=0.15)
    assert table.loc[table.recommended, 'discount'].tolist() == [0.8]


def test_scenario_refuses_unsupported_requests(scenario):
    recommend, context, history = scenario
    table, error = recommend(context, 0.7, history, LinearModel(2.0, -1.0))     # reference has < 3 days
    assert table is None and 'Insufficient' in error
    table, error = recommend(context, 1.0, history.loc[history.discount.eq(1.0)], LinearModel(2.0, -1.0))
    assert table is None and 'Insufficient' in error
    table, error = recommend(context, 1.0, history, LinearModel(0.0, 0.0))      # zero reference proxy
    assert table is None and 'non-positive' in error
    with pytest.raises(ValueError):
        recommend(context, 1.0, history, LinearModel(2.0, -1.0), max_proxy_loss=1.5)


def test_saved_scenario_table_obeys_its_own_rule(main_dir):
    table = pd.read_csv(main_dir / 'illustrative_discount_scenarios.csv')
    meta = json.loads((main_dir / 'scenario_metadata.json').read_text())
    assert (table.sales_value_proxy - table.discount * table.predicted_sales).abs().max() < 1e-12
    reference = table.loc[table.discount.eq(meta['reference_discount']), 'sales_value_proxy'].iloc[0]
    assert table.feasible.eq(table.sales_value_proxy.ge((1 - meta['max_proxy_loss']) * reference - 1e-12)).all()
    best = table.loc[table.feasible].sort_values(['predicted_sales', 'discount'], ascending=[False, False]).iloc[0]
    assert table.loc[table.recommended, 'discount'].tolist() == [best.discount]
    assert table.supported_training_days.ge(3).all()


def test_full_audit_names_missing_prerequisites(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['verify_final_protocol.py', '--root', str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        audit.main()
    message = str(stopped.value)
    for name in ['data/raw/train.parquet', 'data/raw/eval.parquet', 'selected_forecast_model.joblib', 'reproduce.py']:
        assert name in message


def test_packaged_check_names_a_missing_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(packaged, 'MAIN', tmp_path)
    with pytest.raises(SystemExit, match='test_predictions.csv'):
        packaged.run()


def test_assert_based_checks_refuse_optimised_mode(root):
    """With -O every assert is skipped, so the scripts must not print PASS."""
    result = subprocess.run([sys.executable, '-O', str(root / 'scripts/check_saved_results.py')],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'PASS' not in result.stdout
    assert '-O' in result.stderr


@pytest.mark.parametrize('script', ['build_release_manifest.py', 'explore_ensembles.py'])
def test_assert_based_writers_refuse_optimised_mode(root, tmp_path, script):
    """Their checks are asserts too: under -O they would record a verified result without checking it."""
    copy = tmp_path / 'scripts' / script      # a copy, so a missing guard cannot touch the real outputs
    copy.parent.mkdir()
    copy.write_bytes((root / 'scripts' / script).read_bytes())
    result = subprocess.run([sys.executable, '-O', str(copy)], capture_output=True, text=True)
    assert result.returncode != 0 and 'PYTHONOPTIMIZE' in result.stderr
    assert list(tmp_path.rglob('*.json')) == []
