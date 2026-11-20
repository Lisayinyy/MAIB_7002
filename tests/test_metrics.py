"""MAE / MSE / RMSE / WAPE maths in the four places the project computes them."""
import math

import numpy as np
import pytest

import check_saved_results as packaged
import verify_final_protocol as audit
import verify_teammate_baselines as teammate

TRUTH = [1.0, 2.0, 3.0, 4.0]
PREDICTIONS = [1.0, 1.0, 5.0, 4.0]  # errors 0, -1, +2, 0


def records(truth):
    return [{'sale_amount': repr(float(value))} for value in truth]


def test_packaged_metrics_match_hand_calculation():
    assert packaged.metrics(records(TRUTH), PREDICTIONS) == pytest.approx(
        {'MAE': 0.75, 'MSE': 1.25, 'RMSE': math.sqrt(1.25), 'WAPE_pct': 30.0})


@pytest.mark.parametrize('bad', [
    [1.0, -0.1, 5.0, 4.0],            # predictions are clipped at zero before scoring
    [1.0, float('nan'), 5.0, 4.0],
    [1.0, float('inf'), 5.0, 4.0],
    [1.0, 2.0],                       # row count differs from the truth
])
def test_packaged_metrics_reject_invalid_predictions(bad):
    with pytest.raises(AssertionError):
        packaged.metrics(records(TRUTH), bad)


def test_packaged_metrics_reject_empty_input():
    with pytest.raises(AssertionError):
        packaged.metrics([], [])


def test_all_metric_implementations_agree(notebook_ns):
    rng = np.random.default_rng(7002)
    truth = rng.gamma(2.0, 0.5, size=500)
    predictions = np.maximum(0.0, truth + rng.normal(0.0, 0.3, size=500))
    reference = packaged.metrics(records(truth.tolist()), predictions.tolist())
    notebook = notebook_ns['scores'](truth, predictions)
    audited = audit.metrics(truth, predictions)
    baseline = teammate.scores(truth, predictions)
    for key in ['MAE', 'MSE', 'RMSE']:
        for other in (notebook, audited, baseline):
            assert other[key] == pytest.approx(reference[key], rel=1e-12)
    for value in (notebook['WAPE_pct'], audited['WAPE_percent'], baseline['WAPE_percent']):
        assert value == pytest.approx(reference['WAPE_pct'], rel=1e-12)
    assert notebook['n'] == baseline['n'] == 500
    assert reference['RMSE'] == pytest.approx(math.sqrt(reference['MSE']))
    assert reference['WAPE_pct'] == pytest.approx(100 * reference['MAE'] * 500 / truth.sum())


def test_notebook_scores_zero_sales_has_undefined_wape(notebook_ns):
    result = notebook_ns['scores']([0.0, 0.0], [0.0, 1.0])
    assert result['MAE'] == 0.5 and math.isnan(result['WAPE_pct'])


@pytest.mark.parametrize('truth, predictions', [
    ([1.0, 2.0], [1.0]),
    ([1.0, 2.0], [1.0, float('nan')]),
    ([], []),
])
def test_notebook_scores_reject_invalid_input(notebook_ns, truth, predictions):
    with pytest.raises(AssertionError):
        notebook_ns['scores'](truth, predictions)


def test_notebook_predict_clips_at_zero(notebook_ns):
    import pandas as pd

    class Model:
        def predict(self, frame):
            return frame['x'].to_numpy() - 1.0

    frame = pd.DataFrame({'x': [0.0, 1.0, 2.5]})
    assert notebook_ns['predict'](Model(), frame, ['x']).tolist() == [0.0, 0.0, 1.5]


def test_compare_accepts_rounding_noise_but_not_real_differences():
    packaged.compare({'MAE': 0.3}, {'MAE': '0.3'})
    packaged.compare({'MAE': 0.3}, {'MAE': repr(0.3 * (1 + 5e-9))})
    with pytest.raises(AssertionError):
        packaged.compare({'MAE': 0.3}, {'MAE': '0.300001'})


def test_check_keys_detects_duplicates_and_wrong_counts():
    rows = [{'store_id': '1', 'product_id': '2', 'dt': '2024-06-26'},
            {'store_id': '1', 'product_id': '2', 'dt': '2024-06-27'}]
    packaged.check_keys(rows, 2)
    with pytest.raises(AssertionError):
        packaged.check_keys(rows, 3)
    with pytest.raises(AssertionError):
        packaged.check_keys(rows + rows[:1], 3)
