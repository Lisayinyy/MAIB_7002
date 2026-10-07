#!/usr/bin/env python3
"""Recompute packaged scores from predictions; no raw data or ML packages needed.

This verifies artifact consistency, not model training or causal policy value.
For the full raw-data/model audit run verify_final_protocol.py after reproduction.
"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'outputs/final_protocol'
ENS = ROOT / 'outputs/ensemble_exploration'
ALIASES = ['Ridge', 'Decision Tree', 'Random Forest', 'HistGBR', 'CatBoost']
MODELS = ['Ridge Regression', 'Decision Tree Regressor', 'Random Forest Regressor',
          'HistGradientBoostingRegressor', 'CatBoost Regressor']


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def metrics(records, predictions):
    truth = [float(r['sale_amount']) for r in records]
    assert len(truth) == len(predictions) and truth
    assert all(math.isfinite(v) and v >= 0 for v in predictions)
    errors = [p-y for p, y in zip(predictions, truth)]
    mae = math.fsum(abs(e) for e in errors)/len(errors)
    mse = math.fsum(e*e for e in errors)/len(errors)
    return dict(MAE=mae, MSE=mse, RMSE=math.sqrt(mse),
                WAPE_pct=100*math.fsum(abs(e) for e in errors)/math.fsum(abs(y) for y in truth))


def compare(actual, expected):
    for key, value in actual.items():
        assert math.isclose(value, float(expected[key]), rel_tol=1e-8, abs_tol=1e-10), (key, value, expected[key])


def check_keys(records, expected_count):
    keys = [(r['store_id'], r['product_id'], r['dt']) for r in records]
    assert len(keys) == len(set(keys)) == expected_count


def main():
    test = rows(MAIN / 'test_predictions.csv')
    check_keys(test, 2184)
    base_scores = rows(MAIN / 'all_test_scores.csv')
    assert len(base_scores) == 12
    for result in base_scores:
        column = result['Model'] if result['feature_set'] == 'baseline' else result['feature_set']+'::'+result['Model']
        compare(metrics(test, [float(r[column]) for r in test]), result)

    oof = rows(ENS / 'validation_predictions.csv')
    check_keys(oof, 10920)
    folds = defaultdict(list)
    for r in oof:
        folds[r['fold']].append(r)
    assert len(folds) == 5 and all(len(f) == 2184 for f in folds.values())
    cv = rows(ENS / 'validation_scores.csv')
    assert len(cv) == 19 and len({r['Method'] for r in cv}) == 19
    results = {r['Method']: r for r in rows(ENS / 'final_week_scores.csv')}
    calculated_cv = {}
    for candidate in cv:
        weights = json.loads(candidate['weights'])
        assert len(weights) == 5 and all(w >= 0 for w in weights)
        assert math.isclose(sum(weights), 1)
        fold_metrics = []
        for f in folds.values():
            pred = [sum(w*float(r[m]) for w, m in zip(weights, ALIASES)) for r in f]
            fold_metrics.append(metrics(f, pred))
        mean = {k: math.fsum(f[k] for f in fold_metrics)/len(fold_metrics) for k in fold_metrics[0]}
        compare(mean, candidate)
        calculated_cv[candidate['Method']] = mean['MAE']
        pred = [sum(w*float(r['full::'+m]) for w, m in zip(weights, MODELS)) for r in test]
        compare(metrics(test, pred), results[candidate['Method']])

    selected = json.loads((ENS / 'selection_before_final_week.json').read_text())
    winner = min(calculated_cv, key=calculated_cv.get)
    assert winner == selected['method']
    assert selected['selection_uses_final_week'] is False
    selected_predictions = rows(ENS / 'selected_final_week_predictions.csv')
    check_keys(selected_predictions, 2184)
    indexed_test = {(r['store_id'], r['product_id'], r['dt']): r for r in test}
    for r in selected_predictions:
        source = indexed_test[(r['store_id'], r['product_id'], r['dt'])]
        assert math.isclose(float(r['sale_amount']), float(source['sale_amount']), rel_tol=1e-12, abs_tol=1e-12)
        expected = sum(selected['weights'][a]*float(source['full::'+m]) for a, m in zip(ALIASES, MODELS))
        assert math.isclose(float(r['selected_ensemble_prediction']), expected, rel_tol=1e-10, abs_tol=1e-12)
    compare(metrics(selected_predictions, [float(r['selected_ensemble_prediction']) for r in selected_predictions]), results[winner])
    print(f'PASS: {len(base_scores)} baseline/model/ablation score rows; {len(cv)} ensemble candidates; 10,920 validation and 2,184 final-week predictions; winner={winner}.')
    print('This is a saved-artifact consistency check, not a new training run or causal evaluation.')


if __name__ == '__main__':
    main()
