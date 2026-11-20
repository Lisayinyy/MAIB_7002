"""Regression tests: headline numbers recomputed from the saved row-level predictions.

Reference values are those of the verified Docker run of 7 October 2026 (README.md,
docs/PROJECT_SUMMARY.md). A deliberate rerun that changes them should update the
constants here together with the documents that quote them.
"""
from collections import defaultdict
import csv
import json
import math
import re

import pytest

import check_saved_results as packaged

MODEL = dict(zip(packaged.ALIASES, packaged.MODELS))
HEADLINE = {
    # method: (mean validation MAE, final-week MAE, final-week RMSE)
    'Random Forest': (0.3037155824, 0.2929491111, 0.4626890200),
    'CatBoost': (0.3037864543, 0.2877972205, 0.4519051339),
    'RF + CatBoost (50/50)': (0.3014047338, 0.2872878797, 0.4528770432),
}
BASELINE_FINAL_MAE = 0.3141849817
BASELINE_VALIDATION_MAE = 0.3283966510
TOLERANCE = 1e-9


def read(path):
    return packaged.rows(path)


@pytest.fixture(scope='module')
def final_week(main_dir):
    return read(main_dir / 'test_predictions.csv')


@pytest.fixture(scope='module')
def validation(ensemble_dir):
    folds = defaultdict(list)
    for row in read(ensemble_dir / 'validation_predictions.csv'):
        folds[row['fold']].append(row)
    return folds


def blend(row, weights, columns):
    return sum(weight * float(row[column]) for weight, column in zip(weights, columns) if weight)


def validation_mae(validation, weights):
    per_fold = [packaged.metrics(rows, [blend(r, weights, packaged.ALIASES) for r in rows])['MAE']
                for rows in validation.values()]
    assert len(per_fold) == 5
    return math.fsum(per_fold) / 5


def final_scores(final_week, weights):
    columns = ['full::' + name for name in packaged.MODELS]
    return packaged.metrics(final_week, [blend(r, weights, columns) for r in final_week])


def weights_for(*parts):
    lookup = dict(parts)
    return [lookup.get(alias, 0.0) for alias in packaged.ALIASES]


def test_packaged_consistency_check_passes(capsys):
    packaged.main()
    assert capsys.readouterr().out.startswith(
        'PASS: 12 baseline/model/ablation score rows; 19 ensemble candidates; '
        '10,920 validation and 2,184 final-week predictions; winner=RF + CatBoost (50/50).')


@pytest.mark.parametrize('method, weights', [
    ('Random Forest', weights_for(('Random Forest', 1.0))),
    ('CatBoost', weights_for(('CatBoost', 1.0))),
    ('RF + CatBoost (50/50)', weights_for(('Random Forest', 0.5), ('CatBoost', 0.5))),
])
def test_headline_scores_recomputed_from_saved_predictions(method, weights, validation, final_week):
    expected_cv, expected_mae, expected_rmse = HEADLINE[method]
    assert validation_mae(validation, weights) == pytest.approx(expected_cv, abs=TOLERANCE)
    scores = final_scores(final_week, weights)
    assert scores['MAE'] == pytest.approx(expected_mae, abs=TOLERANCE)
    assert scores['RMSE'] == pytest.approx(expected_rmse, abs=TOLERANCE)


def test_baseline_and_relative_improvements(validation, final_week):
    baseline = packaged.metrics(final_week, [float(r['7-day average baseline']) for r in final_week])['MAE']
    assert baseline == pytest.approx(BASELINE_FINAL_MAE, abs=TOLERANCE)
    per_fold = [packaged.metrics(rows, [float(r['sales_mean7']) for r in rows])['MAE'] for rows in validation.values()]
    assert math.fsum(per_fold) / 5 == pytest.approx(BASELINE_VALIDATION_MAE, abs=TOLERANCE)
    improvement = lambda method: 100 * (1 - HEADLINE[method][1] / baseline)
    assert improvement('Random Forest') == pytest.approx(6.76, abs=0.005)           # selected main model
    assert improvement('RF + CatBoost (50/50)') == pytest.approx(8.56, abs=0.005)   # post-hoc supplement


def test_readme_headline_matches_saved_summary(root, ensemble_dir):
    readme = (root / 'README.md').read_text(encoding='utf-8')
    quoted = re.search(r'validation MAE \*\*([\d.]+)\*\*, final-week MAE \*\*([\d.]+)\*\*, '
                       r'final-week RMSE \*\*([\d.]+)\*\*', readme)
    assert quoted, 'README headline sentence changed; update this test with it'
    summary = json.loads((ensemble_dir / 'summary.json').read_text())
    saved = (summary['selected']['CV_mean_MAE'], summary['selected_final_week']['MAE'],
             summary['selected_final_week']['RMSE'])
    assert [float(value) for value in quoted.groups()] == [round(value, 5) for value in saved]
    assert 'post-hoc' in readme and summary['selected']['selection_uses_final_week'] is False


def test_project_summary_table_matches_saved_scores(root, main_dir):
    """docs/PROJECT_SUMMARY.md is what the portfolio site links to and quotes."""
    names = {'Seven-day average baseline': '7-day average baseline', 'Same-weekday baseline': 'Same-weekday baseline',
             'Random Forest Regressor — selected': 'Random Forest Regressor'}
    final = {r['Model']: r for r in read(main_dir / 'model_comparison_test.csv')}
    validation = {r['Model']: float(r['MAE']) for r in read(main_dir / 'chosen_parameters.csv') if r['feature_set'] == 'full'}
    for name in ['7-day average baseline', 'Same-weekday baseline']:
        folds = [float(r['MAE']) for r in read(main_dir / 'baseline_validation_folds.csv') if r['Model'] == name]
        validation[name] = math.fsum(folds) / len(folds)
    row = re.compile(r'^\| ([^|]+?) \| ([\d.*]+) \| ([\d.*]+) \| ([\d.*]+) \| ([\d.*]+)%\*{0,2} \|$', re.M)
    table = (root / 'docs/PROJECT_SUMMARY.md').read_text(encoding='utf-8')
    checked = set()
    for label, cv, mae, rmse, wape in row.findall(table):
        label = label.strip('* ')
        model = names.get(label, label)
        if model not in final:
            continue
        number = lambda text: float(text.strip('*'))
        assert number(cv) == round(validation[model], 6)
        assert number(mae) == round(float(final[model]['MAE']), 6)
        assert number(rmse) == round(float(final[model]['RMSE']), 6)
        assert number(wape) == round(float(final[model]['WAPE_pct']), 4)
        checked.add(model)
    assert checked == set(final) and len(checked) == 7


def test_ensemble_candidates_and_validation_only_selection(ensemble_dir, validation, final_week):
    cv = read(ensemble_dir / 'validation_scores.csv')
    final = {r['Method']: r for r in read(ensemble_dir / 'final_week_scores.csv')}
    plan = json.loads((ensemble_dir / 'exploration_plan.json').read_text())
    weights = {r['Method']: json.loads(r['weights']) for r in cv}
    assert weights == plan['methods'] and plan['weight_column_order'] == packaged.ALIASES
    assert len(weights) == 19 == len({tuple(w) for w in weights.values()})
    for method, vector in weights.items():
        assert len(vector) == 5 and min(vector) >= 0 and math.isclose(sum(vector), 1.0)
    for alias in packaged.ALIASES:  # the five single models are candidates too
        assert weights[alias] == weights_for((alias, 1.0))
    recomputed = {method: validation_mae(validation, vector) for method, vector in weights.items()}
    winner = min(recomputed, key=recomputed.get)
    assert winner == 'RF + CatBoost (50/50)'
    assert [m for m, r in final.items() if r['selected_by_validation'] == 'True'] == [winner]
    for method, value in recomputed.items():
        assert float(final[method]['CV_MAE']) == pytest.approx(value, rel=1e-9)
    # Selection follows validation even though another candidate looks better on the final week.
    final_mae = {method: final_scores(final_week, vector)['MAE'] for method, vector in weights.items()}
    assert min(final_mae, key=final_mae.get) != winner
    assert final_mae['RF + HistGBR + CatBoost (equal)'] < final_mae[winner]


def test_main_model_selection_is_the_validation_minimum_not_the_final_week_minimum(main_dir, final_week):
    chosen = {r['Model']: float(r['MAE']) for r in read(main_dir / 'chosen_parameters.csv') if r['feature_set'] == 'full'}
    selected = json.loads((main_dir / 'selection_before_test.json').read_text())['selected_by_validation']
    assert selected == min(chosen, key=chosen.get) == MODEL['Random Forest']
    final_mae = {name: packaged.metrics(final_week, [float(r['full::' + name]) for r in final_week])['MAE']
                 for name in packaged.MODELS}
    assert min(final_mae, key=final_mae.get) == MODEL['CatBoost']  # reported, but not used for selection


def test_validation_candidates_average_the_150_trials(main_dir):
    trials = read(main_dir / 'validation_trials.csv')
    assert len(trials) == 2 * 5 * 3 * 5
    grouped = defaultdict(list)
    for trial in trials:
        grouped[(trial['feature_set'], trial['Model'], trial['candidate'])].append(float(trial['MAE']))
    assert len(grouped) == 30 and all(len(values) == 5 for values in grouped.values())
    means = {key: math.fsum(values) / 5 for key, values in grouped.items()}
    for row in read(main_dir / 'chosen_parameters.csv'):
        candidates = {key[2]: value for key, value in means.items() if key[:2] == (row['feature_set'], row['Model'])}
        best = min(sorted(candidates, key=int), key=candidates.get)   # ties follow candidate order
        assert row['candidate'] == best
        assert float(row['MAE']) == pytest.approx(candidates[best], rel=1e-12)


def test_final_week_discount_bands_and_saved_slice_scores(main_dir, notebook_ns, final_week):
    import pandas as pd

    none, moderate, deep = notebook_ns['BANDS']
    bands = notebook_ns['discount_bands'](pd.Series([float(r['discount']) for r in final_week])).astype(str).tolist()
    assert bands == [r['discount_band'] for r in final_week]
    assert [bands.count(label) for label in (none, moderate, deep)] == [1186, 757, 241]
    saved = {(r['slice'], r['Model']): r for r in read(main_dir / 'test_slice_scores.csv') if r['dimension'] == 'discount_band'}
    for label in (none, moderate, deep):
        rows = [r for r, band in zip(final_week, bands) if band == label]
        scores = packaged.metrics(rows, [float(r['full::' + MODEL['Random Forest']]) for r in rows])
        packaged.compare(scores, saved[(label, MODEL['Random Forest'])])
