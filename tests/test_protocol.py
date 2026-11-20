"""Fold boundaries, cohort identity and pinned inputs, from the committed artifacts."""
import csv
import json
import re

import pandas as pd
import pytest

import download_public_data

KEYS = ['store_id', 'product_id', 'dt']


@pytest.fixture(scope='module')
def features(main_dir):
    return pd.read_parquet(main_dir / 'feature_frame.parquet')


def test_folds_are_expanding_weekly_and_strictly_chronological(main_dir, protocol, features):
    assert pd.read_csv(main_dir / 'folds.csv').to_dict('records') == protocol['folds']
    folds = pd.read_csv(main_dir / 'folds.csv', parse_dates=['train_start', 'train_end', 'valid_start', 'valid_end'])
    train = features.loc[features.source_split.eq('train')]
    final_week = features.loc[features.source_split.eq('eval')]
    assert folds.fold.tolist() == [1, 2, 3, 4, 5]
    for fold in folds.itertuples():
        assert fold.train_start == train.dt.min() == pd.Timestamp('2024-04-04')
        assert fold.train_end + pd.Timedelta(days=1) == fold.valid_start       # no gap, no overlap
        assert fold.valid_end - fold.valid_start == pd.Timedelta(days=6)        # one week
        assert fold.n_train == int(train.dt.lt(fold.valid_start).sum())
        assert fold.n_valid == int(train.dt.between(fold.valid_start, fold.valid_end).sum()) == 312 * 7
    assert (folds.valid_start.diff().dropna() == pd.Timedelta(days=7)).all()
    assert folds.n_train.is_monotonic_increasing
    assert folds.valid_end.max() == train.dt.max() == pd.Timestamp('2024-06-25')
    assert final_week.dt.min() == pd.Timestamp(protocol['test_period'][0]) == pd.Timestamp('2024-06-26')
    assert final_week.dt.max() == pd.Timestamp(protocol['test_period'][1])
    assert len(train) == 25896 and len(final_week) == 2184


def test_out_of_fold_predictions_cover_exactly_the_validation_weeks(ensemble_dir, protocol, features):
    oof = pd.read_csv(ensemble_dir / 'validation_predictions.csv', parse_dates=['dt'])
    assert len(oof) == 5 * 2184 and not oof.duplicated(KEYS).any()
    for fold in protocol['folds']:
        part = oof.loc[oof.fold.eq(fold['fold'])]
        assert part.dt.min() == pd.Timestamp(fold['valid_start']) and part.dt.max() == pd.Timestamp(fold['valid_end'])
    assert oof.dt.max() < pd.Timestamp(protocol['test_period'][0])          # no final-week row is used for weights
    merged = oof.merge(features[KEYS + ['sale_amount']], on=KEYS, suffixes=('', '_features'), validate='one_to_one')
    assert len(merged) == len(oof) and (merged.sale_amount - merged.sale_amount_features).abs().max() < 1e-12
    assert (oof[['Ridge', 'Decision Tree', 'Random Forest', 'HistGBR', 'CatBoost']] >= 0).all().all()


def test_cohort_is_the_same_312_series_everywhere(root, main_dir, features):
    series = ['store_id', 'product_id']
    teammate = pd.read_csv(root / 'data/metadata/teammate_selected_series.csv')[series]
    selected = pd.read_csv(main_dir / 'selected_series.csv')[series]
    as_set = lambda frame: set(map(tuple, frame[series].drop_duplicates().to_numpy().tolist()))
    assert len(teammate) == len(selected) == 312
    assert as_set(teammate) == as_set(selected) == as_set(features)
    assert sorted(features.store_id.unique().tolist()) == [18, 154, 182, 235, 343]
    assert features.product_id.nunique() == 122


def test_downloader_pins_match_the_locked_protocol(protocol):
    assert download_public_data.REVISION == protocol['data_revision']
    assert download_public_data.SOURCE_HASHES == protocol['source_hashes']
    assert protocol['seed'] == 7002


def test_requirement_pins_match_the_recorded_docker_environment(root, main_dir):
    normalise = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    recorded = dict(line.split('==') for line in
                    (main_dir / 'docker_python_packages.txt').read_text().split() if '==' in line)
    recorded = {normalise(name): version for name, version in recorded.items()}
    pins = {}
    for filename in ['requirements-models.txt', 'requirements-docker.txt']:
        for line in (root / filename).read_text().split():
            if '==' in line:
                name, version = line.split('==')
                pins[normalise(name)] = version
    assert len(pins) == 14
    assert {name: recorded.get(name) for name in pins} == pins
    constraints = dict(line.split('==') for line in (root / 'constraints-docker.txt').read_text().splitlines()
                       if line and not line.startswith('#'))
    assert {normalise(name): version for name, version in constraints.items()} == recorded
    versions = json.loads((main_dir / 'run_manifest.json').read_text())['versions']
    for name in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'pyarrow']:
        assert versions[name] == pins[name]


def test_release_manifest_still_describes_the_saved_results(root, main_dir):
    """The result files of the verified 7 October 2026 run have not been edited since."""
    import hashlib

    manifest = json.loads((main_dir / 'release_manifest.json').read_text())['files']
    results = {name: digest for name, digest in manifest.items() if name.startswith('outputs/')
               or name in {'01_freshretail_project.ipynb', 'data/metadata/teammate_selected_series.csv'}}
    assert len(results) >= 40
    changed = [name for name, digest in results.items()
               if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    assert changed == []
    audit = json.loads((main_dir / 'independent_verification.json').read_text())
    assert audit['status'] == 'PASS'
    assert audit['notebook_sha256'] == manifest['01_freshretail_project.ipynb']


def test_selection_records_do_not_reference_the_final_week(main_dir, ensemble_dir):
    main = json.loads((main_dir / 'selection_before_test.json').read_text())
    with (main_dir / 'chosen_parameters.csv').open(newline='') as handle:
        chosen = [row for row in csv.DictReader(handle) if row['feature_set'] == 'full']
    best = min(chosen, key=lambda row: float(row['MAE']))
    assert main['selected_by_validation'] == best['Model'] == 'Random Forest Regressor'
    supplement = json.loads((ensemble_dir / 'selection_before_final_week.json').read_text())
    assert supplement['selection_uses_final_week'] is False
    assert {name for name, weight in supplement['weights'].items() if weight} == {'Random Forest', 'CatBoost'}
