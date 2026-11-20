"""Shared fixtures for the offline test suite.

Every test reads only files committed to this repository: no raw-data download,
no model fit and no network access. The notebook's helper functions are taken
from the executed notebook itself (the same approach as
scripts/verify_final_protocol.py), so the tests exercise the code that produced
the saved results rather than a copy of it.
"""
from pathlib import Path
import ast
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

NOTEBOOK_CONSTANTS = {'KEYS', 'ROW_KEYS', 'BANDS', 'METRICS', 'FEATURES', 'FEATURE_SETS'}
NOTEBOOK_FUNCTIONS = {'scores', 'predict', 'discount_bands', 'make_features', 'recommend_discount'}


@pytest.fixture(scope='session')
def root():
    return ROOT


@pytest.fixture(scope='session')
def main_dir():
    return ROOT / 'outputs/final_protocol'


@pytest.fixture(scope='session')
def ensemble_dir():
    return ROOT / 'outputs/ensemble_exploration'


@pytest.fixture(scope='session')
def protocol(main_dir):
    return json.loads((main_dir / 'locked_protocol.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def notebook_ns():
    """Namespace holding the notebook's pure helpers and the constants they use."""
    import numpy as np
    import pandas as pd

    notebook = json.loads((ROOT / '01_freshretail_project.ipynb').read_text(encoding='utf-8'))
    code = '\n\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    wanted = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef) and node.name in NOTEBOOK_FUNCTIONS:
            wanted.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in NOTEBOOK_CONSTANTS):
            wanted.append(node)
    namespace = {'pd': pd, 'np': np}
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    exec(compile(module, '01_freshretail_project.ipynb', 'exec'), namespace)
    missing = (NOTEBOOK_CONSTANTS | NOTEBOOK_FUNCTIONS) - set(namespace)
    assert not missing, f'notebook no longer defines {sorted(missing)}'
    return namespace
