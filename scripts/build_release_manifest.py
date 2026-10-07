#!/usr/bin/env python3
"""Refresh the shareable artifact manifest after a verified complete run."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/final_protocol'


def main():
    audit = json.loads((OUT/'independent_verification.json').read_text())
    assert audit['status'] == 'PASS'
    notebook_path = ROOT/'01_freshretail_project.ipynb'
    assert audit['notebook_sha256'] == hashlib.sha256(notebook_path.read_bytes()).hexdigest(), 'Notebook changed since full verification; run verify_final_protocol.py again.'
    nb = json.loads(notebook_path.read_text())
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(code_cells) == audit['notebook_executed_code_cells']
    assert all(c.get('execution_count') is not None for c in code_cells)
    ensemble_audit = json.loads((ROOT/'outputs/ensemble_exploration/verification.json').read_text())
    source_hash = hashlib.sha256('\n'.join(''.join(c['source']) for c in nb['cells']).encode()).hexdigest()
    assert ensemble_audit['notebook_supplement_source_check']['source_sha256'] == source_hash
    assert ensemble_audit['base_model_fit_count'] == 25 and ensemble_audit['selection_matches_validation_minimum']
    manifest_path = OUT/'release_manifest.json'
    candidates = [ROOT/p for p in (
        '01_freshretail_project.ipynb', 'README.md', 'DATA_LICENSE.md',
        'Dockerfile', 'compose.yaml', '.dockerignore', '.gitignore', '.env.example',
        'requirements-models.txt', 'requirements-docker.txt', 'outputs/index.html',
        'data/metadata/teammate_selected_series.csv')]
    for folder in ['scripts', 'docs', '.github', 'outputs/final_protocol', 'outputs/ensemble_exploration']:
        candidates.extend((ROOT/folder).rglob('*'))
    files = {}
    for path in sorted(set(candidates)):
        if not path.is_file() or path == manifest_path or '__pycache__' in path.parts:
            continue
        if path.suffix in {'.joblib', '.log', '.pyc', '.pkl', '.pickle'}:
            continue
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': files,
        'raw_data_included': False,
        'model_binaries_included': False,
        'data_reproduction': 'python scripts/download_public_data.py',
        'complete_reproduction': 'python scripts/reproduce.py',
        'verification': {
            'independent_audit': audit['status'],
            'independent_audit_utc': audit['verified_utc'],
            'notebook_sha256': audit['notebook_sha256'],
            'ensemble_audit_utc': ensemble_audit['verified_utc'],
            'notebook_executed_code_cells': audit['notebook_executed_code_cells'],
            'selected_main_model': audit['selected_model'],
            'ensemble_selection': json.loads((ROOT/'outputs/ensemble_exploration/selection_before_final_week.json').read_text())['method'],
        },
        'scope': '312-series coursework experiment plus post-hoc ensemble supplement; no causal policy evaluation.',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Updated release manifest: {len(files)} shareable files.')


if __name__ == '__main__':
    main()
