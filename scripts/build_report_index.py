#!/usr/bin/env python3
"""Build an offline report landing page from saved experiment tables."""
from pathlib import Path
import csv
import html
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def table(records, columns):
    header = ''.join(f'<th>{html.escape(label)}</th>' for key, label in columns)
    body = []
    for row in records:
        cells = []
        for key, label in columns:
            value = row[key]
            if key in {'MAE', 'RMSE', 'CV_MAE'}:
                value = f'{float(value):.5f}'
            cells.append(f'<td>{html.escape(str(value))}</td>')
        body.append('<tr>'+''.join(cells)+'</tr>')
    return '<div class="scroll"><table><thead><tr>'+header+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'


def main():
    models = read_csv(OUT / 'final_protocol/model_comparison_test.csv')
    ensembles = read_csv(OUT / 'ensemble_exploration/final_week_scores.csv')
    summary = json.loads((OUT / 'ensemble_exploration/summary.json').read_text())
    visible = {'RF + CatBoost (50/50)', 'RF + HistGBR + CatBoost (equal)', 'All five models (equal)', summary['selected']['method']}
    ensembles = [r for r in ensembles if r['Method'] in visible]
    winner = html.escape(summary['selected']['method'])
    formula = html.escape(' + '.join(f'{weight:g} × {name} prediction' for name, weight in summary['selected']['weights'].items() if weight))
    model_table = table(models, [('Model','Model / baseline'), ('MAE','Final-week MAE'), ('RMSE','Final-week RMSE')])
    ensemble_table = table(ensembles, [('Method','Combination'), ('CV_MAE','Validation MAE'), ('MAE','Final-week MAE'), ('RMSE','Final-week RMSE')])
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FreshRetail — project results</title>
<style>body{margin:0;background:#f4f6f8;color:#182a31;font:17px/1.65 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:48px 24px 72px}h1{font-size:clamp(30px,5vw,48px);line-height:1.12;margin:12px 0 22px}h2{margin-top:38px;font-size:24px}a{color:#006552}.tag{color:#006552;font-weight:700;font-size:13px;letter-spacing:.1em;text-transform:uppercase}.lead{max-width:780px;font-size:20px}.card{background:white;padding:24px;border-radius:14px;margin:22px 0;box-shadow:0 1px 4px #0001}.button{display:inline-block;padding:11px 18px;border-radius:7px;background:#006552;color:white;text-decoration:none;margin:6px 12px 6px 0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:13px 10px;border-bottom:1px solid #dfe7e6}th{background:#eef5f3}code{font-size:14px;background:#edf2f4;padding:3px 5px;border-radius:4px}.note{border-left:4px solid #bf8531;padding:12px 18px;background:#fff5e5}small{color:#53666e}li{margin:8px 0}</style></head><body><main>
<div class="tag">HKU · MAIB 7002 · FreshRetailNet-50K</div>
<h1>Daily sales forecasts<br>for discount decisions</h1>
<p class="lead">Compare tomorrow’s discount options for perishable products using observed-sales forecasts and a configurable sales-value constraint.</p>
<a class="button" href="final_protocol/01_freshretail_project.html">Open the complete notebook report</a>
<a href="https://github.com/Lisayinyy/MAIB_7002">Source code &amp; Docker instructions</a>
<div class="card"><strong>312 store–SKU series · 10 numeric features · 5 validation weeks</strong><p>Public observed data only. Final-period benchmark: 26 June–2 July 2024 (2,184 rows). Metrics use normalized sales, not currency or item counts.</p></div>
<h2>Five models and two baselines</h2><p>Random Forest is the original model selected by mean validation MAE. CatBoost has the lowest final-week error among the five individual models.</p>'''+model_table+'''
<h2>Recommended exploratory ensemble</h2><div class="card"><strong>'''+winner+'''</strong><p>Prediction = '''+formula+'''.</p><p>The ensemble is selected by the lowest mean validation MAE among the combinations tested. Final-week errors are reported separately and do not choose the weights.</p></div>'''+ensemble_table+'''
<p class="note">This ensemble is a post-hoc supplementary experiment after the final week had already been viewed. The same validation folds also tuned the base models. Its small improvement still needs confirmation on new dates.</p>
<h2>From prediction to a discount scenario</h2><p>The existing Random Forest demonstration compares historically supported candidate discounts, including full price, and maximizes predicted sales subject to a 5% sales-value-proxy loss limit. This is a hypothetical decision rule; the packaged scenario has not been rebuilt as an ensemble policy.</p>
<p>Historical discounts are not randomized. Forecast accuracy does not establish causal sales lift, actual revenue gains, or reduced food waste.</p>
<h2>Explore or reproduce</h2><ul><li>The complete report above opens immediately using saved outputs.</li><li>Open Jupyter: run <code>docker compose exec lab python scripts/jupyter_url.py</code> and use the printed local login link.</li><li>Check saved scores: <code>docker compose exec lab python scripts/check_saved_results.py</code>.</li><li>Rerun data checks, training, ensembles and reports: <code>docker compose exec lab python scripts/reproduce.py</code>.</li></ul>
<p><a href="final_protocol/model_comparison_test.csv">Model scores (CSV)</a> · <a href="ensemble_exploration/final_week_scores.csv">Ensemble scores (CSV)</a> · <a href="final_protocol/locked_protocol.json">Frozen protocol</a></p>
<small>Data: Dingdong-Inc, <a href="https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K">FreshRetailNet-50K</a>, CC BY 4.0. Cohort and features are derived from the pinned public dataset. Source revision and hashes are in the frozen protocol.</small>
</main></body></html>'''
    (OUT/'index.html').write_text(page, encoding='utf-8')
    print('Built outputs/index.html from saved score tables.')


if __name__ == '__main__':
    main()
