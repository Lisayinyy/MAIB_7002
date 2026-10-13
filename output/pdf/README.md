# A1 exhibition poster

## Current green poster with empirical notebook figures

Print **MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf** at actual size:
**594 x 841 mm**, portrait, one page. Its matching PNG is the preview.

The three-part structure is retained and now shows both project phases:

1. **Problem / Data Cleaning.** Short business context and a larger central
   question on the left; source choice, cohort funnel and cleaning on the right.
2. **Models and Phase One.** Five forecasting approaches: the seven-day mean
   baseline, kNN, Random Forest, CatBoost and their equal-weight ensemble.
   Four empirical plots explain the actual model selection. The final-week
   MAE comparison follows immediately below them.
3. **Results and Implications with Phase Two.** Stock rules determine whether
   to discount; candidate forecasts determine how much. A policy comparison
   and one product's sales/value curves show the resulting business decision.

The abstract neighbour/tree schematics are removed. Charts use actual saved
experiment data, reflowed with larger lettering for the poster. The builder
keeps vector geometry and embedded fonts. Original notebook plot PNGs are
retained in `assets/poster/experiments/notebook_cell_*_original.png` for visual
traceability. Poster plot PDFs, PNGs and editable SVGs are in the same directory.
The numeric provenance is in its `evidence.json`.

### Empirical figure sources

| Poster plot | Source in the final notebook | What it explains |
|---|---|---|
| kNN validation curve | Cell 19, `knn_stage1_summary.csv` | Equal-weight k=25 in a flat tuning region |
| RF depth/leaf heatmap | Cell 25, `rf_tuning_summary.csv` | The chosen depth 10 / minimum leaf 3; small grid differences |
| CatBoost learning curve | Cell 35, `catboost_learning_curves.csv` | Training versus validation error at lr=0.03 |
| RF/CatBoost weight curve | Cell 39, `ensemble_weight_sweep.csv` | 50/50 retained for simplicity near the lowest tested 60/40 result |
| Stock-rule recommendation mix | Cell 61, `recommendation_distribution.csv` | Supported model versus the default stock rules |
| One-product discount depth | Saved case from the same final ensemble | Why the supported 5%-off option has the highest predicted value |

The CatBoost view shows the source's lr=0.03 panel from trees 100-3,000. Its
zero-based logged iteration is displayed as the number of trees added. It is
one validation fold, whereas the locked 500-tree design was selected across
five folds. The RF colour scale covers only a 0.003 MAE range; its full numeric
grid is printed. The ensemble curve is also zoomed, as the actual differences
are small. No data is smoothed and no model is retrained for the poster.

### Business decision and evidence boundaries

Phase One predicts next-day normalized recorded sales using ten inputs.
Phase Two holds the historical context fixed and varies tomorrow's price rate
across 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70 and 0.60. Price rate 0.95 means
5% off. Supported rates require at least three training days within 0.025 for
that store-product. The decision compares predicted sales times price rate,
with zero assumed unit cost in the displayed default case. The code also
supports cost and minimum-gain business settings.

Stockout today of at least one hour forces full price tomorrow. Otherwise no
stockout in seven days requires a supported discount if available; remaining
cases choose the highest eligible predicted value. These policy assumptions
do not measure freshness or stock age. The same 2,184 cases underlie the two
recommendation distributions: 98.3% with model and support, 55.0% with stock
rules. These are recommendation rates, not observed business gains.

The example is store 18 / product 11 / 26 June 2024: a 5% discount predicts
sales 1.261 and value 1.198. The 20%-off candidate predicts more sales but less
value (1.136). Rates of 25%, 30% and 40% off are unsupported for this product.
This saved worked case extends the final notebook's Phase Two logic; it is
not an original plotted case in that notebook. Both its source and values
are identified in the chart evidence file.

All scores follow `pricing-ml-ap` at `a7e26dd`. The five approaches include
one baseline and one ensemble, rather than five independent fitted models.
CatBoost alone is nearly tied with the ensemble; 50/50 is a simple choice on
a flat validation curve, while 60/40 has the smallest measured validation MAE among the tested weights.

The cohort was selected retrospectively from the original training file. The
final evaluation period is 26 June-2 July 2024 and was reused during development.
Sales are normalized and stock-limited; the value proxy is not currency, revenue
or profit. No causal sales, revenue or waste gain is demonstrated. The manager
implication is to use the recommendation as decision support, set business
limits using inventory/cost information, and test it in a controlled store pilot.

### Design and rebuild

The title, member order, green palette and decorative produce asset are retained.
There is no QR code. The requested Figma skills guide spacing, hierarchy,
consistent styling and rendered verification; no Figma source node was supplied
or copied. The Design Report template was consulted as guidance only, following
the user's clarification to focus on the poster.

Plot rebuild (matplotlib 3.11.2, pandas 3.0.6):

```sh
uv run --no-project --with matplotlib==3.11.2 --with pandas==3.0.6 --with pillow --with pypdf==6.20.1 python scripts/build_poster_experiment_figures.py
```

Poster rebuild (ReportLab 4.4.9, pypdf 6.10.0):

```sh
uv run --no-project --with reportlab==4.4.9 --with pypdf==6.10.0 python scripts/revise_fresh_ensemble_poster.py
pdftoppm -scale-to 2800 -singlefile -png output/pdf/MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf output/pdf/MAIB7002_GroupQ_A1_Fresh_Ensemble_revised_preview
```

The body uses Arial if available, otherwise ReportLab Vera. The plot assets use
embedded DejaVu Sans. Check the rendered PDF when rebuilding on another system.
The delivered version is checked for A1 dimensions, all visible embedded fonts,
layout and chart collisions, evidence values, and no QR code.

## Earlier alternative: editorial layout

Print **freshretail_a1_poster.pdf** at actual size: **594 × 841 mm**, portrait,
one page. Text, charts and the QR code are vector objects. The PNG is a preview;
use the PDF for printing. Fonts used in the delivered PDF are embedded.

The poster follows the final `pricing-ml-ap` notebook at commit
[`a7e26dd`](https://github.com/Lisayinyy/MAIB_7002/blob/a7e26dd979af88b4c14f98a7f029f6a64bffaedf/discount_sales_prediction.ipynb).
Its model is the **50/50 Random Forest + CatBoost ensemble**. It does not use the
extended 25/75 model from an earlier branch.

## Content and evidence

- Public data and the five-store, 312-history cohort.
- Forecast inputs, how the two tree methods work, and price selection.
- Time-ordered evaluation, baseline comparison and ensemble-weight experiment.
- Discount recommendations before and after the assumed stock rules.
- A supported 5% discount example for store 18 / product 11 on 26 June 2024.
- Limitations and the next step: inventory/cost data and a controlled pilot.

Scores come from `results/test_summary.csv`; the weight experiment comes from
`results/ensemble_weight_sweep.csv`. The selection, policy and case values are
in `results/visuals/`. The final branch's model, scenario, data, configuration and
evaluation code was checked against the code that produced those CSVs and is
identical. Source-file hashes and the final PDF hash are in `poster_evidence.json`.

The cohort uses the full training period, and the final week was reused during
development. Results therefore describe this shared benchmark. Recommendations
and value comparisons are model scenarios, not measured causal improvements in
sales, revenue or food waste. The stock rules are assumptions, not inventory-age
measurements.

Group Q membership was checked against page 8 of the MAIB7002 project brief:
Aashish Omprakash Pareek, Yuanyuan Yin and Lingyu Chen.

## Optional rebuild

The supplied PDF is ready to print. To edit its content or layout, update
`scripts/build_a1_poster.py`, then run from the repository root:

```sh
uv run --with reportlab==4.4.9 python scripts/build_a1_poster.py
```

The builder reads saved evidence; it does not retrain models. It uses Arial and
Georgia when available on macOS, with ReportLab's bundled Vera fonts as a
portable fallback. Both layouts were rendered and checked. For a new preview,
if Poppler is installed:

```sh
pdftoppm -scale-to 2100 -singlefile -png output/pdf/freshretail_a1_poster.pdf output/pdf/freshretail_a1_poster_preview
```

## Reusable key-finding figures

`FreshRetail_key_findings_visuals.pdf` is a six-page figure pack, not another A1
poster. Five main figures cover cohort selection, forecast results and discount
errors, predicted response versus historical support, stock rules and discount
composition, and one supported product-day recommendation. The sixth is an
optional weather-removal / CatBoost learning-curve experiment.

Editable SVGs, 4800 x 2700 PNGs, an overview, English captions, Chinese speaking
notes and source hashes are in `output/visuals/key_findings/`. The complete asset
archive is `output/visuals/FreshRetail_key_visuals.zip`. Prioritize figures 02-04
for poster space; use 01 and 05 as compact opening and closing illustrations.
