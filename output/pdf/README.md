# A1 exhibition poster

## Current choice: three-part green poster

Print **MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf** at actual size:
**594 x 841 mm**, portrait, one page. Its matching PNG is the preview.

The revised structure follows the group's requested presentation order:

1. **Problem / Data Cleaning.** Problem and business trade-off on the left;
   dataset choice, cohort funnel and cleaning steps on the right.
2. **Models.** The largest section, with five approaches in order: seven-day
   mean baseline, kNN, Random Forest, CatBoost and their 50/50 ensemble. Each
   card explains the mechanism, uses a schematic diagram, and states the
   experiment's actual settings. The input-to-target flow and chronological
   evaluation plan connect the cards to the project.
3. **Results & Implications.** A five-method MAE/RMSE comparison, the 7.9% MAE
   reduction, recommendation rates before/after stock rules, and a 5%-off
   worked case. The final takeaway calls for a controlled store pilot.

There is **no QR code**. The green identity, title and author order are retained;
the header is smaller and uses new generated decorative produce artwork. The
artwork is not project data or a photograph of a participating store. Its PNG
and full generation prompt are in `assets/poster/fresh_produce_hero_v3.*` and
`assets/poster/fresh_produce_hero_v3_prompt.md`. The original header PDF is kept
as an archival asset but is no longer imported by the current builder.

### Evidence and scope

All scores and settings follow the final `pricing-ml-ap` notebook at `a7e26dd`.
The five approaches include one simple baseline and one ensemble, not five
independent fitted models. Random Forest uses 300 trees / depth 10 / minimum
leaf size 3; CatBoost uses 500 trees / depth 5 / learning rate 0.03; kNN uses
25 equally weighted neighbours after feature standardization. CatBoost alone
is almost tied with the ensemble; equal weights were retained for simplicity,
not because 50/50 was the exact validation minimum.

The sample is selected from 50,000 store-product histories using 15-85% days
with price rate below 0.95, at least six discount-state changes, and fewer than
20% zero-sales days. The top five stores by eligible-history count yield 312
histories. The cohort was selected retrospectively from the original training
file. Selection and data-cleaning rules are distinct.

Cleaning checks unique daily keys, missing fields and gaps; caps rates above 1;
masks sales and discounts on zero-price days; builds historical features from
prior observations; then removes rows lacking complete features. The poster
states the rules without inventing per-rule removal counts. There are 25,896
training/validation rows plus 2,184 final-week observations after the seven-day
history requirement.

Settings use five expanding validation weeks. Final-week metrics describe
26 June-2 July 2024, which was reused during development. These are normalized
recorded sales, not units, dollars or unconstrained demand. Recommendation
shares use the same 2,184 cases. The value proxy is predicted sales times price
rate, with zero assumed unit cost; it is not measured revenue or profit.

For the stock policy, a stockout of at least one hour today forces full price
tomorrow. Otherwise no stockout in seven days requires a supported discount if
available; remaining cases choose the highest eligible predicted value. A
supported rate has at least three training days within 0.025 of that rate.
The rules are assumptions: stockouts do not reveal age or freshness. The 5%-off
case is store 18 / product 11 / 26 June 2024. No causal sales, revenue or waste
gain has been demonstrated.

This poster revision changes presentation only; it does not retrain models or
modify the final notebook. Numeric source hashes and provenance are in
`fresh_ensemble_revision_evidence.json`. Model illustrations are explanatory
schematics, not visualizations of particular fitted trees or data samples.

### Rebuild

Verified with ReportLab 4.4.9 and pypdf 6.10.0. From the repository root:

```sh
uv run --no-project --with reportlab==4.4.9 --with pypdf==6.10.0 python scripts/revise_fresh_ensemble_poster.py
pdftoppm -scale-to 2600 -singlefile -png output/pdf/MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf output/pdf/MAIB7002_GroupQ_A1_Fresh_Ensemble_revised_preview
```

The body uses Arial if available, otherwise ReportLab Vera. Check the rendered
PDF when rebuilding on a different system. The delivered version is checked
for one A1 page, embedded visible fonts, no text collisions, correct evidence
values and no QR code. Its charts and text remain vector objects.

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
