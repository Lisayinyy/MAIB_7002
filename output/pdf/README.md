# A1 exhibition poster

## Current choice: revised Fresh Ensemble design

Print **MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf** at actual size:
**594 × 841 mm**, portrait, one page. Its matching PNG is a preview.
This revision uses the group's supplied **Fresh Food. Smarter Discounts.**
design: the original title, fruit artwork and author order are preserved.
The original header is stored in `assets/poster/`, so rebuilding does not depend
on the WeChat cache or on a flattened screenshot.

The main body integrates **key figures 02, 03 and 04** into three large,
sequential panels. Charts are redrawn as vector objects for this A1 layout;
complete landscape figure pages are not shrunk into screenshots. The optional
weather/tuning figure is omitted, and sample selection plus the worked case are
compact opening and closing elements.

The story follows the final `pricing-ml-ap` notebook at `a7e26dd`:

1. **Business question and data.** Help a category manager choose tomorrow's
   discount. FreshRetailNet provides daily sales, discount and stockout records.
   The filters leave 16,684 of 50,000 histories; the five stores with the most
   qualifying histories provide 312.
2. **A better forecast, with difficult cases.** Baseline-to-ensemble MAE bars show
   a 7.9% improvement against the seven-day average. CatBoost alone is nearly
   tied with the equal-weight Random Forest + CatBoost ensemble. A second chart
   shows larger absolute errors on deep-discount days, which also have higher
   average sales. All scores describe the reused final-week benchmark.
3. **Bigger predicted gains, thinner evidence.** Paired plots show the predicted
   response at eight candidate discounts and the proportion of histories with
   historical support. At 40% off, predicted sales are 2.32 times the full-price
   forecast, but only 15 of 312 histories (4.8%) support this rate.
4. **Rules change the recommendation.** A three-step rule diagram and two 100%
   stacked bars show how stock assumptions change the frequency and depth of
   discount recommendations: 98.3% with model + support, 55.0% with stock rules.
5. **Concrete output.** For store 18 / product 11 on 26 June, 5% off gives the
   highest predicted value among eligible candidates. This is a model scenario,
   not an observed gain from an intervention.

Response ratios compare mean predictions over the same **2,184 cases**.
Support uses a different denominator: **312 store-product histories**, requiring
at least three training days within 0.025 of a candidate price rate.
Recommendation shares also use the same 2,184 cases. The value proxy is predicted
normalized sales times price rate, with zero assumed unit cost; it is not currency.

Stockout today of at least one hour takes priority and forces full price tomorrow.
Otherwise, no stockout in the last seven days requires a supported discount when
one is available; remaining cases use the model's highest eligible predicted
value. These rules assume a stock policy. Stockout records do not establish age
or freshness. No causal sales, revenue or waste improvement has been demonstrated.

This revision reads saved experimental evidence; it does not retrain models,
change the final notebook, or modify its pricing rules. The cohort was selected
retrospectively and the final evaluation week was reused during development.
Source hashes and derived values are in `fresh_ensemble_revision_evidence.json`.
The old body is removed from the source PDF stream, so obsolete scores are not
left hidden behind new graphics. The original header, title, fruit art and
member order remain unchanged.

Rebuild from the repository root (verified with ReportLab 4.4.9, pypdf 6.10.0):

```sh
uv run --no-project --with reportlab==4.4.9 --with pypdf==6.10.0 python scripts/revise_fresh_ensemble_poster.py
pdftoppm -scale-to 2600 -singlefile -png output/pdf/MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf output/pdf/MAIB7002_GroupQ_A1_Fresh_Ensemble_revised_preview
```

The body uses Arial if available, otherwise ReportLab Vera. Check the rendered
PDF after rebuilding on a different system. The original header retains its
embedded fonts. The delivered version is checked for A1 size, visible embedded
fonts, text collisions, printing margins, source numbers and QR decoding.

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
