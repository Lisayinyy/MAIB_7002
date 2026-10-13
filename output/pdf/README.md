# A1 exhibition poster

## Current choice: revised Fresh Ensemble design

Print **MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf** at actual size:
**594 × 841 mm**, portrait, one page. Its matching PNG is a preview.
This revision uses the group's supplied **Fresh Food. Smarter Discounts.**
design: the original title, fruit artwork and author order are preserved.
The original header is stored in `assets/poster/`, so rebuilding does not depend
on the WeChat cache or on a flattened screenshot.

The story follows the final `pricing-ml-ap` notebook at `a7e26dd`:

1. **Start with the manager's trade-off.** Unsold perishables and unnecessary
   markdowns motivate tomorrow's discount decision.
2. **Choose histories with price changes.** Public FreshRetailNet records contain
   the sales, discount and stockout information needed. The stated filters reduce
   50,000 histories to 16,684 eligible ones; the top five stores supply 312.
3. **Test assumptions before adding complexity.** Removing weather reduces
   equal-weight kNN validation MAE from 0.3267 to 0.3120 at k=25.
4. **Compare the forecasts.** Random Forest and CatBoost form a simple 50/50
   ensemble. Its final-week MAE is 0.2892 versus the baseline's 0.3142 (7.9% lower);
   CatBoost alone is almost tied. Equal weights are a simplicity choice on a
   flat validation curve, not the exact minimum.
5. **Discover the decision problem.** Maximizing predicted sales times price rate,
   even within supported candidates, recommends discounts on 98.3% of cases.
6. **Test explicit stock rules.** The recommendation share becomes 55.0%. This
   change reflects the assumed rules, not demonstrated business improvement.
   A 5%-off worked example makes the final recommendation concrete.

The two recommendation-rate rings use the same 2,184 store-product-days and
zero assumed unit cost. `stockout_hours_t` describes the current/latest observed
sales day; the recommendation is for tomorrow. Stockout records do not establish
inventory age or freshness.

The figures are drawn from saved experiment results. This poster revision did
not retrain the models or change the final project's pricing code. The editorial
sequence was checked against notebook cells 0-5, 14-20, 38-46 and 54-62.

Source hashes and derived values are in `fresh_ensemble_revision_evidence.json`.
The old body is removed from the source PDF stream, so obsolete scores are not
left hidden behind new graphics. Header pixels match the supplied design.
The revised PDF was checked for A1 size, embedded fonts, final-model results,
text overflow and printing margins.

Rebuild from the repository root (verified with ReportLab 4.4.9, pypdf 6.10.0):

```sh
uv run --with reportlab==4.4.9 --with pypdf==6.10.0 python scripts/revise_fresh_ensemble_poster.py
```

The body uses Arial if available, otherwise ReportLab Vera. The generated layout is checked before publication. The original header keeps its embedded fonts and original artwork.

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
