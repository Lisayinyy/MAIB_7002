# A1 exhibition poster

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
