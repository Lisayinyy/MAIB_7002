# Group Q A1 poster

The editable source is `scripts/build_v2_poster.py`. It builds a single-page, portrait **594 × 841 mm** PDF using ReportLab, embedded Bitstream Vera fonts and vector graphics. The layout uses horizontal sections and ample reading space, rather than a grid of cards.

The generator reads verified V2 evidence from `results/v2/comparison.csv`, `selection.json`, `policy_summary.json`, `features.parquet`, `recommendations.csv`, `discount_response.csv`, and `experiment_plan.json`. It chooses the first recommended-discount case by store, SKU and date, with an explicit first-row fallback when no recommendation exists. Missing metrics or a missing real worked example stop the build. It also includes the frozen transfer check from `results/v2_transfer/` when present. It does not train models, select them by test score, or invent poster numbers.

After the analysis has completed:

```bash
python -m pip install reportlab==4.4.9
python scripts/build_v2_poster.py
```

Run the command in the project environment, which already needs pandas and pyarrow for reading the actual feature data. The PDF is `poster/freshretail_group_q_A1.pdf`. The generator records source hashes and page specifications in `poster/build_manifest.json`. The final `freshretail_group_q_A1_preview.png` is a 2,000-pixel preview; the PDF itself contains vector text and charts suitable for A1 printing. `poster_text.txt` is an extracted text copy for proofreading. `verification.json` records the latest format, metric and rendered-layout checks. Re-render after any source or result change; an earlier verification does not apply to a changed PDF.

The poster covers the business question, public data, chronological evaluation, forecast-to-decision pipeline, a model comparison, a real-data what-if example, the failed transfer improvement and the difference between prediction evidence and demonstrated business effects. The poster is supported by the explanatory notebook; it is not a substitute for each member's required contribution and AI-use statement.

No poster has been submitted or sent for printing by this script.
