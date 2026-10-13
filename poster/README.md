# Group Q A1 poster

The editable source is `scripts/build_v2_poster.py`. It creates one **594 x 841 mm portrait A1 PDF**, with embedded Bitstream Vera fonts and vector artwork. The 11 October revision uses a business question, a produce illustration, a cohort funnel, a four-step decision flow, a compact three-method chart and one real worked example. Main explanations are approximately 22-28 pt; references and supporting notes are smaller.

## Rebuild from saved evidence

Only the Python standard library and ReportLab are required; the poster does not import the model-training environment or refit a model.

```bash
python -m pip install reportlab==4.4.9
python scripts/build_v2_poster.py
```

The generator reads the saved V2 comparison, selection, policy summary, experiment plan, story evidence, row-level validation/test predictions, recommendation table and scenario table. The worked example is the first recommended-discount row ordered by store, SKU and date. Its previous-day observed sales come from the corresponding prior dated target in the saved predictions. The frozen transfer check is included when its saved evidence exists. Missing evidence stops generation rather than introducing invented values.

## Files

- `freshretail_group_q_A1.pdf`: the print-ready PDF.
- `freshretail_group_q_A1_preview.png`: a 2,200-pixel preview; print the PDF, not the PNG.
- `poster_text.txt`: extracted text for proofreading.
- `build_manifest.json`: PDF, generator and result-source hashes.
- `verification.json`: the latest page-size, text, font, source-consistency and rendered-layout checks.

Render and inspect after every change; an earlier verification does not apply to a changed PDF. The PDF must remain exactly one A1 page. Check that names, selected models, rounded scores, case values and limitations agree with the saved evidence. The poster's compact comparison covers the simple baseline, original 50/50 blend and selected 25/75 blend; the explanatory notebook retains all models and detailed experiments.

The poster does not claim observed gains from price changes. It includes the previously viewed benchmark, the failed cross-store improvement and the high frequency of discount recommendations. Each member must separately confirm their required contribution and generative-AI-use statement in the notebook/report.

No poster has been submitted or sent for printing by this script.
