# Bounded refinement of the coursework evidence

Prepared after inspecting the 47-cell narrative notebook, the canonical V2 source and saved plans/results, and the **original eight-page** `maib7002_project_brief.pdf` supplied by the user. The brief was read directly with pypdf using the bundled document runtime; the project dependencies were not changed.

## Decision and scope

Keep the recorded **25% teammate Random Forest + 75% extended CatBoost trained with MAE** design. The new work explains its ingredients and limitations. It must not select a new winner, retune against the already-viewed final week, or replace canonical results in `results/v2`.

The current notebook already has the business story, cohort funnel, decision-time inputs, weekly validation, a clean loss comparison, a worked discount case, an actual forest path, errors, transfer failure and a runnable demonstration. More model families or reinforcement learning would not address its main weaknesses. The useful additions are the following three bounded analyses.

## 1. Fixed-parameter, validation-only ablation

**Question:** Which changes explain predictive performance, and do proposed discount and previous-day activity contain information beyond the other inputs?

This is the most direct improvement to the technical-investigation requirement. The existing comparison between the teammate's base CatBoost and the richer model changes both features and regularization (`l2_leaf_reg=3` versus `5`). It cannot isolate a feature effect. The existing richer RMSE/MAE pair is already matched; reproducing it also provides an integrity check.

The authorized study uses seven CatBoost configurations on the same five expanding validation weeks, for **35 fits total**:

| Configuration | Inputs | Training loss | Purpose |
|---|---:|---|---|
| Base 10 | Original ten inputs | RMSE | Matched reference, with the same regularization as every row below |
| History 18 | Base + eight additional history features | RMSE | Isolate the history-feature package |
| Full 20 | History + store and product categories | RMSE | Isolate the identifier package |
| Full 20 | Same twenty inputs | MAE | Isolate the learning objective |
| No proposed discount | Full minus `discount_next` | MAE | Test the conditional predictive contribution of the proposed target-day price |
| No previous activity | Full minus `activity_t` | MAE | Test yesterday's activity flag |
| Neither field | Full minus both fields above | MAE | Check their combined conditional contribution |

Fixed settings: depth 5, 500 iterations, learning rate 0.03, L2 leaf regularization 5, random seed 0, four CPU threads, no early stopping or validation-target fitting. Store/product columns remain categorical where included. All configurations use the same nonnegative output clipping and original retrospective 312-series cohort.

For each fold, fitting ends strictly before that validation week starts. All seven configurations evaluate the same keyed rows. Historical inputs update with observed prior days as in the existing one-day-ahead protocol; models do not refit daily. The full feature file is filtered to dates before 26 June **before loading the analysis rows**. The script does not score the final week.

The experiment records its hypotheses, exact columns, settings, input/code hashes and output location before fitting. It saves all row-level forecasts, fold scores, paired contrasts and timing, and checks that canonical selection and source/result inputs were unchanged. `--verify-only` recomputes the saved scores without fitting models.

**Primary comparison:** equal-week validation MAE. Show every paired weekly difference, mean change and the count of improved weeks; report RMSE alongside it. A negative `changed − reference` error difference means improvement. A single mean should not conceal folds that move in opposite directions. The reused five weeks are development evidence, not five independent trials or a new untouched test.

**Interpretation boundaries:** removing `discount_next` tests its contribution given the remaining features, which still contain historical prices and sales. Removing `activity_t` tests yesterday's flag, not tomorrow's scheduled promotion. Correlation and redundancy can make a useful input appear unnecessary. A price ablation cannot identify causal elasticity or prove that changing a discount causes the forecasted sales gain. The standalone ablations must not be presented as ablations of the whole blend: its unchanged RF member still uses both base fields.

**Implementation:** `scripts/run_story_ablation.py`, with isolated artifacts under `results/story_ablation/`. Keep this table separate from the canonical 17-method selection table. The notebook should explain the hypotheses before revealing the new values, while acknowledging that this explanatory analysis was designed after prior development.

## 2. Decision-rule sensitivity using saved predictions

**Question:** Is the recommendation stable to reasonable illustrative business thresholds, and what trade-off do those thresholds impose on automatic coverage?

This directly addresses the largest unresolved business weakness. At the current defaults, 1,179 of 2,184 inputs recommend a discount; 967 require stockout review and 38 require model-disagreement review. Among the 1,217 cases without yesterday's stockout, **96.9% recommend a discount**. No current case selects keep-full-price. That is a limitation of the available evidence and the policy design, even though the outputs obey the declared constraints.

Use the saved eight-rate response curves, historical support counts and previous-day stockout signal. No model fits and no new target-outcome evaluation are needed. Declare a small **one-factor-at-a-time** set before calculating results: vary the minimum sales gain, minimum retained sales-value proxy, minimum nearby support count, near-tie tolerance and stockout-review cutoff. Hold the remaining defaults fixed in each scenario. Avoid a large threshold grid and do not choose a replacement default from the resulting table.

Report for each scenario:

- Recommendation, keep-full-price, manual-review and abstention counts, with all input cases retained.
- Automatic coverage and discount share, with denominators stated. If the stockout cutoff changes, also use a common fixed comparison subset so a changed denominator does not masquerade as a changed pricing effect.
- Share of recommendations changed relative to the default and the distribution of recommended price rates.
- Constraint-check counts; all accepted recommendations should satisfy that scenario's declared rules.

The precomputed support count uses the original ±0.025 price window. It supports changing the minimum number of days but not silently changing the window itself. Model agreement is not independent corroboration: the selected blend contains RF, and the reference models share data and related features. The stockout threshold is a review policy, not a measure of tomorrow's stock or inventory age.

**What this establishes:** how the decision procedure behaves under explicit assumptions. **What it cannot establish:** the best threshold, realized sales lift, profit, waste reduction, or the effect of a recommended action. Use a compact coverage/stability figure in the notebook; do not label it an outcome backtest. Root owns this implementation and its artifact path.

## 3. Decision-time EDA and explanation using pre-validation records

**Question:** What information makes this a forecastable but causally difficult discount problem?

Use records strictly before the first validation date, 22 May, for the new exploratory charts. Show normalized-sales scale and variability, observed price variation, availability constraints, and the exact time relationship between a raw row, shifted historical inputs and its next-day target. Include a small real feature calculation that can be explained aloud. These plots should motivate the already-declared design rather than retrospectively claim that its final-period performance justified the choices.

A descriptive sales-by-price plot must explicitly acknowledge confounding and stockout censoring. A difference between discounted and undiscounted days does not estimate an intervention effect. Product identifiers are anonymous; do not invent product names, expiry dates or stock ages.

The existing teammate tuning tables can demonstrate the bias/variance trade-off without rerunning or widening their search. Identify the actual parameter, metric and weekly protocol; distinguish a historical tuning result from the new matched ablations.

**Cohort caveat:** limiting new EDA to early records does not undo selection of the 312-series cohort using all 90 original training days. Preserve that disclosure. A genuinely prospective cohort study would freeze selection before the first validation week or repeat it inside each fold, and would constitute a separate benchmark. It is outside this bounded refinement. Likewise, the already-locked transfer failure is useful evidence and must remain visible; do not redraw its cohort or change the winner after seeing it.

## Rubric coverage and remaining gaps

The original PDF states that understanding and evidence matter more than model count or winning every comparison (pages 1–2). It requires a small worked example and an important technical choice investigated with other conditions fixed (page 1). The weighting and deliverables below come from page 3.

| Criterion | Weight | Existing evidence and bounded improvement | Remaining issue |
|---|---:|---|---|
| Business question and suitable data | 15% | Named manager, store-product-day unit, public observed data, candidate discount input and next-day target; add pre-validation EDA | Daily records support daily scenarios; no intraday bakery or measured-waste claim |
| Technical understanding | 30% | Tree path, live blend arithmetic, defined losses/features; add controlled history/ID/loss/action-field ablations | Every member must personally explain the method and their work; a generated narrative cannot establish this |
| Experimental design, evaluation and errors | 25% | Same-row baselines, chronological folds, paired errors, stockout slices, disclosed transfer failure; new ablations keep controls fixed | Final week and validation periods were reused; do not call them fresh confirmatory evidence. Retrospective cohort caveat remains |
| Business interpretation and limitations | 15% | Explicit recommendation example and review outputs; threshold sensitivity makes assumptions visible | Zero keep-full-price cases and 96.9% discount rate among non-stockout cases remain real weaknesses; no causal policy evaluation |
| Reproducibility | 10% | Pinned public source, code/data hashes, saved results, Docker instructions and earlier clean-copy evidence | Re-execute the expanded notebook and verify new artifacts in the actual delivered Docker copy after integration |
| Poster clarity | 5% | One business question, one useful method/decision diagram, baseline comparison, one controlled experiment and one failure | Final single-page portrait A1 PDF must be visually checked after the new content is added; notebook detail should not overwhelm the poster |

The notebook is sufficient; the brief does not require a separate report (page 3). The A1 poster must be one portrait page, **594 × 841 mm** (page 2). The live demonstration should show a representative input/output, technical steps, an experiment and a failure/limitation (page 2). A small saved-data example must remain usable if an external service is unavailable; a clean copy must follow the README successfully (page 3).

Human completion remains necessary: each member must write a factual contribution and AI-assistance/checking paragraph. Do not invent that a TA discussion or personal verification occurred. The teaching-team feasibility discussion requested on page 1 should be recorded only if it actually happened. These checks are separate from technical readiness and are not a claim of Moodle submission.

## Stop rules

Finish when the declared 35 fits, all contrasts, bounded policy sensitivity, explanatory figures, notebook replay and integrated artifact verification are complete. Report unfavourable results unchanged. Do not add a model family, optimize thresholds for appealing coverage, change blend weights, choose a new cohort from outcomes, or retest the viewed final week to select improvements. No observed causal or business-outcome evidence is created by these additions.

## Completed ablation evidence

The 35 fits subsequently completed in a fresh copy at `/private/tmp/freshretail-refinement-20261011`, using the exact locked dependencies. This avoided unavailable macOS placeholder files in the original virtual environment. The seven input/source artifacts were SHA-256 checked against the live repository before execution. The frozen plan hash is `1c8123c527e951b7dd185e3737a40c5e8f76b532fa3b7b1e8a6a9693852bd5a9`.

| Declared variant | Validation MAE | Validation RMSE |
|---|---:|---:|
| Base 10 / RMSE | 0.30454809 | 0.47221007 |
| History 18 / RMSE | 0.29780957 | 0.46140806 |
| Full 20 / RMSE | 0.29715150 | 0.46014936 |
| Full 20 / MAE | 0.29449465 | 0.45911553 |
| No proposed discount / MAE | 0.31080779 | 0.49954471 |
| No previous activity / MAE | 0.29537482 | 0.46333269 |
| No proposed discount or previous activity / MAE | 0.31044466 | 0.49931335 |

These are equal-week averages, not final-week scores. Adding history reduced MAE by 2.21% and improved all five weeks. Adding IDs reduced it by only 0.22%, improving three of five weeks. Changing the full model's loss from RMSE to MAE reduced validation MAE by 0.89%, improving four of five weeks. Removing the proposed discount increased MAE by 5.54% and worsened every week; removing previous-day activity increased it by 0.30%, worsening four of five weeks. Removing both increased it by 5.42% and worsened every week. The differences are conditional on each reference configuration, so their percentages should not be added together.

The full RMSE and MAE configurations reproduced their canonical validation predictions to a maximum absolute difference of `4.44e-16`. The recorded fitting/output stage took 46.95 seconds, excluding cold imports and earlier environment preparation. All 10,920 row keys and 35 sets of fold metrics passed the script verifier. A separate stdlib `csv`/`math.fsum` calculation, without the project's metric helpers, independently reproduced every MAE and RMSE within `1e-12`.

Thirteen output files were copied back and hash-checked under `results/story_ablation/`. The main readouts are `summary.csv`, `contrasts.csv` and `fold_contrasts.csv`; evidence includes `run_receipt.json`, `canonical_replay_check.json`, `verification.json` and `independent_metric_check.json`. No final-period target was scored and the canonical blend selection was unchanged. The bounded sensitivity, EDA, poster and integrated notebook verification are owned by the main task and must be checked separately before delivery.
