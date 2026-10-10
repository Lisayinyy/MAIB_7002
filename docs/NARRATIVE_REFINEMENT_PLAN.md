# Narrative refinement plan

Reviewed on 11 October 2026. This is a read-only audit and an implementation plan; it does not claim that the proposed figures, additional analyses or notebook chapters have already been produced.

## 1. What the next revision must accomplish

The current 47-cell notebook is a much better introduction than its first version: it starts with a category manager, explains the data and five-store selection, separates prediction from decisions, and acknowledges the unsuccessful transfer check. The next improvement should add the **evidence and reasoning between those chapters**, not another layer of introductory prose.

The teammate's 65-cell notebook feels like an investigation because the reader repeatedly sees a question, an experiment, a result and a reason to retain or reject a choice. Lisa's current version compresses most of that model development into a table of four reference methods. A reader can repeat the conclusion but still cannot explain why the experiment took this particular route.

The desired rhythm for each substantive section is:

1. **Question:** what remains uncertain for the manager or modeller?
2. **Expectation:** why might this particular change help?
3. **Comparison:** what changes and what stays fixed?
4. **Evidence:** a small table or figure computed from identifiable artifacts.
5. **Decision:** what was retained, rejected or left unresolved, and why?

Use a clear source label where needed: *inherited teammate experiment*, *matched V2 experiment*, or *new descriptive analysis of saved results*. A readable story may rearrange explanations; it must not invent a chronological discovery process or present old cached grids as freshly executed experiments.

## 2. Evidence and requirement basis

Files reviewed:

- `discount_sales_prediction.ipynb`: 65 cells, including the teammate's selection, exploratory analysis, successive tuning rounds, blending, forecasting errors and two pricing-rule drafts.
- `02_lisa_pipeline.ipynb`: 47 cells; its saved receipt reports 16 executed code cells, zero errors, three figures and 16 expandable source blocks.
- `scripts/build_lisa_notebook.py`, `scripts/build_story_evidence.py`, `src/finalproject_pricingml/data.py`, `config.py` and `v2.py`.
- Saved tuning summaries under `results/`, and the V2 plan, metrics, keyed predictions, scenario forecasts, policy summary and transfer results.
- `docs/assignment_requirements.md`, checked against extracted text from pages 1–3 of the supplied eight-page `maib7002_project_brief.pdf`.

The brief gives technical understanding 30% and experimental design/error analysis 25%. It explicitly asks for the model's inputs, outputs, learning criterion, assumptions and important settings; a small worked example; a meaningful technical comparison with an expected effect; business interpretation; and a reproducible Docker demonstration. It does not require a large number of sophisticated models. A negative result is acceptable when the reasoning and evidence are clear.

The poster must be a single A1 portrait PDF, 594 × 841 mm. It should summarize the investigation rather than reproduce the notebook's full model catalog. Each member's factual contribution and AI-use statement is still a human declaration; the narrative must not manufacture those accounts or a discussion with a TA.

## 3. Concrete gaps beyond the previous prose rewrite

| Current gap | Why it matters | Specific addition |
|---|---|---|
| The business problem is described, but the proposed service is abstract | The reader needs to know who acts, when, and what a usable answer looks like | A compact decision brief: evening decision for tomorrow; store-product as unit; outputs are a named discount, keep full price, or a review reason; operational benefits remain a motivation |
| Dataset suitability is listed without showing the relevant trading behavior | The transition from public data to a pricing investigation feels assumed | Training-only discount-frequency, support and stockout summaries, plus one actual sales/discount history; explain why variation helps comparison but does not randomize treatment |
| The selection funnel is accurate but detached from a real series | Counts alone do not teach what a series or switch means | Show a short real calendar for an eligible pair and annotate its binary discount switches; retain the existing full-data funnel and exact store counts |
| Feature explanations name groups but do not show their construction | The user still cannot trace raw records to the row that enters a model | A seven-day raw slice for the real case, hand-check the mean and lag, then show the resulting feature row and target; show the decision-time boundary |
| Time ordering is a table rather than a visual explanation | Validation, final-period benchmarking and daily updates are easy to confuse | A rolling-origin calendar with five growing training windows, five validation weeks, final benchmark week and a label that the final period was already viewed |
| Most inherited model experiments are reduced to one reference table | This removes the teammate's strongest explanatory device | Short method-by-method hypothesis/evidence/decision sections using cached grids, with their provenance and limitations |
| Ridge, decision tree and HistGradientBoosting mainly appear in an appendix | The requested comparison of different models lacks an explanation of each model's role | Explain the linear reference, single-tree reference and alternate boosting approach in the main story, without claiming they were separately tuned if only a fixed design was tested |
| Extended history, IDs and loss changes are discussed together | Readers may incorrectly attribute all improvement to CatBoost, IDs or MAE alone | A comparison matrix identifying exactly which inputs, loss and regularization differ; distinguish the controlled loss contrast from the bundled feature/ID change |
| The blend is announced rather than derived | The selected weights can look arbitrary or confused with discount percentages | Show the declared V2 blend candidates and validation MAE, with the base forecasts and one arithmetic example; explain what correlated errors mean for an ensemble |
| Average result tables dominate the conclusion | The reader may overlook small gains, shared hard weeks and transfer failure | A paired five-week comparison and a compact original-cohort versus transfer comparison, with appropriate scales and no claim of significance from five folds |
| Discount over-recommendation is described but its scale is easy to miss | This is the central transition from prediction to business decisions | A policy diagnostic graphic: unrestricted choices, historically supported choices, final automatic recommendations and manual review; retain the 96.9% non-stockout warning |
| The real case shows selection but hides some constraints in columns | The user explicitly wants to understand why nine-tenths price instead of a deeper discount | Annotate support, gain and value thresholds for each rate and identify which model vetoes a deeper candidate |
| Technical appendices preserve mechanics but feel disconnected from the story | The learning objective is to explain the complete pipeline, not only inspect a chart | Link each main-story step to one focused executable cell/helper; keep detailed settings and the RF tree path nearby or in a clearly signposted appendix |

## 4. Proposed detailed chapter structure

Keep an opening route map and a compact table of contents. The main story can contain more sections than the current version, provided each section answers one question and shows one useful piece of evidence. Technical source remains expandable; explanatory tables and essential figures stay visible.

### Act I — Establish the decision and the evidence

**1. Tomorrow's decision: should this product be discounted?**

Start with the manager's time constraint and competing interests, not an algorithm. State that the daily data moved the original bakery idea from half-hourly markdowns to a next-day store-product decision. Separate the measured goal—forecasting normalized observed sales—from desired but unverified revenue/waste outcomes. State an example output format before showing a numerical recommendation.

**2. Why FreshRetailNet, and what can it actually tell us?**

Explain public real data, repeated observations, daily discounts, activity/calendar information and stockout signals. Show one raw row in readable terms. Define normalized sales, price rate and stockout hours; state the missing unit prices, costs, inventory quantities/ages and expiry data. Explain automatic download plus local reuse, rather than implying that training reads a permanently live cloud table.

**3. Why not use every product? From 50,000 histories to five stores.**

Retain the exact funnel: 50,000 → 28,219 after discount-share filter → 17,693 after adding state switches → 16,684 after adding zero-sales filter → 312 eligible histories in the top five stores. The thresholds are price rate **strictly below 0.95**, discounted share in [0.15, 0.85], at least six state switches and zero-sales share below 0.20. The selected stores are 343:64, 18:64, 235:63, 182:61 and 154:60 eligible products; together they involve 122 distinct product IDs and all happen to be in city 0. Explain that stores were ranked by eligible-series count, not model score. Keep the retrospective cohort-selection caveat beside the funnel.

**4. Is there a pattern worth predicting?**

Use only original training-period rows. Show price-rate frequency/support and an actual product history with sales and stockout signals. An optional descriptive sales-by-price-band table can standardize within each series using its training-period mean, with row counts beside every band. State that recorded association is a reason to investigate, not an estimate of causal price elasticity. Do not copy the teammate's initial all-`feat` EDA unchanged: it includes the final benchmark week.

**5. Turn a history into tomorrow's prediction.**

Make Y explicit: normalized observed sales for one store-product on the target day. Candidate discount is an input; the later rule chooses the discount. For the same real case used throughout, display the preceding seven observed sales and calculate yesterday's sales, same-weekday lag and seven-day mean. Explain why tomorrow's actual sales, stockouts and realized weather cannot be predictors. Describe the longer history/ID extension later, when it becomes a modelling hypothesis.

**6. How can we test without pretending to know the future?**

Describe duplicate/date-gap/missing-value checks, invalid-rate treatment and the actual selected-cohort outcome: no invalid-rate/missing-value removals; first seven days removed for lag availability. Show 25,896 training-period rows and 2,184 final-period rows. Draw the five expanding validation windows and the already-viewed final benchmark. Explain that preceding actual days update lags within each week, while model parameters remain fixed at the week's start. This is rolling one-day-ahead forecasting, not a seven-day forecast made once.

### Act II — Build a forecast through explicit technical questions

**7. What if the manager simply used last week's sales?**

Compare seven-day mean with same weekday last week, on the same weekly folds. Define MAE in ordinary language and explain why normalized units require a baseline. Introduce RMSE as sensitivity to large misses, with formulas deferred until the reader knows the purpose. Carry the seven-day mean forward as the reference.

**8. Do we need a complex relationship? Linear and single-tree references.**

Explain Ridge as a regularized linear combination and a decision tree as a set of conditional rules. These are declared fixed designs, not claimed optima. Use the V2 table: Ridge CV MAE about 0.31553, single tree about 0.32359, versus baseline 0.32840. Note that the tree's final-period MAE, about 0.32222, is worse than the seven-day baseline's 0.31418. This is a useful negative result, not something to hide. Explain scaling for Ridge and leaf/depth control for the tree.

**9. Can similar past days predict tomorrow? kNN and the value of feature choice.**

Credit the teammate's cached investigation. Explain standardized distance and k's smoothing trade-off. Use the saved k curves and compare the same equal-weight k=25 with/without weather: CV MAE about 0.32668 → 0.31198. This supports excluding those weather inputs under that setup, not a universal claim that weather never matters. Retain k=25 as the inherited reference. Keep unsupported one-off historical checks out of the main quantitative evidence unless their artifacts are located.

**10. Can averaging trees capture nonlinear patterns more reliably? Random Forest.**

Explain bootstrap sampling, feature subsampling, tree averaging and controls against overly specific leaves. Show the teammate's cached depth/leaf grid, its small numerical range and the selected depth 10/leaf size 3 design at CV MAE about 0.30244. Distinguish hyperparameter selection from split learning. Explain why 300 trees smooth the estimate rather than forming a sequential boosting training curve. A partial actual tree path can connect this chapter to the final worked example.

**11. Can correcting earlier errors help? CatBoost and alternate boosting.**

Explain sequential trees and the learning-rate/iteration trade-off. Attribute the original 12-setting and extended 27-setting grids and learning curves to the teammate; selected reference depth 5, rate 0.03 and 500 iterations yields CV MAE about 0.30330. A validation curve that eventually rises is a concrete overfitting illustration; identify the particular validation fold used. Introduce HistGradientBoosting as an alternate boosting reference, and its monotonic variant as an explicit shape assumption, rather than claiming every boosting method behaved identically.

**12. Lisa's question: is the bottleneck the description of recent trading?**

Describe the added short/long history, variability and category IDs. Show base versus extended specification side by side, including the regularization change. The package comparison is a combined design comparison, not a clean estimate of an individual ID/history feature's benefit. State that IDs can help within recurring stores yet create a transfer concern that will be checked later.

**13. If MAE chooses the winner, should MAE train the model?**

This is the clean controlled technical experiment: same extended inputs and CatBoost settings, loss changes from RMSE to MAE. Show validation MAE and both final-period MAE/RMSE. Validation favors MAE, while the final-period standalone MAE and RMSE favor the RMSE-trained extended model. Explain the trade-off and retain the predeclared validation rule. MAE-trained predictions target a conditional median; multiplying them by a price rate does not create a calibrated expected-revenue forecast.

**14. Can two imperfect forecasts help each other?**

First explain the teammate's 50/50 blend as a near-best simple choice: the historical sweep's exact minimum is 60% RF, so do not call 50/50 its exact validation minimum. Then show the separate V2 declared blend comparison with RF weights 25%, 50% and 75%, using the validation-best non-RF standalone model. Its winner is 25% RF plus 75% extended MAE CatBoost. Weight percentages combine model predictions and are unrelated to the product's discount percentage. Use keyed V2 OOF rows to recompute error correlations and blend scores; avoid joining the older unkeyed OOF file to new rows by position.

**15. What improved, and where does it still fail?**

Present the final comparison with baselines first: selected CV MAE about 0.29402 and final-period MAE about 0.28339 versus teammate 0.30125 and 0.28923. Show the five paired validation weeks and discount-band errors. Explain that about 2% lower forecast MAE is neither 2% extra sales nor 2% extra revenue. Show a mechanically chosen large miss, stockout censoring, and the locked transfer failure. Do not switch the winner after seeing transfer or final-period metrics.

### Act III — Turn forecasts into decisions without overstating evidence

**16. The tempting answer: discount everything. Why is that a problem?**

Derive unrestricted and historically supported choices from the already-saved V2 scenario forecasts. Show how frequently each method recommends discounts and how many unrestricted choices lack relevant price history. Explain confounding by managers' historical decisions and missing stock/expiry context. This is the story's important turn: predictive success is not equivalent to policy success.

**17. What checks should a recommendation pass?**

Show supply review → support → sales/value thresholds → agreement → mildest near-best depth. Define every threshold as an explicit illustrative policy choice. Stockout evidence triggers review, not an unsupported freshness conclusion. Explain that agreement among similar models is a sensitivity check, not a confidence interval. Distinguish complete output coverage from automatic recommendation coverage.

**18. Follow a real case to a specific discount—and a second case to review.**

Continue store 18/product 11/26 June 2024, selected mechanically as the first supported recommendation. Rate 0.90 means 10% off. Show why 0.95 loses on forecast sales, 0.85/0.80 fail the RF value-floor check, and still deeper rates lack support. Then show the first stockout-review case. Report the full cohort's 1,179 automatic discounts, 967 supply reviews and 38 disagreement reviews. Keep the unresolved 96.9% discount share among non-stockout cases prominent; no current case selects full price.

**19. The conclusion a manager can responsibly use.**

Summarize a forecasting improvement within the studied cohort and a transparent recommendation prototype. State the additional observations and prospective evaluation needed before operational pricing claims. Explicitly identify inherited work, Lisa's extension and this explanatory revision. End with a concise runnable demonstration route, reproduction instructions and required personal declarations.

Technical appendices retain the complete 17-method results, parameter/feature dictionary, mathematical definitions, actual RF path and blend arithmetic, descriptive bootstrap caveats, source provenance and Docker commands. They support the main story; they should not be where all methodological reasoning is hidden.

## 5. Five strongest evidence visuals to add or substantially improve

| Visual | Question answered | Exact source and computation | Design and interpretation rule |
|---|---|---|---|
| **An actual series, with the information boundary** | What is a store-product history, and what is known when tomorrow is predicted? | Training rows for a mechanically selected eligible pair; show observed sales, price rate and stockout signals; annotate lag-1, lag-7 and the rolling mean for one forecast date | Three aligned panels with shared dates; label normalized sales and price rate; shade only the target day. No fictional sales or invented product identity |
| **The chronological experiment calendar** | Why five validation rounds, and what data can each model use? | The five dates in `experiment_plan.json`, feature-table start and final-period dates | Horizontal growing train bars and one-week validation blocks; distinct final benchmark block marked already viewed; show cohort selection using the whole 90-day training window so its limitation is not hidden |
| **A model investigation dashboard** | Which technical choice actually changed forecasting quality? | Cached matched kNN weather/k grids; RF depth/leaf grid; original CatBoost learning-curve CSV; V2 controlled RMSE-versus-MAE comparison | Use separate compact panels, each with its own clear hypothesis and provenance. Keep axes honest. For the poster, select the controlled loss contrast rather than miniaturizing all four panels |
| **Why this blend, and how limited is its gain?** | Why 25/75, and does the advantage persist across validation weeks? | V2 keyed OOF predictions and declared blends; validation-fold CSV; final/transfer summaries | A discrete weight comparison beside paired weekly differences. Do not add an undeclared fine-grained weight search or call fold variability a valid confidence interval. Include a clearly separated transfer failure inset or companion chart |
| **From candidate prices to a decision** | Why exactly 10% off, and how often can the assistant decide? | Saved `discount_response.csv`, `recommendations.csv`, policy thresholds and the selected real case | Two aligned axes: predicted sales and value share versus price rate, with 0.95 value floor and unsupported rates shaded; label the RF veto. Pair with a 2,184-case outcome bar and the 96.9% non-stockout warning. These are model scenarios, not realized benefits |

Retain the existing accurate selection funnel. Improve its typography and connect it to the actual-series visual; it does not need another model experiment. Prefer static, exportable vector charts so the same evidence can appear in the notebook and poster.

## 6. Teaching code and evidence plumbing

- Use small read-only helpers that load saved evidence, calculate descriptive summaries and return DataFrames/figures. New chart code need not retrain models.
- Check artifact hashes or explicit keys before combining results. Historical tuning files and V2 comparison files have different provenance and roles; name them accordingly.
- Add a compact evidence manifest mapping each narrative claim/figure to its input files and scope: training-only EDA, historical cached tuning, V2 validation, reused benchmark, transfer or hypothetical policy scenario.
- For raw-to-feature teaching, a small checked extract of actual public observations can be packaged for offline use. It must preserve the true values and carry attribution; it is not a synthetic demonstration dataset.
- Show the actual call chain once: public files → eligible keys → chronological features → expanding validation → selected design → candidate sales forecasts → checked recommendation. Link these steps to the relevant helpers rather than dumping entire modules in visible cells.
- Keep model algorithms available in the notebook, not only the Chinese guide: explain the learning criterion, important settings and at least one actual calculation. Do not imply that plotting a cached metric recreates training.
- For the blend/error-correlation view, use existing V2 keyed validation predictions and only the already-declared candidates. A new diagnostic plot does not authorize retrospective reselection of a model.
- Continue to rebuild narrative evidence after a full experiment rerun, because platform-level numeric differences can change CSV hashes. The notebook's default path should remain a saved-data demonstration plus the two selected worked-example fits.
- Update new execution counts, image counts, figure labels, README links and Docker HTML export after the rewrite. Preserve old delivery receipts as historical rather than silently changing their original verified numbers.

## 7. Claims to keep accurate while adding detail

1. The main target is observed normalized sales, not unconstrained demand, item counts, currency or profit.
2. The five stores were chosen for usable series, not general representativeness, commercial success or highest prediction scores.
3. Full-training-period cohort selection is retrospective and includes validation dates. Chronological model fitting alone does not remove that selection optimism.
4. The benchmark final week was already viewed. A polished story does not restore an untouched holdout.
5. The teammate's initial descriptive discount analysis uses all feature rows; a new training-only figure must be recalculated and may differ.
6. Historical one-off experiments quoted only in prose are weaker provenance than saved grids. Attribute them as notes or leave them out of the central evidence.
7. Native RF and CatBoost importance percentages are not directly comparable measurements. Do not reproduce a side-by-side percentage claim as proof that one model cares twice as much about discount.
8. Similar forecast scores do not prove that the feature set is at a fundamental ceiling, and five validation weeks do not establish universal equivalence.
9. The historical 50/50 blend was a simplicity choice near the sweep minimum. The V2 25/75 blend is the minimum among its declared candidates, not among all possible ensembles.
10. A supported alternative price is still counterfactual for the particular day. Observational support and model agreement do not establish causal uplift.
11. One stockout hour does not establish complete inventory clearance, fresh replenishment or tomorrow's stock age. No stockout for a week does not establish old stock.
12. Price-rate times forecast is a relative sales-value proxy. MAE training and the final mixed-loss ensemble do not justify calling this expected-revenue optimization.
13. Current automated recommendations cover about 54% of all rows, but nearly all non-stockout cases still receive a discount. The selection problem remains unresolved.
14. The transfer failure is a substantive result, not a footnote. A conclusion or poster must not imply successful generalization to other stores.

## 8. Poster story derived from the fuller notebook

Use a single readable narrative rather than a wall of algorithm descriptions:

- **Top:** manager's question and the two-stage solution, with the daily store-product scope.
- **Data:** compact five-store selection funnel and the decision-time information boundary.
- **Method:** predict candidate sales → check support/supply/value/agreement → name a discount or request review.
- **Main evidence:** baseline, teammate blend and Lisa blend; one controlled technical comparison, with the error metric and reused-period status explicit.
- **Concrete case:** the 10% discount example and the specific reason deeper options fail.
- **Limits and next step:** original-cohort gain, failed transfer, no verified causal profit/waste effect, and prospective evaluation needed.

The poster need not show every tuning grid or all 17 result rows. It should support the brief's live demonstration: one business input, the technical path to output, one meaningful experiment and one failure case. Build the final poster from the same verified numerical tables and figure sources as the notebook so the story and numbers cannot drift apart.

## 9. Completion criteria

- A reader can explain why the problem matters, why this dataset and why exactly these five stores.
- Each retained model has a stated role, learning intuition, important technical choice, identifiable evidence and an honest conclusion.
- The difference between a prediction target, a validation score and a pricing recommendation is explicit.
- At least one raw observation can be traced through a feature, a model prediction and a recommendation or review reason.
- Key positives and negatives remain visible: modest original-cohort improvement, standalone loss trade-off, recommendation overuse and unsuccessful transfer.
- Every displayed new figure is computed from actual source/saved observations, with no synthetic training or invented historical narrative.
- The notebook runs from a fresh kernel; the saved-data path works offline in Docker; the rendered HTML remains readable with source collapsed and expandable.
- The A1 poster is rendered and visually inspected, and reports the same cohort, model, scores and limitations as the notebook.

No model fitting, notebook modification, source modification or poster generation was performed for this audit. Only this plan document was added.
