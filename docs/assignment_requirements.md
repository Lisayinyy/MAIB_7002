# MAIB7002 project requirements audit

Checked against the supplied **MAIB7002 Group Project Brief**, eight pages, on **10 October 2026 (Hong Kong)**. Source: `/Users/lisayin/Desktop/HKU/7002_1A/maib7002_project_brief.pdf`. The PDF footer says it was generated on 15 September 2026; this audit does not infer whether a newer Moodle announcement exists.

## Confirmed dates, format, and group allocation

| Item | Confirmed requirement | Source |
|---|---|---|
| Submission | One team member submits for the whole group to Moodle **by the end of 14 October 2026**. Late submissions are not accepted. The PDF does not specify a time zone or a more precise clock time. | p. 3 |
| Exhibition | **16 October 2026**, CP-LT104, 1/F, Block B, Cyberport 4. | pp. 3–4 |
| Group Q presentation | Session 2, Round II, **20:00–21:00**. Complete assigned peer reviews during Round I, **18:30–19:30**. | pp. 4, 8 |
| Poster | **One print-ready, single-page A1 PDF**, portrait, **594 mm wide × 841 mm tall**. | p. 2 |
| Written/technical submission | An explanatory notebook **or** a short report with runnable code, packaged in the Docker project. An explanatory notebook that covers all required elements is sufficient; a separate report is not required. | p. 3 |
| Personal declarations | Each member writes a short paragraph naming their contribution, generative-AI assistance used, and how it was checked; write “none” if no AI was used. | p. 3 |
| Course weighting | Group project is **20% of the course mark**. | p. 3 |

The supplied allocation lists these Group Q members (p. 8): **Aashish Omprakash Pareek, Yuanyuan Yin, Lingyu Chen**. It says group membership is fixed except for teaching-team corrections to enrolment or administrative issues (p. 4).

### Group Q individual peer reviews

| Member | Assigned review 1 | Assigned review 2 |
|---|---|---|
| Aashish Omprakash Pareek | T: Mulan Zhang | Delta: Siyuan Chen |
| Yuanyuan Yin | V: Yinuo Zhao | Y: Zheng Liu |
| Lingyu Chen | Eta: Halton Issac Law | No second review listed |

Source: p. 8. These are individual assignments, separate from formative intra-group feedback (p. 4).

## Assessment weights

| Criterion | Weight |
|---|---:|
| Business question and suitability of data | 15% |
| Technical understanding of chosen methods | 30% |
| Experimental design, evaluation, and error analysis | 25% |
| Business interpretation and limitations | 15% |
| Reproducibility | 10% |
| Poster clarity | 5% |

Source: p. 3. Demonstration and technical Q&A are evidence for technical understanding, evaluation, and reproducibility. Individual marks may be adjusted when there is clear evidence of unequal contribution. A model does **not** have to outperform the baseline for strong marks; understanding and evidence matter more than complexity, dataset size, or interface polish (pp. 2–3).

## Required investigation and content

| Requirement | What to include in this project | Source |
|---|---|---|
| Specific business question | Identify the category manager, daily discount decision, available information, and measurable evidence of success. | p. 1 |
| Feasibility discussion | Record accessible data, observation unit, decision-time inputs, evaluation outcome, baseline, practical constraints, and the short plan discussed with the teaching team. Do not invent a discussion that has not occurred. | p. 1 |
| Technical explanation | Explain inputs, outputs, what the method learns, objective/criterion, assumptions, parameters, and suitability. Define every mathematical symbol used. | p. 1 |
| Small worked example | Show one traceable example such as a prediction, tree path, distance calculation, or learning update, connected to actual project data. | p. 1 |
| Simple baseline | Compare baseline and main method on the same cases and evaluation conditions. | pp. 1–2 |
| Meaningful technical experiment | State an expected effect, vary at least one important technical choice, hold other conditions fixed as far as practical, and explain the measured result. There is no compulsory number of advanced models. | p. 1 |
| Fair supervised evaluation | Keep final test data separate from development and selection; fit preprocessing on training data; choose a split reflecting intended use, such as later dates; exclude unavailable decision-time information. | p. 2 |
| Error/failure analysis | Inspect representative errors or failures, business consequences, reliability, and how examples were chosen without cherry-picking. | p. 2 |
| Business interpretation | Connect technical findings to consequences while separating measured evidence from assumptions about costs, behaviour, or business benefit. | pp. 1–3 |
| Original contribution | Identify what the group built or changed. A demonstration of an existing tool alone is insufficient. Libraries, beyond-class methods, and pretrained models/APIs are allowed when explained; coding from scratch is optional. | p. 1 |

### Required poster content

The A1 poster must communicate (p. 2):

- Project title, group identifier, and members.
- Business question, intended user, and relevant data.
- How the main method works, with a useful diagram or worked example.
- Baseline, key experiment, and most important results.
- Business implications and limitations.

Figures should be readable and explanations concise. Detailed code and the full analysis belong in the notebook/report. The brief specifies no poster template, exact font size, word limit, colour scheme, or citation style.

### Required live demonstration

Use the **submitted Docker environment** to demonstrate (p. 2):

1. A representative business input or case and its model output.
2. The key technical steps from input to output.
3. A comparison or experiment showing an important technical choice.
4. A limitation or failure case and its business implications.

A runnable notebook is sufficient. A separate web/mobile interface is optional, and a previously trained model may be loaded. Include code and instructions to reproduce the group's experiments and training. Every member should be ready to explain the overall approach and their own work.

## Docker submission checklist

The package must contain (p. 3):

- `Dockerfile` and `compose.yaml`.
- Dependency specifications with package versions.
- Project code and notebooks.
- README with exact build/run instructions and how to obtain/place data.
- Documentation for required downloads, external services, model versions, and access costs.
- A small runnable example and saved results, so work can be inspected if an external service is unavailable; marking must not require a paid purchase.
- Evidence that the README was followed successfully from a **fresh copy** of the submission.

The brief does not specify the Moodle upload slots, maximum file size, required archive filename, or whether printed posters are arranged by students. Verify these in Moodle or with the teaching team; do not assume them.

## Relationship to Lisa's earlier project logic

This section describes existing local plans and implementation, not extra course rules.

The earlier plan in `/Users/lisayin/Documents/ChatGPT/7002_1A/freshretail_project/PROJECT_PLAN.md` makes **daily discount selection** the core goal. A sales model estimates sales for supported candidate discounts, and an explicit search selects a candidate while restricting loss of a relative sales-value proxy (lines 7–11, 94–112). It forbids synthetic sales/inventory data and distinguishes observed forecasting accuracy from estimated business gains (lines 126–128). It also proposes action-support checks, fallback recommendations, cross-model disagreement checks, and fixed/calendar-policy comparisons (lines 110–126).

That older plan proposed hourly modeling, a 100-series pilot, and a 60/15/15-day split (lines 37–43, 81–92). These are **historical proposals**, not the current protocol and not requirements in the brief. The canonical delivered notebook later used a reconstructed 312-series daily protocol, five chronological validation weeks, two baselines, and five forecasting models; its main scenario used validation-selected Random Forest, with the 50/50 RF–CatBoost blend retained as a supplementary forecasting experiment. See `/Users/lisayin/Desktop/HKU/7002_1A/group_project/docs/PROJECT_SUMMARY.md`, lines 7–16, 22–28, 49–67, 89–105.

For the new version, a clear explanation should maintain the chain **business decision → historical observations → decision-time features → sales forecast → supported discount comparisons → constrained recommendation → evaluation and limitations**. This follows the user's project logic and the brief's emphasis on technical understanding. A claim of a better model must follow a matched experiment; a cleaner pipeline, more transparent decisions, and more honest evaluation can be improvements even if MAE does not improve.

## Items requiring attention before final submission

- The earlier final evaluation week has already been viewed. The course asks for final data separate from development/selection. Do not rename this reused week an untouched holdout; document its status and the development/evaluation chronology honestly. If genuinely new outcomes are unavailable, explain the limitation rather than fabricate a fresh test.
- A small worked example and line-by-line learning guide should use the actual implemented model and actual data, with symbols and transformations explained.
- Discount decisions need explicit support/fallback rules and must not turn associative forecasts into claims of proven causal sales, profit, or waste improvements.
- Each member's contribution and AI-use paragraph requires their factual account and confirmation. Do not assign work to a person solely because an AI agent produced it.
- Verify the final notebook, poster, and README agree on the selected model, scoring protocol, sample, and results.
- Recheck the final package from a clean copy and record exactly what ran, including whether the full training pipeline or only the saved demonstration was exercised.

No document or assessment was submitted by this audit.
