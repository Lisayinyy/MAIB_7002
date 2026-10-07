# FreshRetailNet project summary

**Canonical results:** the complete Docker run on October 7, 2026, using Linux arm64 and Python 3.11.14. It executed 185 model fits and all 13 notebook code cells, followed by the independent audits. Unless explicitly labeled as an earlier run, the scores below refer to this execution.

## Problem and implemented approach

The project investigates whether historical retail data can support a category manager's decision about whether to discount a perishable SKU tomorrow and, if so, by how much. The intended business objective is to increase sales while limiting the loss of sales value.

The implemented workflow has two components:

1. **Sales forecasting:** predict next-day observed normalized sales for a store–SKU using recent history, calendar information, and a candidate discount.
2. **Constrained discount scenarios:** hold the remaining inputs fixed, compare historically supported discount values, and select the highest predicted sales subject to a limit on a sales-value proxy.

The completed experiment evaluates forecast accuracy. The scenario component illustrates a decision rule; it does not demonstrate real sales uplift, monetary profit, or reduced food waste. No reinforcement learning algorithm is implemented.

**Current model status:** Random Forest remains the validation-selected model in the main notebook, saved model, and scenario demonstration. A later, supplementary experiment recommends a 50/50 Random Forest–CatBoost average for further forecasting evaluation. It has not replaced the main scenario model.

## Data and target

The source is the public [Dingdong-Inc/FreshRetailNet-50K dataset](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), distributed under CC BY 4.0. The experiment pins revision `08c1fab7f9257bc73679d415d65d644165d351d4`; file hashes are recorded in [the locked protocol](../outputs/final_protocol/locked_protocol.json). No synthetic sales labels are used.

The research cohort contains **312 store–SKU series, 122 products, and five stores**—343, 18, 235, 182, and 154—in city 0. The training split covers March 28–June 25, 2024: 28,080 daily observations. After seven warm-up days, 25,896 rows remain for model fitting. The final evaluation period is June 26–July 2, 2024, with 2,184 observations.

To reconstruct the teammate's reported scope, series are selected using all 90 training dates: a discounted-day share of 15%–85%, a zero-sales share below 20%, and at least six consecutive-day discount-state changes. Here, `discount < 0.95` defines a discounted day. The five stores with the most eligible series form the cohort.

This is a **report-based reconstruction**, not an exact reproduction of unavailable teammate preprocessing code. Because cohort selection uses information through June 25, later validation outcomes influence which series are included. Results therefore describe a retrospective research cohort, not one selected before the earliest validation window.

The target, `sale_amount`, is **observed daily normalized sales**. It is not a count of bread units, unconstrained demand, or monetary revenue. Stockouts can suppress observed sales; lagged stockout information does not recover the unobserved demand.

## Ten forecasting inputs

| Input | Meaning at the prediction date |
|---|---|
| `discount` | Target-day discount rate, treated as a planned input or a candidate scenario value |
| `weekday` | Target-day weekday, Monday = 0 through Sunday = 6 |
| `holiday_flag` | Target-day holiday indicator |
| `activity_flag` | Target-day activity indicator, treated as a planned input |
| `sales_lag1` | Previous day's observed sales |
| `sales_lag7` | Observed sales seven days earlier |
| `sales_mean7` | Mean sales over the preceding seven days |
| `discount_lag1` | Previous day's recorded discount |
| `stockout_lag1` | Previous day's recorded stockout hours |
| `series_mean` | Expanding mean of earlier sales for the same store–SKU, including warm-up dates |

History features are calculated within each store–SKU and shifted before taking means. There are no weather, store-ID, SKU-ID, or category-ID inputs. All ten features are numeric; weekday remains an integer in this frozen experiment, including for Ridge. Ridge's scaler is fitted only on each training fold. CatBoost receives no categorical-feature specification.

The recorded target-day discount and activity values are assumed to be known when making the forecast. The public data do not establish their advance availability. Deployment would require verifying that these are genuine planned inputs rather than quantities only available after the day ends.

## Chronological comparison

Five expanding-training folds are used. Training always begins on April 4, after the warm-up period.

| Fold | Training ends | Validation week | Training rows | Validation rows |
|---|---|---|---:|---:|
| 1 | May 21 | May 22–28 | 14,976 | 2,184 |
| 2 | May 28 | May 29–June 4 | 17,160 | 2,184 |
| 3 | June 4 | June 5–11 | 19,344 | 2,184 |
| 4 | June 11 | June 12–18 | 21,528 | 2,184 |
| 5 | June 18 | June 19–25 | 23,712 | 2,184 |

All dates are in 2024. Forecasts are made **one day ahead, repeatedly**: earlier observed days update later-day history features, including within an evaluation week. Models are not refitted within the week. This is not a single forecast of seven future days made at the beginning of the week.

Each of the five models has three predeclared parameter configurations. The lowest mean validation MAE selects its configuration and the main model; exact ties follow candidate order. Selected configurations are refitted on all 25,896 training rows. Predictions are clipped at zero before scoring. The random seed is 7002.

MAE is the primary metric. RMSE measures sensitivity to larger errors; WAPE is `100 × sum(abs(error)) / sum(abs(observed sales))`. Validation metrics are averaged across the five weeks. All model comparisons use the same evaluation rows.

The team had already inspected the final evaluation week before this reconstruction and before the ensemble extension. It is a **previously viewed final-period benchmark**, not a fresh holdout. The recorded selection rules do not use its scores for additional parameter or weight selection.

## Main forecasting results

| Method | Mean validation MAE | Final-period MAE | Final-period RMSE | Final-period WAPE |
|---|---:|---:|---:|---:|
| **Seven-day average baseline** | 0.328397 | 0.314185 | 0.520890 | 32.4360% |
| **Same-weekday baseline** | 0.415021 | 0.399753 | 0.617069 | 41.2699% |
| Ridge Regression | 0.315059 | 0.307541 | 0.503124 | 31.7500% |
| Decision Tree Regressor | 0.321790 | 0.313935 | 0.493794 | 32.4102% |
| **Random Forest Regressor — selected** | **0.303716** | 0.292949 | 0.462689 | 30.2436% |
| HistGradientBoostingRegressor | 0.305520 | 0.289176 | 0.454254 | 29.8541% |
| CatBoost Regressor | 0.303786 | **0.287797** | **0.451905** | **29.7118%** |

Random Forest has the lowest validation MAE, although its difference from CatBoost is only about 0.000071. Its final-period MAE is 6.76% below the seven-day baseline. CatBoost has the lowest final-period MAE among the five individual models, an 8.40% reduction against that baseline; this observation does not change the earlier selection of Random Forest. These are descriptive comparisons, not significance claims.

The selected Random Forest uses 200 trees, maximum depth 10, and a minimum of five observations per leaf. The selected CatBoost configuration uses 400 iterations, depth 4, learning rate 0.05, and L2 leaf regularization 5.

An eight-feature ablation removes target-day `discount` and `activity_flag`, while retaining lagged discount and independently tuning each model with the same search budget. All five full models have lower validation and final-period MAE than their ablations. For Random Forest, final-period MAE changes from 0.310608 without these two inputs to 0.292949 with them, a 5.69% reduction. This shows predictive information under the stated availability assumption; it does not identify a causal promotion effect.

Diagnostic discount bands are none/light (`discount > 0.95`), moderate (`0.80 < discount <= 0.95`), and deep (`discount <= 0.80`). Their final-period counts are 1,186, 757, and 241. This diagnostic boundary differs from the binary discounted-day rule used for cohort selection.

## Supplementary ensemble experiment

The supplementary exploration regenerated 10,920 predictions across the same five validation weeks using the already selected individual-model configurations. It compared fixed equal-weight combinations and a coarse nonnegative weight grid over Random Forest, HistGradientBoosting, and CatBoost. After removing duplicate weight vectors and including single-model endpoints, 19 methods were evaluated.

The lowest mean validation MAE selected:

`ensemble prediction = 0.5 × Random Forest prediction + 0.5 × CatBoost prediction`

| Supplementary choice | Mean validation MAE | Final-period MAE | Final-period RMSE | Final-period WAPE |
|---|---:|---:|---:|---:|
| RF + CatBoost, 50/50 | **0.301405** | **0.287288** | **0.452877** | **29.6592%** |

Its validation MAE is 0.76% below Random Forest's, and it beats both constituent models on MAE in each of the five validation weeks. On the final period, its MAE is 8.56% below the seven-day baseline, 1.93% below Random Forest, and only 0.18% below CatBoost. Its final-period RMSE is slightly worse than CatBoost's.

The equal-weight RF–HistGradientBoosting–CatBoost combination has a lower final-period MAE, 0.286387, but was not selected because its validation MAE is higher than the 50/50 blend's. The recommendation remains the validation-selected 50/50 blend, rather than whichever combination looks best on the final week.

This is **post-hoc exploration**. Individual-model parameters and ensemble weights reuse the same validation folds, so these scores are tuning evidence rather than an independent nested evaluation. The final week had also already been viewed. The blend is a supplementary forecasting recommendation, with no demonstrated causal pricing advantage; the original Random Forest selection and scenario remain intact.

### Comparison with the earlier macOS run

| RF + CatBoost, 50/50 execution | Mean validation MAE | Final-period MAE | Final-period RMSE |
|---|---:|---:|---:|
| Earlier macOS run, Python 3.13.5 | 0.301409790 | 0.287277253 | 0.452900475 |
| Canonical Docker run, October 7, 2026, Python 3.11.14 | 0.301404734 | 0.287287880 | 0.452877043 |

The runs are not numerically identical. Small differences were observed in Random Forest results and combinations containing it; their precise cause has not been isolated. Both runs select Random Forest for the main experiment and the 50/50 blend for the supplement. The Docker run is the reference for the current report; its [Python package inventory](../outputs/final_protocol/docker_python_packages.txt) and [run manifest](../outputs/final_protocol/run_manifest.json) document the environment.

## Illustrative discount decision

The saved scenario uses **Random Forest**, store 18, product 11, June 26, 2024, and `activity_flag = 0`. Candidate rates are rounded to two decimal places and require at least three training days for the same store–SKU and activity state. The highest supported rate is the reference; the example uses 1.00.

For each candidate rate `d`, the rule calculates `V(d) = d × predicted normalized sales(d)`. It maximizes predicted sales subject to `V(d) >= 0.95 × V(reference)`. The 5% tolerance is an illustrative assumption. `V` is a dimensionless sales-value proxy, not verified currency revenue.

| Candidate rate | Supporting training days | Predicted normalized sales | Sales-value proxy | Proxy change vs. reference |
|---|---:|---:|---:|---:|
| 1.00 — reference | 18 | 1.152685 | 1.152685 | 0.00% |
| 0.99 | 15 | 1.234863 | 1.222515 | +6.06% |
| 0.89 — selected by rule | 3 | 1.235686 | 1.099761 | −4.59% |

Rates 0.94 and 0.97 are also evaluated in the saved output. Rate 0.89 means 89% of the reference price, or an 11% discount. Its predicted sales exceed those at 0.99 by only about 0.000823, so the rule accepts a substantial proxy trade-off for a very small modeled sales difference. This illustrates the need for uncertainty-aware or minimum-improvement rules before operational use; it does not establish that the deeper discount is commercially preferable.

The hypothetical predictions under alternative rates are not observed sales outcomes. Historical support reduces extrapolation but does not remove confounding, and rounding rates pools nearby historical discounts. Without verified monetary prices, costs, remaining inventory, and disposal measurements, the project does not report actual revenue, profit, ROI, or food-waste reductions.

## Evidence and verification

The current main experiment is in [the executed notebook](../01_freshretail_project.ipynb), with results under [outputs/final_protocol](../outputs/final_protocol/). Files directly under `outputs/` belong to an older 100-series pilot and should not be mixed with this comparison.

For this summary, the October 7 Docker run's saved row-level evidence was independently checked without retraining or changing source outputs:

- All 12 baseline, full-model, and ablation score rows were recomputed from the 2,184 [final-period predictions](../outputs/final_protocol/test_predictions.csv).
- All 19 supplementary methods' five-fold and final-period metrics were recomputed from [validation predictions](../outputs/ensemble_exploration/validation_predictions.csv), saved weights, and the main final-period predictions.
- The selected blend's saved predictions were matched by store, SKU, and date; targets and the 50/50 arithmetic agreed. Metric differences were below `1e-12`.
- Both baseline validation MAEs were recomputed from the saved validation rows. The current [main verification report](../outputs/final_protocol/independent_verification.json) records feature, cohort, fold, saved-model, and scenario checks, together with successful execution of all 13 code cells in the integrated notebook.

No scoring discrepancy was found. The principal limits are retrospective cohort selection, assumed advance availability of promotion inputs, a previously viewed final benchmark, repeated validation use for tuning, stockout-censored sales, and observational rather than causal evidence. New untouched dates and a credible method of evaluating price interventions would be needed before claiming an effective deployed pricing policy.
