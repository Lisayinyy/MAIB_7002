# Random forest build-out: next-day sales for discounted perishables

Record of how the random forest was tuned and tested, with the numbers behind each decision. The goal, data, store selection, features, splits, metrics and baseline are the same as for kNN and are described in `knn_build_out.md` (sections 1–7).

## 1. Shared setup in brief

- **Target:** sales on day t+1 for the same store and product.
- **Data:** 312 store-product series in 5 stores, 28,080 rows.
- **Features:** the same 10 as the locked kNN (sales on day t, sales 7 days before, 7-day average, average to date, stockout hours on day t, discount on day t, discount on day t+1, activity on day t, holiday on day t+1, weekday of day t+1).
- **Validation:** five time-ordered weekly folds (May 22 – June 25), each trained on all earlier rows.
- **Test:** the unseen week, June 26 – July 2.
- **Baseline:** mean of the last 7 days' sales. Validation MAE 0.328.

## 2. How the model works

A random forest builds many decision trees, each on a random sample of the training rows, and averages their predictions.

- **Split rule:** squared error. Each split is chosen to give the largest drop in the variance of sales in the two resulting groups. Gini and entropy are the classification equivalents and do not apply to a continuous target.
- **No scaling needed:** trees split on one feature at a time, so feature units do not matter.
- **No gradient descent:** trees are built by choosing splits directly, so there is no loss curve over training steps.
- **What limits over-fitting:** maximum depth and minimum rows per leaf. Either stops trees from memorising noise; they play the role k played for kNN.

## 3. Tuning

All figures are average MAE over the five validation weeks, with 300 trees.

### Round 1: leaf size, features per split, weather (36 combinations, no depth limit)

| Finding | Detail |
|---|---|
| Best setting | Leaf ≥ 10, half the features per split, no weather: 0.3031 |
| Same setting with weather | 0.3058 |
| Very small leaves (1 row) | 0.3057–0.3122 |
| Very large leaves (40 rows) | 0.3077–0.3095 |
| Robustness | All 36 combinations beat the baseline in all five weeks |

Weather made every setting worse, as it did for kNN, though by less (about 0.003 against about 0.015).

### Round 2: depth and leaf size (16 combinations, half the features per split)

| Maximum depth | Leaf ≥ 3 | Leaf ≥ 5 | Leaf ≥ 10 | Leaf ≥ 20 |
|---|---|---|---|---|
| 10 | 0.3024 | 0.3027 | 0.3035 | 0.3052 |
| 15 | 0.3030 | 0.3031 | 0.3032 | 0.3049 |
| 20 | 0.3043 | 0.3036 | 0.3034 | 0.3048 |
| No limit | 0.3044 | 0.3034 | 0.3031 | 0.3051 |

- The whole grid spans 0.003, about a seventh of the week-to-week variation (0.02).
- Depth and leaf size trade off: with small leaves a depth cap helps slightly; with leaves of 10 it makes no difference.
- Without a limit, trees grow to 27–41 levels depending on leaf size.
- Plot: `results/rf_tuning.png`. Its vertical axis covers only 0.3024–0.3052, so the lines look further apart than they are.

### Further checks (one-off runs, change against 0.3024)

| Setting | Value tried | MAE |
|---|---|---|
| Random seed | Four other seeds | 0.3026–0.3028 |
| Number of trees | 50 | 0.3040 |
| | 100 | 0.3028 |
| | 1,000 | 0.3025 |
| Shallower depth | 4 | 0.3230 |
| | 6 | 0.3087 |
| | 8 | 0.3040 |
| Features tried per split | A third | 0.3037 |
| | Three quarters | 0.3030 |
| | All | 0.3048 |
| Rows sampled per tree | Half | 0.3017 |
| | Three quarters | 0.3023 |
| Split rule | Absolute error (100 trees) | 0.3019 |
| Target | Sales ÷ 7-day average | 0.3033 |
| Tree type | Extra-trees variant | 0.3069 |

- Depth 10 is a real minimum: depths 8, 6 and 4 are progressively worse.
- 300 trees is enough.
- Seed noise is about 0.0004; no alternative improves on the chosen design by more than 0.001.

## 4. Locked design

- 10 features, no scaling
- 300 trees, maximum depth 10, at least 3 rows per leaf
- Half the features tried at each split, squared-error splits

Validation by week:

| Validation week | Baseline | kNN (k=25) | Random forest |
|---|---|---|---|
| May 22–28 | 0.309 | 0.295 | 0.290 |
| May 29–Jun 4 | 0.316 | 0.299 | 0.291 |
| Jun 5–11 | 0.362 | 0.350 | 0.336 |
| Jun 12–18 | 0.327 | 0.301 | 0.290 |
| Jun 19–25 | 0.328 | 0.316 | 0.306 |
| Average MAE | 0.328 | 0.312 | 0.302 |
| Average WAPE | 34.5% | 32.7% | 31.7% |
| Average MSE | 0.270 | 0.238 | 0.219 |

The order is the same in every week: random forest, then kNN, then the baseline. Plot: `results/validation_weeks.png`.

## 5. Unseen test week (June 26 – July 2)

Fitted once on all 25,896 training rows; scored on 2,184 rows.

| Method | MAE | WAPE | MSE | RMSE |
|---|---|---|---|---|
| Random forest | 0.292 | 30.2% | 0.216 | 0.465 |
| kNN (k=25) | 0.304 | 31.4% | 0.240 | 0.490 |
| Baseline (7-day average) | 0.314 | 32.4% | 0.271 | 0.521 |
| Same weekday last week | 0.400 | 41.3% | 0.381 | 0.617 |

- Against the baseline: MAE 7.0% lower, MSE 20.5% lower.
- Against kNN: MAE 3.8% lower, MSE 10.1% lower.
- The validation gap over the baseline was 7.9% on MAE, so nearly all of it carried over. kNN kept about two thirds of its gap.

### By tomorrow's discount

| Discount | Rows | Random forest MAE | kNN MAE | Baseline MAE |
|---|---|---|---|---|
| None (0.95–1.0) | 1,186 | 0.235 | 0.245 | 0.271 |
| Moderate (0.8–0.95) | 757 | 0.301 | 0.309 | 0.303 |
| Deep (below 0.8) | 241 | 0.546 | 0.581 | 0.561 |

Average sales, actual against predicted:

| Discount | Actual | Random forest | kNN | Baseline |
|---|---|---|---|---|
| None | 0.810 | 0.816 | 0.805 | 0.887 |
| Moderate | 0.997 | 1.027 | 0.987 | 0.998 |
| Deep | 1.659 | 1.567 | 1.495 | 1.391 |

- The forest beats the baseline on deep-discount days, where kNN did not, and comes closest to the actual spike.
- On moderate discounts it is level with the baseline.
- Errors on deep-discount days are more than twice those on ordinary days for every method.

### By day

| Day | Random forest | kNN | Baseline |
|---|---|---|---|
| Wed Jun 26 | 0.282 | 0.287 | 0.315 |
| Thu Jun 27 | 0.247 | 0.267 | 0.279 |
| Fri Jun 28 | 0.286 | 0.285 | 0.283 |
| Sat Jun 29 | 0.330 | 0.347 | 0.342 |
| Sun Jun 30 | 0.348 | 0.359 | 0.403 |
| Mon Jul 1 | 0.265 | 0.272 | 0.276 |
| Tue Jul 2 | 0.287 | 0.311 | 0.300 |

### By store

| Store | Random forest | kNN | Baseline |
|---|---|---|---|
| 18 | 0.295 | 0.310 | 0.313 |
| 154 | 0.261 | 0.261 | 0.258 |
| 182 | 0.293 | 0.300 | 0.329 |
| 235 | 0.286 | 0.301 | 0.310 |
| 343 | 0.326 | 0.345 | 0.358 |

### Feature importance

| Feature | Importance |
|---|---|
| 7-day average sales | 43.4% |
| Average sales to date | 20.9% |
| Sales on day t | 13.9% |
| Discount on day t+1 | 10.5% |
| Sales 7 days before | 5.9% |
| Discount on day t | 2.1% |
| Weekday of day t+1 | 1.5% |
| Stockout hours on day t | 0.8% |
| Holiday on day t+1 | 0.8% |
| Activity on day t | 0.2% |

Sales history sets the level; tomorrow's discount is the main driver of the change. This matches the kNN feature-removal test.

## 6. Findings and limits

1. **Best model:** the random forest beats kNN and the baseline on validation (every week) and on the unseen week.
2. **Stable:** its lead over the baseline held from validation to test (7.9% to 7.0% on MAE).
3. **Settings barely matter:** every combination tried beat the baseline; depth and leaf size moved MAE by 0.003 at most. As with kNN, the feature set mattered more than the settings.
4. **Better on the days that matter:** it is the only model ahead of the baseline on deep-discount days, which makes it the better engine for phase 2.
5. **Still under-predicts deep discounts:** 1.57 against an actual 1.66, and errors there are more than twice the ordinary-day error.
6. **No edge on moderate discounts:** level with the 7-day average.
7. **Phase 2 caveats unchanged:** discounts were not assigned at random, so the response is correlational; and choosing an "optimal" discount needs a margin assumption, since the model predicts volume only.
8. **Scope:** five stores in one city, one unseen week. Results may not carry to other cities or seasons.

## 7. Files

| File | Purpose |
|---|---|
| `scripts/06_tune_rf.py` | Five-fold tuning over depth and leaf size, and the tuning plot |
| `scripts/07_plot_validation_weeks.py` | Per-week validation plot for baseline, kNN and random forest |
| `scripts/08_test_rf.py` | Locked forest on the test week, with kNN and baseline side by side |
| `results/rf_tuning_leaf_features_*.csv` | Round 1 results (leaf size, features per split, weather) |
| `results/rf_tuning_summary.csv`, `results/rf_tuning_folds.csv`, `results/rf_tuning.png` | Round 2 results (depth and leaf size) |
| `results/validation_weeks.png` | Per-week validation plot |
| `results/test_summary.csv`, `results/test_predictions.csv` | Test week metrics and row-level predictions for all four methods |

The "Further checks" table was a one-off run and is not saved in a script.
