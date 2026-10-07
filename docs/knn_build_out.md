# kNN build-out: next-day sales for discounted perishables

Record of how the kNN model was built, tuned and tested, with the numbers behind each decision.

## 1. Goal

Help a Dingdong category manager decide whether to discount a perishable product tomorrow, by predicting next-day sales for a store-product given tomorrow's discount.

- **Success test:** lower next-day sales error than a simple baseline.
- **Phase 2 (not yet built):** feed candidate discount rates into the model to suggest a discount. This is a model-based claim, not something the data can verify.

Two limits set by the data:

- The dataset has one discount rate per day, with no intraday markdown. "End-of-day discount" therefore means "tomorrow's daily discount".
- Discounts were not assigned at random, so the model's response to discount is a correlation, not a proven cause.

## 2. Data

Source: FreshRetailNet-50K (Dingdong, Hugging Face, CC BY 4.0).

| Item | Value |
|---|---|
| Series (store-product) | 50,000 |
| Stores / cities / products | 898 / 18 / 865 |
| Training period | 90 days, 2024-03-28 to 2024-06-25 |
| Unseen test period | 7 days, 2024-06-26 to 2024-07-02 (`eval.parquet`) |
| Missing values | None |
| Rows with no discount (exactly 1.0) | About 48% |
| Rows with at least one stockout hour | 44% |

- `sale_amount` is normalised, not real units, so errors only mean something relative to a baseline.
- Stockouts mean recorded sales understate true demand on those days. Stockout days were kept.

## 3. Store and product selection

A day counts as "discounted" when the discount is below 0.95. A series is usable when:

- it is discounted on 15–85% of days (removes never- and always-discounted products),
- it switches between discounted and not at least 6 times,
- it has zero sales on fewer than 20% of days.

Result: 4,997 series are never discounted, 1,872 always discounted, and 16,684 of 50,000 pass.

The top 5 stores by usable series, all in city 0:

| Store | Series | Usable |
|---|---|---|
| 343 | 152 | 64 |
| 18 | 164 | 64 |
| 235 | 162 | 63 |
| 182 | 160 | 61 |
| 154 | 131 | 60 |

Working set: 312 series, 122 products, 28,080 rows after building features.

## 4. Does discount carry signal?

Sales relative to each series' own average, by discount rate (top 5 stores):

| Discount rate | Rows | Relative sales |
|---|---|---|
| 0.95–1.0 (none) | 16,906 | 0.86 |
| 0.9–0.95 | 3,938 | 1.06 |
| 0.8–0.9 | 6,248 | 1.17 |
| 0.7–0.8 | 2,072 | 1.44 |
| 0.6–0.7 | 897 | 1.57 |
| below 0.6 | 203 | 3.09 |

## 5. Target and features

**Target:** sales on day t+1 for the same store and product. One row is one store-product-day. Every feature is known on the evening of day t.

**Why history features are needed.** The raw columns describe the day, not the product. kNN with raw columns only scored MAE 0.558 against a baseline of 0.339 in an early check; adding the product's own sales history brought it to 0.364.

**Locked feature set (10):**

| Feature | Day |
|---|---|
| Sales | t |
| Sales 7 days before the target day | t−6 |
| Mean sales over the last 7 days | up to t |
| Mean sales to date | up to t |
| Stockout hours | t |
| Discount | t |
| Discount (the manager's decision) | t+1 |
| Activity flag | t |
| Holiday flag | t+1 |
| Weekday | t+1 |

**Considered and left out:**

| Item | Reason |
|---|---|
| Weather on day t (rain, temperature, humidity) | Removing it lowered error (section 8) |
| Weather on day t+1 | No gain; 0.88–0.97 correlated with day t |
| Activity flag on day t+1 | No gain; largely repeats tomorrow's discount (92% of activity days are discounted) |
| Stockout hours on day t+1 | Not known in advance; would leak the answer |
| Hourly sales and stock lists, product ids | Excluded from the first model by design |

## 6. Cleaning and splits

- No duplicate store-product-days, no date gaps, no missing values in the 312 series.
- The first 7 days of each series are dropped because the 7-day features need a full week (2,184 rows).
- Standardization is fitted on the training rows of each fold only.

Splits are by time. For each validation week the model trains on every row dated before that week.

| Part | Dates | Rows |
|---|---|---|
| Train only | Apr 4 – May 21 | 14,976 |
| Validation week 1 | May 22 – 28 | 2,184 |
| Validation week 2 | May 29 – Jun 4 | 2,184 |
| Validation week 3 | Jun 5 – 11 | 2,184 |
| Validation week 4 | Jun 12 – 18 | 2,184 |
| Validation week 5 | Jun 19 – 25 | 2,184 |
| Test (unseen) | Jun 26 – Jul 2 | 2,184 |

**Why not random splits with replacement.** Neighbouring days of the same product share most of their history, so random splits put near-copies on both sides. Mean MAE over five rounds, all 13 original features:

| k | Time-ordered | Random, no overlap | Random with replacement |
|---|---|---|---|
| 1 | 0.429 | 0.377 | 0.139 |
| 5 | 0.342 | 0.303 | 0.261 |
| 15 | 0.327 | 0.294 | 0.280 |
| 40 | 0.327 | 0.296 | 0.291 |

Random splits flatter the model and pick the wrong k (k=1 looks best with replacement and is worst in honest testing).

## 7. Metrics and baseline

- **MAE:** average absolute error, in sales units. Main tuning criterion.
- **WAPE:** total absolute error divided by total sales.
- **MSE:** reported as the academic metric. It penalises large misses more heavily.

Baseline: the mean of the last 7 days' sales. It beats "same weekday last week" on every fold.

| Validation week | 7-day average MAE |
|---|---|
| 1 | 0.309 |
| 2 | 0.316 |
| 3 | 0.362 |
| 4 | 0.327 |
| 5 | 0.328 |
| Average | 0.328 |

## 8. Tuning

All figures are average MAE over the five validation weeks.

### Round 1: neighbours and feature weighting, with weather (13 features)

A feature weight multiplies the standardized feature, so a weight of 2 makes it count twice as much in the distance.

| Candidate | Best k | MAE | Weeks beating baseline |
|---|---|---|---|
| Baseline | – | 0.328 | – |
| Equal weights | 25 | 0.327 | 3 of 5 |
| Prior weighting (sales averages, discounts, holiday, weekday ×2) | 25 | 0.318 | 5 of 5 |
| Evidence-led (sales averages, tomorrow's discount ×2; weather, stockout ×0.5) | 25 | 0.312 | 5 of 5 |

An earlier prior that boosted weather and stockout hours made the model worse (0.332 at k=15 against 0.327 for equal weights).

### Round 2: weather removed (10 features)

| Candidate | MAE with weather | MAE without |
|---|---|---|
| Equal weights | 0.327 | 0.312 |
| Prior weighting | 0.318 | 0.311 |
| Evidence-led | 0.312 | 0.309 |

- Dropping weather mattered more than any weighting.
- Once weather is gone the three weightings are within 0.003, well inside the week-to-week variation (about 0.02). Weighting was dropped.
- Weather has no measurable link to the day's sales lift: correlation of about −0.01 with sales divided by the 7-day average.

### The k curve (10 features, equal weights)

| k | 3 | 5 | 10 | 15 | 25 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|---|---|
| MAE | 0.347 | 0.330 | 0.319 | 0.314 | 0.312 | 0.312 | 0.313 | 0.316 |

Small k follows noise; large k averages over products that are too different. The bottom is flat from 15 to 60. MSE also picks k=25. Plot: `results/knn_stage1_three_weightings.png`.

### Further checks (one-off runs, k=25, change against 0.3120)

| Check | MAE | Change |
|---|---|---|
| Sales on day t at ×2 | 0.3125 | +0.0005 |
| Add activity t+1 and holiday t+1, both ×2 | 0.3129 | +0.0009 |
| Manhattan distance | 0.3105 | −0.0015 |
| Closer neighbours count more | 0.3115 | −0.0005 |
| Weekday as seven yes/no columns | 0.3174 | +0.0054 |
| Weekday removed | 0.3096 | −0.0024 |
| Min-max scaling | 0.3225 | +0.0105 |
| Robust scaling | 0.3103 | −0.0017 |
| Target = sales ÷ 7-day average, log-scaled features | 0.3079 | −0.0041 |
| Train on last 28 days only | 0.3156 | +0.0036 |

No option moved MAE by more than 0.004, so tuning stopped here.

### Which features carry the model (remove one at a time, k=25)

| Feature removed | Change in MAE |
|---|---|
| Discount on day t+1 | +0.0139 |
| 7-day average sales | +0.0095 |
| Sales on day t | +0.0082 |
| Average sales to date | +0.0024 |
| Discount on day t | +0.0007 |
| Sales 7 days before | +0.0004 |
| Holiday on day t+1 | +0.0001 |
| Stockout hours on day t | −0.0002 |
| Activity on day t | −0.0009 |
| Weekday of day t+1 | −0.0024 |

Tomorrow's discount is the most important single feature.

## 9. Locked design

- 10 features (section 5), standard scaling
- Euclidean distance, all neighbours counted equally
- k = 25
- Validation: MAE 0.312, WAPE 32.7%, MSE 0.238, against baseline MAE 0.328, WAPE 34.5%, MSE 0.270. Ahead in all five weeks.

## 10. Unseen test week (June 26 – July 2)

Fitted once on all 25,896 training rows; scored on 2,184 rows.

| Method | MAE | WAPE | MSE | RMSE |
|---|---|---|---|---|
| kNN (k=25) | 0.304 | 31.4% | 0.240 | 0.490 |
| Baseline (7-day average) | 0.314 | 32.4% | 0.271 | 0.521 |
| Same weekday last week | 0.400 | 41.3% | 0.381 | 0.617 |

kNN is 3.2% better than the baseline on MAE and 11.5% better on MSE. The validation gap was 5% on MAE, so about a third did not carry over.

By tomorrow's discount:

| Discount | Rows | kNN MAE | Baseline MAE | Actual mean sales | kNN mean prediction | Baseline mean prediction |
|---|---|---|---|---|---|---|
| None (0.95–1.0) | 1,186 | 0.245 | 0.271 | 0.810 | 0.805 | 0.887 |
| Moderate (0.8–0.95) | 757 | 0.309 | 0.303 | 0.997 | 0.987 | 0.998 |
| Deep (below 0.8) | 241 | 0.581 | 0.561 | 1.659 | 1.495 | 1.391 |

- kNN is better on four of seven days and in four of five stores.
- The whole gain comes from non-discounted days, where the baseline over-predicts.
- On discounted days kNN is slightly worse on MAE. It moves in the right direction on deep discounts but under-shoots.

## 11. Findings and limits

1. **Success test met:** tuned kNN beats the 7-day average on validation and on the unseen week.
2. **The margin is modest:** 3–5% on MAE, 11–15% on MSE.
3. **Feature choice mattered more than any setting:** removing weather was the largest single gain; k, distance measure and weighting were all within noise once it was gone.
4. **Prior expectations were partly wrong:** weather and stockout hours did not help; sales history and tomorrow's discount did.
5. **Weak on the days that matter most:** kNN does not beat the baseline on discounted days, because averaging 25 neighbours pulls predictions towards typical sales. This limits it as the engine for phase 2.
6. **Selection optimism:** many options were compared on the same five weeks, so validation figures are slightly flattering. The test week is the honest number.
7. **No gradient descent:** kNN stores the training rows and has no iterative training, so there is no loss curve. The k curve is a search over a setting, not a training curve.

## 12. Files

| File | Purpose |
|---|---|
| `scripts/01_select_series.py` | Store and series selection |
| `scripts/02_build_features.py` | Cleaning, features, splits |
| `scripts/03_tune_knn.py` | Five-fold tuning (currently equal weights against sales on day t ×2) |
| `scripts/04_plot_knn_stage1.py` | Tuning plot |
| `scripts/05_test_knn.py` | Locked model on the test week |
| `results/knn_stage1_with_weather_*` | Round 1 results (13 features, three weightings) |
| `results/knn_stage1_three_weightings_*` | Round 2 results (10 features, three weightings) |
| `results/knn_stage1_*` | Latest run (equal weights against sales on day t ×2) |
| `results/knn_test_summary.csv`, `results/knn_test_predictions.csv` | Test week metrics and row-level predictions |

The checks in "Further checks", the feature-removal table, the random-split comparison and the weather t+1 and activity t+1 tests were one-off runs and are not saved in a script.
