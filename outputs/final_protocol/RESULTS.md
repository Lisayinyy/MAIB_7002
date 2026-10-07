# Final protocol results

| Model                         |    MAE |   WAPE_pct |    MSE |   RMSE |   MAE_improvement_vs_7day_pct |
|:------------------------------|-------:|-----------:|-------:|-------:|------------------------------:|
| 7-day average baseline        | 0.3142 |    32.4360 | 0.2713 | 0.5209 |                        0.0000 |
| Same-weekday baseline         | 0.3998 |    41.2699 | 0.3808 | 0.6171 |                      -27.2348 |
| Ridge Regression              | 0.3075 |    31.7500 | 0.2531 | 0.5031 |                        2.1148 |
| Decision Tree Regressor       | 0.3139 |    32.4102 | 0.2438 | 0.4938 |                        0.0796 |
| Random Forest Regressor       | 0.2929 |    30.2436 | 0.2141 | 0.4627 |                        6.7590 |
| HistGradientBoostingRegressor | 0.2892 |    29.8541 | 0.2063 | 0.4543 |                        7.9600 |
| CatBoost Regressor            | 0.2878 |    29.7118 | 0.2042 | 0.4519 |                        8.3988 |

### Executed findings

- **Validation-selected model:** Random Forest Regressor; five-week mean MAE **0.3037**, versus **0.3284** for the 7-day average.
- **Final-period score for this model:** MAE **0.2929**, WAPE **30.24%**, MSE **0.2141**, RMSE **0.4627**.
- **Relative MAE improvement vs 7-day average:** **+6.76%**. Positive values mean lower prediction error; they do not mean more sales.
- **Validation consistency:** beats the 7-day baseline in **5 of 5** weeks.
- **Promotion ablation:** adding target-day discount and activity changes final-period MAE from **0.3106** to **0.2929** (+5.69% improvement).
- **Lowest observed final-period MAE, descriptive only:** CatBoost Regressor, **0.2878**. Model selection still uses validation.

### Discount-day diagnostics for the selected model

- None/light (>0.95): MAE 0.2362; +12.87% relative MAE improvement vs 7-day average (n=1,186).
- Moderate (0.80, 0.95]: MAE 0.3026; +0.24% relative MAE improvement vs 7-day average (n=757).
- Deep (<=0.80): MAE 0.5422; +3.29% relative MAE improvement vs 7-day average (n=241).

### What the experiment supports

This is a reproducible comparison of observed-sales forecasts on 312 selected store–SKU series.
The scenario function shows how a forecast can support a constrained discount choice. It does
not show that changing a discount will cause the predicted sales, improve real revenue, or reduce waste.

### Limitations

1. The sample is retrospectively selected from the full 90-day train period, all in one city; early CV folds do not have a prospectively selected cohort.
2. The team previously inspected the final week. It is a shared historical benchmark, with only seven target dates, not a newly untouched test or a significance claim.
3. Features are a documented reconstruction of the report; the original feature-generation source and ID list were not available for exact verification.
4. Planned target-day discount/activity availability is assumed. Promotion assignment is observational and may be confounded.
5. Recorded sales are censored by stock availability and normalized. There are no verified unit prices, inventory balances, causal demand curves or revenue/waste outcomes.
6. The hyperparameter search is deliberately small; the weekday representation and feature set were frozen before this run.

**Next evidence needed for deployment:** a later untouched period, documented decision-time inputs and price/inventory data, followed by a controlled policy evaluation. These are beyond this coursework forecasting comparison.
