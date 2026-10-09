"""Create the English submission notebook from explicit, reviewable teaching cells."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))


md('''
# Daily discount decisions for perishable food
**MAIB7002 · Group Q · Lisa's evaluation and pipeline extension**

Aashish Omprakash Pareek · Yuanyuan Yin · Lingyu Chen

We estimate next-day observed sales, decide when the evidence is sufficient to compare discounts,
and choose a supported discount subject to a sales-value constraint. The business user is a fresh-food
category manager. Better forecasts are measured; higher revenue or lower waste is not established.

This notebook extends Aashish's `pricing-ml-ap` commit `5b52099` without replacing his notebook or helper modules.
The Chinese guides in `docs/` explain both versions line by line. This English notebook is the presentation and runnable demonstration.

**Reading order:** business question → data and decision timing → models → matched evaluation → errors → decision rules → real case → limitations.
''')
code('''
from pathlib import Path
import sys, json
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    raise RuntimeError("Open this notebook from the repository root")
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from finalproject_pricingml import config as C, v2
OUT = ROOT / "results" / "v2"
pd.set_option("display.precision", 5)
%matplotlib inline

# False: verify saved artifacts and run a small live model demonstration offline.
# True: rebuild features and rerun all 66 model fits; requires the two verified raw files.
RERUN_FULL_EXPERIMENT = False
if RERUN_FULL_EXPERIMENT:
    v2.run(OUT)
verification = v2.verify_saved(OUT)
display(verification)
''')
md('''
## 1. Business decision and evidence

The manager first asks **whether a discount can be responsibly recommended**, then **how deep it should be**.
Sales history, previous-day stockout hours and activity, tomorrow's calendar, and the proposed daily discount
are available inputs. Tomorrow's realized sales, weather and stockouts are not predictors.

Success for the forecasting component means lower **mean absolute error (MAE)** than the 7-day mean baseline on the same rows.
The decision component is assessed for historical support, complete output coverage, transparent reasons and constraint compliance.
Its predicted gains are scenarios, not an observed policy return.

The dataset has no inventory ages, batch-level expiry dates or unit prices. A week without a stockout does not mean that stock is a week old.
''')
md('''
## 2. Public data and unit of observation

Source: [Dingdong FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), CC BY 4.0.
One row is one store-product-day. `sale_amount` is normalized observed sales; **an MAE of 0.28 does not mean 0.28 loaves or dollars**.
Stockouts censor observed sales: customers' unmet demand is not recorded as sales.

We use the teammate's 312 store-product series in five stores in city 0, selected for discount variation from the original 90-day training period.
This retrospective cohort makes comparison possible but creates selection optimism within the validation period and limits generalization.
Public raw files are verified against fixed SHA-256 values. The small derived feature table and saved predictions are included for offline inspection.
''')
code('''
feat = pd.read_parquet(OUT / "features.parquet")
plan = json.loads((OUT / "experiment_plan.json").read_text())
selection = json.loads((OUT / "selection.json").read_text())
comparison = pd.read_csv(OUT / "comparison.csv")
oof = pd.read_csv(OUT / "validation_predictions.csv", parse_dates=["dt"])
test_predictions = pd.read_csv(OUT / "test_predictions.csv", parse_dates=["dt"])
print("Series:", len(feat[C.KEY].drop_duplicates()), "Stores:", feat.store_id.nunique())
display(feat.groupby(["split", "val_fold"]).agg(rows=("dt", "size"), first=("dt", "min"), last=("dt", "max")))
display(pd.DataFrame(plan["environment"].items(), columns=["package", "version"]))
''')
md('''
## 3. Feature timing and a real input row

The date in a row is the **target day**, called day t+1. `sales_t` is yesterday's observed sales;
`sales_lag7` is sales seven days before the target date. Rolling windows are shifted **before** aggregation.
For example, `shift(1).rolling(7).mean()` excludes the target day's sales.

The 10 base inputs match the teammate implementation. In particular, `activity_t` is **yesterday's** activity flag;
the older main branch used the target day's planned activity. Identical baseline scores do not establish identical feature sets.

The history extension adds sales at lags 2 and 3, means over 3 and 14 days, 7-day variability,
the difference between the 3-day and 7-day means, mean previous discount and recent stockout-day count.
It also passes store and product IDs to CatBoost as **categories**, not numeric magnitudes.
There are 18 numeric inputs and two categorical inputs. All models keep exactly the same benchmark rows.
''')
code('''
example = feat[feat.split == "test"].sort_values(C.ROW_KEY).iloc[[0]]
display(example[C.ROW_KEY + C.FEATURES + v2.EXTRA + v2.IDS + [C.TARGET]].T.rename(columns={example.index[0]: "real case"}))
feature_dictionary = pd.DataFrame({
    "group": ["Base"]*len(C.FEATURES) + ["Additional history"]*len(v2.EXTRA) + ["Categorical ID"]*len(v2.IDS),
    "feature": C.FEATURES + v2.EXTRA + v2.IDS})
display(feature_dictionary)
''')
md(r'''
## 4. What each method learns

| Method | Inputs and learning | Main technical choice |
|---|---|---|
| 7-day mean baseline | Average the previous seven observed sales | No fitted model |
| Same-weekday baseline | Reuse sales from seven days earlier | No fitted model |
| Ridge | A linear combination of standardized inputs with a squared-coefficient penalty | alpha = 10 |
| Decision tree | Recursive feature splits reduce squared error; predict a leaf mean | Depth 6; at least 20 rows per leaf |
| Random Forest (RF) | Average 300 bootstrap-trained trees | Depth 10; leaf size 3; half the features per split |
| HistGradientBoosting | Add trees that correct the current model's errors | 300 iterations; 15 leaves; learning rate 0.05 |
| CatBoost | Sequential symmetric trees, with categorical target statistics for ID-enabled variants | Depth 5; 500 iterations; learning rate 0.03 |
| kNN reference | Average the 25 nearest standardized training rows | Scaling fitted inside each training fold |

For boosting, $F_m(x)=F_{m-1}(x)+\eta h_m(x)$: $x$ is the input row, $F_m$ the prediction after step $m$,
$h_m$ the new tree, and $\eta$ the learning rate. The new tree approximates the direction that reduces the training loss.
With squared loss the residual magnitude matters strongly; MAE is less sensitive to individual large errors and targets a conditional median.
Consequently a MAE-trained forecast is not automatically an expected demand or expected revenue estimate.

For a blend, $\hat y=w\hat y_{RF}+(1-w)\hat y_{CB}$: $\hat y$ is predicted normalized sales and $w$ the RF weight.
The components can make different errors, so averaging can help; their errors are strongly correlated, so gains are limited.
''')
md('''
## 5. Recorded experiment and chronological evaluation

Eleven standalone designs and four blends were declared in code before this run's final-period scoring; two baselines are reported first.
The five validation weeks begin on May 22, May 29, June 5, June 12 and June 19, 2024.
Each fit uses only dates before its validation week. We average weekly MAE and choose the smallest value.
The final-period dates are June 26–July 2. Forecasts are **rolling one day ahead**: actual earlier days update the lags within a week,
while the model itself is fitted only at the beginning of that week. This is not a seven-day-ahead forecast.

**Development history matters:** the final week and validation folds have already been used by earlier versions.
This extension is exploratory and its final-period results are a shared benchmark, not a new untouched holdout.
The new selection is saved before final scoring to prevent additional within-run test-based changes; this does not erase past exposure.

The controlled technical experiment compares **RMSE versus MAE training loss with identical extended features, depth, iterations, learning rate and regularization**.
We expected MAE training to help the primary absolute-error metric, potentially at the expense of large-error sensitivity.
The extended feature package is also compared with the base package, but several features change together, so that contrast does not identify an individual feature's causal contribution.
''')
code('''
display(pd.DataFrame(plan["candidates"]).fillna("-"))
print("Selection:", selection["selected_model"])
print("Best standalone:", selection["best_standalone"])
print("Locked before this run's final scoring:", selection["selected_before_test_scoring_utc"])
display(comparison[comparison.model.isin(["CatBoost history + IDs", "CatBoost history + IDs MAE"])][
    ["model", "validation_mae", "test_mae", "test_rmse"]])
''')
md(r'''
## 6. Results

For $n$ observed rows, actual sales $y_i$ and predictions $\hat y_i$:

$$MAE=\frac{1}{n}\sum_i|y_i-\hat y_i|,\qquad RMSE=\sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2}.$$

WAPE is $\sum_i|y_i-\hat y_i|/\sum_i|y_i|$. Lower is better for all three.
Validation MAE chooses the model. RMSE diagnoses large misses and WAPE provides relative scale.
We do not switch the winner just because another design has a lower final-period RMSE.
''')
code('''
display(comparison[["model", "validation_mae", "test_mae", "test_rmse", "test_wape", "selected"]])
winner = selection["selected_model"]
metrics = comparison.set_index("model")
for ref in ["7-day mean", "Teammate blend 50/50"]:
    gain = 100 * (metrics.loc[ref, "test_mae"] - metrics.loc[winner, "test_mae"]) / metrics.loc[ref, "test_mae"]
    print(f"Final-period MAE reduction versus {ref}: {gain:.2f}%")
folds = pd.read_csv(OUT / "validation_folds.csv")
paired = folds[folds.model.isin(["Teammate blend 50/50", winner])].pivot(index="fold", columns="model", values="mae")
display(paired)
print("Weeks better than teammate:", int((paired[winner] < paired["Teammate blend 50/50"]).sum()), "of 5")

show = ["7-day mean", "kNN reference", "Teammate RF", "Teammate CatBoost", "Teammate blend 50/50", winner]
fig, ax = plt.subplots(figsize=(10, 4.5))
pos = np.arange(len(show)); width = .36
ax.barh(pos+width/2, metrics.loc[show, "validation_mae"], width, label="Validation", color="#82988f")
ax.barh(pos-width/2, metrics.loc[show, "test_mae"], width, label="Final-period benchmark", color="#127263")
ax.set_yticks(pos, ["7-day mean", "kNN", "Teammate RF", "Teammate CatBoost", "Teammate 50/50", "Lisa 25/75"])
ax.invert_yaxis(); ax.set_xlim(0, .36); ax.set_xlabel("MAE (normalized observed sales)")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=180); plt.show()
''')
md('''
## 7. Errors and practical limits

Inspect all discount bands, all stores and all days, rather than selecting only favorable cases.
The largest errors below are selected mechanically by absolute error.
Target-day stockout status is used **only to diagnose errors after the fact**, never as a model input.

The paired interval resamples entire store-product series, keeping their seven days together.
It is descriptive: five stores and one week are too little to establish broad reliability, and shared-store shocks and selection optimism are not corrected.
''')
code('''
slices = pd.read_csv(OUT / "error_slices.csv")
display(slices[slices["slice"] == "discount"])
display(slices[(slices["slice"] == "store") & (slices.model == winner)])
display(pd.read_csv(OUT / "largest_errors.csv").head(5))
display(json.loads((OUT / "paired_comparison.json").read_text()))
''')
md('''
## 8. A live, traceable model example

For the first store-product-day in the final period, refit only the selected blend members on the original training rows.
No parameter is changed. Compare the reproduced prediction to the saved result.
The member predictions, blend arithmetic and RF tree path expose how the final number is constructed.
This can run using the included small feature table without downloading the full raw dataset.
''')
code('''
training = feat[feat.split == "train"]
specs = {s["name"]: s for s in plan["candidates"]}
members = selection["blend_weights"].get(winner, {winner: 1.})
live_models = {}
example_values = []
from threadpoolctl import threadpool_limits
for name, weight in members.items():
    spec = specs[name]
    with threadpool_limits(limits=4):
        model = v2.estimator(spec).fit(training[v2.columns(spec)], training[C.TARGET])
    live_models[name] = model
    prediction = float(v2.predict(model, example, spec)[0])
    example_values.append({"member": name, "weight": weight, "prediction": prediction, "weighted_part": weight*prediction})
example_table = pd.DataFrame(example_values)
display(example_table)
live_prediction = example_table.weighted_part.sum()
case_key = example.iloc[0]
saved_case = test_predictions[(test_predictions.store_id == case_key.store_id) &
    (test_predictions.product_id == case_key.product_id) & (test_predictions.dt == case_key["dt"])].iloc[0]
print(f"Blend = {live_prediction:.6f}; actual observed sales = {case_key[C.TARGET]:.6f}")
print("Maximum replay difference:", abs(live_prediction - saved_case[winner]))
assert np.isclose(live_prediction, saved_case[winner], atol=1e-6)

forest = live_models.get("Teammate RF")
if forest is not None:
    x = example[C.FEATURES].to_numpy()
    tree = forest.estimators_[0]
    leaf = tree.apply(x)[0]
    print("First RF tree leaf prediction:", float(tree.tree_.value[leaf, 0, 0]))
    print("RF prediction = mean of", len(forest.estimators_), "tree predictions:",
          float(np.mean([t.predict(x)[0] for t in forest.estimators_])))
    path = tree.decision_path(x).indices
    for node in path:
        feature = tree.tree_.feature[node]
        if feature >= 0:
            value, threshold = x[0, feature], tree.tree_.threshold[node]
            print(f"{C.FEATURES[feature]} = {value:.4f} {'<=' if value <= threshold else '>'} {threshold:.4f}")
''')
md(r'''
## 9. Whether to discount, then how much

1. **Check recent supply evidence.** A stockout yesterday triggers supply review because sales may be censored. It does not prove all stock was sold or that tomorrow's stock is fresh.
2. **Check historical support.** Each candidate needs at least three training days within ±0.025 for this store-product. A supported full-price reference is also required.
3. **Compare scenarios.** Hold known context fixed, vary only the candidate rate $d$, and compute forecast $\hat y(d)$.
4. **Require agreement.** The selected model, RF and reference CatBoost must each predict at least 5% more sales while keeping $d\hat y(d)$ at least 95% of their own full-price value.
5. **Choose depth.** Maximize the selected forecast within those candidates; use the highest rate (mildest discount) within 1% of the best sales prediction.

Here $d=0.9$ means **10% off**, not 90% off. $d\hat y(d)$ is a relative sales-value proxy; it is neither currency nor profit.
The thresholds are explicit illustrative choices, not learned optimal business rules. Cross-model agreement is a sensitivity check, not a confidence interval.

Each input produces one output: a discount, keep full price, or a named review/abstention reason.
We never infer inventory age from a stockout count. Candidate support reduces extrapolation but does not remove confounding.
''')
code('''
policy = json.loads((OUT / "policy_summary.json").read_text())
recs = pd.read_csv(OUT / "recommendations.csv", parse_dates=["dt"])
responses = pd.read_csv(OUT / "discount_response.csv", parse_dates=["dt"])
display(pd.DataFrame(policy["status_counts"].items(), columns=["status", "rows"]))
print(f"Explicit output coverage: {policy['output_rows']}/{policy['rows']}")
print(f"Actionable recommendation coverage: {policy['decision_coverage']:.1%}")
print(f"Manual review / abstention: {policy['manual_review_or_abstain_share']:.1%}")
print("These are decision diagnostics, not realized policy rewards.")
''')
md('''
## 10. A real discount scenario and a failure case

Use the first supported recommendation in key order, not the most impressive predicted gain.
Only the recorded discount has an observed sales outcome; alternative-price outcomes are unknown.
Then inspect the first supply-review case to demonstrate when the system declines to decide.
''')
code('''
case = recs[recs.status == "recommend_discount"].sort_values(C.ROW_KEY).iloc[0]
mask = ((responses.store_id == case.store_id) & (responses.product_id == case.product_id) & (responses.dt == case["dt"]))
case_curve = responses.loc[mask].sort_values("rate", ascending=False).copy()
case_curve["selected_value_proxy"] = case_curve.rate * case_curve.selected_sales
display(case.to_frame("recommendation"))
display(case_curve[["rate", "support_days", "supported", "selected_sales", "rf_sales", "cat_sales", "selected_value_proxy"]])
actual_case = feat[(feat.store_id == case.store_id) & (feat.product_id == case.product_id) & (feat.dt == case["dt"])]
display(actual_case[C.ROW_KEY + ["sales_t", "sales_mean7", "stockout_hours_t", "discount_next", C.TARGET]])
display(recs[recs.status == "review_stockout"].sort_values(C.ROW_KEY).head(1))

fig, ax = plt.subplots(figsize=(8, 3.5))
supported = case_curve[case_curve.supported]
ax.plot(supported.rate, supported.selected_sales, "o-", color="#127263", label="Supported candidate forecasts")
ax.scatter([case.recommended_rate], [case.predicted_sales], s=100, facecolors="none", edgecolors="#c76a20", linewidths=2, label="Recommendation")
ax.set_xlabel("Price / full price (1.0 = no discount)"); ax.set_ylabel("Predicted normalized sales")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "worked_scenario.png", dpi=180); plt.show()
''')
md('''
## 11. Transfer stress test after model selection

After locking the winner, a separate check chose 100 eligible series outside city 0 using a fixed SHA-256 ordering of store-product keys from training data only.
The three required estimators were fitted on the original 312 series, with no fitting on the new series and no retuning.
The resulting 700 evaluation rows span 93 new stores in 16 other cities, including 47 product IDs not seen by the fitted models.
Earlier observed days of each new series still supply the lag features, as required for rolling daily prediction.

The new blend does **not** outperform the teammate blend here, and its RMSE is worse than even the 7-day mean.
Thus the measured improvement applies to the original cohort; it does not establish general transfer to new stores.
We retain the original validation selection and report this failure instead of changing the winner after seeing these outcomes.
This is a locked transfer stress test, not a claim that no group member ever saw these public data before.
''')
code('''
transfer_dir = ROOT / "results" / "v2_transfer"
display(pd.read_csv(transfer_dir / "summary.csv"))
display(json.loads((transfer_dir / "run_receipt.json").read_text()))
''')
md('''
## 12. What improved, and what remains unproved

The extension provides a reproducible matched comparison, a real worked example, richer past-only features,
an objective aligned with MAE, keyed blend alignment and a complete decision output with abstention.
The teammate's original code and saved results remain available and are credited.

The strongest unresolved issues are observational discount confounding, stockout censoring, limited geography and repeated use of evaluation periods.
Most eligible cases still receive a discount; review rules do not demonstrate that these discounts are beneficial.
MAE forecasts target a median, so multiplying them by price cannot be called expected-revenue maximization.
Stock quantity, replenishment, expiry and cost data are required to assess actual inventory, profit or waste outcomes.
A prospective pilot or defensible causal evaluation is needed before operational use.

No reinforcement-learning model is used. The available data do not provide a verified sequential inventory environment or reliable off-policy policy-value estimate.
''')
md('''
## 13. Reproduce and demonstrate

From a clean repository copy:
```bash
docker compose up --build -d
docker compose exec lab python scripts/jupyter_v2_url.py
docker compose exec lab python scripts/run_v2.py --verify-only
docker compose exec lab python -m unittest discover -s tests -v
```

The saved-results demonstration and live worked example run offline. To reproduce all data preparation and 66 fits:
```bash
docker compose exec lab python scripts/download_public_data.py
docker compose exec lab python scripts/run_v2.py
docker compose exec lab python scripts/execute_lisa_notebook.py
```

See `docs/assignment_requirements.md` for the brief, the Chinese walkthroughs for study, and `docs/CONTRIBUTIONS_TEMPLATE.md` for mandatory personal declarations.
This notebook and the poster are prepared materials; the authors must verify their contributions and submit through Moodle themselves.
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, ROOT / "02_lisa_pipeline.ipynb")
print(f"Wrote {len(cells)} cells")
