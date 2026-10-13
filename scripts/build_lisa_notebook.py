"""Build the business-first English notebook from reviewable narrative and code cells."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip(), metadata={
        "jupyter": {"source_hidden": True}, "tags": ["story-code"]}))


md(r'''
# Should we discount tomorrow?

MAIB7002 · Group Q

Aashish Omprakash Pareek · Yuanyuan Yin · Lingyu Chen

We use FreshRetailNet-50K to predict next-day sales for a store-product, then compare candidate discounts using those predictions. The notebook follows the original model comparison, tests an extension with more sales history, and examines why a lower forecast error is not enough to justify a pricing policy.

| Sections | Contents |
|---|---|
| 1–4 | Goal, data exploration, features and evaluation |
| 5–8 | Model comparisons, extension and error analysis |
| 9–14 | Discount recommendations, sensitivity checks and limits |
| Appendices | Worked model calculations, full results and reproduction commands |

Code and saved outputs are included throughout. The HTML copy has expandable code blocks.
''')

md(r'''
## 1. Goal

Help a Dingdong category manager choose tomorrow's discount for a perishable product in one store. Unsold food has a limited selling window, but a discount can also reduce the value of sales that would have happened at full price. The aim is to sell more without giving away too much value.

We split the work into two parts:

1. Predict next-day sales. Compare models with the previous seven days' average on the same later dates. Select by validation MAE, the average absolute error, and report RMSE to check large misses.
2. Compare candidate discounts. Use the selected model to estimate sales at each rate, then apply history, value and model-agreement checks. The output is a specific discount, full price, or a request for the manager to review the case.

The original bakery idea involved half-hourly markdowns. This dataset records one rate per day, so our scope is a daily decision. Its records can test forecast accuracy; they cannot establish whether our proposed discounts would increase actual revenue or reduce waste.
''')

md(r'''
## 2. Data

We selected [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) because it provides the daily sales, discount rates, stockout records and store-product identifiers needed to forecast next-day sales and compare discount scenarios for perishable products.

We first explore the full dataset, select a manageable group of stores, and compare that sample with the wider source before preparing the model inputs.
''')

md(r'''
### 2.1 The full FreshRetailNet dataset

The source covers 898 stores in 18 cities, with 50,000 store-product histories in the original 90-day training file. A history follows one product in one store across dates. We begin with the full source so the later five-store sample has a clear context.

The tables group the available fields by their role in the project. The donut shows the composition of the 50,000 store-product histories by category, with smaller categories grouped as Other. The horizontal chart shows how the 898 stores are spread across cities. These are two different denominators.

Category and city identifiers are anonymized. City 0 is highlighted because the later selection happens to fall there, not because we selected by city.
''')

code(r'''
from pathlib import Path
import sys, json, hashlib, subprocess
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    raise RuntimeError("Open this notebook from the repository root")
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from finalproject_pricingml import config as C, v2
from finalproject_pricingml import storytelling as story
from finalproject_pricingml import data_exploration, model_labels
OUT = ROOT / "results" / "v2"
pd.set_option("display.precision", 5)
%matplotlib inline

# Read verified saved results; the appendix performs three modest worked-example fits.
# Full experiments use the explicit CLI recipe in appendix C, including dependent diagnostics.
verification = v2.verify_saved(OUT)
feat = pd.read_parquet(OUT / "features.parquet")
plan = model_labels.structure(json.loads((OUT / "experiment_plan.json").read_text()))
selection = model_labels.structure(json.loads((OUT / "selection.json").read_text()))
comparison = model_labels.study_order(model_labels.frame(pd.read_csv(OUT / "comparison.csv")))
metrics = comparison.set_index("model")
winner = selection["selected_model"]
test_predictions = model_labels.frame(pd.read_csv(OUT / "test_predictions.csv", parse_dates=["dt"]))
evidence = json.loads((OUT / "story_evidence.json").read_text())
for key, filename in [("features","features.parquet"), ("discount_response","discount_response.csv"),
                      ("recommendations","recommendations.csv"), ("policy_summary","policy_summary.json")]:
    assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest() == evidence["inputs_sha256"][key], "Rebuild story evidence after changing results"
assert evidence["inputs_sha256"]["raw_train"] == plan["raw_sha256"]["train"]
policy = json.loads((OUT / "policy_summary.json").read_text())
recs = pd.read_csv(OUT / "recommendations.csv", parse_dates=["dt"])
responses = pd.read_csv(OUT / "discount_response.csv", parse_dates=["dt"])
print("Saved experiment verified. No full model search is run by default.")

# This compact full-data summary is rebuilt from the same verified public file.
data_overview = json.loads((ROOT / "results/data_eda/overview.json").read_text())
assert data_overview["inputs_sha256"]["raw_train"] == plan["raw_sha256"]["train"]
assert data_overview["inputs_sha256"]["features"] == hashlib.sha256((OUT/"features.parquet").read_bytes()).hexdigest()
''')

code(r'''
overview_figure, scope_table, field_groups = data_exploration.overview(data_overview)
display(scope_table[["Coverage", "Full training file"]])
display(field_groups)
plt.show()
''')

md(r'''
The dataset combines sales and price records with stockout, promotion, calendar and weather information. Its hourly arrays describe sales and availability, while the price-rate field is daily, so the project uses a daily discount decision. The sales values are already normalized by the source and are not direct item counts or currency.

With 50,000 histories available, the next step is to identify a smaller group that contains both regular selling and repeated changes in discount status.
''')

md(r'''
### 2.2 Selecting five stores

We need histories with price variation. An always-discounted product gives little comparison with ordinary trading, while a never-discounted product gives little information about discounts.

We keep the original study's screening rules so both pipelines use the same cases:

| Rule | Reason |
|---|---|
| Discounted on 15%–85% of days | Retain observations in both price states |
| At least six state switches | Avoid relying on one long promotion period |
| Zero sales on fewer than 20% of days | Focus on products with regular recorded sales |

For this screen, discounted means a price rate below 0.95. Exactly 0.95 is in the other group. A move from 0.90 to 0.80 is not a switch because both rates are below the threshold. The other group includes small discounts as well as full price, so this screen does not guarantee a full-price comparison. We check support for each candidate separately when making recommendations.

We then rank stores by their number of eligible series and take the top five. All five happen to be in city 0; neither city nor model performance was a selection criterion. Only eligible products in those stores are retained.
''')

code(r'''
funnel = evidence["selection_funnel"]
display(pd.DataFrame([
    ["Public training histories", funnel["all_series"]],
    ["Keep 15%–85% discounted days", funnel["after_discount_share_filter"]],
    ["Also require at least six state switches", funnel["after_discount_share_and_switch_filters"]],
    ["Also require fewer than 20% zero-sales days", funnel["after_all_three_filters"]],
    ["Eligible histories in the selected five stores", funnel["selected_series_in_top_five"]],
], columns=["Selection stage", "Store-product series"]))
stores = pd.DataFrame(funnel["top_five_stores_ranked"])
display(stores[["store_id", "city_id", "n_eligible"]].rename(columns={
    "store_id": "Store ID", "city_id": "City ID", "n_eligible": "Eligible products"}))
print("Distinct products across these stores:", funnel["selected_unique_products"])
assert len(feat[C.KEY].drop_duplicates()) == funnel["selected_series_in_top_five"]
fig, ax = plt.subplots(figsize=(9, 2.2))
ax.axis("off")
steps = [("50,000", "public histories"),
         (f'{funnel["after_all_three_filters"]:,}', "eligible histories"),
         ("5", "stores with most eligible histories"), ("312", "selected histories")]
for i, (count, label) in enumerate(steps):
    x = .12 + i*.25
    ax.text(x, .64, count, ha="center", fontsize=22, color="#127263", fontweight="bold")
    ax.text(x, .27, label.replace(" with most ", "\nwith most\n"), ha="center", fontsize=10)
    if i < 3:
        ax.annotate("", xy=(x+.17,.6), xytext=(x+.08,.6), arrowprops={"arrowstyle":"->", "color":"#82988f"})
fig.tight_layout(); plt.show()
''')

md(r'''
The screen leaves 16,684 eligible series across the source data. The selected five stores contribute 312 series covering 122 different products.

These are practical thresholds from the original study, not optimized settings. They exclude sparse-selling products and produce a narrow sample, so the results do not represent every store or product. Discount decisions within the sample were still made by managers, not randomized.

Selection uses all 90 days of the original training file, including the later validation weeks. This is a retrospective cohort. A stricter future study would select it before the first validation week or repeat selection inside each fold. Here we retain it for a matched comparison with the original pipeline.
''')

md(r'''
### 2.3 What the selected sample represents

The selection rules favour histories with regular sales and repeated discount-state changes. We compare the resulting sample with the full source to see how much coverage is lost and how its trading conditions differ.

Category shares count distinct store-product histories, not transactions or sales value. The category labels are anonymous IDs. The behaviour comparison uses the same early dates for both groups, ending before the first validation week; the denominators are shown in the table.

The connected dots show how each category's share changes between the source and the selected sample. A longer line means a larger difference in representation, not a change in sales.
''')

code(r'''
composition_figure, coverage_table, behaviour_table = data_exploration.composition(data_overview)
display(scope_table)
display(coverage_table.sort_values("Full histories", ascending=False).head(8).style
        .format({"Full share": "{:.1%}", "Selected share": "{:.1%}"})
        .hide(axis="index")
        .set_caption("Eight most common source categories; the chart groups the rest as Other."))
display(behaviour_table.style.format({"Share": "{:.2%}",
        "Matching days": "{:,.0f}", "Eligible days": "{:,.0f}"}).hide(axis="index"))
plt.show()
''')

md(r'''
The five-store sample contains 122 distinct products and 21 of the source's 32 first-level category IDs, all in city 0. Its category mix differs from the source: category 4 accounts for 8.65% of selected histories versus 21.78% overall, while category 16 accounts for 20.19% versus 14.35%. The IDs do not reveal category names.

Across the early dates, zero-sales records are less common in the selected sample (2.15% versus 5.04%). Stockout-day shares are almost identical (46.43% versus 46.39%), and discounted-day shares are 40.14% versus 42.88%. Selecting regular-selling histories therefore does not remove the problem of availability limiting observed sales.

The main experiment describes these eligible products in five stores. It cannot be assumed to represent every store or product in FreshRetailNet.

The full-source composition describes the original training file. Restricting the behavioural plots to early dates does not undo the retrospective selection, which used all 90 days. These are descriptive checks; they do not introduce a new cohort or change the model-selection results.
''')

md(r'''
### 2.4 Sales, discounts and stockouts

Within the selected histories, we compare recorded sales across three price bands. A price rate of 0.90 means 10% off; lower rates mean deeper discounts. We use feature rows from 4 April to 21 May, after the seven-day history-building period and before the first validation week.

For this plot, each day's sales are divided by that series' mean over the same period. This puts products with different sales levels on a comparable scale. The table also shows the number of observations and the share of days with stockouts in each price band.
''')

code(r'''
eda_figure, eda_table = story.training_eda(feat)
display(eda_table)
plt.show()
''')

md(r'''
Deeper-discount days have higher relative sales in this sample, which supports testing discount as a predictor. Managers chose when to discount, however, so the pattern does not isolate the effect of changing price. Stockouts occur on 56.0% of records in the deepest-discount band, compared with 42.3% in the near/full-price band. Recorded sales may therefore be limited by availability and can understate unconstrained demand.

The rescaling above is only for the chart; the prediction target stays in the source's normalized scale. These counts differ from the original notebook's exploratory table because that table used all feature rows. Restricting this chart to early dates does not undo the full-period cohort selection.
''')

md(r'''
Do the five stores show the same trading conditions? The heatmap below separates them, using the same 4 April–21 May window as the preceding chart. Each cell is a percentage of that store's selected product-day records, so stores with different numbers of eligible products remain comparable. Darker cells mean a higher share; they are not model scores.
''')

code(r'''
# Descriptive target-day outcomes from early dates only; no validation or final-period rows.
early_store = feat.loc[pd.to_datetime(feat["dt"]) < pd.Timestamp("2024-05-22")].copy()
assert early_store["dt"].min() == pd.Timestamp("2024-04-04")
assert early_store["dt"].max() == pd.Timestamp("2024-05-21")
assert early_store["discount_next"].gt(0).all() and early_store["discount_next"].le(1).all()
store_pattern = (early_store.assign(
    discounted=early_store["discount_next"].lt(0.95),
    stockout=early_store["target_stockout_hours"].gt(0),
    zero_sales=early_store["target_sales"].eq(0))
    .groupby("store_id").agg(
        records=("target_sales", "size"),
        discounted=("discounted", "mean"),
        stockout=("stockout", "mean"),
        zero_sales=("zero_sales", "mean")))
assert len(store_pattern) == 5 and store_pattern["records"].sum() == 312 * 48

from matplotlib.colors import LinearSegmentedColormap
fig, ax = plt.subplots(figsize=(9, 4.4))
heat_values = store_pattern[["discounted", "stockout", "zero_sales"]].to_numpy()
heat_colors = LinearSegmentedColormap.from_list("retail_shares", ["#f1f7f3", "#71b1a2", "#116656"])
heat = ax.imshow(heat_values, cmap=heat_colors, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3), ["Discount < 0.95", "Any stockout", "Zero sales"])
ax.set_yticks(range(5), [f"Store {i}  ·  {int(r.records):,} records" for i, r in store_pattern.iterrows()])
ax.tick_params(length=0, pad=10, labelsize=10)
for row in range(5):
    for col in range(3):
        value = heat_values[row, col]
        ax.text(col, row, f"{value:.1%}", ha="center", va="center", fontsize=12,
                color="white" if value > .6 else "#173f35")
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
colorbar = fig.colorbar(heat, ax=ax, fraction=.035, pad=.04)
colorbar.set_ticks([0, .25, .5, .75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
colorbar.outline.set_visible(False)
ax.set_title("Trading patterns within the five-store sample", loc="left", pad=18, fontsize=13)
fig.text(.02, .02, "4 Apr–21 May 2024 · shares of selected product-day records within each store", fontsize=9, color="#395b50")
fig.tight_layout(rect=(0, .06, 1, 1))
plt.show()
''')

md(r'''
All five stores have few zero-sales records, but their stockout shares range from 41.5% to 49.2%. Even within one city, the observed availability conditions differ. This motivates checking forecast errors by store later in the notebook.
''')

md(r'''
### 2.5 Data cleaning

The pipeline sorts records by store, product and date, then checks for duplicate dates, gaps and missing values. These checks stop the build if they fail. The recorded preprocessing also caps price rates above one and marks zero-rate sales and prices as missing before creating historical features; neither rate issue occurs in the selected cohort.

The cleaning audit below covers the selected records from both the original training file and the seven-day evaluation file: 30,264 raw rows in total. The earlier EDA scope table covers only the 28,080 original training rows. All row loss comes from the first seven dates of each series, which do not yet have a complete set of historical inputs. Their raw observations remain available to construct later features. Zero-sales and stockout days are retained.

The resulting table contains 25,896 training-period rows and 2,184 final-period rows. The next section explains what each input and target represents.
''')

code(r'''
cleaning = evidence["cleaning"]
display(pd.DataFrame([
    ["Selected raw store-product-day rows", cleaning["raw_selected_rows"]],
    ["Selected raw price rates above one", cleaning["raw_selected_discount_above_one_rows"]],
    ["Selected raw zero price rates", cleaning["raw_selected_zero_discount_rows"]],
    ["Missing values in audited selected columns", cleaning["raw_selected_missing_values_in_audited_columns"]],
    ["Rows removed before complete features", cleaning["rows_removed_before_saved_features"]],
    ["Saved feature rows", cleaning["saved_feature_rows"]],
], columns=["Audit item", "Count"]))
''')

md(r'''
## 3. Target and features

The target, Y, is next-day normalized observed sales for the same store and product. Tomorrow's discount is an input. During evaluation it is the recorded rate; later we replace it with candidate rates to make what-if predictions.

Each row is dated by the day being predicted. "Yesterday" always means the immediately preceding day, when the manager makes the decision.

| Input | Information it provides |
|---|---|
| Yesterday's sales, same weekday last week, recent averages | Recent sales level and pattern |
| Yesterday's discount, stockout hours and activity | Recent trading conditions |
| Tomorrow's weekday, holiday and proposed discount | Calendar context and the candidate action |
| Tomorrow's actual sales or stockouts | Excluded from the predictors; known only afterwards |

The example is the first final-period row when sorted by store, product and date. We show its actual sales only to check the prediction afterwards. The product is anonymized, so we use its identifier.
''')

code(r'''
example = feat[feat.split == "test"].sort_values(C.ROW_KEY).iloc[[0]]
case_key = example.iloc[0]
readable = {
    "Store": case_key.store_id, "Product": case_key.product_id,
    "Day being predicted": str(case_key["dt"].date()),
    "Yesterday's sales": case_key.sales_t, "Previous 7-day average": case_key.sales_mean7,
    "Yesterday's stockout hours": case_key.stockout_hours_t,
    "Recorded target-day price rate": case_key.discount_next,
    "Actual target-day sales (evaluation only)": case_key[C.TARGET],
}
display(pd.DataFrame(readable.items(), columns=["Case information", "Value"]))
''')

md(r'''
### 3.1 Check one feature row

Within this store-product's history, `sales_t` is the previous day's sales, `sales_lag7` is sales seven days before the target date, and `sales_mean7` averages the preceding seven days. Each window ends before the target date. Another store selling the same product gets a separate calculation.

The next tables show the historical observations and recompute the features against the saved input row. The seven-day average is also our main baseline.
''')

code(r'''
case_history, feature_checks = story.feature_walkthrough(feat, example)
display(case_history)
display(feature_checks)
''')

md(r'''
The recalculated values match the saved row. The target day's sales are kept separately for scoring and do not enter these history features.
''')

md(r'''
We evaluate rolling one-day-ahead forecasts. Before predicting each day, the previous day's observed sales become available and update the history. The fitted model stays fixed within the evaluation week. These results therefore describe daily forecasts, not a seven-day forecast made before the week begins.
''')

md(r'''
## 4. Baselines and evaluation

The main baseline predicts tomorrow's sales as the previous seven days' mean. We also test sales from the same weekday last week. A learned model has to improve on these rules to justify its added complexity.

MAE measures the average absolute miss and is our selection criterion. RMSE gives more weight to large misses. Both are in the dataset's normalized sales scale, and lower is better.

We use five weekly validation folds. Each model trains on all feature rows before a validation week, and preprocessing is fitted on those training rows only. Settings and blend weights are selected by mean validation MAE. The final week provides a common benchmark after selection.

That final week has already been viewed during earlier development. Recording the new selection before this run's final scoring does not make it an untouched holdout. The five validation weeks have also been reused for multiple comparisons.
''')

code(r'''
timeline_figure, timeline_table = story.split_timeline(feat, plan)
display(timeline_table)
plt.show()
''')

md(r'''
### 4.1 Time-ordered splits

Random splits would mix nearby dates with overlapping sales histories across training and evaluation. The timeline above keeps training dates earlier than the scored dates; each candidate is evaluated on the same observations. Standardization is fitted separately inside each fold for models that need it.

The earlier cohort selection and reuse of the final week still limit this evaluation, despite the chronological model fits.
''')

code(r'''
display(comparison[comparison.model.isin(["7-day mean baseline", "Same-weekday baseline"])][
    ["model", "validation_mae", "test_mae", "test_rmse"]].rename(columns={"test_mae":"final_period_mae", "test_rmse":"final_period_rmse"}))
''')

md(r'''
The seven-day mean beats the same-weekday baseline here, so it is the main reference for the model comparison. Both remain in the results table.
''')

md(r'''
## 5. Model comparison

We begin with two simple baselines, then test kNN, Random Forest and CatBoost using the same ten base inputs. This establishes whether learning relationships between sales history, discounts and context improves on a recent-sales average. RF is short for Random Forest.

The table lists the baselines first, followed by the individual models. We then examine their settings, test richer features and different training losses, and finally combine forecasts. These are comparisons to test; adding complexity does not guarantee a better score.

The scores come from the recorded comparison on the same validation and final-period dates. The tuning charts show saved experiments; running this notebook does not repeat those searches.
''')

code(r'''
reference_names = ["Same-weekday baseline", "7-day mean baseline", "kNN", "Random Forest", "CatBoost (base, RMSE)"]
display(metrics.loc[reference_names, ["validation_mae", "test_mae", "test_rmse"]].rename(
    columns={"test_mae":"final_period_mae", "test_rmse":"final_period_rmse"}))
''')

md(r'''
All three individual models have lower validation MAE than the seven-day average. Random Forest has the lowest validation MAE among these three; CatBoost has the lowest final-period MAE. We keep validation MAE as the selection rule and examine the recorded tuning results next.
''')

md(r'''
### 5.1 k-nearest neighbours

kNN finds training days with similar sales history, price conditions and calendar context, then averages their observed sales. It stores the training rows; it does not build trees or learn weights through gradient descent.

A small neighbourhood can follow noise; a large one can mix days that differ too much. Features are scaled before calculating distance, so a large numerical range, such as a weather measurement's units, does not dominate the comparison.

The original experiments varied neighbour count and feature sets over the same five validation weeks. The chart compares results with and without weather.
''')

code(r'''
knn_figure, knn_table = story.knn_tuning(ROOT)
display(knn_table)
plt.show()
''')

md(r'''
With 25 neighbours and equal feature weights, removing weather reduced validation MAE from about 0.32668 to 0.31198, against a seven-day baseline of 0.32840. This result applies to these inputs and this setup; weather may still matter for food demand. The neighbour-count curve is fairly flat after weather is removed.

The reference keeps ten inputs, 25 neighbours, uniform averaging, equal feature weights, Euclidean distance and a standard scaler fitted on training rows. The appendix shows the actual neighbours and their average for our example.
''')

md(r'''
### 5.2 Random Forest

A tree can group days through conditions such as recent average sales followed by a price threshold, without requiring similarity across every feature. It predicts the mean sales in each leaf. A Random Forest averages trees trained on resampled rows and feature subsets.

Deep trees with tiny leaves can memorise unusual days. The original experiments varied depth and minimum leaf size to limit this. The chart labels its colour scale because the score differences are small.
''')

code(r'''
forest_figure, forest_table = story.rf_tuning(ROOT)
display(forest_table)
plt.show()
''')

md(r'''
The saved grid selects depth 10 and at least three rows per leaf, with validation MAE about 0.30244. The earlier leaf-size and feature-subset comparison also favoured removing weather. Nearby settings score similarly, so this is a useful reference setting, not a universal optimum.

The reference uses 300 trees, half the input features at each split and squared-error splitting. More forest trees add estimates to the average; they do not sequentially correct earlier predictions.
''')

md(r'''
### 5.3 CatBoost

CatBoost adds shallow trees one after another to reduce the current prediction error. Depth controls each tree's complexity, learning rate controls its contribution, and iterations control the number of steps. This can capture interactions in tabular data, though extra trees can eventually fit training noise.

The original search tested 12 settings, then 27 shallower, slower combinations. The plot also includes a saved learning curve from one validation fold. That curve diagnoses training behaviour on that fold; it is not another set of independent test weeks.
''')

code(r'''
cat_figure, cat_table = story.catboost_tuning(ROOT)
display(cat_table)
plt.show()
''')

md(r'''
The reference uses depth 5, learning rate 0.03 and 500 iterations. Validation MAE is about 0.30330, close to the forest's score. The learning curves show training error continuing to fall after validation error starts rising: more fitting can hurt later forecasts. This does not establish a prediction ceiling for every product.

We retain this regularized reference to test its contribution to a blend. It has ten numeric inputs; categorical store and product IDs enter only in the extended model.
''')

md(r'''
### 5.4 Other model families

We also test Ridge, a single decision tree and Histogram Gradient Boosting to check whether the two tree ensembles justify their complexity. These are fixed reference designs; this run did not tune them again.

| Model | What it tests | Setting that matters |
|---|---|---|
| Ridge | Whether a linear relationship is enough | Scaled inputs and a penalty on large coefficients |
| Decision tree | Whether a few conditional rules are enough | Depth and minimum leaf size |
| HistGradientBoosting | Whether another boosting implementation competes | Learning rate, number of stages and leaf count |
| Monotonic HistGradientBoosting | Whether constraining the price response helps | With other inputs fixed, a higher price rate cannot raise predicted sales |

The monotonic constraint makes price scenarios easier to interpret. It remains a modelling assumption: it does not remove confounding, and forecast error still needs to be checked.
''')

code(r'''
other_references = ["7-day mean baseline", "Ridge", "Decision Tree", "Histogram Gradient Boosting", "Histogram Gradient Boosting (monotonic)"]
display(metrics.loc[other_references, ["validation_mae", "test_mae", "test_rmse"]].rename(
    columns={"test_mae":"final_period_mae", "test_rmse":"final_period_rmse"}))
''')

md(r'''
Ridge improves on the recent-sales baseline. The single tree's final-period MAE is worse than the baseline, while neither HistGradientBoosting variant wins the validation comparison. We keep these results, including the unsuccessful constraint.

The forest and CatBoost remain close. The next comparisons change the information CatBoost receives and the loss it trains on.
''')

md(r'''
## 6. Extended features and training loss

Two products can have the same seven-day average while one is rising and the other falling. The same average can also hide different levels of day-to-day variation. We add history to distinguish these cases, along with store and product IDs treated as categories.

The added history includes sales two and three days ago, three- and fourteen-day averages, seven-day variation, a short-versus-long trend, previous discount averages and recent stockout-day counts. All use earlier observations. The extended model has 18 numeric inputs and two categorical IDs.

The full experiment compares eleven standalone designs, four blends and two baselines, retaining Ridge, the decision tree and HistGradientBoosting. The full table is in the appendix. Candidates and blend rules were recorded before this run's final-period scoring.

The initial history-and-ID package also changed regularization, so its result cannot isolate each feature's contribution. We can isolate the loss comparison: the two extended CatBoost models below have the same inputs and other settings, but train with RMSE or MAE loss. After this, we compare the best standalone design with three declared RF blend weights.
''')

code(r'''
loss_rows = ["CatBoost (history + IDs, RMSE)", "CatBoost (history + IDs, MAE)"]
display(metrics.loc[loss_rows, ["validation_mae", "test_mae", "test_rmse"]].rename(
    columns={"test_mae":"final_period_mae", "test_rmse":"final_period_rmse"}))
''')

md(r'''
MAE training gives lower validation MAE, making it the stronger standalone candidate under our selection rule. Its final-period MAE and RMSE are both worse than the RMSE-trained version. We retain the validation-based choice and report that trade-off.

Absolute-error training targets a conditional median. Multiplying this forecast by a price rate later gives a decision proxy; it should not be interpreted as mathematically expected revenue.
''')

md(r'''
### 6.1 Feature and loss controls

The initial history-and-ID experiment changed several settings together. A later validation-only experiment separates them, using the same five expanding folds and fixed CatBoost depth, iterations, learning rate, regularization and seed.

It adds history to the ten base inputs, then categorical IDs, and compares losses. It also removes the proposed discount, previous-day activity or both from the full MAE design. History tests changes beyond the seven-day mean, IDs recurring store-product context, and discount information beyond recent sales. Removing activity tests what yesterday's activity adds after those inputs are present. Each comparison can also reveal that an expected gain is absent.

These are additional checks on already-used validation weeks. They do not score the final period, alter the canonical 17-method comparison, select new blend weights or create an untouched evaluation.
''')

code(r'''
ablation_dir = ROOT / "results" / "story_ablation"
subprocess.run([sys.executable, str(ROOT / "scripts" / "run_story_ablation.py"), "--verify-only"],
               check=True, capture_output=True, text=True)
ablation = pd.read_csv(ablation_dir / "summary.csv")
ablation_contrasts = pd.read_csv(ablation_dir / "contrasts.csv")
display(ablation[["variant", "n_features", "loss", "validation_mae", "validation_rmse", "folds"]])
display(ablation_contrasts[["contrast", "reference", "changed", "mae_delta",
                            "relative_mae_change_pct", "weeks_changed_lower_mae", "folds"]])
print("Contrast convention: changed minus reference; a negative MAE difference is an improvement.")
print("Input/output provenance and validation metrics verified; no final-period scoring or reselection.")
''')

code(r'''
ablation_by_variant = ablation.set_index("variant")
explanatory_pairs = [
    ("Base 10 / RMSE", "History 18 / RMSE", "Adding recent-history features"),
    ("History 18 / RMSE", "Full 20 / RMSE", "Adding store and product IDs after history"),
    ("Full 20 / RMSE", "Full 20 / MAE", "Changing the full design from RMSE to MAE training"),
    ("Full 20 / MAE", "No proposed discount / MAE", "Removing tomorrow's proposed discount"),
    ("Full 20 / MAE", "No previous activity / MAE", "Removing yesterday's activity flag"),
]
findings = []
for reference, changed, label in explanatory_pairs:
    before = float(ablation_by_variant.loc[reference, "validation_mae"])
    after = float(ablation_by_variant.loc[changed, "validation_mae"])
    direction = "lower" if after < before else "higher" if after > before else "unchanged"
    relative = abs(after-before)/before*100
    findings.append(f"- **{label}:** average validation MAE moves from {before:.5f} to {after:.5f} "
                    f"({relative:.2f}% {direction}).")
display(Markdown("\n".join(findings)))
''')

md(r'''
Each result depends on the comparison preceding it: IDs are tested after history has been added, for example. Removing discount or activity measures its contribution to prediction under this protocol. It does not estimate the causal sales effect of changing a discount.

We use these checks to explain the design and retain the previously selected model for the benchmark and pricing examples. Repeated use of the validation weeks limits generalization from these findings.
''')

md(r'''
### 6.2 Combining the forecasts

A blend can help when the models make different errors. If both miss the same sales spike in the same direction, averaging leaves that miss. We compare errors on validation rows predicted using earlier training dates, then score the declared blend choices.

We first combine Random Forest and CatBoost with the ten base inputs. An equal-weight blend was retained as a simple near-best setting in that sweep; the exact saved minimum was at 60% RF. We then combine Random Forest with the validation-best standalone alternative, CatBoost with history, IDs and MAE loss, testing 25%, 50% and 75% RF weights. These sweeps use different CatBoost configurations.

In the labels below, **base** means the ten original inputs, **history + IDs** means the added historical features and categorical store/product identifiers, and **RMSE / MAE** identifies the training loss. Ensemble weights are always written in **RF / CatBoost** order.
''')

code(r'''
blend_figure, blend_table = story.blend_diagnostics(ROOT)
display(blend_table)
plt.show()
blend_names = ["RF + CatBoost (base, 50/50)",
               "RF + CatBoost (history + IDs, MAE; 25/75)",
               "RF + CatBoost (history + IDs, MAE; 50/50)",
               "RF + CatBoost (history + IDs, MAE; 75/25)"]
display(metrics.loc[blend_names, ["validation_mae", "test_mae", "test_rmse"]].rename(
    columns={"test_mae":"final_period_mae", "test_rmse":"final_period_rmse"}))
''')

md(r'''
Validation selects 25% RF and 75% extended MAE CatBoost from the declared choices. The stronger standalone forecast gets more weight, with a smaller contribution from the forest.

We keep these weights for the final-period comparison. Nearby candidates differ only slightly, so the weights should not be described as universally optimal. They come from forecast error, not feature importance or product discount rates. The worked example shows the exact weighted average.
''')

md(r'''
## 7. Forecast comparison

We compare the selected 25% RF / 75% extended CatBoost blend with both baselines and the original 50/50 blend on the same cases. This measures forecast accuracy; we examine the pricing recommendations separately below.
''')

code(r'''
show = ["Same-weekday baseline", "7-day mean baseline", "RF + CatBoost (base, 50/50)", winner]
display(metrics.loc[show, ["validation_mae", "test_mae", "test_rmse"]].rename(
    columns={"test_mae":"final_period_mae", "test_rmse":"final_period_rmse"}))
folds = model_labels.frame(pd.read_csv(OUT / "validation_folds.csv"))
paired = folds[folds.model.isin(["RF + CatBoost (base, 50/50)", winner])].pivot(index="fold", columns="model", values="mae")
for ref in ["7-day mean baseline", "RF + CatBoost (base, 50/50)"]:
    improvement = 100*(metrics.loc[ref,"test_mae"]-metrics.loc[winner,"test_mae"])/metrics.loc[ref,"test_mae"]
    print(f"Final-period MAE reduction versus {ref}: {improvement:.2f}%")
print("Validation weeks with lower MAE than the base-feature 50/50 blend:", int((paired[winner] < paired["RF + CatBoost (base, 50/50)"]).sum()), "out of 5")
plot_names = ["Same-weekday baseline", "7-day mean baseline", "kNN", "Random Forest",
              "CatBoost (base, RMSE)", "CatBoost (history + IDs, MAE)",
              "RF + CatBoost (base, 50/50)", winner]
fig, ax = plt.subplots(figsize=(11, 6)); pos = np.arange(len(plot_names)); width = .36
ax.barh(pos+width/2, metrics.loc[plot_names,"validation_mae"], width, label="Validation", color="#82988f")
ax.barh(pos-width/2, metrics.loc[plot_names,"test_mae"], width, label="Final-period benchmark", color="#127263")
ax.set_yticks(pos, ["Same-weekday baseline", "7-day mean baseline", "kNN", "Random Forest",
                   "CatBoost (base, RMSE)", "CatBoost (history + IDs, MAE)",
                   "RF + CatBoost (base, 50/50)", "RF + CatBoost\n(history + IDs, MAE; 25/75)"])
ax.invert_yaxis(); ax.set_xlim(0,.46); ax.set_xlabel("MAE (normalized observed sales)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); plt.show()
''')

md(r'''
The selected blend has lower MAE than the original blend in all five validation weeks and reduces final-period MAE by about 2%. This is a modest gain within the selected cohort. It does not imply improvement on every metric, product or future store, or establish optimal weights beyond this comparison.
''')

md(r'''
## 8. Error analysis

Deep discounts coincide with larger sales spikes and higher forecast error here. We report every recorded discount band and the largest absolute errors, so the comparison includes difficult cases as well as the overall average.

Target-day stockouts can help explain a miss after the outcome is known. They are used only in this diagnosis and are never supplied to the model before prediction.
''')

code(r'''
slices = model_labels.frame(pd.read_csv(OUT / "error_slices.csv"))
display(slices[slices["slice"] == "discount"][["value","model","mae","rmse","n"]])
largest = model_labels.frame(pd.read_csv(OUT / "largest_errors.csv"))
display(largest[C.ROW_KEY + [C.TARGET,"discount_next","target_stockout_hours",winner,"selected_abs_error"]].head(3))
''')

md(r'''
### 8.1 Errors by group

The plots use the same final-period cases to check where errors are concentrated. A large miss on a high-sales day can dominate RMSE, and a lower overall MAE can coexist with weaker results in one store or discount band.

These are checks made after observing the outcomes. They inform reliability and future data needs without adding unavailable information to the original forecasts.
''')

code(r'''
error_figure, error_table = story.error_diagnostics(feat, test_predictions, winner)
display(error_table)
plt.show()
''')

md(r'''
We report these slices and a mechanically selected failure alongside the average score. Manual review remains part of the decision process; a few stores or one difficult week cannot establish performance at future locations.
''')

md(r'''
An underestimated sales spike or stockout-censored history can also distort a price recommendation. We next feed candidate discounts into the model and check what its sales predictions actually imply for the suggested price.
''')

md(r'''
## 9. Discount scenarios

For each store-product-day, hold the known features fixed and vary tomorrow's price rate. The candidates include full price, 5% off, 10% off and deeper discounts. We compare two rules: choose the highest forecast sales, or the highest price rate × forecast sales. Each is tried with all candidates and then with historical price support. The review rules have not yet been applied.

This repeats the original pipeline's diagnostic with the extension's saved forecasts. There are no new fits or observed sales. Historical discounts reflect managers' decisions and factors missing from the data; changing the price input alone does not identify a causal discount effect.
''')

code(r'''
naive_rows = []
for supported_only in [False, True]:
    pool = responses[responses.supported].copy() if supported_only else responses.copy()
    pool["value_proxy"] = pool.rate * pool.selected_sales
    for criterion in ["selected_sales", "value_proxy"]:
        # Highest rate first gives a milder discount if forecasts tie exactly.
        ordered = pool.sort_values(C.ROW_KEY + [criterion,"rate"], ascending=[True]*len(C.ROW_KEY)+[False,False])
        chosen = ordered.drop_duplicates(C.ROW_KEY)
        naive_rows.append({"Search": "Historically supported rates" if supported_only else "All candidate rates",
            "Choose highest": "Forecast sales" if criterion == "selected_sales" else "Sales-value proxy",
            "Cases": len(chosen), "Discount recommended (%)": round(100*(chosen.rate<1).mean(),1),
            "Choices without support": int((~chosen.supported).sum())})
display(pd.DataFrame(naive_rows))
''')

md(r'''
The sales rule recommends a discount on 100% of cases, with or without historical support. The value-proxy rule falls from 99.8% to 99.4%. Support removes unsupported choices, but barely changes whether a discount is recommended. We add the following decision checks; their business effects still need to be tested.
''')

md(r'''
## 10. Decision rules

The extension applies these rules in order:

| Check | Rule |
|---|---|
| Supply | At least one stockout hour yesterday triggers manual review of tomorrow's availability |
| Historical support | Require at least three training days for the same store-product within ±0.025 of a candidate rate, and a supported full-price comparator |
| Sales and value | Require at least 5% predicted sales gain and 95% of the full-price sales-value proxy |
| Model agreement | The selected blend, reference Random Forest and reference CatBoost must each pass the sales and value checks against their own full-price forecasts |
| Discount depth | Among passing candidates, choose the mildest discount within 1% of the highest predicted sales |

The sales-value proxy is price rate × predicted normalized sales. It compares scenarios for the same product, but has no currency unit and excludes costs. The thresholds are illustrative business assumptions. Model agreement is a sensitivity check, not a confidence interval.

Stockouts can censor observed sales. They do not establish tomorrow's freshness or the age of remaining stock, so we use them to request review rather than infer inventory age.
''')

code(r'''
decision_figure, decision_table = story.decision_flow()
display(decision_table)
plt.show()
''')

md(r'''
## 11. Worked example: 10% off

Store 18, product 11, 26 June 2024 is the first supported recommendation in store-product-date order. It is the same case used earlier, chosen by order rather than predicted gain. Yesterday had zero stockout hours, so the supply check passes.

A price rate of 0.90 means 10% off (九折). Only the recorded price has an observed sales outcome; the other rows are model scenarios.
''')

code(r'''
case = recs[recs.status == "recommend_discount"].sort_values(C.ROW_KEY).iloc[0]
mask = (responses.store_id == case.store_id) & (responses.product_id == case.product_id) & (responses.dt == case["dt"])
case_curve = responses.loc[mask].sort_values("rate", ascending=False).copy()
full = case_curve[case_curve.rate == 1.].iloc[0]
settings = plan["decision_parameters"]
case_curve["selected_gain_pct"] = 100*(case_curve.selected_sales/full.selected_sales-1)
case_curve["selected_value_share"] = case_curve.rate*case_curve.selected_sales/full.selected_sales
case_curve["rf_value_share"] = case_curve.rate*case_curve.rf_sales/full.rf_sales
case_curve["passes_all_models"] = case_curve.supported & (case_curve.rate < 1)
for col in ["selected_sales","rf_sales","cat_sales"]:
    case_curve["passes_all_models"] &= (case_curve[col]/full[col]-1 >= settings["min_sales_gain"])
    case_curve["passes_all_models"] &= (case_curve.rate*case_curve[col]/full[col] >= settings["min_value_share"])
def explain(row):
    if row.rate == 1: return "Full-price comparator"
    if not row.supported: return "Too little nearby price history"
    if not row.passes_all_models: return "Fails at least one model's gain/value check"
    return "Selected" if np.isclose(row.rate,case.recommended_rate) else "Passes; lower forecast sales"
case_curve["Decision"] = case_curve.apply(explain,axis=1)
display(case_curve[["rate","support_days","selected_sales","selected_gain_pct","selected_value_share","rf_value_share","Decision"]].rename(columns={
    "rate":"Price rate", "support_days":"Nearby historical days", "selected_sales":"Predicted sales",
    "selected_gain_pct":"Predicted gain (%)", "selected_value_share":"Selected value / full price",
    "rf_value_share":"RF value / full price"}).reset_index(drop=True))
print(f"Recommendation: {100*(1-case.recommended_rate):.0f}% off; predicted sales {case.predicted_sales:.3f}.")
fig, ax = plt.subplots(figsize=(8,3.6)); supported = case_curve[case_curve.supported]
ax.plot(supported.rate,supported.selected_sales,"o-",color="#127263",label="Supported candidate forecasts")
ax.scatter([case.recommended_rate],[case.predicted_sales],s=110,facecolors="none",edgecolors="#c76a20",linewidths=2,label="Selected: 10% off")
ax.set_xlabel("Price / full price (1.0 = no discount)"); ax.set_ylabel("Predicted normalized sales")
ax.legend(frameon=False); fig.tight_layout(); plt.show()
''')

md(r'''
Both 5% and 10% off pass. The forecast at 10% off is higher by more than the 1% near-tie allowance, so the rule selects it. At 15% and 20% off, the RF value proxy falls below 95% of its full-price reference. Deeper discounts lack historical support.

This explains the recommendation. The observed record cannot establish that 10% off is the optimal price.
''')

md(r'''
## 12. Review decisions

The table shows the first case in store-product-date order sent for supply review because of yesterday's stockout, followed by the counts for all 2,184 cases.
''')

code(r'''
display(recs[recs.status == "review_stockout"].sort_values(C.ROW_KEY)[C.ROW_KEY+["status","reason"]].head(1))
counts = pd.DataFrame(policy["status_counts"].items(),columns=["Decision", "Cases"])
counts["Share (%)"] = 100*counts.Cases/policy["rows"]
display(counts)
no_stockout = feat[feat.split == "test"][C.ROW_KEY+["stockout_hours_t"]].merge(recs,on=C.ROW_KEY,validate="one_to_one")
no_stockout = no_stockout[no_stockout.stockout_hours_t < settings["stockout_hours_review_threshold"]]
print(f"All {len(recs):,} input cases receive an explicit output.")
print(f"Automatic decision coverage: {policy['decision_coverage']:.1%}")
print(f"Among {len(no_stockout):,} cases without yesterday's stockout, {(no_stockout.status == 'recommend_discount').mean():.1%} still recommend a discount.")
''')

md(r'''
The policy returns an output for every case:

- 1,179 discount recommendations (54.0%).
- 967 supply reviews after a stockout.
- 38 reviews because the models disagree.

No case selects keep-full-price. Among the 1,217 cases without a previous-day stockout, 96.9% still receive a discount recommendation. The checks make the reasons explicit, but do not establish that the policy distinguishes beneficial discounts from unnecessary ones. Models trained on similar observational data can share the same bias.
''')

md(r'''
### 12.1 Rule sensitivity

We replay ten scenarios on the same 2,184 cases: the saved policy and nine changes to individual rules. The forecasts stay fixed. We vary the sales-gain hurdle, value floor, required history, near-tie tolerance and stockout threshold to check their effect on recommendation coverage and discount depth.

No models are refitted and no alternative-policy outcomes are observed. This is a sensitivity check, with no threshold tuning or change to the default policy.
''')

code(r'''
subprocess.run([sys.executable, str(ROOT / "scripts" / "build_story_diagnostics.py"), "--verify-only"],
               check=True, capture_output=True, text=True)
sensitivity_figure, sensitivity_table = story.policy_sensitivity(ROOT)
display(sensitivity_table)
plt.show()
''')

md(r'''
Requiring ten nearby historical days reduces discount recommendations from 1,179 to 926. Raising the review threshold from one stockout hour to eight increases them to 1,699. The rules therefore affect coverage substantially.

We retain the original thresholds. These counts show how the decisions change; they cannot tell us which setting would produce higher sales, profit or lower waste.
''')

md(r'''
## 13. Transfer to other stores

The original five stores are all in city 0. After locking the model, a fixed ordering of eligible training-data keys selected 100 series outside that city. The fitted models still use only the original cohort. Earlier observations from the new stores provide their daily lag inputs; their evaluation outcomes are not used for tuning.

The check contains 700 rows from 93 new stores in 16 other cities, including 47 product IDs unseen during fitting. This is a transfer stress test. We do not claim that these public records were never viewed earlier in the project.
''')

code(r'''
transfer_dir = ROOT / "results" / "v2_transfer"
display(model_labels.study_order(model_labels.frame(pd.read_csv(transfer_dir/"summary.csv")))[["model","mae","rmse","rows"]])
''')

md(r'''
The selected RF + CatBoost blend with history, IDs and MAE loss has MAE 0.28902, against 0.28842 for the base-feature 50/50 blend. Its RMSE, 0.48863, is also worse than the seven-day baseline's 0.48356. The original-cohort improvement does not carry over here. We retain the validation-selected design without reselecting after this check.

## 14. Findings and limits

The extension improves forecast error within the original five-store cohort and produces a supported discount or an explicit review reason. The worked example selects 10% off. Generalization remains limited: the cohort was selected retrospectively using the full original training period, and the final benchmark week had already been viewed.

A business pilot needs availability, replenishment, expiry and cost data, plus prospective pricing evidence to test the recommendations. Forecast errors and scenario curves do not demonstrate realized profit or waste reduction, or justify automatic rollout across stores.

The project progresses from baselines and individual forecasting models to richer history, categorical IDs, loss comparisons and forecast combinations. We then use the chosen forecast in discount scenarios, apply review rules and check transfer to other stores. The saved experiments and helper modules document each step.
''')

md(r'''
## Technical appendix A. How the model produces a number

For an input row $x$, Random Forest averages predictions from trees trained on bootstrap samples. Each tree follows feature thresholds to a leaf and predicts the mean training target in that leaf.

CatBoost adds trees one at a time: $F_m(x)=F_{m-1}(x)+\eta h_m(x)$, where $F_m$ is the prediction after step $m$, $h_m$ is the new tree and $\eta$ is the learning rate. The extended model treats store and product IDs as categories.

Our forecast is $\hat y=w\hat y_{RF}+(1-w)\hat y_{CB}$, where $w=0.25$, $\hat y_{RF}$ is the forest forecast and $\hat y_{CB}$ is the extended CatBoost forecast. The output is normalized observed sales.

Below, we refit the two selected models on the original training rows and predict the earlier case at its recorded price. The output shows each model's prediction, the blend calculation and the first forest tree's path. It uses the included feature table, so no raw-data download is needed.
''')

code(r'''
training = feat[feat.split == "train"]
specs = {s["name"]: s for s in plan["candidates"]}
members = selection["blend_weights"].get(winner,{winner:1.})
live_models = {}; example_values = []
from threadpoolctl import threadpool_limits
for name, weight in members.items():
    spec = specs[name]
    with threadpool_limits(limits=4):
        model = v2.estimator(spec).fit(training[v2.columns(spec)],training[C.TARGET])
    live_models[name] = model
    prediction = float(v2.predict(model,example,spec)[0])
    example_values.append({"Member":name,"Weight":weight,"Prediction":prediction,"Weighted part":weight*prediction})
example_table = pd.DataFrame(example_values); display(example_table)
live_prediction = example_table["Weighted part"].sum()
saved_case = test_predictions[(test_predictions.store_id == case_key.store_id) &
    (test_predictions.product_id == case_key.product_id) & (test_predictions.dt == case_key["dt"])].iloc[0]
print(f"Blend: {live_prediction:.6f}; observed outcome: {case_key[C.TARGET]:.6f}")
print("Replay difference from saved forecast:",abs(live_prediction-saved_case[winner]))
assert np.isclose(live_prediction,saved_case[winner],atol=1e-6)
forest = live_models.get("Random Forest")
if forest is not None:
    x = example[C.FEATURES].to_numpy(); tree = forest.estimators_[0]; leaf = tree.apply(x)[0]
    print("First forest tree's leaf prediction:",float(tree.tree_.value[leaf,0,0]))
    print("Forest mean across",len(forest.estimators_),"trees:",float(np.mean([t.predict(x)[0] for t in forest.estimators_])))
    for node in tree.decision_path(x).indices:
        feature = tree.tree_.feature[node]
        if feature >= 0:
            value, threshold = x[0,feature],tree.tree_.threshold[node]
            print(f"{C.FEATURES[feature]} = {value:.4f} {'<=' if value <= threshold else '>'} {threshold:.4f}")
''')

md(r'''
### A.1 A second worked example: actual kNN neighbors

kNN first standardizes numeric feature $j$: $z_j=(x_j-\mu_j)/s_j$, where $x_j$ is its input value and $\mu_j$ and $s_j$ are the training mean and standard deviation. The Euclidean distance to training row $i$ is $d_i=\sqrt{\sum_j(z_j-z_{ij})^2}$, where $z_{ij}$ is that row's standardized value of feature $j$.

Let $N_{25}(x)$ be the 25 closest training rows to input $x$. The prediction is $\hat y(x)=\frac{1}{25}\sum_{i\in N_{25}(x)}y_i$, where $y_i$ is neighbor $i$'s observed sales. Neighbors can come from other store-products because distance uses their numeric trading features. We fit the scaler on the original training rows only.

The calculation below uses the same real case. It shows the distances and how the neighbor average produces a forecast. Only the first five neighbors are displayed; the sum and prediction use all 25. Similar trading days do not by themselves identify a causal price effect.
''')

code(r'''
neighbors, neighbor_calculation = story.live_knn_example(feat, example)
display(neighbors)
display(neighbor_calculation)
''')

md(r'''
The blend combines tree predictions; kNN averages sales from nearby historical cases. Both predict normalized observed sales and are scored against the same target and baselines. We use the validation results to choose between designs.
''')

md(r'''
## Technical appendix B. Metrics, inputs and the full comparison

For $n$ evaluated rows, let $y_i$ be observed sales and $\hat y_i$ the forecast for row $i$:

$$MAE=\frac1n\sum_i|y_i-\hat y_i|,\qquad RMSE=\sqrt{\frac1n\sum_i(y_i-\hat y_i)^2}.$$

WAPE is $\sum_i|y_i-\hat y_i|/\sum_i|y_i|$. Lower is better for all three. We select by validation MAE and report RMSE and WAPE as additional checks. Both baselines appear first in the table.

Rolling features are shifted before aggregation to exclude the target day's sales. The ten base inputs include yesterday's activity flag. The older main-branch version used target-day planned activity. Its matching baseline scores do not imply that every input is defined the same way.
''')

code(r'''
display(comparison[["model","validation_mae","test_mae","test_rmse","test_wape","selected"]])
display(pd.DataFrame({"Group":["Base"]*len(C.FEATURES)+["Additional history"]*len(v2.EXTRA)+["Categorical ID"]*len(v2.IDS),
                      "Feature":C.FEATURES+v2.EXTRA+v2.IDS}))
display(pd.DataFrame(plan["candidates"]).fillna("—"))
display(paired)
''')

md(r'''
### B.1 Training objectives and settings

The linear model is $\hat y_i=b+\sum_j\beta_j z_{ij}$, where $b$ is the intercept, $\beta_j$ is a learned coefficient and $z_{ij}$ is a standardized input. Ridge minimizes $\sum_i(y_i-\hat y_i)^2+\alpha\sum_j\beta_j^2$: squared prediction error plus a penalty on large coefficients. The penalty strength is $\alpha$; the intercept is not penalized.

A squared-error tree chooses thresholds that reduce variation within its leaves. For leaf $L$, the quantity is $\sum_{i\in L}(y_i-\bar y_L)^2$, where $\bar y_L$ is the mean training target in that leaf. A split is useful when the two child leaves have a lower combined total. Random Forest repeats this process with resampled rows and subsets of features, then averages the trees.

Boosting adds trees to reduce the chosen training loss. Squared error penalizes large misses more heavily; absolute error grows linearly with the size of a miss. RMSE and squared-error loss have the same minimizer for a fixed set of predictions, although their implementation and optimization can differ. CatBoost also regularizes the trees and handles the store and product categories in the extended model.

MAE training targets a conditional median. Combining that CatBoost forecast with the forest does not guarantee either a conditional mean or an exact conditional median.

| Design | Main settings in the recorded experiment | Why the settings matter |
|---|---|---|
| Ridge | StandardScaler; alpha 10 | Make the coefficient penalty meaningful across input scales |
| Decision tree | Depth 6; minimum 20 rows per leaf | Limit narrow, unstable rules |
| kNN | StandardScaler; 25 neighbors; equal weights | Control distance comparability and local averaging |
| Random Forest | 300 trees; depth 10; minimum leaf 3; half the features per split | Average varied trees while limiting each tree's complexity |
| HistGradientBoosting | 300 stages; learning rate 0.05; at most 15 leaves; minimum leaf 20; L2 1 | Balance sequential corrections against complexity |
| Histogram Gradient Boosting (monotonic) | Same settings; non-increasing prediction in proposed price rate | Test an explicit price-response shape restriction |
| CatBoost (base, RMSE) | 500 stages; depth 5; learning rate 0.03; L2 3; RMSE loss | Keep the base-feature RMSE configuration fixed |
| CatBoost (history + IDs) | Same depth/stages/rate; L2 5; 18 numeric inputs and two IDs; RMSE or MAE | Test richer context and a controlled loss contrast |

We clip every model's forecast at zero. This prevents negative predicted sales but does not ensure an appropriate response to price changes. The full candidate table above also includes the depth-4 CatBoost configuration and loss variants.
''')

md(r'''
### B.2 What each comparison tests

The baseline comparison asks whether a learned model improves the forecast. Comparisons between model families test different representations at the declared settings; the recorded tuning grids compare settings within a family. The matched loss experiment changes only the training loss. The validation-only ablation holds other settings fixed while changing feature groups. The initial history/ID comparison also changed regularization, so it cannot isolate the feature effect.

We keep the design with the lowest average validation MAE under the declared protocol. Final-period RMSE, error slices and transfer results describe where it works and where it fails. These forecast comparisons do not measure the business return from deploying the discount policy.
''')

md(r'''
The models use the recorded settings, with kNN scaling fitted within each validation fold. This notebook runs no new parameter search.

The paired bootstrap below resamples store-product series, keeping each series' seven final observations together. It does not adjust for shocks shared by a store or optimism from repeated model selection. With five stores and one week, we treat the intervals as descriptive.
''')

code(r'''
display(model_labels.structure(json.loads((OUT/"paired_comparison.json").read_text())))
display(slices[(slices["slice"] == "store") & (slices.model == winner)][["value","mae","rmse","n"]])
''')

md(r'''
## Technical appendix C. Reproduce the evidence

The target, cohort, selected design and benchmark scores are unchanged from the saved experiment. By default, this notebook reads the saved results and fits three models for the worked examples: the two blend members and the fixed kNN reference. It does not repeat the historical tuning grids or the separate validation-only feature experiment. The selection funnel can be rebuilt from the verified public raw files with `scripts/build_story_evidence.py`.

From a complete repository copy:

```bash
docker compose up --build -d
docker compose exec lab python scripts/jupyter_v2_url.py
docker compose exec lab python scripts/run_v2.py --verify-only
docker compose exec lab python scripts/execute_lisa_notebook.py
```

To regenerate the complete experiment and story evidence:

```bash
docker compose exec lab python scripts/download_public_data.py
docker compose exec lab python scripts/run_v2.py
docker compose exec lab python scripts/run_v2_transfer_check.py --force
docker compose exec lab python scripts/run_story_ablation.py --force
docker compose exec lab python scripts/build_story_evidence.py
docker compose exec lab python scripts/build_dataset_eda.py
docker compose exec lab python scripts/build_story_diagnostics.py
docker compose exec lab python scripts/build_lisa_notebook.py
docker compose exec lab python scripts/execute_lisa_notebook.py
```

The raw files are downloaded once, verified and reused. The first image build needs network access; the saved-data demonstration runs offline. The report and lab use separate image snapshots. Export changes from the lab container before recreating it, then rebuild the report image to display regenerated output. The README gives the full export and data-volume instructions.

### C.1 Dataset reference

Source: [Dingdong's FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), licensed CC BY 4.0. These are public retail records, not simulated demand. They provide repeated observations of sales and discounts, with calendar, activity and stockout information.

| Item | Scope |
|---|---|
| Observation | One store-product-date |
| Series | One store-product's history across dates |
| Full dataset | 50,000 series, 898 stores, 18 cities |
| Original training file | 28 March–25 June 2024, 90 days |
| Evaluation file | 26 June–2 July 2024, seven days |

The same product in two stores has two separate sales histories. The source sales field is already normalized: 1.2 does not mean 1.2 items, kilograms or dollars. We use it directly as the target. Stockouts can make these recorded sales lower than customer demand.

The column `discount` is a price rate. A value of 0.90 means charging 90% of the regular price, or 10% off; 1.00 is full price and 0.80 is 20% off. Lower rates mean deeper discounts.

For a tomorrow forecast, we use past sales, discounts, stockout hours and activity, plus tomorrow's calendar and proposed rate. Stockout hours do not tell us the stock quantity, batch age or tomorrow's freshness. The anonymized identifiers also give no store names, customer profiles or product shelf lives.

The download script retrieves a fixed Hugging Face revision, verifies it and caches it locally. Small derived files are included so the notebook and its worked examples can run offline.
''')

code(r'''
display(pd.DataFrame(plan["environment"].items(),columns=["Package","Recorded experiment version"]))
print("Saved experiment verification:",verification["status"])
print("Feature rows:",len(feat),"; validation predictions:",verification["oof_rows"],"; final-period cases:",verification["test_rows"])
print("Selection was recorded before this run's final scoring:",selection["selected_before_test_scoring_utc"])
print("Original full experiment fits:",json.loads((OUT/"run_receipt.json").read_text()).get("fitted_models"))
''')

md(r'''
Generative AI assisted substantially with the code, review and presentation. The submission requires each member's own short paragraph stating their actual contribution, the AI assistance used and how they checked it. `docs/CONTRIBUTIONS_TEMPLATE.md` is still a template for those accounts. See also the requirements checklist and Chinese code walkthroughs.
''')

md(r'''
### Before submission

Each member should be able to explain the business question, the five-store selection, an input row, the validation rule, the selected blend and the reason for the example discount or review decision. Automated checks verify execution; personal understanding still needs to be demonstrated.

Record a feasibility discussion with teaching staff only if it took place. Moodle submission is a separate step.
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},
    "title":"Should we discount tomorrow?"})
nbf.write(nb,ROOT/"02_lisa_pipeline.ipynb")
print(f"Wrote {len(cells)} story cells")
