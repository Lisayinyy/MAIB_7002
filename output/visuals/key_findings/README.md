# Key findings for the FreshRetail poster

Five main figures and one optional experiment, based on the group's final
[`pricing-ml-ap` notebook at a7e26dd](https://github.com/Lisayinyy/MAIB_7002/blob/a7e26dd979af88b4c14f98a7f029f6a64bffaedf/discount_sales_prediction.ipynb).
The charts use saved experimental evidence. This revision does not train or
change a model, and it does not modify the final notebook.

Each figure has a 4800 x 2700 PNG for placement and an SVG with editable text
and vector chart marks. The six-page PDF is in
`output/pdf/FreshRetail_key_findings_visuals.pdf`. These are figure assets,
not a replacement A1 poster. `all_key_findings.png` is the contact sheet.

## Recommended story and captions

| Figure | Priority | What it explains | English caption |
|---|---|---|---|
| 01 Sample selection | Compact introduction | Why price changes and regular sales matter, then why these five stores | We selected regularly selling products with repeated discount changes. |
| 02 Forecast results and errors | Main result | The overall gain and the remaining difficulty on deep-discount days | The ensemble lowers MAE by 7.9%; CatBoost alone is nearly tied. Absolute errors remain largest on deep-discount days. |
| 03 Response and support | Main narrative turning point | The model's largest predicted gains occur where few histories support the rate | At 40% off, predicted sales are 2.32 times the full-price forecast, but only 15 of 312 histories meet the support rule. |
| 04 Rules and recommendations | Main decision method | Explicit stock assumptions affect both the frequency and depth of discounts | Stock rules change discount recommendations from 98.3% to 55.0% of cases; their business value remains untested. |
| 05 One-product decision | Concrete ending | How the final system selects a specific discount | For this product-day, predicted value peaks at 5% off among eligible rates. |
| 06 Model complexity | Optional supporting figure | Why more weather features and more boosting trees were not automatically better | Choose complexity using later validation dates. |

If space is tight, give figures **02, 03 and 04** the most room. Use 01 and 05
as smaller opening/closing illustrations. Keep 06 in the notebook or presentation
backup. Do not add all tuning grids to the poster.

## 中文讲解顺序

1. 先解释为什么用这个数据集：它同时记录销量、折扣和缺货。为了能比较折扣变化，筛选经常有销售、折扣又有变化的商品，再选合格商品最多的五家店。312 是门店-商品的历史序列数，不是 312 行数据。
2. 模型有没有进步：组合模型的整体 MAE 比七天均值低 7.9%，但 CatBoost 单独已经很接近；深折扣组的绝对误差仍然较大，而且这组销量也更大。
3. 为什么不能直接用模型挑最深折扣：模型在 40% off 下给出很乐观的预测，但只有 15/312 条商品历史满足该档折扣的支持条件。预测收益和证据覆盖面要一起看。
4. 所以我们检验规则怎样影响决策：加入根据当天缺货及最近七天缺货记录设定的规则后，推荐打折的比例从 98.3% 变成 55.0%。这是建议发生变化，不是已经证明业务变好。
5. 最后落到具体输出：这个门店、这个商品、这一天，5% off 的预测价值高于其他合格候选折扣，所以系统推荐 5% off。还需要真实实验验证。

## Reading the figures correctly

- Discount rate is the fraction of full price: rate 0.95 means 5% off.
- Forecast target and error metrics use normalized observed sales. They are not
  item counts, currency, unconstrained demand or profit.
- Figure 02's final evaluation week is 26 June-2 July 2024, 2,184 cases, reused
  during development. The cohort was selected retrospectively from the training
  file. It is not an untouched holdout with nested cohort selection.
- Figure 03's ratios compare mean predictions over the same 2,184 cases.
  Historical support is a different denominator: 312 store-product histories.
  A supported rate requires at least three training days within 0.025 of it.
  Highlighting 40% off is visual emphasis, not another eligibility threshold.
- Figure 04's rules use the current/latest observed day to recommend tomorrow's
  price. They assume a stock policy; the data does not measure age or freshness.
  Recommendation shares use any rate below 1; the cohort's discounted-day filter
  uses rate below 0.95. These intentionally have different purposes.
- Figures 03-05 use zero assumed unit cost. Sales times price rate is a value
  proxy; no causal sales uplift, revenue increase or waste reduction is measured.
- Figure 05 is the deterministic first eligible worked case, not chosen for a
  large gain. Its y-axis is zoomed and labeled; compare the printed values.
- Figure 06's left plot uses five validation weeks. Its right plot is the saved
  **MAE** learning curve on one validation fold, zoomed to 100-3,000 trees; it is
  not RMSE and does not replace the five-fold model selection.

`evidence.json` records exact source hashes, notebook cells and versions.
`error_by_discount.csv` contains the newly summarized error diagnostic.

## Rebuild

From the repository root:

```sh
uv run --no-project --with matplotlib==3.11.2 --with pandas==3.0.6 --with pillow --with pypdf==6.20.1 python scripts/build_key_findings_visuals.py
```

The builder reads the repository's saved CSVs and verifies a fixed copy of the
final notebook. For offline use, pass `--notebook /path/to/the/a7e26dd/notebook.ipynb`.
The optional `--git-dir` and `--evidence-commit` arguments read the same evidence
from an existing local Git object store if cloud-backed file hydration is slow.
