# Teammate version: independent evaluation audit

审计对象：`5b52099ab73e57c5afbb1915d0bc4a0adbde9032`，2026-10-10。

**结论：这是一个写作清晰、结构良好、数值能够复核的销量预测项目；最需要改进的是预测结果怎样变成折扣建议，以及怎样表达证据的边界。不能直接把它称为已验证的库存优化或减少浪费系统。**

本次没有修改队友的 `src/`、notebook、`results/`，没有重新跑完整 tuning grid。独立从保存的逐行预测重算指标，核对全部已保存的 tuning 汇总，在内存中从公共原始 parquet 重建 cohort 和 features，并检查决策函数的覆盖率。复核成功证明保存结果内部一致，不能代替重新训练对所有预测的逐项复现。

## 1. 已经验证的分数

每个验证周有 2,184 行；五周 OOF 合计 10,920 行。最终 benchmark 有 2,184 行，日期为 2024-06-26 至 2024-07-02。MAE/RMSE 是**归一化 observed sales 的误差**，不是商品件数、利润、折扣分类准确率或价格策略收益。

| 方法 | 五周平均验证 MAE | 最终周 MAE | 最终周 RMSE | 最终周 WAPE |
|---|---:|---:|---:|---:|
| 7-day mean baseline | 0.328397 | 0.314185 | 0.520890 | 32.436% |
| Same-weekday baseline | — | 0.399753 | 0.617069 | 41.270% |
| kNN, k=25 | 0.311976 | 0.303985 | 0.490022 | 31.383% |
| Random Forest | 0.302436 | 0.292348 | 0.464595 | 30.182% |
| CatBoost | 0.303303 | 0.289649 | 0.460264 | 29.903% |
| RF + CatBoost, 50/50 | **0.301253** | **0.289232** | **0.459262** | **29.860%** |

集成相对 7-day mean 的最终周 MAE 降低 **7.942%**，MSE 降低 **22.263%**。集成的验证 MAE 相对单独 RF 降低 **0.391%**，幅度很小。

精确的权重扫描最低点其实是 **60% RF + 40% CatBoost：0.3011774501**。50/50 为 **0.3012533324**，只差 0.0000758823。选择 50/50 可以是简洁性判断，应写“选择简单且接近最低误差的方案”，不应写成“权重扫描严格最优”。50/50 在五周里有 **4 周**优于 RF、**5 周**优于 CatBoost；第四周输给 RF。

可追溯证据：`results/test_predictions.csv`、`results/test_summary.csv`、`results/oof_predictions.csv`、`results/ensemble_weight_sweep.csv`。审计脚本还检查了 **158 条 tuning 汇总行**与其五折原始分数一致，六种最终方法的 MAE/MSE/RMSE/WAPE 一致，11 种集成权重一致。

## 2. 与我们原版本是否是同一个实验

**同一个 cohort、同一组日期、同一 target，但并非完全相同的 feature set。**

从 raw data 重建得到同样的 312 个 store-product series、25,896 个训练行、2,184 个最终周行。两边所有 2,184 个 benchmark row keys、targets、discounts 和两个 baseline 的逐行预测一致。比较全部 28,080 个 feature rows，另外九个对应 feature 的最大差异为零。

唯一实际 feature 差异很重要：

| 决策信息 | 队友版本 | 我们原版本 |
|---|---|---|
| Activity | 昨天的 `activity_t` | 目标日计划中的 `activity_flag` |
| 何时可得 | 昨天已观察到 | 必须额外假设明日活动已安排 |
| 数值不同的行 | 两列在全部 28,080 行里有 **4,297 行**不同，最终周里有 **304 行**不同 | 同左 |

证据：`src/finalproject_pricingml/data.py:83`；原项目 `scripts/build_submission_notebook.py:258` 和 `:272`；本仓库 `scripts/build_notebook.py:667–677` 声称“same ten inputs under different names”，应修正。相同 baseline 不能单独证明完整 feature pipeline 相同，本次是逐行比较后才确认九个 feature 一致。

原版本 Docker 的 50/50 集成为验证 MAE 0.3014047、最终周 MAE 0.2872879；队友为 0.3012533 / 0.2892322。**队友验证略好，原版本最终周略好。**这不足以证明哪套未来表现更好，更不能把差异全部归因于模型参数，因为 activity 的时点也不同。下一版应固定同一 feature set，控制其他因素后再比较模型。

## 3. 按后果排序的发现

### P1 — 用 stockout 推断库存新鲜程度，证据不成立

`scenario.py:97–105,110–119` 把“今天至少一个小时缺货”解释为“明天库存新鲜”，把“七天都没有缺货”解释为“库存已经放了一周，因此必须打折”。`scripts/build_notebook.py:766–770,885` 把这些作为 operational facts，并说数据支持第一条。

缺货记录是**availability**，不是批次年龄、补货、剩余库存或报损。每天补充新货且持续售出，也可能七天从未空架；上午短暂缺货后补货，也不证明晚上已清空。所需的 inventory age、replenishment 和 waste 没有在这套特征里观察到。训练表里“继续折扣后又缺货”的相关性也不能证明继续折扣导致缺货。

后果：规则会强制改变折扣，却没有证据确认这些商品老化或明天不需要折扣。当前最终周有 967 个“今天缺货”行、195 个“本周未缺货”行，即 **53.2%** 的行先被规则限制。

建议：保留可观察事实的名称，例如 `recent_stockout`；把它用作预测风险/人工复核提示。撤下“无缺货就强制折扣”和库存年龄推断。只有获得真实 stock-on-hand、batch age、expiry 或 replenishment 后，才能实现基于保质期的硬约束。教学展示可以保留原规则作 clearly labelled hypothetical policy，不能作为已证实业务结论。

### P1 — 现有预测评估不能验证建议折扣的收益或减少浪费

`scenario.py:1–7` 的警示写得正确：这是历史相关性下的 what-if。可是后续叙述出现“a discount does lift sales”“data's case for rules”等更强因果措辞，见 `scripts/build_notebook.py:794–801,885`。同一个模型生成折扣候选的结果，再选择该模型认为最好的候选，测到的是**模型内部乐观预测**，不是 policy value。

下一版应把两个问题分开：

1. 在观察到的实际折扣下预测明天销量：可以通过实际 target 验证。
2. 改成未发生的另一个折扣：只能作为 assumption-dependent scenario；没有反事实标签，不能用相同 MAE 证明业务有效。

同 SKU 内折扣日/非折扣日对比仍有时间变化的混杂；它是诊断，不是“within-series causal check”。森林各树之间的预测 spread 也不是“免费的 prediction interval”（`scripts/build_notebook.py:881`），更不是折扣增量效果的置信区间。可校准 observed-price forecast intervals，但价格干预仍需额外识别假设或未来试验。

### P2 — `guarded_recommend` 和销量最大化规则组合时会静默丢失行

代码位置：`scenario.py:38–50` 中 `max_sales_within_value` 从传入表寻找 1.0 的 baseline；`:122–132` 对“本周无缺货”的行先移除 1.0；`:143–144` 再将这个过滤后的表交给销量规则。baseline 消失后 value floor 变成 NaN，没有任何候选通过。

已用真实 cohort 的 row keys、support 和 stock states，配合恒定预测值做**纯函数覆盖率探针**。该探针不是训练数据、不是经验 sales-effect 实验：

| 规则组合 | 返回行 | 缺失行 |
|---|---:|---:|
| Supported max value | 2,184 | 0 |
| Supported max sales within value | 2,184 | 0 |
| Guarded max value（队友 notebook 当前实际调用） | 2,184 | 0 |
| Guarded max sales within value | 1,989 | **195** |

**当前 notebook 调的是 max value，因此它没有这 195 行缺失的问题。**这是现有 API 的可组合性缺陷，恰好会影响我们“尽量增加销量，同时约束价值损失”的目标。修复应先从原始候选表保存 reference value，再独立过滤动作；最后断言一行输入对应一条决策，包括明确的 `abstain/manual_review`。

所有 312 个 series 在本次数据都支持 1.0，故“full price 没有历史支持”当前造成的缺失为零。新数据若不支持 1.0，现有 support + soldout 组合也可能返回空集合；需要显式处理而不是声称这里已经发生。

### P2 — Validation 是反复使用过的开发集，不足以量化提升可信度

`data.py:19–45` 用完整 90 天训练期选择 series，再回头评估从 May 22 开始的验证周。对于这个已固定的 retrospective cohort 可以接受，但不是在第一个验证周之前就能确定的 prospective selection。若主张完全按当时可得信息部署，cohort 应在 first validation cutoff 前冻结，或每个 outer fold 单独选择。

`scripts/build_notebook.py:413–469,512–537` 先在五周上选 CatBoost 参数，再用相同五周的 OOF 调集成权重。OOF 能防止某个模型直接见到自己那一行的 target，但不能消除 hyperparameter search 对同一验证集的 selection optimism。还检查了 weather、kNN feature weights、距离、目标变换、RF settings、seeds 等。

正面之处是 notebook 后面已经承认最终周是 shared benchmark；前面“unseen”“touched exactly once”“Every design choice ... without looking”的口径应统一，不能用执行顺序重新赋予已看过的数据独立性。原始最终周已被讨论过，下一版任何在它上面的提升也只能作为 **retrospective benchmark**。

建议：固定一个小而有理由的候选集合，提前记录 selection criterion，保留逐周配对差异；如增加嵌套时间验证，明确它仍来自已研究过的样本。不要用“差异小于跨周标准差”来证明模型等价，也不要把四/五周改善说成已确认统计显著。更强的泛化结论需要真正未参与选择的新日期或新门店。

### P2 — 展示业务目标和实际优化目标不完全相同

队友 §15 提供两个规则，但 §16 主结果是 `max_value`，即最大化 `predicted_sales * (discount - cost)`。我们早期目标是最大化销量并限制价值损失，这对应另一种规则 `max_sales_within_value`，两者可以给出不同答案。

`sale_amount` 经过标准化且没有实际单位售价；`discount × predicted_sales` 是 value proxy。设定 cost 为 0.5 或 0.7 是假设，不是数据估计出来的 SKU cost。不能写成已优化实际 revenue、margin 或 waste。下一版应把目标、成本假设和默认阈值写在配置里，并在输出里显示 chosen action、参考预测、预计差异、支持度与不可判断原因。

### P3 — 可复现性和边界处理还可以更完整

| 项目 | 现状及影响 | 针对性改进 |
|---|---|---|
| OOF 身份 | `evaluate.py:147–155` 按数组位置混合；保存 CSV 没有 store/product/date。当前 OOF targets 顺序与重建顺序一致，但无法从 CSV 单独断言模型间身份一致。 | 保存 keys + fold；按 keys one-to-one join，assert targets 和顺序。 |
| Tuning 的缓存来源 | `data.py:105–113` 只检查缓存存在与 `stockout_days7` 列，未检查 raw/feature-code hashes。默认 notebook 读 CSV；一些原始 weather grids、互动 ablations 没有完整重算入口。 | 写 run manifest：原始数据 revision/hash、feature signature、依赖版本、seed、参数和选模规则。 |
| Full rerun 声明 | `RERUN_TUNING=True` 重算主要 grids，但最早 weather/weighting 比较仍直接读 CSV；notebook 自己承认 one-off checks 未脚本化。 | 区分“重算主结果”与“重算所有历史探索”；不承诺单一 flag 重现所有数值。 |
| 默认 ensemble | `models.py:75–80` 声称默认 locked ensemble，却调用默认 depth6/lr.05/500 CatBoost，而 notebook locked 为 depth5/lr.03/500。Notebook 显式传 members，当前结果安全。 | 参数统一到已保存 configuration；禁止第二份不同的默认 locked model。 |
| 折扣边界 | `config.py:22` 以 `<0.95` 定义 discounted，band 把 `0.95` 当 moderate；`scenario.py:166–167` 把 `>=0.95` 当 stop。 | 明确 action 是 any markdown 还是至少5%折扣，统一边界。 |
| 缺失输出统计 | `value_counts(normalize=True)` 忽略 missing，`.cat.codes` 把 missing 变成-1；未来候选为空会扭曲 agreement/discount shares。当前 max-value 没有缺行。 | coverage 单独报告，以全部输入为分母；缺失 decision 不进入“更深折扣”。 |
| Docker 保存 | README 挂载 `data`，未挂载 notebook/results；`docker run --rm` 结束后容器内新结果与编辑会消失。 | 把源代码/outputs 持久化或提供明确导出路径。 |

## 4. 值得保留的设计

- 时间切分、过去销量 shift、group-wise rolling/expanding 用法正确；九个共有 features 与原版本逐行一致。
- 标准化放在 sklearn Pipeline 中，因此每个 fold 只对训练行 fit；目标日 stockout 未作为特征。
- 同样的 cohort、目标、validation folds 和两种 baseline，适合可控模型比较。
- Random Forest、CatBoost、kNN 的调参叙事、学习曲线、按折扣/日期/门店的误差诊断很有教学价值。
- OOF 权重扫描比用 benchmark 选权重更合适；简单 50/50 是合理的复杂度选择。
- 支持限制要求每个 SKU 在相近价格至少出现三次，比任意扫所有价格更谨慎。它仍只保证局部历史出现，不能保证因果可比或条件上下文 overlap。
- 已明确数据是 normalized observed sales、有 stockout censoring、没有随机折扣实验。这些边界应贯穿所有报告和 poster。

## 5. 可复核命令

只检查已保存结果，不依赖 ML 包、不训练模型：

```bash
python scripts/audit_teammate_saved_results.py
```

在已安装锁定环境并备好原始数据时，重建 cohort/features、检查 coverage、对照我们原版本：

```bash
.venv/bin/python scripts/audit_teammate_saved_results.py \
  --features \
  --original-root /Users/lisayin/Desktop/HKU/7002_1A/group_project
```

本次机器可读证据保存在 `docs/teammate_saved_results_audit.json`。标准库部分和 feature 部分均 PASS；程序只读取输入文件，feature table 在内存构建，未覆盖队友 cached features 或实验结果。

## 6. 下一版优先做什么

1. 固定队友 feature contract，明确预测的是观察销量；保留两种 baseline 和同一时间协议。
2. 少量有明确理由的实验，例如针对深折扣大误差的损失函数比较、历史趋势/波动特征、对照我们原模型；先用开发周选择，再统一报旧 benchmark，避免无限调参。
3. 建立完整、可审计的决策层：reference value 永远独立保留，support/count/feasibility 明确，折扣选择与安全回退按相同目标执行，全部输入得到 recommendation 或 abstention。
4. 去掉未观察到的“库存老化”结论。库存信息不足时，以“需要补充库存/expiry 信息”作为复核原因。
5. 将重建脚本、逐行预测、配置与 hashes 一起输出；notebook 展示既有 baseline、prediction metrics，也有 action coverage 和 limitations。
6. Poster 报告能证实的 forecast improvement；折扣输出展示为 decision-support scenario。**更好可以来自更清晰的逻辑、更可靠的复现和正确的决策边界；只有实际实验支持时才声称分数更好。**

## 7. V2 实现后的独立复核

V2 为独立的 `src/finalproject_pricingml/v2.py` 和 `results/v2/`，原队友实现保留。审计看到的选择是 **25% Teammate RF + 75% CatBoost history + IDs MAE**：增加过去销量趋势/波动、历史平均折扣、过去缺货天数与 categorical store/product IDs，并比较 MAE objective。不是先训练一个折扣分类器；它先预测，再通过可行性/风险检查决定是否能提供建议。

| 项目 | 独立复算结果 |
|---|---:|
| 选中方案验证 MAE | 0.2940187483 |
| 已看过的最终周 MAE | 0.2833894724 |
| 已看过的最终周 RMSE | 0.4560514364 |
| 相对队友 50/50 的最终周 MAE 差值 | -0.0058427713 |
| 逐行 recommendation 输出 | 2,184 / 2,184 |
| 自动给出折扣建议 | 1,179 |
| 昨天缺货，要求复核供应 | 967 |
| 模型间敏感性检查不一致，复核 | 38 |

独立检查使用标准库实现指标与决策条件，**没有调用 V2 的 `scores`、`choose_action` 或 `verify_saved` 来给自己背书**：17 种方法的 MAE/RMSE/WAPE 与逐折 MAE、四种 blend 的全部 OOF/test 公式、row keys、1,179 条折扣建议的 support/gain/value 条件和“销量相差1%以内优先更轻折扣”的选择全部 PASS。8 个决策边界单元测试全部通过。

额外做了未来信息扰动检查：对 store18/product11 的原始行建立临时副本，改变 June29–July2 的四天销量和折扣。June29 及以前的 87 行新历史特征完全不变；June30 以后特征按预期变化，证明探针不是空检验。真实 raw files 未修改。这是**特征函数的 unit fixture**，不是自造数据参与模型实验。代码中的新 rolling/lag 全部先 shift，再汇总；读取 eval 文件本身不等于泄漏，关键是某日输入只用之前已经观察到的日期。

V2 修复了主要决策问题：保留 full-price comparator；缺少支持、zero reference 和模型不一致时显式 abstain；不再从“七天没缺货”推断库存年龄；stockout 触发 supply review，而不是宣称自动 full-price 最优。审计最初指出 plan/default 参数重复的漂移风险，最终实现已将 plan parameters 直接传入决策函数，并在验证时绑定 feature/code/plan hash；固定配置重跑后分数不变。

仍须清楚表达三点：

- 自动给出建议的 coverage 为 **53.98%**；不是所有行都得到“折扣或不折扣”的业务决策。在此 benchmark 上 `keep_full_price` 实际出现 **0 次**，可处理的案例仍大多偏向折扣，不能声称已解决原始关联模型偏好普遍打折的问题。
- 模型一致性是 sensitivity check，不是校准置信度或因果证据。它们在同一个观察数据上训练，会共享偏差。
- 同一五周继续选特征、loss 和 blend 是探索性开发；保存选择顺序不使旧数据重新成为全新 holdout。series bootstrap 也不能消除共同门店冲击或多次选模偏差。

复核命令和证据：

```bash
.venv/bin/python scripts/audit_teammate_saved_results.py --v2 --v2-feature-time
.venv/bin/python -m unittest discover -s tests -v
```

机器可读结果：`docs/v2_independent_audit.json`。

## 8. 锁定方案的跨门店压力检查：提升没有普遍迁移

为了测试新方案是否只适合原来五家店，额外进行一次**不调参、不改 winner** 的 transfer stress check：

- 从 train.parquet 中符合原 usable 标准、`city_id != 0` 的 series 开始，以固定 SHA-256 排序选择100个；选择不看 eval outcomes。
- 读取新 cohort 的 eval 之前，保存 plan 和 selected-key hash。
- 三个所需 estimator 都只在原来的312个 series、June26之前的25,896行上 fit。
- 预测新100个 series 的 June26–July2；允许用每个 series 当时已有的过去销量构造 one-day-ahead history。
- 总计93家新店、16个其他城市、700行；47个 product IDs 在模型训练中从未出现。没有丢行、重抽样选 cohort 或根据分数换模型。

| 方法 | 新 cohort MAE | 新 cohort RMSE | 新 cohort WAPE |
|---|---:|---:|---:|
| 7-day mean | 0.296668 | 0.483557 | 36.332% |
| Same weekday | 0.362629 | 0.575504 | 44.410% |
| 队友 RF + CatBoost 50/50 | **0.288419** | **0.470964** | **35.322%** |
| 已锁定 V2 25/75 | 0.289017 | 0.488632 | 35.395% |

**V2 在这个跨城市压力检查中没有胜过队友集成。**MAE 差约0.000598；RMSE 更高，甚至略高于7-day baseline。可以报告原 cohort 的改善；不能声称跨门店普遍领先。新城市、未见门店和未见 SKU 会同时改变数据分布，单凭这次结果不能断言是哪一种 feature 导致差异，也不能据此重调参数后把同一 cohort 再称为全新测试。

本项目已接触整份公开数据及部分相关结果，因此这里称为 **train-selected cross-store stress check**，不保证整个项目历史中完全 untouched。首次 plan hash 为 `c3b929ab6bef2e9cad826fff463f306bcd632eaef63301752838bd8bb5da723d`。V2 最终 source/plan hash 更新后做了固定配置复现，保留首次 plan 并验证 selected-key hash 不变；复现不是第二次新的独立评估。

脚本：`scripts/run_v2_transfer_check.py`。数据与追溯：`results/v2_transfer/experiment_plan.json`、`first_experiment_plan.json`、`selected_series.csv`、`predictions.csv`、`summary.csv`、`by_city.csv`、`run_receipt.json`。复现必须显式传 `--force`，且不会修改 V2 的选模结果。
