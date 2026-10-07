# FreshRetailNet: daily discount decision support

Forecast next-day observed sales for perishable products, then compare candidate discounts under a sales-value constraint. This coursework project uses the public FreshRetailNet-50K dataset, two baselines, five regression models, and a supplementary ensemble experiment.

**Recommended exploratory combination: 50% Random Forest + 50% CatBoost**, selected by mean validation MAE among the tested combinations. In the verified Docker run (7 October 2026): validation MAE **0.30140**, final-week MAE **0.28729**, final-week RMSE **0.45288**. This is a post-hoc extension: the final week had already been viewed, and new dates are needed to confirm the small improvement. The original validation-selected Random Forest and its illustrative discount scenario remain separately reported. The earlier macOS run gave MAE 0.28728; the Docker run has a small numerical difference and the same selected combination. Full environment versions are saved in `outputs/final_protocol/docker_python_packages.txt`.

## Open with Docker

Install and start Docker Desktop (or Docker Engine with Compose), then:

```bash
git clone https://github.com/Lisayinyy/MAIB_7002.git
cd MAIB_7002
docker compose up --build -d
```

Open **[http://localhost:18080](http://localhost:18080)** for the results overview and complete executed notebook report. Saved results are included; no training or dataset download is required to read them. The first image build downloads the Python environment.

For the editable Jupyter notebook, run the following and open its printed login link:

```bash
docker compose exec lab python scripts/jupyter_url.py
```

Jupyter uses port **18888** and a generated login token; it does not share the old port 8888 environment. To download the pinned public data, rerun the full experiment (including ensembles), independently verify it, and refresh the HTML:

```bash
docker compose exec lab python scripts/reproduce.py
```

This overwrites the notebook's executed outputs and current result files in your cloned directory. A full run includes 185 model fits and depends on CPU speed; raw files total approximately 109.6 MiB. To stop: `docker compose down`.

A Dockerfile is the build recipe and needs the repository files as its build context. Share this repository (or its complete ZIP), rather than the Dockerfile alone. See the [Docker guide](docs/DOCKER_GUIDE.md) for alternative ports, persistence and login help, and the [project summary](docs/PROJECT_SUMMARY.md) for the final research narrative.

## Shared repository

The team's shared repository is [Lisayinyy/MAIB_7002](https://github.com/Lisayinyy/MAIB_7002). Use this repository for subsequent project code, notebooks, documentation, and reviewed results.

Teammates can clone this repository into any local directory. Raw public data can be obtained with `scripts/download_public_data.py`; no personal paths or credentials are required.

## Project overview

The main deliverable is `01_freshretail_project.ipynb`. It contains the experiment, results, figures, interpretation, limitations, and a clearly marked supplementary ensemble section. The current experiment reconstructs the teammate's reported data and evaluation protocol while comparing our five models: Ridge Regression, Decision Tree Regressor, Random Forest Regressor, HistGradientBoostingRegressor, and CatBoost Regressor.

This is a **report-based reconstruction**, not an exact reproduction of the teammate's unavailable preprocessing source code. Cohort counts, date ranges, discount-band counts, and reported baseline scores are independently checked in the notebook. The final evaluation week has already been viewed by the team; it is not a previously unseen test set.

## Run the project

Run these commands from the `group_project` directory in the same Python environment used by the notebook:

```bash
python -m pip install -r requirements-models.txt
python scripts/reproduce.py
```

The downloader uses only the Python standard library. It retrieves the official public `train` and `eval` Parquet files at the pinned dataset revision, verifies their SHA-256 hashes, and skips files that already match. Downloads are first written to a temporary file; an existing data file is replaced only after the new file passes verification. Use `python scripts/download_public_data.py --check-only` for an offline integrity check.

For an interactive run, open `01_freshretail_project.ipynb` in JupyterLab, select the Python environment with these dependencies, and use **Kernel → Restart Kernel and Run All Cells**. The command-line runner also executes in a clean `python3` kernel and saves cell outputs back to this notebook. To register the active Python environment as that kernel if necessary:

```bash
python -m ipykernel install --user --name python3 --display-name "Python 3 (FreshRetailNet)"
```

To export the executed notebook as a standalone HTML report:

```bash
python -m nbconvert --to html --embed-images 01_freshretail_project.ipynb --output 01_freshretail_project_report.html
```

The HTML export uses saved notebook outputs. Run the notebook first when fresh results are needed. Rerunning writes updated notebook outputs and files under `outputs/final_protocol/`; archived notebooks and older results are retained separately.

## Frozen experiment protocol

| Item | Current protocol |
|---|---|
| Data | Official FreshRetailNet-50K `train` (2024-03-28–2024-06-25) and `eval` (2024-06-26–2024-07-02); no synthetic sales labels |
| Cohort selection | On all 90 training dates, `discount < 0.95` defines a discounted day; discounted share 15%–85%, zero-sale share <20%, at least six consecutive-day discount-state changes; the first observation is not counted as a change |
| Stores | The five stores with the most eligible series: 343, 18, 235, 182, 154 |
| Cohort size | 312 store–SKU series, 122 products, city 0; 28,080 raw training observations |
| Model-ready training data | 2024-04-04–2024-06-25 after seven warm-up days; 25,896 rows |
| Validation | Five expanding-training weekly folds; validation weeks start 2024-05-22, 05-29, 06-05, 06-12, and 06-19; each has 2,184 rows |
| Final evaluation | 2024-06-26–2024-07-02; 2,184 rows |
| Forecast timing | Rolling one day ahead; earlier observed days update later-day features, including within each validation/evaluation week; models are not refitted within a week |
| Primary baseline | Trailing seven-day mean of observed sales, using `shift(1).rolling(7)` |
| Secondary baseline | Same-weekday sales seven days earlier, using `shift(7)` |
| Parameter selection | Three predeclared configurations per model per feature set; lowest mean MAE across five validation weeks; ties follow candidate order |
| Final fits | Refit selected parameters on all model-ready training rows; report final-week scores without using them for additional tuning |
| Metrics | MAE is primary; WAPE (%), MSE, and RMSE provide additional diagnostics |
| Prediction handling | Clip predictions at zero before scoring; random seed 7002 |

Cohort selection intentionally uses all 90 training days to match the report. Consequently, later validation-period outcomes influence which series enter this fixed research cohort. This is a retrospective cohort comparison, not an evaluation of a cohort chosen before the earliest validation week.

The full feature set has **10 numeric inputs**: `discount`, `weekday`, `holiday_flag`, `activity_flag`, `sales_lag1`, `sales_lag7`, `sales_mean7`, `discount_lag1`, `stockout_lag1`, and `series_mean`. The last feature is a past-only expanding sales mean, including the raw warm-up dates. Weekday is encoded Monday=0 through Sunday=6. No weather, store ID, SKU ID, or category ID is fed to these models; CatBoost uses the same numeric feature representation as the other models. Ridge's scaler is fitted on each training fold only.

All five models are also tuned and refitted with an **8-feature ablation**, removing only target-day `discount` and `activity_flag`. Their recorded values are treated as known planned inputs in the full experiment; the public data do not establish their actual advance availability. Lagged discount remains in the ablation.

The reported diagnostic bands use right-closed boundaries: none/light `discount > 0.95`, moderate `0.80 < discount <= 0.95`, and deep `discount <= 0.80`. Final-week counts are 1,186, 757, and 241 respectively. These band boundaries differ from the binary rule used for cohort selection.

## Outputs and provenance

**Use `outputs/final_protocol/` for the current 312-series experiment.** Files directly under `outputs/` belong to the older 100-series pilot and must not be mixed with the current results. Earlier notebooks and supporting files are preserved under `outputs/backups/`.

The shared repository excludes the old pilot files and backups. `outputs/index.html` is the generated current-results landing page. `outputs/ensemble_exploration/` contains the supplementary combination experiment, including out-of-fold predictions, the validation-selected weights and final-week predictions. Its weights are applied to freshly generated base-model predictions when the full notebook is rerun.

Key current outputs include:

- `locked_protocol.json`: source revision and hashes, feature sets, folds, fixed parameter grids, and assumptions.
- `selected_series.csv`, `selected_daily_observations.parquet`, `feature_frame.parquet`, and `folds.csv`: cohort, inputs, and time boundaries.
- `validation_trials.csv`, `validation_candidates.csv`, `chosen_parameters.csv`, and `selection_before_test.json`: validation evidence and the pre-final-evaluation selection record.
- `model_comparison_test.csv`: both baselines followed by the five full models; `all_test_scores.csv` also includes ablations.
- `promotion_ablation_test.csv`, `test_slice_scores.csv`, `daily_test_mae.csv`, and `test_predictions.csv`: promotion comparison, diagnostic slices, and row-level predictions.
- `selected_forecast_model.joblib`: the fitted full-feature model selected by validation MAE.
- `RESULTS.md`, `run_manifest.json`, and PNG figures: generated findings, execution provenance, and plots.
- `illustrative_discount_scenarios.csv` and `scenario_metadata.json`: a restricted scenario demonstration and its assumptions.

## Interpretation

The target is **observed daily normalized sales**, not unconstrained demand, unit counts, or monetary revenue. Stockouts may suppress the observed target. Better forecast errors do not establish the causal effect of changing a discount.

The optional scenario demonstration compares historically supported discount values using the selected sales model and a dimensionless sales-value proxy. Its 5% proxy-loss tolerance is an illustrative assumption, not a currency-denominated business guarantee or validated optimal pricing policy. Neither this analysis nor the final-period comparisons prove that a recommendation will increase actual sales or reduce food waste. Full assumptions and limitations are retained in the notebook.

## Verify the saved execution

To recompute packaged metrics from saved predictions without downloading data or training models:

```bash
python scripts/check_saved_results.py
```

This checks artifact consistency, including all 19 ensemble candidates; it does not establish that training has been rerun. In Docker, prefix the command with `docker compose exec lab`.

Run `python scripts/verify_final_protocol.py` after executing the notebook. This independently reconstructs all ten features from original observations, checks the time boundaries and selection rule, recomputes saved scores, and verifies the saved model and scenario output. The result is written to `outputs/final_protocol/independent_verification.json`.

Data attribution: [Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), CC BY 4.0, revision `08c1fab7f9257bc73679d415d65d644165d351d4`. The source hashes are recorded in `outputs/final_protocol/locked_protocol.json` and the downloader.
