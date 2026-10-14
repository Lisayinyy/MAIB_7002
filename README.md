# Next-day sales for discounted perishables

HKU MAIB 7002 final project. Should a category manager discount a perishable product tomorrow? We predict next-day sales for a store-product given tomorrow's discount, compare models against a simple baseline, and (phase 2) use the best model to suggest a discount.

**Start with the notebook:** [`discount_sales_prediction.ipynb`](discount_sales_prediction.ipynb). The A1 poster is at [`poster/pricing_ml_poster_A1.pdf`](poster/pricing_ml_poster_A1.pdf) (preview: [`poster/pricing_ml_poster_A1_preview.png`](poster/pricing_ml_poster_A1_preview.png)). It walks through the data, the feature design, each model's tuning, the decisions taken, the comparison on an unseen week, and phase 2, where the model is turned into a discount recommendation with stock rules around it. GitHub renders it with all outputs, so nothing needs to be installed to read it.

## Layout

| Path | What |
|---|---|
| `discount_sales_prediction.ipynb` | The presentation notebook |
| `src/finalproject_pricingml/` | Helper package the notebook calls: `data` (selection, features, splits), `evaluate` (metrics, time-ordered cross-validation, out-of-fold predictions, test-week scoring), `models` (one factory per model: kNN, random forest, CatBoost, ensemble), `scenario` (phase 2 what-if discounts and stock guardrails), `plots`, `config` |
| `results/` | Every tuning grid and test-week prediction as CSV, plus the figures |
| `data/processed/` | The 0.5 MB feature table the models use (312 series, 28,080 store-product-days) and the selected series |
| `data/raw/` | The two public parquet files, or `freshretail_raw.zip` holding them; see Data below |
| `docs/` | Original build-out notes for kNN and the random forest, with the numbers behind each decision |
| `poster/` | The A1 poster and the script that builds it |
| `scripts/` | `build_notebook.py` (regenerates the notebook), `build_poster.py`, `make_submission.sh`, and the original step-by-step scripts |
| `Dockerfile` | Reproducible environment with JupyterLab |

## Data

Source: [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong, CC BY 4.0), two parquet files totalling 110 MB. They are too large for GitHub, so the notebook's first cell calls `data.ensure_raw_data()`, which tries, in order:

1. the files are already in `data/raw/` and their SHA-256 matches the pinned revision: nothing is done;
2. `data/raw/freshretail_raw.zip` is present (it is in the hand-in folder, 66 MB): extract once and verify;
3. download the two files from the pinned Hugging Face revision (`08c1fab7`) and verify.

A second run therefore costs one hash check and no download. If none of the three succeeds, the notebook still runs on the shipped `data/processed/features.parquet`.

## Running locally

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                      # creates .venv with all dependencies, including JupyterLab
uv run jupyter lab           # open discount_sales_prediction.ipynb
```

Or open the notebook in VS Code and pick the `.venv` interpreter as the kernel.

Every tuning grid is saved in `results/`, so the notebook runs in well under a minute by default. Set `RERUN_TUNING = True` in the first cell to recompute everything from the feature table (about a minute more).

## Running in Docker

```bash
docker build -t pricing-ml .
docker run --rm -p 8888:8888 pricing-ml
```

Then open the `http://127.0.0.1:8888/lab?token=...` link printed in the terminal. The data folder is copied into the image, so no volume mount is needed.

## Hand-in package

`scripts/make_submission.sh` writes `GroupQ-PricingML-submission.zip` one level above the project. It unpacks into a `GroupQ-PricingML/` folder holding the project with the processed table and `freshretail_raw.zip`, but without the git history, the virtual environment or the extracted parquet files (about 70 MB). Unzip it anywhere and run the Docker commands above from inside that folder.

## Adding a model

1. Add a factory to `src/finalproject_pricingml/models.py` that returns an unfitted estimator with `fit` and `predict`.
2. Tune it with `evaluate.cross_validate(feat, factory, grid)` on the same five weekly folds.
3. Score the locked design with `evaluate.fit_predict_test` and add it to the comparison in section 12 of the notebook.

The notebook compares kNN, random forest, CatBoost and a 50/50 forest + CatBoost ensemble, then builds phase 2 (section 14): the discount what-if, restricted to discounts each series has run, with stock guardrails (no discount after a sell-out, a discount required after a week without one) and cost and minimum-gain parameters. The appendix explains how CatBoost and the ensemble work and how to reproduce every result. The notebook is generated from `scripts/build_notebook.py`; edit that file, rebuild, then execute the notebook.
