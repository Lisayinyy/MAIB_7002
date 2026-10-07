# Next-day sales for discounted perishables

HKU MAIB 7002 final project. Should a category manager discount a perishable product tomorrow? We predict next-day sales for a store-product given tomorrow's discount, compare models against a simple baseline, and (phase 2) use the best model to suggest a discount.

**Start with the notebook:** [`discount_sales_prediction.ipynb`](discount_sales_prediction.ipynb). It walks through the data, the feature design, each model's tuning, the decisions taken, and the comparison on an unseen week. GitHub renders it with all outputs, so nothing needs to be installed to read it.

## Layout

| Path | What |
|---|---|
| `discount_sales_prediction.ipynb` | The presentation notebook |
| `src/finalproject_pricingml/` | Helper package the notebook calls: `data` (selection, features, splits), `evaluate` (metrics, time-ordered cross-validation, out-of-fold predictions, test-week scoring), `models` (one factory per model: kNN, random forest, CatBoost, ensemble), `scenario` (phase 2 what-if discounts), `plots`, `config` |
| `results/` | Every tuning grid and test-week prediction as CSV, plus the figures |
| `docs/` | Original build-out notes for kNN and the random forest, with the numbers behind each decision |
| `scripts/` | The original step-by-step scripts the package was extracted from (kept for reference) |
| `Dockerfile` | Reproducible environment with JupyterLab |

## Data

The raw data is not in the repository. Download [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong, CC BY 4.0) and place the two files at:

```
data/raw/train.parquet
data/raw/eval.parquet
```

The notebook builds `data/processed/` from them on first use.

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
docker run --rm -p 8888:8888 -v "$PWD/data:/app/data" pricing-ml
```

Then open the `http://127.0.0.1:8888/lab?token=...` link printed in the terminal.

## Adding a model

1. Add a factory to `src/finalproject_pricingml/models.py` that returns an unfitted estimator with `fit` and `predict`.
2. Tune it with `evaluate.cross_validate(feat, factory, grid)` on the same five weekly folds.
3. Score the locked design with `evaluate.fit_predict_test` and add it to the comparison in section 12 of the notebook.

The notebook currently compares kNN, random forest, CatBoost and a 50/50 forest + CatBoost ensemble, and ends with a first draft of the phase 2 discount what-if.
