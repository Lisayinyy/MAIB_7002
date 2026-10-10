# Daily discount decisions for perishable food

HKU MAIB7002 · Group Q · **Lisa's review and pipeline extension**

The workflow is **predict tomorrow's observed sales → check whether discount evidence is sufficient → compare supported rates → recommend or request review**. The intended user is a fresh-food category manager. Forecast errors are measured; predicted discount gains are scenarios, not proven sales, revenue or waste improvements.

This branch extends Aashish's `pricing-ml-ap` commit `5b52099ab73e57c5afbb1915d0bc4a0adbde9032`. His notebook, helper modules and saved results remain unchanged. See [his original README](docs/TEAMMATE_README.md), [the independent audit](docs/TEAMMATE_EVALUATION_AUDIT.md), and the Chinese learning guides below. Nothing has been submitted to Moodle.

## Start here

1. **Learn the teammate version:** [逐行代码讲解](docs/TEAMMATE_CODE_WALKTHROUGH_ZH.md).
2. **Understand Lisa's logic:** [中文故事与答辩讲解](docs/PROJECT_STORY_ZH.md), then [源码逐步说明](docs/LISA_PIPELINE_WALKTHROUGH_ZH.md).
3. **Read the project story:** [02_lisa_pipeline.ipynb](02_lisa_pipeline.ipynb), or its [executed HTML report](results/v2/02_lisa_pipeline.html). Follow the manager's question, public-data suitability, five-store selection, feature arithmetic, chronological evaluation, each model's investigation, controlled ablations, and a specific discount recommendation. The HTML has chapter navigation and expandable Python. Worked model calculations and full results remain in the technical appendix.
4. **Review the A1 poster:** [single-page portrait PDF](poster/freshretail_group_q_A1.pdf), [preview](poster/freshretail_group_q_A1_preview.png), [editable generator](scripts/build_v2_poster.py).
5. **Check submission requirements:** [brief checklist](docs/assignment_requirements.md) and [personal contribution/AI declarations](docs/CONTRIBUTIONS_TEMPLATE.md). Each member must complete and confirm their own paragraph.

## Run with Docker

Install Docker Desktop, start its engine, and obtain the **whole repository**, not just the Dockerfile.

```bash
git clone --branch codex/freshretail-review-v2 https://github.com/Lisayinyy/MAIB_7002.git
cd MAIB_7002
docker compose up --build -d
```

Open **http://localhost:19080** for the report and poster. For editable Jupyter:

```bash
docker compose exec lab python scripts/jupyter_v2_url.py
```

Open the printed localhost link. It includes this server's random login token; do not post the token publicly. Port **19888** is separate from the earlier project at 18888. Authentication remains enabled and both services bind to localhost.

```bash
# Check saved metrics, row keys, selection, source hash and decision constraints.
docker compose exec lab python scripts/run_v2.py --verify-only
# Exercise decision edge cases using small unit-test fixtures, not synthetic training data.
docker compose exec lab python -m unittest discover -s tests -v
# Verify the new validation-only controlled experiment and saved-policy sensitivity.
docker compose exec lab python scripts/run_story_ablation.py --verify-only
docker compose exec lab python scripts/build_story_diagnostics.py --verify-only
# Execute the presentation notebook in a fresh kernel (three modest example fits).
docker compose exec lab python scripts/execute_lisa_notebook.py
```

The default notebook uses saved public-data artifacts and refits the two selected members plus a fixed kNN reference for real worked examples. It does **not** rerun the full tuning grid or the 35-fit ablation. These steps need no external data service once the image has been built. First image build requires Internet access to free public package registries.

To rebuild the full raw-data pipeline:

```bash
# Downloads ~110 MiB of public data, verifies a pinned revision and SHA-256.
docker compose exec lab python scripts/download_public_data.py
# Eleven standalone candidates × five validation fits and one final fit = 66 fits.
docker compose exec lab python scripts/run_v2.py
# Repeat the locked transfer check without retuning; --force explicitly replaces that run's outputs.
docker compose exec lab python scripts/run_v2_transfer_check.py --force
# Seven fixed CatBoost designs × five validation weeks = 35 diagnostic fits.
# No final-period scoring or new winner selection in this experiment.
docker compose exec lab python scripts/run_story_ablation.py --force
# Rebuild the selection funnel and scenario diagnostics against the new output hashes.
docker compose exec lab python scripts/build_story_evidence.py
docker compose exec lab python scripts/build_story_diagnostics.py
# Update notebook output and poster after results change.
docker compose exec lab python scripts/build_lisa_notebook.py
docker compose exec lab python scripts/execute_lisa_notebook.py
docker compose exec lab python scripts/build_v2_poster.py
```

Raw data persist in a Docker named volume, avoiding host-folder permission differences. `docker compose down` preserves that volume; `down -v` would delete it. Notebooks/results edited **inside** the running container persist across a stop/start, but are lost if the container is removed or recreated. Export them before that:

```bash
docker compose cp lab:/app/02_lisa_pipeline.ipynb ./02_lisa_pipeline.ipynb
docker compose cp lab:/app/results ./container-results
# Stop without deleting containers:
docker compose stop
# Remove containers after exporting anything you need:
docker compose down
```

The report service serves the files saved in its image. Re-executing the notebook in `lab` does not update `report` automatically. To publish your regenerated artifacts locally, copy `lab:/app/results/.` into `./results/` and `lab:/app/poster/.` into `./poster/`, then run `docker compose up --build -d` again. Export notebook edits first as shown above. The default report already contains the delivered results.

To change host ports, copy `.env.example` to `.env` and edit them before startup. The [current refinement Docker check](docs/REFINEMENT_DOCKER_VERIFICATION.json) ran all 31 notebook code cells and three live examples without network or raw data, verified both new result sets, and rebuilt the poster. For local Python development, `uv sync --frozen` uses the version-locked environment; run commands with `uv run`. Python and uv base images are pinned by digest in the Dockerfile. See [Docker verification](docs/DELIVERY_VERIFICATION.json) for exactly what was tested.

## Measured result and scope

All rows below use the same 312 series, five chronological validation weeks, and 2,184 final-period cases. MAE and RMSE are in **normalized observed sales**, not physical units or currency. Rows are in baseline-first order.

| Method | Validation MAE | Final-period MAE | Final-period RMSE |
|---|---:|---:|---:|
| 7-day mean baseline | 0.32840 | 0.31418 | 0.52089 |
| Same weekday baseline | 0.41502 | 0.39975 | 0.61707 |
| Teammate 50/50 RF + CatBoost | 0.30125 | 0.28923 | 0.45926 |
| Lisa: 25% RF + 75% extended CatBoost (MAE) | **0.29402** | **0.28339** | **0.45605** |

The selected blend improves MAE on all five original validation weeks. Relative to the teammate blend, its mean validation MAE is 2.40% lower and final-period MAE 2.02% lower. It adds past-only history features and categorical store/product IDs; its CatBoost member trains with MAE loss. The controlled RMSE-versus-MAE loss comparison keeps the extended features and other CatBoost settings fixed. [Full comparison: 17 methods including baselines](results/v2/comparison.csv).

A full Docker Linux ARM64 rerun reproduced the same winner (validation MAE 0.294015; final-period MAE 0.283404). Small platform differences occur: the largest absolute MAE difference across all 17 methods was 0.000044. The table above uses the saved macOS run; [Docker comparison](docs/docker_comparison.csv) records the rerun separately.

**The final week was already seen in previous development.** This is an exploratory extension using a shared benchmark. Saving selection before this run's final scoring prevents new within-run choices based on that score, but does not undo prior exposure or repeated validation reuse.

**The improvement did not transfer to the extra cohort.** After selection, a fixed training-data-only SHA ordering chose 100 series across 93 new stores in 16 other cities. On their 700 final-period rows, the locked new blend scored MAE **0.28902** / RMSE **0.48863**, versus the teammate blend's **0.28842 / 0.47096**. No retuning or winner change followed. The stress check is not claimed globally untouched across the entire project's history. [Transfer evidence](results/v2_transfer/summary.csv).

## Decision rules and limitations

A supported candidate has at least three nearby historical training discounts for that store-product. We compare predicted sales and the relative proxy `price_rate × predicted_sales`, requiring at least 5% predicted sales gain and 95% of the full-price proxy under the selected model and both reference models. Among near-best sales options, choose the mildest discount.

A previous-day stockout requests a supply check; missing price support or model disagreement produces an explicit review/abstention. A week without stockout is **not** treated as evidence of old stock. Every input gets a record: 2,184 outputs, with 1,179 discount scenarios, 967 stockout reviews and 38 disagreement reviews in this run. The automated decision coverage is 53.98%; no current row selected keep-full-price. Among the 1,217 cases with no previous-day stockout, 96.9% still suggest a discount, an unresolved warning about using observational associations for intervention decisions.

The thresholds are illustrative, not learned business optima. MAE training targets a conditional median; the proxy is not an expected-revenue estimate. There is no measured causal uplift, profit, inventory aging, food-waste outcome or RL policy return. Reliable deployment needs further stock/cost/expiry information and a prospective or defensible causal evaluation.

## Evidence and files

- `discount_sales_prediction.ipynb`, `src/finalproject_pricingml/{data,evaluate,models,scenario,...}.py`: teammate original.
- `02_lisa_pipeline.ipynb`, `src/finalproject_pricingml/v2.py`: extension and worked demonstration.
- `results/v2/experiment_plan.json`, `selection.json`, `run_receipt.json`: source/data hashes, declared choices, selection chronology and actual fits.
- `results/v2/validation_predictions.csv`, `test_predictions.csv`: keyed row-level evidence.
- `scripts/audit_teammate_saved_results.py`, `docs/*audit*.json`: independent numerical and feature-timing checks.
- `tests/test_v2_decisions.py`: support, value-floor, abstention and near-tie edge cases.
- `scripts/build_lisa_notebook.py`, `scripts/build_v2_poster.py`: editable artifact generators.
- `scripts/build_story_evidence.py`, `results/v2/story_evidence.json`: independently rebuilt selection-funnel counts and diagnostics from saved discount scenarios; no new model fitting. The narrative revision preserves the existing model scores and selection.
- `scripts/run_story_ablation.py`, `results/story_ablation/`: seven fixed CatBoost designs, 35 validation-only fits and paired contrasts separating historical inputs, store/product IDs, training loss and discount/activity inputs. These diagnose the frozen design; they do not select a new winner or establish causal uplift.
- `scripts/build_story_diagnostics.py`, `results/story_diagnostics/`: ten rule settings applied to the same saved forecasts, including a replay of the original policy. No model fits or observed policy outcomes; the delivered thresholds are unchanged.
- [Expanded-story verification](docs/REFINEMENT_VERIFICATION.json): current notebook, controls, rendering and delivery checks. [Earlier narrative verification](docs/STORY_REVISION_VERIFICATION.json) remains the historical record of the shorter 16-code-cell version.
- `DATA_LICENSE.md`: FreshRetailNet attribution. No raw data or login tokens are committed.

Generative AI assisted substantially with this extension's code, review, experiments and presentation materials. Automated checks have been run; they do not replace each member's personal understanding, contribution declaration and final review. The group brief requires a one-page A1 portrait poster, runnable Docker project and personal declarations by the end of **14 October 2026**.
