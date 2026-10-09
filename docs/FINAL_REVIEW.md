# Final independent handoff review

Reviewed on 10 October 2026 (Hong Kong). Scope: README, the new notebook and its generator, Docker configuration, frozen V2/transfer outputs, artifact links, and the supplied project-brief checklist. This review changes no implementation or experiment result.

**Technical handoff passes: the analysis, teaching artifacts and recorded Docker checks are ready for group review, and the implementation branch is published and independently read back. A final documentation commit follows this review. Formal assignment submission additionally requires each member's personal contribution/AI declaration and human review.**

## What was independently checked

| Check | Result |
|---|---|
| Selected model | Consistent: 25% teammate RF + 75% extended CatBoost trained with MAE. |
| Original-cohort scores | Consistent: validation MAE 0.2940187483; final-period MAE 0.2833894724; RMSE 0.4560514364. README rounding matches the CSV. |
| Baselines and controlled experiment | Baselines are first; matched rows and chronological folds are stated. Extended CatBoost RMSE/MAE variants keep features and hyperparameters fixed apart from loss. |
| V2 hashes | Current `v2.py` matches the experiment plan; the saved feature file matches its hash; selection matches the plan hash. |
| Transfer hashes | Current source, original plan and selection match the transfer plan; the receipt matches that plan. The initial cohort hash is preserved across the fixed replay. |
| Independent numerical audit | All 17 methods, four blend formulas, row keys, and all 1,179 discount recommendations pass independent checks. See `docs/v2_independent_audit.json`. |
| Feature timing | Future-raw-value perturbation leaves earlier extra features unchanged; later features change. No actual raw files were altered. |
| New notebook | 24 cells; 10 code cells executed in order; zero stored errors; two inline figures. A real case refits two selected models and displays RF tree decisions. |
| Poster evidence | PDF hash, generator hash and every recorded source hash match the current files. Existing poster format/render audit reports one A1 portrait page with embedded fonts. This pass checks the audit binding, not a new render. |
| Original teammate work | The original notebook and helper modules remain unchanged; provenance and author attribution are visible. |
| Paths | Main notebook, HTML, poster, generators, CSVs, declarations, learning guides and `docs/DELIVERY_VERIFICATION.json` exist at their linked paths. |
| Docker receipt | Reviewed the completed receipt: Linux arm64 offline demonstration, fresh public download and 66-fit rerun, clean-copy Compose build, service health and non-root named-volume write all passed. This is the runner's recorded execution evidence; this review does not claim an additional independent image build. |
| GitHub branch | Independent `git ls-remote` confirms `refs/heads/codex/freshretail-review-v2` at `d0e2e5a7110042aaf004d7aea3ab9afb918289a9`. The README clone branch therefore exists. Final documentation and any PR/CI state are recorded by the completing delivery task. |

## Scientific claims are appropriately limited

The README, notebook and poster distinguish normalized observed sales from product units or money, a reused benchmark from an untouched holdout, and model-based discount scenarios from measured business impact. V2's better score on the original cohort is supported by its saved predictions; it is not evidence of higher profit, less waste or causal uplift.

The transfer failure is included rather than hidden: V2 MAE 0.289017 / RMSE 0.488632 versus teammate MAE 0.288419 / RMSE 0.470964 on 700 rows from 100 selected series in other cities. The selection was not changed after this result.

Output completeness and decision coverage are kept distinct. All 2,184 rows receive a record, but only 1,179 receive a discount recommendation; 967 require supply review and 38 require model-disagreement review. Automated decision coverage is 53.98%. There are no keep-full-price decisions in the current benchmark. Agreement among correlated models is not described as a calibrated confidence interval.

MAE targets a conditional median. The materials now correctly avoid equating price times a MAE forecast with expected-revenue optimization. Continued validation reuse and shared-store dependence are disclosed; the series-bootstrap interval is descriptive, not proof that all deployment populations improve.

## Concrete review fixes already applied

1. The notebook had claimed to display a CatBoost trajectory while showing only member predictions and an RF tree path. The unsupported trajectory wording was removed from both notebook and generator.
2. The report service and lab have separate image snapshots. README now explains that lab regeneration does not automatically update the report, and describes copying results/poster files back before rebuilding.
3. `.gitattributes` now enforces LF for text and binary handling for parquet/PDF/images, protecting hash-bound source, plans and CSVs from automatic CRLF conversion on checkout.

These changes do not change the selected model or its evidence.

## Docker and offline behavior: exact scope

- The image uses digest-pinned Python and uv bases and `uv sync --frozen`; the small derived feature table and saved results are included. The default new notebook does not need the raw download and performs only two live model fits for its worked example.
- A first build still needs Internet access for public base images, apt and Python packages. Rebuilding all features and 66 fits requires the two public raw files; the downloader pins a dataset revision, validates SHA-256, and requires no paid account or credentials.
- Raw data use the named volume `raw-data:/app/data`, rather than inheriting permissions from a host bind mount. The volume survives ordinary `docker compose down`; `down -v` would remove it. Notebooks and generated results in lab persist while that container exists; they do not survive container removal unless exported. These limits are stated in the README.
- Both services bind to localhost, Jupyter retains token authentication, and the new ports do not collide with the older project by default.
- The static HTML export contains optional remote JavaScript such as MathJax from a CDN. Its text, tables and embedded plots are saved; polished formula rendering in that exported HTML can depend on the browser's network/cache. The required offline runnable example is the Jupyter notebook, and the saved PDF poster is self-contained.
- `docs/DELIVERY_VERIFICATION.json` records actual execution against implementation commit `d0e2e5a7110042aaf004d7aea3ab9afb918289a9`: a clean local Git clone without hardlinks, no host virtual environment or raw files copied, Compose build with existing package/image cache, both services healthy, and the named volume writable by the non-root user.
- The offline container used `network=none` and no raw files: all 17 saved methods and eight tests passed; all 10 notebook code cells ran without errors, including two model fits and two inline images; the poster rebuilt. A separate public-data download and full 66-fit Linux arm64 run also passed. Its selected model is unchanged. Linux validation/test MAE are 0.2940153897 / 0.2834041515, compared with saved macOS 0.2940187483 / 0.2833894724; the README discloses close numerical rather than bitwise reproduction.
- Only macOS arm64 and Linux arm64 via Docker Desktop are recorded as tested at this point. Ubuntu amd64 CI is configured but not yet established as passed by this review. The report page and HTML have recorded browser checks; the receipt explicitly leaves final Jupyter UI login pending. Do not silently promote those remaining checks to verified.

## Publication scope

The implementation branch is pushed and the README clone branch resolves to the independently observed SHA above. Final review/receipt documentation is being committed after this pass. The parent task records the final documentation SHA, PR and CI outcome separately; none is assumed here. The technical checks have no remaining blocker requiring another experiment.

This is a publication/readback item, not a request for new modeling or hypothetical extra tests. No additional tuning is needed for this review.

## Before formal assignment submission

The supplied brief requires the one-page A1 portrait poster, explanatory runnable notebook/code in Docker, dependency/download instructions, a worked example, baseline comparison, technical experiment, error analysis and personal declarations. The delivered materials cover the technical and presentation elements. The reused final week remains a disclosed methodological limitation; it must not be relabelled as fresh.

`docs/CONTRIBUTIONS_TEMPLATE.md` is deliberately unfinished: each member must state their actual contribution, AI use and checks in their own factual paragraph. Automated agent verification is not evidence that a student has personally understood or reviewed the work. The team must read the final notebook/poster and complete those declarations before submission. No Moodle submission or printing has occurred.
