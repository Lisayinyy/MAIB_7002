# Run the shared project with Docker

This guide describes the Docker workflow for [Lisayinyy/MAIB_7002](https://github.com/Lisayinyy/MAIB_7002). The repository includes an executed notebook, an HTML report, figures, and result tables, so reviewing the current experiment does not require retraining.

The Dockerfile needs the rest of the repository as its build context. Downloading only the Dockerfile is not enough: it copies the dependency files, notebook, scripts, and saved results into the image. Docker Compose also mounts your local checkout, so notebook edits and regenerated outputs persist on your computer.

## 1. Clone and start

Install Docker Desktop, or Docker Engine with the Compose plugin, and start Docker. Then run:

```bash
git clone https://github.com/Lisayinyy/MAIB_7002.git
cd MAIB_7002
docker compose up --build -d
```

If you already have this project checkout, run the Compose command from that directory instead of cloning it again. The first build needs internet access to obtain the Python base image, Linux packages, and Python dependencies.

The two local services use these default addresses:

| Service | Host address | Purpose |
|---|---|---|
| Report | <http://localhost:18080> | Browse saved reports and results |
| JupyterLab | <http://localhost:18888> | View, edit, or run the notebook; token login required |

The report links to the current 312-series experiment. Its direct HTML address is <http://localhost:18080/final_protocol/01_freshretail_project.html>.

The services bind to the local computer only. This setup uses **18080 and 18888** on the host; it does not change or stop an existing Jupyter service on port 8888. Port 8888 inside the lab container is separate from the host's port 8888.

## 2. Open the notebook

**Linux file ownership:** the lab defaults to UID/GID 1000. If your Linux account uses different IDs, run `id -u` and `id -g`, copy `.env.example` to `.env`, and set `LOCAL_UID` and `LOCAL_GID` to those numbers before starting Compose. This lets the non-root notebook process write to your checkout. Docker Desktop on macOS/Windows normally handles host file sharing; leave these defaults unless file permissions require otherwise. Jupyter's runtime home is a writable temporary directory inside the lab container; project files persist through the bind mount.

Get the actual Jupyter login URL, including its automatically generated token:

```bash
docker compose exec lab python scripts/jupyter_url.py
```

Open the URL printed by the command. Jupyter authentication remains enabled; no fixed token or password is added to the repository. The token is local runtime information and may change when the container is recreated.

The notebook opens as `01_freshretail_project.ipynb`. Saved tables and figures are available immediately. Running cells requires the inputs described below. Notebook saves write through the lab service's read/write bind mount to your checkout; the report service mounts that checkout read-only.

To inspect service status or startup problems:

```bash
docker compose ps
docker compose logs --tail=100 report lab
```

## 3. Reproduce the complete experiment

Run the complete pipeline inside the running lab container:

```bash
docker compose exec lab python scripts/reproduce.py
```

The pipeline performs these steps in order:

1. Download or verify the official public `train.parquet` and `eval.parquet` at the pinned FreshRetailNet revision.
2. Execute the notebook in a clean Python kernel and save its outputs.
3. Independently verify the generated features, predictions, parameter selection, metrics, and scenario example.
4. Check the saved result artifacts, export the executed notebook to HTML, and refresh the report index.

The raw downloads total **114,876,411 bytes, approximately 109.6 MiB**. An existing file with the correct SHA-256 hash is reused. A download is installed only after verification, so a failed download does not replace an existing data file. This step needs internet access only when a required verified input is missing.

The experiment includes two baselines, five models with 10 features, and an 8-feature promotion ablation. It performs 150 validation fits, 10 final fits and 25 supplementary ensemble-validation fits (185 fits in total). Training takes longer than opening the saved report; wait for the pipeline's final success message. The last evaluation week was already viewed by the team, and the experiment is a report-based reconstruction of the teammate's protocol, not a reproduction of unavailable teammate source code.

The notebook and files under `outputs/final_protocol/` are updated in your checkout. Save any local changes you want to retain before rerunning. Refresh the browser after the HTML export finishes; notebook saves alone do not automatically refresh the exported HTML.

For a new one-off container instead of the running lab container:

```bash
docker compose run --rm lab python scripts/reproduce.py
```

This creates a fresh container process but **reuses the same mounted project files and verified data**. It does not erase earlier outputs or force new downloads. Avoid running this command at the same time as another notebook execution or pipeline that writes the same output files. For an independent run with separate files, use a separate repository checkout.

## 4. What is shared and what stays local

The image includes the project code and saved review artifacts. Raw source data, trained model binaries, backups, caches, and archive packages are excluded from the image build context. Git also ignores local raw data and model binaries. The public-data downloader supplies the raw inputs when a full run is requested.

The small curated observation and feature Parquet files under `outputs/final_protocol/` are retained for inspection. They are derived from the public dataset and are distinct from the complete raw train/eval downloads. Existing local raw files and regenerated model files remain available to the lab through its bind mount even though they are excluded from the image.

The full independent verification script also checks a trained model file generated by the notebook. Accordingly, opening the saved HTML or notebook is an immediate review path; it is not a claim that the complete model verification has run on your machine. Run `scripts/reproduce.py` to generate the model and perform that verification in your environment.

FreshRetailNet-50K is provided by Dingdong under CC BY 4.0. See the project README for the dataset link, pinned revision, attribution, experiment details, and limitations.

## 5. Alternative ports

Copy `.env.example` to `.env` and change `REPORT_PORT` or `JUPYTER_PORT` if either default host port is busy. Restart with `docker compose up -d`; the Jupyter URL helper reads the configured host port.

## 6. Stop or restart

Stop and remove the project's service containers:

```bash
docker compose down
```

This leaves the bind-mounted notebook, data, and results on your computer. No `-v` option is needed. To start the services again:

```bash
docker compose up -d
```

If you change the Dockerfile or dependency files, rebuild with:

```bash
docker compose up --build -d
```

A Dockerfile or documented command does not by itself establish that a build or full training run succeeded. Use the actual build output, service health, and pipeline verification results to assess each run.
