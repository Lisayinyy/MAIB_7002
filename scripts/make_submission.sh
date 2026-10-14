#!/usr/bin/env bash
# Builds the hand-in zip: the project folder with the processed table and data/raw/freshretail_raw.zip,
# without the git history, the virtual environment, or the extracted parquet files (the notebook's first
# cell extracts them once from the zip). Usage: scripts/make_submission.sh [output.zip]
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-../pricing-ml-submission.zip}"
rm -f "$out"
zip -r -9 "$out" . -x ".git/*" ".venv/*" "data.nosync/*" "data/raw/*.parquet" "data/raw/*.part" "*/__pycache__/*" "*.pyc" \
    "*/.ipynb_checkpoints/*" ".ipynb_checkpoints/*" "catboost_info/*" ".claude/*" ".vscode/*" "*.DS_Store"
echo "wrote $out ($(du -h "$out" | cut -f1))"
