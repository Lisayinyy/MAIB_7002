#!/usr/bin/env bash
# Builds the hand-in zip. It unpacks into a GroupQ-PricingML/ folder holding the project with the processed table and
# data/raw/freshretail_raw.zip, without the git history, the virtual environment or the extracted parquet files (the
# notebook's first cell extracts them once from the zip). Usage: scripts/make_submission.sh [output.zip]
set -euo pipefail
cd "$(dirname "$0")/.."
name="GroupQ-PricingML"
out="$(cd .. && pwd)/${1:-$name-submission.zip}"
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
rsync -a --copy-links \
    --exclude ".git" --exclude ".venv" --exclude "data.nosync" --exclude "data/raw/*.parquet" --exclude "data/raw/*.part" \
    --exclude "__pycache__" --exclude "*.pyc" --exclude ".ipynb_checkpoints" --exclude "catboost_info" \
    --exclude ".claude" --exclude ".vscode" --exclude ".DS_Store" --exclude "dockerfiles-common" \
    ./ "$stage/$name/"
rm -f "$out"
(cd "$stage" && zip -q -r -9 "$out" "$name")
echo "wrote $out ($(du -h "$out" | cut -f1))"
