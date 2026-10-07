#!/usr/bin/env python3
"""Download official data, execute the notebook, verify, and refresh reports."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("Running:", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def main():
    run(sys.executable, "scripts/download_public_data.py")
    run(sys.executable, "scripts/run_notebook.py")
    run(sys.executable, "scripts/verify_final_protocol.py")
    run(sys.executable, "scripts/check_saved_results.py")
    run(sys.executable, "-m", "nbconvert", "--to", "html", "--embed-images",
        "01_freshretail_project.ipynb", "--output", "01_freshretail_project.html",
        "--output-dir", "outputs/final_protocol")
    run(sys.executable, "scripts/build_report_index.py")
    run(sys.executable, "scripts/build_release_manifest.py")
    print("PASS: full experiment and independent verification completed.", flush=True)


if __name__ == "__main__":
    main()
