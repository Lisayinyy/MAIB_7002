"""Run the complete exploratory comparison, or independently audit its saved scores."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finalproject_pricingml import v2

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--output-dir", type=Path, default=v2.OUT)
    a = p.parse_args()
    if a.verify_only:
        print(v2.verify_saved(a.output_dir))
    else:
        v2.run(a.output_dir)
