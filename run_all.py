"""Reproduce every table and figure in the manuscript.

    python run_all.py            # UCI + NHANES
    python run_all.py --uci      # UCI only
    python run_all.py --nhanes   # NHANES only

Outputs go to results/tables and results/figures.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ckd import experiment_nhanes, experiment_uci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", action="store_true")
    ap.add_argument("--nhanes", action="store_true")
    a = ap.parse_args()
    both = not (a.uci or a.nhanes)
    t0 = time.time()
    if a.uci or both:
        print("== UCI Chronic Kidney Disease ==")
        print(json.dumps(experiment_uci.run(), indent=2, default=float))
    if a.nhanes or both:
        print("== NHANES 2017-2018 external validation ==")
        print(json.dumps(experiment_nhanes.run(), indent=2, default=float))
    print(f"Done in {time.time() - t0:.0f} s. See results/.")


if __name__ == "__main__":
    main()
