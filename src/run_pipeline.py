"""
run_pipeline.py
===============
Run the full pipeline end-to-end:

    1. wrangle_lcf.py              - build LCF expenditure-share CSV
    2. compute_group_inflation.py  - compute archetype inflation rates

MM23 and HCI are manually cleaned in Excel (data/cleaned/*.xlsx) and are
read directly by the scripts above via data_loaders.py — no script needed.

Chart generation is kept separate; run generate_report_figures.py or
visualise_inflation.py manually after the pipeline completes.

Usage:
    python src/run_pipeline.py
"""

import pathlib
import subprocess
import sys
import time

SRC = pathlib.Path(__file__).parent
SCRIPTS = ["wrangle_lcf.py", "compute_group_inflation.py"]


def run(script: str) -> None:
    print(f"\n{'=' * 60}\n  RUNNING: {script}\n{'=' * 60}")
    start = time.time()
    result = subprocess.run([sys.executable, str(SRC / script)])
    if result.returncode != 0:
        print(f"\n  FAILED: {script} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  DONE: {script} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    for s in SCRIPTS:
        run(s)
    print("\nAll done. Outputs in data/output/")
