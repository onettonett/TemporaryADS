"""
run_pipeline.py
===============
Pipeline runner: executes LCF + CPIH/HCI steps in sequence.

Usage:
    python src/run_pipeline.py           # run all steps
    python src/run_pipeline.py lcf       # run LCF only
    python src/run_pipeline.py mm23      # run MM23/CPIH only
    python src/run_pipeline.py hci       # run HCI only
    python src/run_pipeline.py inflation # compute group inflation

All outputs go to data/interim/ (intermediate) and data/processed/ (analysis-ready).
Raw data in data/raw/ is never modified.
"""

import sys
import time
import importlib


STEPS = {
    "lcf": ("wrangle_lcf", "LCF (Living Costs and Food Survey)"),
    "mm23": ("wrangle_mm23", "MM23 (CPIH Price Indices)"),
    "hci": ("wrangle_hci", "HCI (Household Costs Indices)"),
    "inflation": ("compute_group_inflation", "Group-Specific Inflation Rates"),
}


def run_step(module_name: str, description: str) -> None:
    """Import and run a wrangling module's main() function."""
    start = time.time()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - start
        print(f"\n  [{description}] completed in {elapsed:.1f}s\n")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [{description}] FAILED after {elapsed:.1f}s: {e}\n")
        raise


def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]

    # Determine which steps to run
    if not args:
        steps_to_run = list(STEPS.keys())
    else:
        steps_to_run = [a for a in args if a in STEPS]
        unknown = [a for a in args if a not in STEPS]
        if unknown:
            print(f"Unknown steps: {unknown}")
            print(f"Available: {list(STEPS.keys())}")
            sys.exit(1)

    print("=" * 60)
    print("DATA WRANGLING PIPELINE")
    print("Measuring Differential Inflation by Household Type")
    print("=" * 60)
    print(f"\nSteps to run: {', '.join(steps_to_run)}")

    total_start = time.time()

    for step_key in steps_to_run:
        module_name, description = STEPS[step_key]
        print(f"\n{'─' * 60}")
        print(f"STEP: {description}")
        print(f"{'─' * 60}")
        run_step(module_name, description)

    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"ALL DONE — total time: {total_elapsed:.1f}s")
    print("=" * 60)
    print("\nOutputs:")
    print("  data/interim/   — intermediate extracted data")
    print("  data/processed/ — analysis-ready merged datasets")


if __name__ == "__main__":
    main()
