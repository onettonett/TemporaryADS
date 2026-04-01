#!/bin/bash
# ============================================================================
# QUICK START: Measuring Differential Inflation by Household Type
# ============================================================================
# Runs the complete data wrangling, analysis and figure generation pipeline.
# Expected runtime: ~10-20 minutes depending on your machine
# ============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$PROJECT_ROOT/src"

echo "============================================================================"
echo "DIFFERENTIAL INFLATION ANALYSIS PIPELINE"
echo "============================================================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Choose Python interpreter from the active environment.
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "ERROR: '$PYTHON_BIN' not found. Activate your environment first."
    echo "  Conda: conda activate ads_easter"
    echo "  venv:  source .venv/bin/activate"
    exit 1
fi

echo "Python: $("$PYTHON_BIN" --version)"
echo ""

# STEP 1: Wrangle raw data + compute group inflation
echo "--- Step 1: Data Wrangling + Group Inflation ---"
cd "$SRC"
"$PYTHON_BIN" run_pipeline.py
echo ""

# STEP 2: Exploratory charts (6 phases)
echo "--- Step 2: Exploratory Visualisation ---"
"$PYTHON_BIN" visualise_inflation.py
echo ""

# STEP 3: Publication-quality report figures
echo "--- Step 3: Report Figures ---"
"$PYTHON_BIN" generate_report_figures.py
echo ""

# SUMMARY
echo "============================================================================"
echo "PIPELINE COMPLETE"
echo "============================================================================"
echo ""
echo "Outputs:"
echo "  data/processed/          -- parquet datasets"
echo "  data/processed/charts/   -- exploratory charts (6 phases)"
echo "  data/processed/report_figures/ -- publication-quality figures"
echo ""
