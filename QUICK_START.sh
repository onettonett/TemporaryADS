#!/bin/bash
# ============================================================================
# QUICK START: Measuring Differential Inflation by Household Type
# ============================================================================
# This script runs the complete data wrangling and analysis pipeline.
# Expected runtime: ~10-20 minutes depending on your machine
# ============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$PROJECT_ROOT/src"
DATA="$PROJECT_ROOT/data"

echo "============================================================================"
echo "DIFFERENTIAL INFLATION ANALYSIS PIPELINE"
echo "============================================================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SRC"
echo ""

# Choose Python interpreter from the active environment.
# Prefer `python` so conda/venv activation is respected on macOS.
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "ERROR: '$PYTHON_BIN' not found. Activate your environment first."
    echo "  Conda: conda activate ads"
    echo "  venv:  source .venv/bin/activate"
    exit 1
fi

echo "Python interpreter:"
command -v "$PYTHON_BIN"
echo "Python version:"
"$PYTHON_BIN" --version
echo ""

# STEP 1: Extract and clean data from raw sources

echo "───────────────────────────────────────────────────────────────────────────"
echo "STEP 1: Data Extraction & Cleaning"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "This loads LCF shares and CPIH/HCI prices, then builds group inflation."
echo ""
echo "Running: python run_pipeline.py"
echo ""

cd "$SRC"
"$PYTHON_BIN" run_pipeline.py

echo ""
echo "✅ Step 1 complete. Outputs in data/interim/ and data/processed/"
echo ""

# STEP 2: Compute group-specific inflation rates

echo "───────────────────────────────────────────────────────────────────────────"
echo "STEP 2: Computing Group-Specific Inflation Rates"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "This is the core analysis step. It combines:"
echo "  • LCF expenditure shares (by household archetype)"
echo "  • MM23 CPIH price indices (monthly)"
echo "  • COICOP concordance mapping"
echo ""
echo "Calculates: inflation_rate = Σ [ expenditure_share × price_change ]"
echo ""
echo "Running: python compute_group_inflation.py"
echo ""

"$PYTHON_BIN" compute_group_inflation.py

echo ""
echo "✅ Step 2 complete. Main outputs:"
echo "   • data/processed/group_inflation_rates.parquet"
echo "   • data/processed/inflation_decomposition.parquet"
echo "   • data/processed/archetype_inflation_summary.parquet"
echo ""

# STEP 3: Visualise results

echo "───────────────────────────────────────────────────────────────────────────"
echo "STEP 3: Visualisation (Five-Phase Strategy)"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "Generates 32 charts across five phases:"
echo "  • Phase 1: Sample and Coverage"
echo "  • Phase 2: Expenditure Heterogeneity"
echo "  • Phase 3: The Price Environment"
echo "  • Phase 4: Group-Specific Inflation"
echo "  • Phase 5: Validation Against HCI"
echo ""
echo "Running: python visualise_inflation.py"
echo ""

"$PYTHON_BIN" visualise_inflation.py

echo ""
echo "✅ Step 3 complete. Charts saved to: data/processed/charts/"
echo ""

# VERIFICATION & SUMMARY

echo "───────────────────────────────────────────────────────────────────────────"
echo "VERIFICATION & SUMMARY"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""

# Check that key outputs exist
if [ -f "$PROJECT_ROOT/data/processed/group_inflation_rates.parquet" ]; then
    echo "✅ group_inflation_rates.parquet exists"
    "$PYTHON_BIN" -c "
import pandas as pd
df = pd.read_parquet('$PROJECT_ROOT/data/processed/group_inflation_rates.parquet')
print(f'   Rows: {len(df):,}')
print(f'   Archetype dimensions: {df[\"archetype_name\"].nunique()}')
print(f'   Years covered: {sorted(df[\"year\"].unique())}')
print(f'   Mean inflation: {df[\"inflation_rate\"].mean():.2f}%')
print(f'   Range: {df[\"inflation_rate\"].min():.2f}% to {df[\"inflation_rate\"].max():.2f}%')
"
else
    echo "❌ group_inflation_rates.parquet NOT found"
fi

echo ""

# Count charts
chart_count=$(find "$PROJECT_ROOT/data/processed/charts/" -name "*.png" 2>/dev/null | wc -l)
echo "✅ Charts generated: $chart_count (in data/processed/charts/)"

echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo "✅ PIPELINE COMPLETE"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "📊 RESULTS SUMMARY"
echo ""
echo "Key findings in: data/processed/group_inflation_rates.parquet"
echo "  • Rows: one per (archetype, year)"
echo "  • Columns: archetype_name, archetype_value, year, inflation_rate"
echo ""
echo "Charts for dissertation (data/processed/charts/):"
echo "  • p2_4_basket_income_quintile.png  — expenditure shares Q1–Q5"
echo "  • p4_series_income_quintile.png    — group inflation lines vs CPIH"
echo "  • p4_gap_*.png                     — annual inflation gap bar charts"
echo "  • p4_cumulative_*.png              — cumulative purchasing power loss"
echo "  • p5_*_comparison.png              — validation against HCI"
echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo "NEXT STEPS FOR YOUR DISSERTATION"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. LOAD & EXPLORE RESULTS (Python)"
echo "   import pandas as pd"
echo "   inflation = pd.read_parquet('data/processed/group_inflation_rates.parquet')"
echo "   inflation.groupby('archetype_name')['inflation_rate'].mean()"
echo ""
echo "2. VALIDATE FINDINGS"
echo "   ✓ Does Q1 (poorest) face higher inflation than Q5 (richest)?"
echo "   ✓ Is 2022-2023 spike visible for food/energy inflation?"
echo "   ✓ Does your pensioner result match HCI Retired?"
echo ""
echo "3. WRITE YOUR DISSERTATION"
echo "   ✓ Include PNG charts as Figures from data/processed/charts/"
echo "   ✓ Follow structure in ANALYSIS_README.md"
echo "   ✓ Document limitations in Methodology"
echo ""
echo "4. KEY REFERENCES"
echo "   📖 ANALYSIS_README.md - pipeline overview"
echo "   📖 COMPLETION_STATUS.md - component status"
echo "   📖 src/compute_group_inflation.py - methodology comments"
echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "Questions? Check ANALYSIS_README.md for documentation."
echo ""
