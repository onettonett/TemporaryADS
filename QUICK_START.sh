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

# Check that we can access Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi

echo "Python version:"
python3 --version
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
python3 run_pipeline.py

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

python3 compute_group_inflation.py

echo ""
echo "✅ Step 2 complete. Main outputs:"
echo "   • data/processed/group_inflation_rates.parquet"
echo "   • data/processed/inflation_decomposition.parquet"
echo "   • data/processed/archetype_inflation_summary.parquet"
echo ""

# STEP 3: Analyze and visualize results

echo "───────────────────────────────────────────────────────────────────────────"
echo "STEP 3: Analysis & Visualization"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "Generates:"
echo "  • CSV summary tables (for dissertation)"
echo "  • PNG line charts (inflation trends by income, tenure, region, age)"
echo "  • Heatmaps (inflation inequality over time)"
echo "  • K-means clustering (household profiles by inflation loss)"
echo ""
echo "Running: python analyze_inflation_inequality.py"
echo ""

python3 analyze_inflation_inequality.py

echo ""
echo "✅ Step 3 complete. Outputs in:"
echo "   • data/analysis/     (CSV tables)"
echo "   • plots/             (PNG charts)"
echo ""

# VERIFICATION & SUMMARY

echo "───────────────────────────────────────────────────────────────────────────"
echo "VERIFICATION & SUMMARY"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""

# Check that key outputs exist
if [ -f "$PROJECT_ROOT/data/processed/group_inflation_rates.parquet" ]; then
    echo "✅ group_inflation_rates.parquet exists"
    python3 -c "
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

# Count CSV files
csv_count=$(find "$PROJECT_ROOT/data/analysis/" -name "*.csv" 2>/dev/null | wc -l)
echo "✅ CSV files generated: $csv_count"

# Count PNG files
png_count=$(find "$PROJECT_ROOT/plots/" -name "*.png" 2>/dev/null | wc -l)
echo "✅ PNG charts generated: $png_count"

echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo "✅ PIPELINE COMPLETE"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "📊 RESULTS SUMMARY"
echo ""
echo "Key findings in: data/processed/group_inflation_rates.parquet"
echo "  • Rows: one per (archetype, year)"
echo "  • Columns: archetype_name, archetype_value, year, inflation_rate, ..."
echo ""
echo "CSV tables for dissertation:"
echo "  • data/analysis/inflation_by_income_quintile.csv"
echo "  • data/analysis/inflation_by_tenure.csv"
echo "  • data/analysis/inflation_by_region.csv"
echo "  • data/analysis/inflation_by_hrp_age_band.csv"
echo "  • data/analysis/archetype_clusters.csv"
echo ""
echo "PNG charts for dissertation:"
echo "  • plots/inflation_by_income_quintile.png"
echo "  • plots/inflation_by_tenure.png"
echo "  • plots/inflation_by_region.png"
echo "  • plots/inflation_by_age_band.png"
echo "  • plots/inflation_heatmap_*.png"
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
echo "   ✓ Does Q5 (richest) face lower inflation than Q1 (poorest)?"
echo "   ✓ Is 2022-2023 spike visible for food/energy inflation?"
echo "   ✓ Does your pensioner result match ONS pensioner CPI?"
echo ""
echo "3. WRITE YOUR DISSERTATION"
echo "   ✓ Use CSV tables in Results section"
echo "   ✓ Include PNG charts as Figures"
echo "   ✓ Follow structure in ANALYSIS_README.md"
echo "   ✓ Document limitations in Methodology"
echo ""
echo "4. KEY REFERENCES"
echo "   📖 ANALYSIS_README.md - comprehensive guide (300+ lines)"
echo "   📖 COMPLETION_STATUS.md - what's been built, next steps"
echo "   📖 src/compute_group_inflation.py - methodology comments"
echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "Questions? Check ANALYSIS_README.md for detailed documentation."
echo "Ready to write your dissertation!"
echo ""
