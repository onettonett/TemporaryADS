# One Size Fits None — UK Differential Inflation (2015-2023)

Construct group-specific inflation series using **LCF microdata** expenditure
weights and **ONS MM23 CPIH** item-level indices under a **Laspeyres**
framework (lagged weights). Validated against ONS **Household Costs Indices (HCI)**.

## Repository layout

```
src/                 # pipeline scripts (see below)
data/raw/            # raw LCF, MM23, HCI files (not in repo)
data/cleaned/        # manually-curated Excel files (MM23_cleaned.xlsx, HCI_cleaned.xlsx)
data/output/         # CSV results + chart subfolders
docs/                # methodology and limitations notes
notebooks/           # LCF data exploration (export_lcf_to_excel.ipynb)
```

MM23 and HCI are manually wrangled once in Excel and saved to
`data/cleaned/*.xlsx`, then read directly by `data_loaders.py`. There are no
intermediate parquet files; all outputs go to `data/output/` as CSV.

## Pipeline scripts

| Script | Purpose |
|---|---|
| `wrangle_lcf.py` | Clean LCF, compute COICOP expenditure shares, build 3 archetype dimensions (tenure_type, income_quintile, hrp_age_band) |
| `compute_group_inflation.py` | Laspeyres: combine LCF shares with CPIH price changes |
| `data_loaders.py` | Read CPIH/HCI Excel + LCF share CSV (no external state) |
| `run_pipeline.py` | Runs LCF wrangling + inflation computation in sequence |
| `visualise_inflation.py` | 5-phase exploratory charts → `data/output/charts/` |
| `generate_report_figures.py` | Publication-quality figures for IEEE report |

## Quick start

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the pipeline
python src/run_pipeline.py

# 3. Generate charts (after pipeline)
python src/visualise_inflation.py
python src/generate_report_figures.py
```

## Outputs

All results land in `data/output/`:

- `household_inflation.csv` — 45,500 household-year rows, 12 COICOP shares, 3 archetypes, personal inflation proxy
- `group_inflation.csv` — group-level weighted-mean shares + inflation rate
- `group_inflation_breakdown.csv` — COICOP category contributions to each group's inflation
- `charts/` — exploratory charts
- `report_figures/` — publication figures
