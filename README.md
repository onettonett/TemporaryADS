# One Size Fits None — UK Differential Inflation (2015-2023)

Construct group-specific inflation series using **LCF microdata** expenditure weights and **ONS MM23 CPIH** item-level indices under a **Laspeyres** framework (lagged weights). Validated against ONS **Household Costs Indices (HCI)**.

## Repository layout
- `src/` -- pipeline scripts (see below)
- `data/raw/` -- raw LCF, MM23, HCI files (not included in repo)
- `data/interim/` -- intermediate extracted parquets
- `data/processed/` -- analysis-ready parquets + charts + report figures
- `docs/` -- methodology and limitations notes

## Pipeline scripts

| Script | Purpose |
|---|---|
| `wrangle_lcf.py` | Clean LCF, compute COICOP expenditure shares, build 3 archetype dimensions |
| `wrangle_mm23.py` | Extract CPIH monthly indices from MM23 |
| `wrangle_hci.py` | Extract ONS HCI validation benchmarks |
| `compute_group_inflation.py` | Laspeyres: combine LCF shares with CPIH price changes |
| `visualise_inflation.py` | 6-phase exploratory charts (sample, baskets, prices, inflation, HCI validation, clustering) |
| `generate_report_figures.py` | Publication-quality figures for IEEE report |
| `run_pipeline.py` | Runs all wrangling + inflation computation in sequence |

## Quick start

```bash
bash QUICK_START.sh
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate ads_easter
```

Or with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running scripts manually

```bash
python src/run_pipeline.py            # wrangle all data + compute inflation
python src/visualise_inflation.py     # exploratory charts -> data/processed/charts/
python src/generate_report_figures.py # report figures -> data/processed/report_figures/
```
