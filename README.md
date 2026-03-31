# One Size Fits None — UK Differential Inflation (2015–2023)

Construct group-specific inflation series using **LCF microdata** expenditure weights and **ONS MM23 CPIH** item-level indices under a **Laspeyres** framework (lagged weights).

## Repository layout (high level)
- `src/`: pipeline scripts
- `data/`: inputs/outputs (raw data not included in this repo)
- `plots/`: generated figures
- `docs/`: methodology and limitations notes

## Quick start (recommended)
Run the full pipeline:

```bash
bash QUICK_START.sh
```

## Environment setup (pick one)

### Option A: Conda (best if you already use Anaconda)

```bash
conda env create -f environment.yml
conda activate ads
python -V
```

### Option B: venv + pip (works everywhere)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -V
```

## Running scripts manually

```bash
python src/run_pipeline.py       # wrangle LCF, MM23, HCI → parquets
python src/visualise_inflation.py  # five-phase visualisation → 32 charts
```

## Outputs
- Core results (parquet): `data/processed/`
  - `group_inflation_rates.parquet`
  - `inflation_decomposition.parquet`
  - `archetype_inflation_summary.parquet`
- Figures (PNG): `data/processed/charts/`

## Notes on key group definitions
See `docs/methodology.md` and `docs/limitations.md` for details (e.g., `is_pensioner` and benefit-based `is_disability` / `is_carer` flags).

# Instructions on Running this ADS Project

- This project uses Anaconda (which you can install from here if you don't have it alreaday: https://www.anaconda.com/download)

### Environment setup
```bash
conda env create -f environment.yml
conda activate ads
python -V
```