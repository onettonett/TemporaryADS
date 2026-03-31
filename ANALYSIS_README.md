# Project Overview (Concise)

Pipeline builds household-type inflation using LCF expenditure weights and CPIH/HCI price indices.

## Steps
1. `wrangle_lcf.py` – clean LCF and compute COICOP share columns and archetype flags.
2. `wrangle_mm23.py` – load CPIH monthly indices.
3. `wrangle_hci.py` – load ONS HCI indices (optional).
4. `compute_group_inflation.py` – combine LCF shares with CPIH price changes (Laspeyres, lagged weights); outputs:
   - `data/processed/group_inflation_rates.parquet`
   - `data/processed/inflation_decomposition.parquet`
   - `data/processed/archetype_inflation_summary.parquet`
5. `visualise_inflation.py` – five-phase visualisation strategy (sample coverage, expenditure heterogeneity, price environment, group-specific inflation, HCI validation); outputs 32 charts to `data/processed/charts/`.

## Run
```bash
python src/run_pipeline.py
python src/visualise_inflation.py
```
