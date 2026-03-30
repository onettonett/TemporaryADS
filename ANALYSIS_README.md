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
5. `analyze_inflation_inequality.py` – summary CSVs and line charts per archetype.
6. `visualise_inflation.py` – additional charts from the processed inflation/decomposition files.
7. `data_exploration_deep.py` – HCI/CPIH exploratory plots.

## Run
```bash
python src/run_pipeline.py
python src/analyze_inflation_inequality.py
python src/visualise_inflation.py
python src/data_exploration_deep.py
```
