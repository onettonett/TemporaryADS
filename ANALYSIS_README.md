# Measuring Differential Inflation by Household Type and Region

## Project Overview

This project constructs **household-archetype-specific inflation rates** by combining:
- **LCF (Living Costs and Food Survey)** – household expenditure shares by COICOP category
- **MM23 (ONS CPIH)** – monthly price indices at the COICOP division level
- **FRS (Family Resources Survey)** – household income and benefit data
- **Custom archetype classification** – tenure, pension status, income quintile, region, age, disability, etc.

The result is a panel of inflation rates stratified by household type, answering the question: *"How much did purchasing power actually fall for each household archetype, given what they actually spend on?"*

---

## Why This Matters

The national CPI treats all 28 million households as having identical spending patterns. This is obviously false:
- **Pensioners** spend much more on food and heating; less on transport and recreation
- **Private renters** face monthly rent inflation; owner-occupiers with mortgages face interest-rate shocks
- **Low-income households** spend more (as a share) on food and energy, which have volatile price growth
- **Regional variation** – housing costs, transport, and local services vary widely

By computing group-specific inflation rates, you measure **inequality** in the inflation experience and can identify which households lost the most purchasing power.

---

## Data Pipeline Architecture

```
Raw data (data/raw/)
    ↓
Step 1: Extract & clean (wrangle_*.py)
    ├→ wrangle_lcf.py       → LCF households + archetypes
    ├→ wrangle_frs.py       → FRS household income + benefits
    ├→ wrangle_mm23.py      → MM23 CPIH monthly price indices
    └→ wrangle_hci.py       → HCI (optional, for robustness)
    ↓
Interim data (data/interim/)
    ↓
Step 2: Compute group inflation (compute_group_inflation.py)
    ├→ Load LCF expenditure shares by archetype
    ├→ Load CPIH price indices
    ├→ Apply COICOP concordance mapping
    ├→ Compute annual price changes per COICOP
    └→ Calculate: inflation_g,t = Σ_i [ share_i,g,t × price_change_i,t ]
    ↓
Analysis-ready data (data/processed/)
    ├→ group_inflation_rates.parquet
    ├→ inflation_decomposition.parquet
    └→ archetype_inflation_summary.parquet
    ↓
Step 3: Analyze & visualize (analyze_inflation_inequality.py)
    ├→ Compute inequality gaps between archetype groups
    ├→ Generate line charts (inflation trends by income, tenure, region, age)
    ├→ Cluster households by inflation loss
    └→ Produce summary tables (CSV exports)
```

---

## COICOP Concordance Table

The **concordance table** (defined in `compute_group_inflation.py`) maps LCF expenditure categories to price index series:

| LCF Category | LCF Column Name | MM23 Price Index | COICOP Code |
|---|---|---|---|
| Food | `share_01_food_non_alcoholic` | `food_non_alcoholic` | 01 |
| Alcohol & Tobacco | `share_02_alcohol_tobacco` | `alcohol_tobacco` | 02 |
| Clothing & Footwear | `share_03_clothing_footwear` | `clothing_footwear` | 03 |
| Housing, Fuel & Power | `share_04_housing_fuel_power` | `housing_fuel_power` | 04 |
| Furnishings | `share_05_furnishings` | `furnishings` | 05 |
| Health | `share_06_health` | `health` | 06 |
| Transport | `share_07_transport` | `transport` | 07 |
| Communication | `share_08_communication` | `communication` | 08 |
| Recreation & Culture | `share_09_recreation_culture` | `recreation_culture` | 09 |
| Education | `share_10_education` | `education` | 10 |
| Restaurants & Hotels | `share_11_restaurants_hotels` | `restaurants_hotels` | 11 |
| Miscellaneous | `share_12_misc_goods_services` | `misc_goods_services` | 12 |

**Key Design Decision**: The mapping is 1:1 at the COICOP division level (12 categories). This aligns with LCF granularity and standard ONS publishing practice. If finer-grained analysis is needed (e.g., 01.1.1 = bread & cereals), extend the concordance and LCF archetype data structures.

---

## The Core Inflation Calculation

For each household archetype `g` in year `t`:

$$\pi_{g,t} = \sum_{i=1}^{12} w_{i,g,t-1} \times \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}} \times 100$$

Where:
- $w_{i,g,t-1}$ = expenditure share for COICOP category $i$ in group $g$ (from LCF, lagged)
- $P_{i,t}$ = price index for category $i$ in year $t$ (from MM23, averaged over FY)
- $P_{i,t-1}$ = price index for category $i$ in year $t-1$

This is a **Laspeyres-type index** with household-group-specific weights. It answers: "If household type $g$ spent money according to their 2019 basket, how much did those exact goods cost in 2023?"

---

## Running the Pipeline

### Option 1: Run the full pipeline
```bash
cd src
python run_pipeline.py
```
This runs: FRS → LCF → MM23 → HCI (if available) → compute group inflation.

### Option 2: Run individual steps
```bash
python run_pipeline.py lcf mm23        # Just LCF and MM23
python run_pipeline.py inflation       # Just compute group inflation
```

### Option 3: Run analysis after data is ready
```bash
python compute_group_inflation.py      # Compute inflation rates
python analyze_inflation_inequality.py # Generate charts and summary stats
```

---

## Outputs

After running `compute_group_inflation.py`, you'll find:

### Parquet Files (for further analysis)
- **`group_inflation_rates.parquet`**
  Columns: `archetype_name`, `archetype_value`, `year`, `inflation_rate`, `n_coicop`, `mean_share`, `mean_price_change`
  This is the main result: one row per (archetype, year).

- **`inflation_decomposition.parquet`**
  Detailed breakdown: (archetype, year, COICOP) → contribution to inflation.
  Useful for "attributing" where inflation comes from (e.g., "food inflation drove +0.8pp of pensioner inflation").

- **`archetype_inflation_summary.parquet`**
  Summary statistics by archetype dimension (tenure, income, region, etc.).

### CSV Files (from `analyze_inflation_inequality.py`)
- `inflation_inequality_gaps.csv` – spread between highest/lowest inflation within each archetype
- `inflation_by_income_quintile.csv` – inflation rates for Q1–Q5, by year
- `inflation_by_tenure.csv` – inflation for each tenure type
- `inflation_by_region.csv` – inflation for each broad region
- `inflation_by_hrp_age_band.csv` – inflation by age of household reference person
- `archetype_clusters.csv` – k-means clustering of archetype groups by inflation experience

### Visualizations (PNG, in `plots/`)
- `inflation_by_income_quintile.png` – line chart showing Q1 typically faces higher inflation
- `inflation_by_tenure.png` – line chart comparing renters vs. owners
- `inflation_by_region.png` – regional variation in inflation
- `inflation_by_age_band.png` – inflation trends by HRP age
- `inflation_heatmap_income_quintile.png` – 2D heatmap (archetype × year)
- `inflation_heatmap_tenure.png` – 2D heatmap (tenure × year)

---

## Key Findings to Look For

When you run the analysis, pay attention to:

1. **Income inequality**: Does the bottom quintile consistently face higher inflation than the top quintile? The literature (Hobijn & Lagakos 2005, Jaravel 2019) predicts yes, because low-income households spend more on food and energy.

2. **Tenure effects**: Do private renters face systematically higher inflation than owner-occupiers? This is especially visible during rental-market upswings or interest-rate shocks.

3. **Age effects**: Do pensioners (who spend more on food and heating) face higher inflation in years when food/energy prices spike (2008, 2021–2023)?

4. **Regional variation**: How large are regional differences? Is London substantially different from the North East?

5. **Inflation inequality over time**: Did inflation inequality *widen* during major price shocks (2022–2023)? Or *narrow*?

6. **Clustering insights**: Which household profiles cluster together in terms of inflation experience? Are "low-income private renters" a distinct group from "high-income owner-occupiers"?

---

## Validation & Robustness

### Benchmarking
- **Pensioner inflation**: ONS publishes a pensioner-specific CPI. Your weighted calculation should approximately match this as a sanity check.
- **National aggregate**: If you weight all household archetypes by population share, your result should closely match the national CPIH inflation.

### Sensitivity checks
1. **Fixed-weight vs. chained-weight**: Currently uses lagged weights (Laspeyres). Try setting weights to a fixed base year (e.g., 2015) and compare results. The gap reveals substitution behaviour.
2. **Alternative COICOP mappings**: If LCF and price indices diverge in their category definitions, try aggregating to higher levels (e.g., 1-digit COICOP) and re-running.
3. **Weighting by household counts**: Currently uses survey weights (for population representation). Try unweighted to check if small subgroups drive results.

### Known Limitations
1. **No substitution**: If beef prices surge and households switch to chicken, your COICOP-level weights won't capture this (elasticity assumption).
2. **Aggregate price data**: You're using national price indices. In reality, a pensioner in Newcastle and one in London face different prices, but we have no group-specific regional price data.
3. **Survey errors in LCF**: Non-response bias, under-reporting of alcohol/tobacco, small sample sizes in some region × income cells.
4. **Quality adjustment**: ONS adjusts prices for quality changes nationally. Different groups may experience quality changes differently (e.g., low-income households may shift to lower-quality goods).

---

## Dissertation Structure (for your MSc)

Here's how to write this up:

1. **Introduction** (2 pages)
   - Why average CPI is misleading
   - Policy relevance (different groups respond to inflation differently)

2. **Literature Review** (3–4 pages)
   - Inflation inequality: IFS, Jaravel, others
   - Household heterogeneity in price sensitivity
   - UK-specific context (energy crisis 2022–2023)

3. **Data & Methodology** (4–5 pages)
   - LCF structure, sampling, representativeness
   - CPIH index construction, base year, coverage
   - COICOP concordance: mapping decisions, any aggregations
   - Laspeyres index formula, lagging of weights
   - Archetype classification scheme
   - Limitations section

4. **Results** (6–8 pages)
   - Line charts showing inflation divergence across income, tenure, region
   - Summary tables (mean inflation, spread, volatility by group)
   - Attribution: which COICOP categories drove group-specific inflation
   - Clustering results: household profiles by inflation loss
   - Key finding: Which groups lost the most purchasing power, and when?

5. **Discussion** (3–4 pages)
   - Interpretation of results in policy context
   - Who bears the cost of inflation under the current CPI framework?
   - Robustness: fixed-weight vs. chained, sensitivity to concordance choices
   - Limitations and directions for future work

6. **Conclusion** (1 page)

---

## Next Steps

1. **Run the full pipeline** (should take 5–15 minutes depending on your machine):
   ```bash
   python run_pipeline.py
   ```

2. **Check outputs**:
   ```bash
   ls data/processed/group_inflation_rates.parquet
   ls plots/inflation_by_*.png
   ```

3. **Explore the results**:
   ```bash
   python analyze_inflation_inequality.py
   ```

4. **Load and visualize yourself** (Python):
   ```python
   import pandas as pd
   inflation = pd.read_parquet("data/processed/group_inflation_rates.parquet")
   # Your analysis here
   ```

5. **Write up findings** – use the CSV tables and PNGs in your dissertation.

---

## Questions?

If the pipeline breaks:
- Check that all raw data files are present in `data/raw/LCF/`, `data/raw/FRS/`, `data/raw/MM23/`
- Verify that `COICOP_CONCORDANCE` in `compute_group_inflation.py` matches your LCF column names
- Run individual wranglers to debug: `python wrangle_lcf.py`, `python wrangle_mm23.py`, etc.

If results seem off:
- Validate that MM23 inflation rates are plausible (e.g., food inflation in 2022 was ~8%, should see this)
- Check that LCF archetype aggregations make sense (quintile 5 should have higher income share on luxury goods)
- Spot-check a few households' share calculations by hand

---

## References

- Hobijn, B., & Lagakos, D. (2005). Inflation inequality in the United States. *Review of Income and Wealth*, 51(4).
- Jaravel, X. (2019). The unequal gains from product innovations. *Journal of Political Economy*, 127(2).
- IFS various reports on UK inflation inequality (http://www.ifs.org.uk/)
- ONS CPIH documentation: https://www.ons.gov.uk/economy/inflationandpriceindices/articles/retcpiandcpihmethodguide/2015-03-16
