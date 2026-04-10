## LCF wrangling (`src/wrangle_lcf.py`)

### Purpose
`wrangle_lcf.py` builds a harmonised, multi-year panel of Living Costs and Food Survey (LCF) derived data spanning financial years **2015/16 to 2023/24**, and produces an analysis-ready file of **COICOP expenditure shares** with a comprehensive set of **household archetype classification variables** for differential inflation analysis.

The outputs are used downstream to construct household-type-specific spending baskets that can be linked to CPIH sub-indices (from MM23).

### Pipeline steps (6 stages)

1. **Extract** - load dvhh Stata files, select target household-level variables
2. **Extract** - load dvper Stata files, select target person-level variables (age, sex)
3. **Interim save** - save raw extracted data before transformations
4. **Clean** - handle negative expenditures, validate denominators, flag zeros
5. **Shares + Archetypes** - compute COICOP shares, filter implausible households, build household archetype flags
6. **QA** - cell-size checks, share-sum validation

### Inputs (raw)
For each year 2015-2023 it reads two Stata files from `data/raw/LCF/`:

- **`dvhh`** (derived household file): household expenditure + household demographics
- **`dvper`** (derived person file): person-level demographics (age, sex) used only to pick up HRP (person 1) age

The script lowercases all column names on read and never modifies the raw inputs on disk.

### Outputs
The script writes three parquet datasets:

- `data/interim/lcf_household.parquet`
  Household-year rows containing a selected subset of `dvhh` columns plus:
  - `year` (integer, 2015-2023)
  - `fy` (financial year string like `2015/16`)

- `data/interim/lcf_person.parquet`
  Person-year rows containing a selected subset of `dvper` columns plus:
  - `year` (integer, 2015-2023)
  - `person_age`, `person_sex_code` (used only to recover HRP demographics)

- `data/processed/household_inflation.parquet`
  Household-year rows that start from `lcf_household` and add:
  - `share_*` COICOP division expenditure shares (domain-filtered)
  - Three archetype variables (see below)

### What is extracted (column selection decisions)

From `dvhh` it attempts to keep:

- **Identifiers and weights**
  - `case` (household case ID)
  - `weighta` (annual grossing weight)

- **Household demographics**
  - `a049` (household size - total persons)
  - `a121` (tenure type code)

- **Income / housing payment variables**
  - `p389p` (gross normal weekly household income)
  - `eqincdmp` (equivalised income, modified OECD scale)
  - `b010` (rent, gross - used for COICOP 04 rent/energy decomposition)

- **Expenditure totals**
  - `hh_total_coicop_expenditure` (total COICOP expenditure)

- **COICOP division totals**
  - `p600t`-`p612t` (division totals; `p601t`-`p612t` are used to compute shares)

From `dvper` it attempts to keep:

- `case` (link key), `person` (person number within household)
- `a005p` (age), `a006p` (sex)

### Expenditure cleaning

Before computing shares, the pipeline applies these cleaning steps:

1. **Negative total expenditure** (`hh_total_coicop_expenditure < 0`) is set to NaN. These are implausible for spending basket construction.
2. **Negative division totals** (`p601t`-`p612t < 0`) are set to NaN.
3. **Plausibility filter**: households with total expenditure below £30/week (likely incomplete diary) or above £3,000/week (likely data-entry error) are removed before computing shares.
4. **Denominator consistency check**: if `hh_total_coicop_expenditure` diverges from `sum(p601t:p612t)` by more than 1% for a substantial proportion of households, the division sum is used as the denominator instead, for internal consistency.
5. **Zero/missing total expenditure** households are flagged (shares will be NaN). These households had no valid diary expenditure.

### COICOP division expenditure share construction (`share_*`)

For each household-year observation:

$$\text{share}_{d}=\frac{\text{division expenditure } (p60dt)}{\text{total COICOP expenditure (cleaned denominator)}}$$

**Household filtering**: after computing shares, we remove households with demonstrably incomplete or erroneous diary records: negative total expenditure (7), zero food expenditure (296), or zero housing & utilities expenditure (50) — totalling 350 households (0.76%). Standard per-column winsorisation is inappropriate here because expenditure shares are compositional (summing to unity per household); clipping individual shares would break the budget constraint.

**Validation**: the pipeline checks that share sums are approximately 1.0 and reports deviations.

### Archetype variables (household classifications)

The analysis uses **three** archetype dimensions, chosen because they are (a) conceptually independent, (b) directly relevant to differential inflation exposure, and (c) populated with adequate cell sizes (≥100 households/year) across the 2015-2023 panel.

#### 1. Tenure type (`tenure_type`)
Based on `a121`: social_rent, private_rent, own_outright, own_mortgage. Rent-free households (~50/year, below the 100-observation minimum) are excluded by leaving them unmapped (NaN/"unknown").

#### 2. Income quintile (`income_quintile`)
**Weighted** quintiles using equivalised income (`eqincdmp`, modified OECD scale) and survey weights (`weighta`). Computed within each year using cumulative weight approach. Falls back to unweighted gross income if equivalised income is unavailable. Rows with missing/zero/negative income are NaN.

#### 3. HRP age band (`hrp_age_band`)
Continuous HRP age grouped into: under_30, 30_to_49, 50_to_64, 65_to_74, 75_plus.

#### Descriptive columns (carried through but not used as archetype dimensions)
- `household_size` (from `a049`): total persons in household
- `hrp_age`: continuous HRP age before banding

### Quality assurance

- **Cell-size check**: for each archetype group x year, warns if fewer than 100 unweighted observations (unreliable group estimates).
- **Share diagnostics**: reports valid observation count, share sum statistics.
- **Summary statistics**: weighted mean shares, archetype distributions printed at pipeline completion.
