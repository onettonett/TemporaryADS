## LCF wrangling (`src/wrangle_lcf.py`)

### Purpose
`wrangle_lcf.py` builds a harmonised, multi-year panel of Living Costs and Food Survey (LCF) derived data spanning financial years **2015/16 to 2023/24**, and produces an analysis-ready file of **COICOP expenditure shares** with a comprehensive set of **household archetype classification variables** for differential inflation analysis.

The outputs are used downstream to construct household-type-specific spending baskets that can be linked to CPIH sub-indices (from MM23).

### Pipeline steps (6 stages)

1. **Extract** - load dvhh + dvper Stata files, select target variables
2. **Extract** - load person-level data for disability, employment, carer status
3. **Interim save** - save raw extracted data before transformations
4. **Clean** - handle negative expenditures, validate denominators, flag zeros
5. **Shares + Archetypes** - compute COICOP shares, filter implausible households, build all household group flags
6. **QA** - cell-size checks, share-sum validation

### Inputs (raw)
For each year 2015-2023 it reads two Stata files from `data/raw/LCF/`:

- **`dvhh`** (derived household file): household expenditure + household demographics
- **`dvper`** (derived person file): person-level demographics, economic position, disability benefits, carer status

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
  - Disability benefit amounts (b403, b405, b421, b552, b553)
  - Carer's Allowance (b343)
  - Economic position (a206, a015)

- `data/processed/lcf_expenditure_shares.parquet`
  Household-year rows that start from `lcf_household` and add:
  - `share_*` COICOP division expenditure shares (domain-filtered)
  - Full archetype variables (see below)

### What is extracted (column selection decisions)

From `dvhh` it attempts to keep:

- **Identifiers and weights**
  - `case` (household case ID)
  - `weighta` (annual grossing weight)

- **Household demographics**
  - `a049` (household size - total persons)
  - `a062` (family type - detailed: 1 man, 1 woman, couples+children etc.)
  - `a121`, `a122` (tenure type codes)
  - `gorx` (Government Office Region, 1-12)
  - Age-banded person counts (sum to `a049`):
    - `a040` (children under 2), `a041` (children 2-5), `a042` (children 5-18)
    - `a043` (adults 18-45), `a044` (adults 45-60), `a045` (adults 60-65)
    - `a046` (adults 65-70), `a047` (adults 70+)
  - `a054` (number of workers in household)
  - `sexhrp` (sex of HRP)
  - `a065p` (age band of HRP, coded)
  - `a071` (NS-SEC of HRP)

- **Income / housing payment variables**
  - `p389p` (gross normal weekly household income)
  - `eqincdmp` (equivalised income, modified OECD scale)
  - `b010` (rent, gross)
  - `b020` (mortgage payments)

- **Expenditure totals**
  - `p630tp` (total COICOP expenditure)

- **COICOP division totals**
  - `p600t`-`p612t` (division totals; `p601t`-`p612t` are used to compute shares)

- **COICOP class-level totals**
  - 24 selected `ck*t` variables for granular CPIH matching

From `dvper` it attempts to keep:

- `case` (link key), `person` (person number), `a005p` (age), `a006p` (sex), `a200` (household size)
- `a206` (economic position, ILO), `a015` (employment position)
- `b403` (DLA self-care), `b405` (DLA mobility), `b421` (Attendance Allowance)
- `b552` (PIP care), `b553` (PIP mobility)
- `b343` (Carer's Allowance)

### Expenditure cleaning

Before computing shares, the pipeline applies these cleaning steps:

1. **Negative total expenditure** (`p630tp < 0`) is set to NaN. These are implausible for spending basket construction.
2. **Negative division totals** (`p601t`-`p612t < 0`) are set to NaN.
3. **Denominator consistency check**: if `p630tp` diverges from `sum(p601t:p612t)` by more than 1% for a substantial proportion of households, the division sum is used as the denominator instead, for internal consistency.
4. **Zero/missing total expenditure** households are flagged (shares will be NaN). These households had no valid diary expenditure.

### COICOP division expenditure share construction (`share_*`)

For each household-year observation:

$$\text{share}_{d}=\frac{\text{division expenditure } (p60dt)}{\text{total COICOP expenditure (cleaned denominator)}}$$

**Household filtering**: after computing shares, we remove households with demonstrably incomplete or erroneous diary records: negative total expenditure (7), zero food expenditure (296), or zero housing & utilities expenditure (50) — totalling 350 households (0.76%). Standard per-column winsorisation is inappropriate here because expenditure shares are compositional (summing to unity per household); clipping individual shares would break the budget constraint.

**Validation**: the pipeline checks that share sums are approximately 1.0 and reports deviations.

### Archetype variables (household classifications)

#### 1. Tenure type (`tenure_type`)
Based on `a121`: social_rent, private_rent, own_outright, own_mortgage, rent_free, unknown.

#### 2. Pensioner flag (`is_pensioner`)
Multi-signal definition: `hrp_age >= 66` OR any person aged 65+ (`a046 + a047 > 0`, i.e. adults 65-70 + adults 70+) OR HRP economic position is retired (a206 in {6,7}) OR HRP employment position is retired (a015 == 2). This captures early retirees and aligns with ONS HCI methodology.

#### 3. Income quintile (`income_quintile`)
**Weighted** quintiles using equivalised income (`eqincdmp`, modified OECD scale) and survey weights (`weighta`). Computed within each year using cumulative weight approach. Falls back to unweighted gross income if equivalised income is unavailable. Rows with missing/zero/negative income are NaN.

#### 4. Children and composition
- `n_children` (sum of `a040` + `a041` + `a042`: children under 2 + 2-5 + 5-18), `has_children` (boolean)
- `n_adults` (sum of `a043` + `a044` + `a045` + `a046` + `a047`: adults across all age bands)
- `is_single_parent` (1 adult + at least 1 child)
- `hh_composition`: single_parent / couple_with_children / no_children

#### 5. Disability (`is_disability`)
Aggregated from person-level: household is flagged if **any** person receives DLA self-care (b403), DLA mobility (b405), Attendance Allowance (b421), PIP care (b552), or PIP mobility (b553). Both DLA and PIP are captured because DLA was gradually replaced by PIP during the study period.

#### 6. Carer household (`is_carer`)
Household where any person receives Carer's Allowance (b343 > 0).

#### 7. Employment status (`employment_status`)
Aggregated from person-level economic position (a206): all_working / has_unemployed / all_retired / mixed.

#### 8. Region (`region`, `region_broad`)
- `region`: 12 Government Office Regions from `gorx`
- `region_broad`: aggregated to North / Midlands / South / London / Devolved (for adequate cell sizes)

#### 9. HRP age band (`hrp_age_band`)
Continuous HRP age grouped into: under_30, 30_to_49, 50_to_64, 65_to_74, 75_plus.

#### 10. COVID flag (`is_covid_year`)
Expanded to cover both FY 2019/20 (year==2019) and FY 2020/21 (year==2020), since COVID effects began in March 2020 and extended across both financial years.

### Quality assurance

- **Cell-size check**: for each archetype group x year, warns if fewer than 100 unweighted observations (unreliable group estimates).
- **Share diagnostics**: reports valid observation count, share sum statistics.
- **Summary statistics**: weighted mean shares, archetype distributions printed at pipeline completion.
