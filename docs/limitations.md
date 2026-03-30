## LCF wrangling limitations and assumptions (`src/wrangle_lcf.py`)

### Resolved in v2 (previously listed as limitations)

- **Income quintiles are now weighted**: `income_quintile` is computed using `weighta` (survey grossing weights) and `eqincdmp` (equivalised income, modified OECD scale) within each year. This produces population-representative quintiles that adjust for household size and composition.

- **Expenditure cleaning now applied**: negative total expenditures are set to NaN, negative division totals are set to NaN, and shares are Winsorised at [1st, 99th] percentiles within each year to address diary artefacts.

- **Denominator consistency validated**: the pipeline checks whether `p630tp` agrees with the sum of division totals (`p601t`-`p612t`) and uses the division sum when they diverge by >1%, ensuring share numerators and denominator are internally consistent.

- **Disability classification now available**: `is_disability` flag constructed from person-level DLA (b403, b405), Attendance Allowance (b421), and PIP (b552, b553) receipt. Both DLA and PIP captured to handle the gradual transition between the two benefit systems over 2015-2023.

### Remaining limitations

- **HRP identification**: assumes the household reference person is `person == 1` in `dvper`. If this convention differs in some years/releases, `hrp_age` / `hrp_sex` (and therefore `is_pensioner`) will be misassigned.

- **Income restrictions**: only `eqincdmp > 0` observations are assigned an income quintile; zero/negative incomes are left as NaN. This excludes a small number of households with reported zero or negative equivalised income.

- **Disability definition is benefit-based, not condition-based**: the LCF does not ask about long-standing illness or disability directly. The `is_disability` flag relies on disability benefit receipt (DLA/PIP/AA), which under-counts disabled people who do not claim benefits.

- **Carer classification is benefit-based**: `is_carer` relies on Carer's Allowance receipt (b343 > 0), which under-counts unpaid carers who do not claim. Only ~2.7% of households are flagged.

- **Share sums not exactly 1.0**: mean share sum is ~0.992 because the division sum denominator may exclude very small COICOP items not captured in `p601t`-`p612t`. This is a minor issue (< 1% deviation on average).

- **Winsorisation is within-year only**: percentile bounds are computed separately per year. If a genuine structural shift occurs (e.g., energy crisis 2022), the bounds will adapt, which is desirable but means extreme values in crisis years may be less aggressively clipped.

- **Rent-free tenure has small cell sizes**: only ~50 households per year, below the 100-observation minimum for reliable group estimates. Consider merging with another tenure category or dropping from analysis.

- **"COVID years" flag covers two FYs**: both 2019/20 and 2020/21 are flagged (`is_covid_year`). The 2019/20 LCF fieldwork ended March 2020, so it mostly pre-dates COVID but the final weeks overlap with the first lockdown.

- **Single-parent definition is structural**: `is_single_parent` = 1 adult + at least 1 child. This includes lone parents living alone with children but would miss lone parents in multi-adult households (e.g., living with grandparents).

- **Employment status derived from ILO position (a206)**: a206 codes 0 ("not recorded") are excluded from working/retired/unemployed counts, which may undercount in some categories. Children and non-respondents both code as 0.

- **No equivalisation of expenditure shares**: shares are computed from raw household expenditure. Unlike income quintiles (which use equivalised income), the shares themselves are not adjusted for household size. This is standard practice (ONS HCI also uses raw expenditure shares).
