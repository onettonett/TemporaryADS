## LCF wrangling limitations and assumptions (`src/wrangle_lcf.py`)

### Resolved in v2 (previously listed as limitations)

- **Income quintiles are now weighted**: `income_quintile` is computed using `weighta` (survey grossing weights) and `eqincdmp` (equivalised income, modified OECD scale) within each year. This produces population-representative quintiles that adjust for household size and composition.

- **Expenditure cleaning now applied**: negative total expenditures are set to NaN, negative division totals are set to NaN. Implausible households are removed via domain-based filters (zero food, zero housing, negative total expenditure — 350 records, 0.76%) rather than per-column winsorisation, which would break the compositional sum-to-1 constraint of expenditure shares.

- **Denominator consistency validated**: the pipeline checks whether `hh_total_coicop_expenditure` agrees with the sum of division totals (`p601t`-`p612t`) and uses the division sum when they diverge by >1%, ensuring share numerators and denominator are internally consistent.

### Remaining limitations

- **HRP identification**: assumes the household reference person is `person == 1` in `dvper`. If this convention differs in some years/releases, `hrp_age` / `hrp_sex` (and therefore `hrp_age_band`) will be misassigned.

- **Income restrictions**: only `eqincdmp > 0` observations are assigned an income quintile; zero/negative incomes are left as NaN. This excludes a small number of households with reported zero or negative equivalised income.

- **Share sums not exactly 1.0**: mean share sum is ~0.992 because the division sum denominator may exclude very small COICOP items not captured in `p601t`-`p612t`. This is a minor issue (< 1% deviation on average).

- **No per-column outlier treatment**: we deliberately chose not to winsorise or IQR-clip individual expenditure shares. For zero-inflated categories (Education 95.5% zeros, Rent 76%), IQR collapses and flags legitimate subpopulations as outliers. Genuine extreme spenders (e.g., housing-poor renters at 80%+ housing share) are retained, as their inflation experience is central to the research question. Group averaging over 500+ households per cell provides natural robustness.

- **Rent-free tenure excluded**: only ~50 households per year, below the 100-observation minimum for reliable group estimates. These households are excluded from the tenure dimension (mapped to NaN/"unknown" rather than retained as a fourth group).

- **COVID disruption in 2019/20 and 2020/21**: the 2019/20 LCF fieldwork ended March 2020 and overlaps the first UK lockdown; 2020/21 was entirely affected. Inflation estimates for those two years reflect a distorted expenditure basket and should be interpreted with caution.

- **No equivalisation of expenditure shares**: shares are computed from raw household expenditure. Unlike income quintiles (which use equivalised income), the shares themselves are not adjusted for household size. This is standard practice (ONS HCI also uses raw expenditure shares).
