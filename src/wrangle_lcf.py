"""
wrangle_lcf.py
==============
Extract, clean, harmonise and merge LCF household-level expenditure and
demographic data across financial years 2015/16 - 2023/24.

The LCF provides COICOP-classified expenditure data used to construct
household-type-specific spending baskets.  These baskets are later combined
with CPIH sub-indices (from MM23) to compute differential inflation rates.

Pipeline steps
--------------
1. Extract  - load dvhh + dvper Stata files, select target variables
2. Clean    - handle negatives, zeros, validate denominators
3. Shares   - compute COICOP division expenditure shares, filter implausible records
4. Classify - build household archetype flags (disability, children,
              tenure, pensioner, income quintile, employment, region, age)
5. QA       - cell-size checks, share-sum validation, temporal diagnostics

Outputs
-------
- data/interim/lcf_household.parquet   - household-level panel with
      demographics + COICOP division totals (p601t-p612t) + class-level (ck*)
- data/interim/lcf_person.parquet      - person-level panel (demographics +
      disability benefits + economic position)
- data/processed/lcf_expenditure_shares.parquet - household-level expenditure
      shares by COICOP division, with archetype classification variables,
      domain-filtered and quality-checked, ready for linking to CPIH indices.

Raw data in data/raw/LCF/ is never modified.
"""

# Library imports
import pathlib
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LCF"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# File-path mapping
# dvhh = derived household-level file (expenditure + demographics)
# dvper = derived person-level file (individual demographics)
# Use the main annual files, NOT quarterly or urbanrural variants.

# Information about the LCF data files
# DVHH contains one row per household {expendicture, per household demographics}.
# DVPER contains one row per person in the household {per individualdemographics}.

LCF_DVHH = {
    2015: RAW / "LCF_2015" / "stata" / "stata11_se" / "2015-16_dvhh_ukanon.dta",
    2016: RAW / "LCF_2016" / "stata" / "stata13_se" / "2016_17_dvhh_ukanon.dta",
    2017: RAW / "LCF_2017" / "stata" / "stata11_se" / "dvhh_ukanon_2017-18.dta",
    2018: RAW / "LCF_2018" / "stata" / "stata13" / "2018_dvhh_ukanon.dta",
    2019: RAW / "LCF_2019" / "stata" / "stata13" / "lcfs_2019_dvhh_ukanon.dta",
    2020: RAW / "LCF_2020" / "stata" / "stata13" / "lcfs_2020_dvhh_ukanon.dta",
    2021: RAW / "LCF_2021" / "stata" / "stata13_se" / "lcfs_2021_dvhh_ukanon.dta",
    2022: RAW / "LCF_2022" / "stata" / "stata13_se" / "dvhh_ukanon_2022.dta",
    2023: RAW / "LCF_2023" / "stata" / "stata13_se" / "dvhh_ukanon_v2_2023.dta",
}

LCF_DVPER = {
    2015: RAW / "LCF_2015" / "stata" / "stata11_se" / "2015-16_dvper_ukanon.dta",
    2016: RAW / "LCF_2016" / "stata" / "stata13_se" / "2016_17_dvper_ukanon.dta",
    2017: RAW / "LCF_2017" / "stata" / "stata11_se" / "dvper_ukanon_2017-18.dta",
    2018: RAW / "LCF_2018" / "stata" / "stata13" / "2018_dvper_ukanon201819.dta",
    2019: RAW / "LCF_2019" / "stata" / "stata13" / "lcfs_2019_dvper_ukanon201920.dta",
    2020: RAW / "LCF_2020" / "stata" / "stata13" / "lcfs_2020_dvper_ukanon202021.dta",
    2021: RAW / "LCF_2021" / "stata" / "stata13_se" / "lcfs_2021_dvper_ukanon202122.dta",
    2022: RAW / "LCF_2022" / "stata" / "stata13_se" / "dvper_ukanon_2022-23.dta",
    2023: RAW / "LCF_2023" / "stata" / "stata13_se" / "dvper_ukanon_202324_2023.dta",
}

# COICOP division-level total expenditure columns (weekly, GBP)
# p601t = COICOP 01 (Food & non-alcoholic beverages)
# p602t = COICOP 02 (Alcoholic beverages & tobacco)
# ...
# p612t = COICOP 12 (Miscellaneous goods & services)
# p600t = COICOP 00 (total, used as denominator cross-check)

# NOTE: cannot use f-string with range() because p610t != p6010t.
COICOP_DIVISION_COLS = [
    "p600t",   # COICOP 00 (all items total)
    "p601t",   # COICOP 01
    "p602t",   # COICOP 02
    "p603t",   # COICOP 03
    "p604t",   # COICOP 04
    "p605t",   # COICOP 05
    "p606t",   # COICOP 06
    "p607t",   # COICOP 07
    "p608t",   # COICOP 08
    "p609t",   # COICOP 09
    "p610t",   # COICOP 10 (Education)
    "p611t",   # COICOP 11 (Restaurants & hotels)
    "p612t",   # COICOP 12 (Miscellaneous goods & services)
]
COICOP_DIVISION_LABELS = {
    "p601t": "01_food_non_alcoholic",
    "p602t": "02_alcohol_tobacco",
    "p603t": "03_clothing_footwear",
    "p604t": "04_housing_fuel_power",
    "p605t": "05_furnishings",
    "p606t": "06_health",
    "p607t": "07_transport",
    "p608t": "08_communication",
    "p609t": "09_recreation_culture",
    "p610t": "10_education",
    "p611t": "11_restaurants_hotels",
    "p612t": "12_misc_goods_services",
}

# Household demographic/classification variables (dvhh)

DVHH_SOURCE_COLS = [
    "case",         # household case ID
    "weighta",      # annual grossing weight

    "a049",         # household size (total persons)
    "a121",         # tenure type 1 (social/private rent, own outright/mortgage)

    # Income / housing
    "p389p",        # gross normal weekly household income (fallback for quintiles)
    "eqincdmp",     # equivalised income (modified OECD scale, primary for quintiles)
    "hh_total_coicop_expenditure",  # total COICOP expenditure
    "b010",         # rent (gross) — needed for COICOP 04 rent/energy decomposition
]

DVHH_READABLE_NAMES = {
    "case": "household_id",
    "weighta": "household_weight",
    "a049": "household_size",
    "a121": "tenure_type1_code",
    "p389p": "hh_income_gross_weekly",
    "eqincdmp": "hh_income_equivalised_oecd_mod",
    "hh_total_coicop_expenditure": "hh_total_coicop_expenditure",
    "b010": "rent_gross_weekly",
}

# Person-level variables (dvper)

DVPER_SOURCE_COLS = [
    "case",         # Household ID (used to link individual's records to household records).
    "person",       # Person number within the household (e.g., person 1, person 2).
    "a005p",        # Person's age (needed for HRP age band).
    "a006p",        # Person's sex.
]

DVPER_READABLE_NAMES = {
    "case": "household_id",
    "person": "person_id",
    "a005p": "person_age",
    "a006p": "person_sex_code",
}

# Minimum cell size for reliable group estimates

MIN_CELL_SIZE = 100


# Helper functions

def _safe_cols(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Return the subset of *wanted* columns that actually exist in *df*."""
    available = {c.lower() for c in df.columns}
    return [c for c in wanted if c in available]


def load_stata(path: pathlib.Path) -> pd.DataFrame:
    """Read Stata .dta and lowercase all column names."""
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    return df


# Step 1: Extraction

def extract_dvhh(year: int) -> pd.DataFrame:
    """Load LCF dvhh file for *year* and select target variables."""
    path = LCF_DVHH[year]
    df = load_stata(path)

    wanted = DVHH_SOURCE_COLS + COICOP_DIVISION_COLS

    # Keeps only the wanted columns that are in that year's file. 
    keep = _safe_cols(df, wanted)

    df = df[keep].copy()

    # Rename demographic columns to human-readable names.
    # COICOP division cols (p601t-p612t) and CK cols are kept as-is
    # since they are referenced by name in downstream share calculations.
    df = df.rename(columns=DVHH_READABLE_NAMES, errors="ignore")

    # Formats the financial year label (e.g. "2015/16")
    df["year"] = year
    df["fy"] = f"{year}/{str(year + 1)[-2:]}"
    return df


def extract_dvper(year: int) -> pd.DataFrame:
    """Load LCF dvper file for *year* and select target variables."""
    path = LCF_DVPER[year]
    df = load_stata(path)

    # Keeps only the wanted columns that are in that year's file.
    keep = _safe_cols(df, DVPER_SOURCE_COLS)

    df = df[keep].copy()

    # Rename person-level columns to human-readable names.
    df = df.rename(columns=DVPER_READABLE_NAMES, errors="ignore")

    df["year"] = year
    return df

# DATA CLEANING

def clean_expenditure(dvhh: pd.DataFrame) -> pd.DataFrame:
    """
    Clean expenditure data before computing shares.

    Steps:
    - Set negative total expenditure (p630tp) to NaN (implausible for spending basket construction).
    - Set negative division totals (p601t-p612t) to NaN.
    - Flag households with zero total expenditure (no diary information) as excluded from share calculations.
    - Validate denominator: if p630tp deviates from sum(p601t:p612t) by more than 1%, use the division sum instead for consistency.
    """
    df = dvhh.copy()

    # Data cleaning step: sets negative total expenditure to NaN.
    if "hh_total_coicop_expenditure" in df.columns:

        # Counts the number of households with negative total expenditure.
        n_neg_total = (df["hh_total_coicop_expenditure"] < 0).sum()

        if n_neg_total > 0:
            print(f"    Cleaning: {n_neg_total} households with negative total expenditure (negative total expenditure-> NaN)")
            
            # Sets the total expenditure to NaN for households with negative total expenditure.
            df.loc[df["hh_total_coicop_expenditure"] < 0, "hh_total_coicop_expenditure"] = np.nan

    # Data cleaning: cleans negative values in each COICOP division to NaN.

    # Gets the columns for each COICOP division.
    coicop_cols = [c for c in COICOP_DIVISION_LABELS.keys() if c in df.columns]

    # Cycles through each COICOP column, sets all households with negative values for that COICOP column to NaN.
    for col in coicop_cols:

        # Counts the number of households with negative values in the COICOP division.
        n_neg = (df[col] < 0).sum()

        # If there are any negative values in the COICOP division, sets them to NaN.
        if n_neg > 0:
            print(f"Cleaning: {n_neg} negative values in {col} (COICOP column) -> NaN")
            df.loc[df[col] < 0, col] = np.nan

    # Plausibility filter on total expenditure
    # LCF records weekly expenditure. Values outside plausible bounds are
    # almost certainly data errors (incomplete diaries, annual figures entered
    # as weekly, transcription errors) rather than genuine extreme spenders.
    # We remove these BEFORE computing shares so the shares themselves are
    # uncontaminated. Implausible households (zero food, zero housing,
    # negative total expenditure) are removed later by
    # filter_implausible_households(). Genuine extreme spenders
    # (e.g. fuel-poor pensioners at 45% energy share) are real households
    # and suppressing them would directly hide the inequality we are measuring.
    PLAUSIBILITY_LOWER = 30.0    # below £30/week is almost certainly an incomplete diary
    PLAUSIBILITY_UPPER = 3000.0  # above £3,000/week is almost certainly a data entry error

    if "hh_total_coicop_expenditure" in df.columns:
        valid_mask = df["hh_total_coicop_expenditure"].notna()
        n_too_low = (df.loc[valid_mask, "hh_total_coicop_expenditure"] < PLAUSIBILITY_LOWER).sum()
        n_too_high = (df.loc[valid_mask, "hh_total_coicop_expenditure"] > PLAUSIBILITY_UPPER).sum()
        if n_too_low > 0:
            print(f"    Plausibility filter: {n_too_low} households with total expenditure "
                  f"< £{PLAUSIBILITY_LOWER:.0f}/week removed (likely incomplete diary)")
            df.loc[df["hh_total_coicop_expenditure"] < PLAUSIBILITY_LOWER,
                   "hh_total_coicop_expenditure"] = np.nan
        if n_too_high > 0:
            print(f"    Plausibility filter: {n_too_high} households with total expenditure "
                  f"> £{PLAUSIBILITY_UPPER:.0f}/week removed (likely data entry error)")
            df.loc[df["hh_total_coicop_expenditure"] > PLAUSIBILITY_UPPER,
                   "hh_total_coicop_expenditure"] = np.nan

    # Denominator consistency check
    # Compare p630tp to the sum of divisions. If they diverge by > 1%
    # it means p630tp includes/excludes items the divisions don't.
    # Remember p630tp is the total weekly household expenditure.

    if "hh_total_coicop_expenditure" in df.columns and len(coicop_cols) > 0:

        # coicop_cols_sum is the sum of the COICOP divisions.
        coicop_cols_sum = df[coicop_cols].sum(axis=1)

        # Only check the households where both the official total expenditure and the COICOP columns sum are positive.
        positive_totals = (df["hh_total_coicop_expenditure"] > 0) & (coicop_cols_sum > 0)

        # Calculate official total expenditure to COICOP columns sum ratio.
        expenditure_ratio = df.loc[positive_totals, "hh_total_coicop_expenditure"] / coicop_cols_sum[positive_totals]

        # Count the number of households where the official total expenditure to COICOP columns sum ratio is less than 0.99 or greater than 1.01.
        n_inconsistent = ((expenditure_ratio < 0.99) | (expenditure_ratio > 1.01)).sum()

        # Basically, check for consistency between the official total expenditure and the COICOP columns sum.
        # If consistent, use the official total expenditure as the denominator.
        # Otherwise, use the COICOP columns sum as the denominator.
        if n_inconsistent > 0:
            pct_inconsistent = 100 * n_inconsistent / positive_totals.sum()
            print(f"Denominator check: {n_inconsistent} households ({pct_inconsistent:.1f}%) where COICOP columns sum differs from official total expenditure by >1%.")

            # Use coicop_cols_sum as the denominator for all rows to ensure consistency; store the chosen denominator explicitly.
            df["_total_expenditure"] = coicop_cols_sum.replace(0, np.nan)

        else:
            # If all households are consistent, just use the official total expenditure (as the denominator).
            df["_total_expenditure"] = df["hh_total_coicop_expenditure"].replace(0, np.nan)

    # No COICOP columns, so just use the official total expenditure as the denominator.
    elif "hh_total_coicop_expenditure" in df.columns:
        df["_total_expenditure"] = df["hh_total_coicop_expenditure"].replace(0, np.nan)

    else:
        # if there are no COICOP columns, use the official total expenditure as the denominator.
        coicop_cols_sum = df[coicop_cols].sum(axis=1) if coicop_cols else pd.Series(np.nan, index=df.index)
        df["_total_expenditure"] = coicop_cols_sum.replace(0, np.nan)

    # Flag the households which have a total expenditure of 0 (NaN)
    df["_zero_expenditure"] = df["_total_expenditure"].isna()
    n_zero = df["_zero_expenditure"].sum()
    if n_zero > 0:
        print(f"    Flagged: {n_zero} zero/missing total expenditure households: shares will be NaN.")

    return df


# Step 3: Expenditure shares + Domain-based household filtering

def compute_expenditure_shares(dvhh: pd.DataFrame) -> pd.DataFrame:
    """
    Compute COICOP division-level expenditure shares for each household.

    share_d = expenditure_on_division_d / total_COICOP_expenditure

    Uses the cleaned _total_expenditure denominator (from clean_expenditure).
    Returns the original DataFrame with added share_* columns.
    """
    df = dvhh.copy()

    total = df["_total_expenditure"]

    # Calculate the share of each COICOP division for each household.
    for col, label in COICOP_DIVISION_LABELS.items():
        if col in df.columns:
            df[f"share_{label}"] = df[col] / df["_total_expenditure"]

    # Split COICOP 04 into rent vs energy+other
    # b010 (rent_gross_weekly) is actual rent paid by tenants.
    # For owners this is zero, so their share_04_actual_rent = 0 correctly.
    # The remainder (p604t - b010) captures energy, maintenance, water.
    if "rent_gross_weekly" in df.columns and "p604t" in df.columns:
        rent_col = df["rent_gross_weekly"].fillna(0).clip(lower=0)
        p604 = df["p604t"].fillna(0).clip(lower=0)
        # Ensure rent component cannot exceed total COICOP 04
        rent_capped = rent_col.clip(upper=p604)
        energy_other = (p604 - rent_capped).clip(lower=0)
        total = df["_total_expenditure"]
        df["share_04_actual_rent"]     = rent_capped   / total
        df["share_04_energy_other"]    = energy_other  / total

    # Ensure the shares sum to (very near) 1 for each household.
    share_cols = [c for c in df.columns if c.startswith("share_")]

    if share_cols:
        share_sum = df[share_cols].sum(axis=1)

        # Warn when the shares deviate from 1.0 by more than 0.01.
        valid = share_sum.notna()
        if valid.any():
            deviation = (share_sum[valid] - 1.0).abs()
            n_bad = (deviation > 0.01).sum()
            if n_bad > 0:
                print(f"    Share validation: {n_bad} households where shares deviate from 1.0 by >0.01")

    return df

def filter_implausible_households(df: pd.DataFrame) -> pd.DataFrame:
    """Remove households with demonstrably incomplete or erroneous diary records.

    Expenditure shares are compositional (summing to 1.0 per household),
    which precludes standard univariate cleaning methods like winsorisation
    or per-column IQR — both break the budget constraint.  Instead we apply
    household-level exclusion criteria grounded in domain knowledge:

    1. Negative total expenditure  — arithmetically impossible.
    2. Zero food expenditure       — no household spends £0 on food over a
       2-week LCF diary period; indicates an incomplete diary record.
    3. Zero housing & utilities    — every UK household pays energy costs
       at minimum (gas, electricity, water), even outright owners.

    Returns the filtered DataFrame (removed rows are printed for transparency).
    """
    n_before = len(df)

    # Build exclusion mask
    neg_exp = df["p600t"] <= 0 if "p600t" in df.columns else pd.Series(False, index=df.index)
    zero_food = df["share_01_food_non_alcoholic"] == 0
    zero_housing = df["share_04_housing_fuel_power"] == 0

    exclude = neg_exp | zero_food | zero_housing

    n_neg   = neg_exp.sum()
    n_food  = zero_food.sum()
    n_hous  = zero_housing.sum()
    n_total = exclude.sum()

    print(f"    Domain-based household filter:")
    print(f"      Negative total expenditure:   {n_neg:>5}")
    print(f"      Zero food expenditure:        {n_food:>5}")
    print(f"      Zero housing & utilities:     {n_hous:>5}")
    print(f"      Combined (with overlap):      {n_total:>5} of {n_before:,} "
          f"({100 * n_total / n_before:.2f}%)")

    result = df[~exclude].copy()
    print(f"      Remaining households:         {len(result):>5}")
    return result
        

# CLASSIFY HOUSEHOLDS INTO DEMOGRAPHICS

def _aggregate_person_to_household(dvper: pd.DataFrame) -> pd.DataFrame:
    """
    Extract HRP age from person-level data (person_id == 1).

    Returns one row per (case, year) with hrp_age.
    """
    per = dvper.copy()
    hrp = per[per["person_id"] == 1][["household_id", "year"]].copy()

    if "person_age" in per.columns:
        hrp = hrp.merge(
            per.loc[per["person_id"] == 1, ["household_id", "year", "person_age"]],
            on=["household_id", "year"], how="left",
        ).rename(columns={"person_age": "hrp_age"})

    hrp = hrp.loc[:, ~hrp.columns.duplicated()]
    return hrp


def _weighted_quintiles(
    df: pd.DataFrame,
    value_col: str,
    weight_col: str,
    year_col: str = "year",
    n_quantiles: int = 5,
) -> pd.Series:
    """
    Compute weighted quantile groups within each year.

    Uses cumulative weight approach: sorts by value within each year,
    computes cumulative weight share, and assigns quintile based on
    which weight threshold the household falls into (20%, 40%, ...).

    Returns a Series of quintile labels (1-5), with NaN for rows where
    the value is missing/non-positive.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)

    for yr in df[year_col].unique():
        yr_mask = (
            (df[year_col] == yr)
            & df[value_col].notna()
            & (df[value_col] > 0)
        )
        if yr_mask.sum() < n_quantiles:
            continue

        subset = df.loc[yr_mask, [value_col, weight_col]].copy()
        subset[weight_col] = subset[weight_col].fillna(1)  # fallback

        # Sort by income value
        subset = subset.sort_values(value_col)

        # Cumulative weight share
        cum_weight = subset[weight_col].cumsum()
        total_weight = cum_weight.iloc[-1]
        cum_share = cum_weight / total_weight

        # Assign quintile: 1 for [0, 0.2], 2 for (0.2, 0.4], etc.
        thresholds = np.arange(1, n_quantiles + 1) / n_quantiles
        quintile = np.searchsorted(thresholds, cum_share.values, side="left") + 1
        quintile = np.clip(quintile, 1, n_quantiles)

        result.loc[subset.index] = quintile.astype(float)

    return result


def add_lcf_archetypes(
    dvhh: pd.DataFrame, dvper: pd.DataFrame
) -> pd.DataFrame:
    """
    Add archetype classification variables to the LCF household data
    for differential inflation analysis.

    Archetype dimensions:
    1. Tenure type      - social rent / private rent / own outright /
                          own with mortgage
    2. Income quintile  - weighted, equivalised (modified OECD)
    3. HRP age band     - Under 30 / 30-49 / 50-64 / 65-74 / 75+

    """
    df = dvhh.copy()

    # Aggregate person-level data to household
    print("    Aggregating person-level data (HRP demographics)...")
    hh_person = _aggregate_person_to_household(dvper)
    df = df.merge(hh_person, on=["household_id", "year"], how="left")

    # Group 1: Tenure type (from a121)
    # a121 codes: 1=council rent, 2=HA rent, 3=private rent unfurnished,
    # 4=private rent furnished, 5=owned outright, 6=buying with loan help,
    # 7=owned with mortgage, 8=rent free
    if "tenure_type1_code" in df.columns:
        tenure_map = {
            1: "social_rent",
            2: "social_rent",
            3: "private_rent",
            4: "private_rent",
            5: "own_outright",
            6: "own_mortgage",
            7: "own_mortgage",
        }
        df["tenure_type"] = df["tenure_type1_code"].map(tenure_map).fillna("unknown")
    else:
        df["tenure_type"] = "unknown"

    # Group 2: Income quintile (weighted, equivalised)
    # Uses equivalised income (modified OECD scale) from eqincdmp, which
    # adjusts for household size and composition.  Falls back to gross
    # income (p389p) if equivalised is unavailable.
    # Quintiles are computed using survey weights (weighta) so they
    # represent population quintiles, not sample quintiles.
    income_col = "hh_income_equivalised_oecd_mod" if "hh_income_equivalised_oecd_mod" in df.columns else "hh_income_gross_weekly"
    weight_col = "household_weight" if "household_weight" in df.columns else None

    if income_col in df.columns and weight_col in df.columns:
        print(f"    Computing weighted income quintiles using "
              f"{income_col} + {weight_col}...")
        # Exclude households with zero/negative income so they don't pollute Q1
        df.loc[df[income_col] <= 0, income_col] = np.nan
        df["income_quintile"] = _weighted_quintiles(
            df, value_col=income_col, weight_col=weight_col
        )
    elif income_col in df.columns:
        # Fallback: unweighted quintiles
        print(f"    Computing unweighted income quintiles using "
              f"{income_col} (no weights available)...")
        quintiles = pd.Series(np.nan, index=df.index, dtype=float)
        for yr in df["year"].unique():
            mask = (
                (df["year"] == yr)
                & df[income_col].notna()
                & (df[income_col] > 0)
            )
            if mask.sum() > 0:
                q = pd.qcut(
                    df.loc[mask, income_col],
                    q=5, labels=False, duplicates="drop",
                ) + 1
                quintiles.loc[mask] = q.astype(float)
        df["income_quintile"] = quintiles
    else:
        df["income_quintile"] = np.nan

    # Group 3: HRP age band
    if "hrp_age" in df.columns:
        bins = [0, 30, 50, 65, 75, 200]
        labels = ["under_30", "30_to_49", "50_to_64", "65_to_74", "75_plus"]
        df["hrp_age_band"] = pd.cut(
            df["hrp_age"], bins=bins, labels=labels, right=False
        )
        # Convert to string for consistent handling; NaN stays as NaN
        df["hrp_age_band"] = df["hrp_age_band"].astype(
            pd.CategoricalDtype(categories=labels, ordered=True)
        )
    else:
        df["hrp_age_band"] = np.nan

    return df


# Step 5: Quality assurance

def qa_cell_sizes(df: pd.DataFrame) -> None:
    """
    Check that each archetype group x year has at least MIN_CELL_SIZE
    unweighted observations.  Print warnings for small cells.
    """
    group_vars = [
        ("tenure_type", None),
        ("income_quintile", None),
        ("hrp_age_band", None),
    ]

    warnings_found = []
    for col, _ in group_vars:
        if col not in df.columns:
            continue
        counts = df.groupby(["year", col], observed=True).size().reset_index(name="n")
        small = counts[counts["n"] < MIN_CELL_SIZE]
        if len(small) > 0:
            for _, row in small.iterrows():
                warnings_found.append(
                    f"    WARNING: {col}={row[col]}, year={int(row['year'])} "
                    f"-> n={row['n']} (below {MIN_CELL_SIZE})"
                )

    if warnings_found:
        print(f"\n  Cell-size warnings ({len(warnings_found)} small cells):")
        # Show at most 15 warnings to avoid flood
        for w in warnings_found[:15]:
            print(w)
        if len(warnings_found) > 15:
            print(f"    ... and {len(warnings_found) - 15} more")
    else:
        print("\n  Cell-size check: all group x year cells >= "
              f"{MIN_CELL_SIZE} observations")


def qa_share_diagnostics(df: pd.DataFrame) -> None:
    """Print summary diagnostics for expenditure shares."""
    share_cols = [c for c in df.columns if c.startswith("share_")]
    if not share_cols:
        return

    print("\n  Share diagnostics:")
    # How many non-NaN share rows
    n_valid = df[share_cols[0]].notna().sum()
    n_total = len(df)
    print(f"    Valid share observations: {n_valid:,} / {n_total:,} "
          f"({100*n_valid/n_total:.1f}%)")

    # Share sum check
    share_sum = df[share_cols].sum(axis=1)
    valid_sums = share_sum.dropna()
    if len(valid_sums) > 0:
        print(f"    Share sum: mean={valid_sums.mean():.4f}, "
              f"min={valid_sums.min():.4f}, max={valid_sums.max():.4f}")


# Main pipeline

def main() -> None:
    print("=" * 60)
    print("LCF Data Wrangling Pipeline (v2 - full archetypes)")
    print("=" * 60)

    years = sorted(LCF_DVHH.keys())

    # 1. Extract dvhh data
    print("\n[1/6] Extracting household expenditure data (dvhh)...")
    dvhh_frames = []
    for yr in years:
        frame = extract_dvhh(yr)
        print(f"  {yr}/{yr+1-2000:02d}: {len(frame):,} households, "
              f"{frame.shape[1]} vars")
        dvhh_frames.append(frame)
    dvhh = pd.concat(dvhh_frames, ignore_index=True)
    print(f"  TOTAL: {len(dvhh):,} household-year observations")

    # 2. Extract dvper data
    print("\n[2/6] Extracting person-level data (dvper)...")
    dvper_frames = []
    for yr in years:
        frame = extract_dvper(yr)
        print(f"  {yr}/{yr+1-2000:02d}: {len(frame):,} persons")
        dvper_frames.append(frame)
    dvper = pd.concat(dvper_frames, ignore_index=True)
    print(f"  TOTAL: {len(dvper):,} person-year observations")

    # 3. Save interim files
    print("\n[3/6] Saving interim files...")
    dvhh.to_parquet(INTERIM / "lcf_household.parquet", index=False)
    dvper.to_parquet(INTERIM / "lcf_person.parquet", index=False)
    print(f"  Saved: {INTERIM / 'lcf_household.parquet'}")
    print(f"  Saved: {INTERIM / 'lcf_person.parquet'}")

    # 4. Clean expenditure data
    print("\n[4/6] Cleaning expenditure data...")
    cleaned = clean_expenditure(dvhh)

    # 5. Compute shares, filter implausible households, add archetypes
    print("\n[5/6] Computing expenditure shares & filtering...")
    analysis = compute_expenditure_shares(cleaned)
    analysis = filter_implausible_households(analysis)
    analysis = add_lcf_archetypes(analysis, dvper)

    # Drop internal working columns
    internal_cols = [c for c in analysis.columns if c.startswith("_")]
    analysis = analysis.drop(columns=internal_cols, errors="ignore")

    analysis.to_parquet(
        PROCESSED / "lcf_expenditure_shares.parquet", index=False
    )
    print(f"  Saved: {PROCESSED / 'lcf_expenditure_shares.parquet'}")

    # 6. Quality assurance
    print("\n[6/6] Quality assurance...")
    qa_share_diagnostics(analysis)
    qa_cell_sizes(analysis)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Years covered: {min(years)}/{min(years)+1-2000:02d} to "
          f"{max(years)}/{max(years)+1-2000:02d}")
    print(f"  Total households: {len(analysis):,}")
    print(f"  Columns in analysis file: {analysis.shape[1]}")

    share_cols = [c for c in analysis.columns if c.startswith("share_")]
    print(f"\n  COICOP expenditure shares (weighted mean across all years):")
    for col in sorted(share_cols):
        if "household_weight" in analysis.columns:
            valid = analysis[col].notna() & analysis["household_weight"].notna()
            wmean = (
                (analysis.loc[valid, col] * analysis.loc[valid, "household_weight"]).sum()
                / analysis.loc[valid, "household_weight"].sum()
            )
            print(f"    {col.replace('share_', '')}: {wmean:.4f}")
        else:
            print(f"    {col.replace('share_', '')}: "
                  f"{analysis[col].mean():.4f}")

    # Archetype distributions
    print(f"\n  Tenure distribution:")
    if "tenure_type" in analysis.columns:
        for t, n in analysis["tenure_type"].value_counts().items():
            pct = 100 * n / len(analysis)
            print(f"    {t}: {n:,} ({pct:.1f}%)")

    print(f"\n  HRP age band:")
    if "hrp_age_band" in analysis.columns:
        for t, n in analysis["hrp_age_band"].value_counts().items():
            pct = 100 * n / len(analysis)
            print(f"    {t}: {n:,} ({pct:.1f}%)")

    print(f"\n  Households per year:")
    for yr in years:
        n = (analysis["year"] == yr).sum()
        print(f"    {yr}/{yr+1-2000:02d}: {n:,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
