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
3. Shares   - compute COICOP division expenditure shares, Winsorise
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
      Winsorised and quality-checked, ready for linking to CPIH indices.

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

# ── File-path mapping ───────────────────────────────────────────────────────
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

# ── COICOP division-level total expenditure columns (weekly, GBP) ──────────
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

# COICOP class-level columns (ck*t) - more granular, for matching CPIH
# sub-indices.  These 24 columns are common across all 9 years.
CK_COLS = [
    "ck1211t", "ck1313t", "ck1314t", "ck1315t", "ck1316t", "ck1411t",
    "ck2111t", "ck3111t", "ck3112t", "ck4111t", "ck4112t", "ck5111t",
    "ck5113t", "ck5115t", "ck5116t", "ck5212t", "ck5213t", "ck5214t",
    "ck5215t", "ck5216t", "ck5221t", "ck5222t", "ck5223t", "ck5316t",
]

# ── Household demographic/classification variables (dvhh) ─────────────────

DVHH_SOURCE_COLS = [
    "case",         # household case ID
    "weighta",      # annual grossing weight

    # Household composition
    "a049",         # household size (total persons)
    "a062",         # family type (detailed: 1 man, 1 woman, couples+children etc.)
    "a121",         # tenure type 1 = { {1,2}, {3,4}, {5}, {6,7}, {8} } = {social rent, private rent, own outright, own mortgage, rent free}
    "a122",         # tenure type 2 (more detailed but I don't think we'll be using this extra detail)
    "gorx",         # Government Office Region (1-12)

    # Counts of individuals in each age band (in the household).
    # Age bands are {<2, 2-5, 5-18, 18-45, 45-60, 60-65, 65-70, 70+}.
    # Sum of the counts in each age band {a040, a041, a042, a043, a044, a045, a046, a047} should equal a049.
    "a040",         # number of children under 2
    "a041",         # number of children aged 2 to under 5
    "a042",         # number of children aged 5 to under 18
    "a043",         # number of adults aged 18 to under 45
    "a044",         # number of adults aged 45 to under 60
    "a045",         # number of adults aged 60 to under 65
    "a046",         # number of adults aged 65 to under 70
    "a047",         # number of adults aged 70 and over

    "a054",         # number of workers in household

    # HRP stands for household reference person (it's the person chosen to represent the household).
    # To me, designing the groups using the HRP rather than checking every individual in the household feels lazy but perhaps it works for some grouping?

    "sexhrp",       # sex of HRP (from dvhh directly)
    "a065p",        # (coded) age band of HRP
    "a071",         # NS-SEC of HRP (NS-SEC stands for National Statistics Socio-Economics Classification)

    # Income / housing
    "p389p",        # gross normal weekly household income
    "eqincdmp",     # equivalised income (modified OECD scale)
    "hh_total_coicop_expenditure",       # total COICOP expenditure
    "b010",         # rent (gross)
    "b020",         # mortgage payments
]

DVHH_READABLE_NAMES = {

    # Keys / weights
    "case": "household_id",
    "weighta": "household_weight",

    # Household composition
    "a049": "household_size",
    "a062": "family_type_code",
    "a121": "tenure_type1_code",
    "a122": "tenure_type2_code",
    "gorx": "region_code",

    # Age-banded counts
    "a040": "n_children_under_2",
    "a041": "n_children_2_to_4",
    "a042": "n_children_5_to_17",
    "a043": "n_adults_18_to_44",
    "a044": "n_adults_45_to_59",
    "a045": "n_adults_60_to_64",
    "a046": "n_adults_65_to_69",
    "a047": "n_adults_70_plus",
    "a054": "n_workers_in_household",

    # HRP (Household Reference Person)
    "sexhrp": "hrp_sex_code",
    "a065p": "hrp_age_band_code",
    "a071": "hrp_nssec_code",

    # Income / housing
    "p389p": "hh_income_gross_weekly",
    "eqincdmp": "hh_income_equivalised_oecd_mod",
    "hh_total_coicop_expenditure": "hh_total_coicop_expenditure",
    "b010": "rent_gross_weekly",
    "b020": "mortgage_payments_weekly",
}

# ── Person-level variables (dvper) ─────────────────────────────────────────

DVPER_SOURCE_COLS = [
    "case",         # Household ID (used to link individual's records to household records).
    "person",       # Person number within the household (e.g., person 1, person 2).
    "a005p",        # Person's age.
    "a006p",        # Person's sex.
    "a200",         # Number of people in this household (household size).

    # Work / labour-market status
    "a206",         # Main economic status {1 = self-employed, 2 = full-time employed, 3 = part-time employed, 4 = unemployed, 5 = government training, 6 = retired, 7 = retired/under pension age, 0 = not recorded}.
    "a015",         # Employment position/status (e.g. working, retired, full-time education, other).

    # Disability-related benefits (weekly GBP amounts = 0 usually means not received)
    # Basically, DLA was replaced by PIP from 2013 onwards (gradual transition through 2015-2023), so both neccessary because there's variation.
    "b403",         # Disability Living Allowance (DLA) - care component.
    "b405",         # Disability Living Allowance (DLA) - mobility component.
    "b421",         # Attendance Allowance.
    "b552",         # Personal Independence Payment (PIP) - daily living / care component.
    "b553",         # Personal Independence Payment (PIP) - mobility component.

    # Carer-related benefit
    "b343",         # Carer's Allowance (weekly amount).
]

DVPER_READABLE_NAMES = {
    # Keys
    "case": "household_id",
    "person": "person_id",
    "a200": "household_size_from_person_file",

    # Demographics
    "a005p": "person_age",
    "a006p": "person_sex_code",

    # Economic position
    "a206": "economic_position_ilo_code",
    "a015": "employment_position_code",

    # Disability-related benefits (weekly amounts)
    "b403": "dla_self_care_weekly",
    "b405": "dla_mobility_weekly",
    "b421": "attendance_allowance_weekly",
    "b552": "pip_daily_living_weekly",
    "b553": "pip_mobility_weekly",

    # Carer status proxy
    "b343": "carers_allowance_weekly",
}

# ── Region labels ──────────────────────────────────────────────────────────

REGION_LABELS = {
    1: "North East",
    2: "North West",
    3: "Yorkshire and the Humber",
    4: "East Midlands",
    5: "West Midlands",
    6: "Eastern",
    7: "London",
    8: "South East",
    9: "South West",
    10: "Wales",
    11: "Scotland",
    12: "Northern Ireland",
}

# When cell sizes are too small, we attempt grouping regions into broader categories.
REGION_BROAD = {
    1: "North",                             # North East
    2: "North",                             # North West
    3: "North",                             # Yorkshire and the Humber
    4: "Midlands",                          # East Midlands
    5: "Midlands",                          # West Midlands
    6: "South (excluding London)",          # Eastern
    7: "London",                            # London (kept separate - very economically distinct from the rest of the UK)
    8: "South (excluding London)",          # South East
    9: "South (excluding London)",          # South West
    10: "Wales",                            # Wales
    11: "Scotland",                         # Scotland
    12: "Northern Ireland",                 # Northern Ireland
}

# ── Winsorisation bounds ───────────────────────────────────────────────────

WINSOR_LOWER = 0.01   # 1st percentile
WINSOR_UPPER = 0.99   # 99th percentile

# ── Minimum cell size for reliable group estimates ─────────────────────────

MIN_CELL_SIZE = 100


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _safe_cols(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Return the subset of *wanted* columns that actually exist in *df*."""
    available = {c.lower() for c in df.columns}
    return [c for c in wanted if c in available]


def load_stata(path: pathlib.Path) -> pd.DataFrame:
    """Read Stata .dta and lowercase all column names."""
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_dvhh(year: int) -> pd.DataFrame:
    """Load LCF dvhh file for *year* and select target variables."""
    path = LCF_DVHH[year]
    df = load_stata(path)

    wanted = DVHH_SOURCE_COLS + COICOP_DIVISION_COLS + CK_COLS

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

    # --- Denominator consistency check ---
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


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Expenditure shares + Winsorisation
# ═══════════════════════════════════════════════════════════════════════════

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

def winsorise_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise expenditure shares at 1st and 99th percentiles by year."""
    res = df.copy()
    # These are the expenditure shares that we just computed with compute_expenditure_shares()
    share_cols = [c for c in res.columns if c.startswith("share_")]

    # Helper function only winsorises if enough values for the year to be confident about the extremities.
    def winsorise_helper(share_values):
        if len(share_values) < 20:
            return share_values
        lo = share_values.quantile(WINSOR_LOWER)
        hi = share_values.quantile(WINSOR_UPPER)
        return share_values.clip(lo, hi)

    # Apply winsorisation to each share column by year
    for col in share_cols:
        res[col] = res.groupby("year")[col].transform(winsorise_helper)

    return res
        

# CLASSIFY HOUSEHOLDS INTO DEMOGRAPHICS

def _aggregate_person_to_household(dvper: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate person-level data to household level for disability,
    carer status, and employment composition flags.

    Returns one row per (case, year) with:
    - hrp_age, hrp_sex       : HRP demographics (person == 1)
    - hrp_econ_pos            : HRP economic position (a206)
    - hrp_emp_position        : HRP employment position (a015)
    - is_disability           : any person receives DLA/PIP/AA
    - is_carer                : any person receives Carer's Allowance
    - n_working, n_retired,
      n_unemployed, n_adults_per : counts for employment composition
    """
    per = dvper.copy()

    # HRP demographics: HRP has person_id == 1
    hrp = per[per["person_id"] == 1][["household_id", "year"]].copy()

    # Adds HRP information.
    if "person_age" in per.columns:
        hrp = hrp.merge(per.loc[per["person_id"] == 1, ["household_id", "year", "person_age"]], on=["household_id", "year"], how="left").rename(columns={"person_age": "hrp_age"})

    if "person_sex_code" in per.columns:
        hrp = hrp.merge(per.loc[per["person_id"] == 1, ["household_id", "year", "person_sex_code"]], on=["household_id", "year"], how="left").rename(columns={"person_sex_code": "hrp_sex_code"})

    if "economic_position_ilo_code" in per.columns:
        hrp = hrp.merge(per.loc[per["person_id"] == 1, ["household_id", "year", "economic_position_ilo_code"]], on=["household_id", "year"], how="left").rename(columns={"economic_position_ilo_code": "hrp_economic_position_ilo_code"})

    if "employment_position_code" in per.columns:
        hrp = hrp.merge(per.loc[per["person_id"] == 1, ["household_id", "year", "employment_position_code"]], on=["household_id", "year"], how="left").rename(columns={"employment_position_code": "hrp_employment_position_code"})

    # Remove any duplicate columns from sequential merges
    hrp = hrp.loc[:, ~hrp.columns.duplicated()]

    # Captures DLA (self-care b403 + mobility b405), Attendance Allowance
    # (b421), and PIP (care b552 + mobility b553).

    # DISABILITY FLAG implemented - defined as any person in the household receiving a disability benefit.    
    # 3 notable systems of disability benefits:
    # DLA was the old system of Non-means tested disability benefits: Care component (b403) and Mobility component (b405).
    # PIP is the new system of means tested disability benefits: Daily Living component (b552) and Mobility component (b553).
    # Basically, DLA was gradually replaced by PIP over [2015,2023] so both needed.
    # AA is for people who have reached state pension age and have a severe disability that requires someone to care for them.

    disability_cols = _safe_cols(per, ["dla_self_care_weekly", "dla_mobility_weekly","attendance_allowance_weekly", "pip_daily_living_weekly", "pip_mobility_weekly",])

    if disability_cols:

        # Returns true if any person in the household receives any kind of disability benefit.
        per["_any_disability_benefit"] = (per[disability_cols].fillna(0).gt(0).any(axis=1))
        hh_disability = (
            per.groupby(["household_id", "year"])["_any_disability_benefit"]
            .any()
            .reset_index()
            .rename(columns={"_any_disability_benefit": "is_disability"})
        )
        hrp = hrp.merge(hh_disability, on=["household_id", "year"], how="left")
        hrp["is_disability"] = hrp["is_disability"].fillna(False)
    else:
        hrp["is_disability"] = False

    # Carer flag (any individual receives Carer's Allowance)
    if "carers_allowance_weekly" in per.columns:
        per["_is_carer"] = per["carers_allowance_weekly"].fillna(0) > 0
        hh_carer = (
            per.groupby(["household_id", "year"])["_is_carer"]
            .any()
            .reset_index()
            .rename(columns={"_is_carer": "is_carer"})
        )
        hrp = hrp.merge(hh_carer, on=["household_id", "year"], how="left")
        hrp["is_carer"] = hrp["is_carer"].fillna(False)
    else:
        hrp["is_carer"] = False

    # Employment flag implemented - defined as any individual being employed, retired, or unemployed.
    # ILO codes from database: 0=not recorded, 1=self-employed, 2=FT employee, 3=PT employee, 4=unemployed, 5=govt training, 6=retired, 7=retired/under pension age
    if "economic_position_ilo_code" in per.columns:

        # Decision: We decided to include government training as working.
        per["_is_working"] = per["economic_position_ilo_code"].isin([1, 2, 3, 5])
        per["_is_retired"] = per["economic_position_ilo_code"].isin([6, 7])
        per["_is_unemployed"] = per["economic_position_ilo_code"] == 4

        agg_emp = per.groupby(["household_id", "year"]).agg(
            n_working=("_is_working", "sum"),
            n_retired=("_is_retired", "sum"),
            n_unemployed=("_is_unemployed", "sum"),
        ).reset_index()

        hrp = hrp.merge(agg_emp, on=["household_id", "year"], how="left")
    else:
        hrp["n_working"] = np.nan
        hrp["n_retired"] = np.nan
        hrp["n_unemployed"] = np.nan

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
    Add comprehensive archetype classification variables to the LCF
    household data for differential inflation analysis.

    Household groups constructed:
    1. Tenure type          - social rent / private rent / own outright /
                              own with mortgage
    2. Pensioner status     - using multiple signals (age, retirement count,
                              economic position)
    3. Income quintile      - weighted, equivalised (modified OECD)
    4. Children             - has children / single parent / couple with
                              children / childless
    5. Disability           - any person receives DLA/PIP/AA
    6. Carer household      - any person receives Carer's Allowance
    7. Employment status    - all-working / has-unemployed / all-retired /
                              mixed
    8. Region               - 12 GORs + broad grouping
    9. HRP age band         - Under 30 / 30-49 / 50-64 / 65-74 / 75+
    10. COVID period        - FY 2019/20 and 2020/21
    """
    df = dvhh.copy()

    # ── Aggregate person-level data to household ──
    print("    Aggregating person-level data (disability, employment)...")
    hh_person = _aggregate_person_to_household(dvper)
    df = df.merge(hh_person, on=["household_id", "year"], how="left")

    # ═══════════════════════════════════════════════════════════════════
    # Group 1: Tenure type (from a121)
    # ═══════════════════════════════════════════════════════════════════
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
            8: "rent_free",
        }
        df["tenure_type"] = df["tenure_type1_code"].map(tenure_map).fillna("unknown")
    else:
        df["tenure_type"] = "unknown"

    # ═══════════════════════════════════════════════════════════════════
    # Group 2: Pensioner household (improved multi-signal definition)
    # ═══════════════════════════════════════════════════════════════════
    # A household is classified as pensioner if ANY of:
    # - HRP age >= 66 (state pension age)
    # - Any person aged 65+ in HH (a046 + a047 > 0)
    # - HRP economic position (a206) is retired (6 or 7)
    # - HRP employment position (a015) is retired (2)
    # This captures early retirees and aligns with the ONS HCI definition
    # of "retired households".
    cond_age = df["hrp_age"].fillna(0) >= 66 if "hrp_age" in df.columns \
        else pd.Series(False, index=df.index)
    # n_adults_65_to_69 = adults 65-70, n_adults_70_plus = adults 70+ (NOT "number retired")
    cond_over65 = (
        df[["n_adults_65_to_69", "n_adults_70_plus"]].fillna(0).sum(axis=1) > 0
    ) if "n_adults_65_to_69" in df.columns and "n_adults_70_plus" in df.columns \
        else pd.Series(False, index=df.index)
    cond_econ = df["hrp_econ_pos"].isin([6, 7]) if "hrp_econ_pos" in df.columns \
        else pd.Series(False, index=df.index)
    cond_emp = df["hrp_emp_position"] == 2 if "hrp_emp_position" in df.columns \
        else pd.Series(False, index=df.index)

    df["is_pensioner"] = cond_age | cond_over65 | cond_econ | cond_emp

    # ═══════════════════════════════════════════════════════════════════
    # Group 3: Income quintile (weighted, equivalised)
    # ═══════════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════════
    # Group 4: Children and household composition
    # ═══════════════════════════════════════════════════════════════════
    # LCF age-banded person counts:
    #   a040 = children under 2
    #   a041 = children aged 2 to under 5
    #   a042 = children aged 5 to under 18
    #   a043 = adults 18-45, a044 = adults 45-60, a045 = adults 60-65,
    #   a046 = adults 65-70, a047 = adults 70+
    # Total children = a040 + a041 + a042
    # Total adults   = a043 + a044 + a045 + a046 + a047
    child_cols = _safe_cols(df, ["n_children_under_2", "n_children_2_to_4", "n_children_5_to_17"])
    adult_cols = _safe_cols(df, ["n_adults_18_to_44", "n_adults_45_to_59", "n_adults_60_to_64", "n_adults_65_to_69", "n_adults_70_plus"])

    if child_cols:
        df["n_children"] = df[child_cols].fillna(0).sum(axis=1).astype(int)
        df["has_children"] = df["n_children"] > 0
    else:
        df["n_children"] = 0
        df["has_children"] = False

    if adult_cols:
        df["n_adults"] = df[adult_cols].fillna(0).sum(axis=1).astype(int)
    else:
        df["n_adults"] = np.nan

    # Single parent: 1 adult + at least 1 child
    df["is_single_parent"] = (df["n_adults"] == 1) & df["has_children"]

    # Household composition category
    conditions = [
        df["is_single_parent"],
        df["has_children"] & (df["n_adults"] >= 2),
        ~df["has_children"],
    ]
    choices = ["single_parent", "couple_with_children", "no_children"]
    df["hh_composition"] = np.select(conditions, choices, default="unknown")

    # ═══════════════════════════════════════════════════════════════════
    # Group 5: Disability (already merged from person aggregation)
    # ═══════════════════════════════════════════════════════════════════
    # is_disability is already in df from _aggregate_person_to_household
    # Ensure it's boolean
    df["is_disability"] = df["is_disability"].fillna(False).astype(bool)

    # ═══════════════════════════════════════════════════════════════════
    # Group 6: Carer household (already merged)
    # ═══════════════════════════════════════════════════════════════════
    df["is_carer"] = df["is_carer"].fillna(False).astype(bool)

    # ═══════════════════════════════════════════════════════════════════
    # Group 7: Employment composition
    # ═══════════════════════════════════════════════════════════════════
    # Based on aggregated person-level economic position (a206)
    if "n_working" in df.columns:
        n_work = df["n_working"].fillna(0)
        n_ret = df["n_retired"].fillna(0)
        n_unemp = df["n_unemployed"].fillna(0)
        n_ad = df["n_adults"].fillna(0) if "n_adults" in df.columns \
            else n_work + n_ret + n_unemp

        conditions_emp = [
            (n_work > 0) & (n_ret == 0) & (n_unemp == 0),  # all working
            n_unemp > 0,                                     # has unemployed
            (n_ret > 0) & (n_work == 0) & (n_unemp == 0),   # all retired
        ]
        choices_emp = ["all_working", "has_unemployed", "all_retired"]
        df["employment_status"] = np.select(
            conditions_emp, choices_emp, default="mixed"
        )
    else:
        df["employment_status"] = "unknown"

    # ═══════════════════════════════════════════════════════════════════
    # Group 8: Region
    # ═══════════════════════════════════════════════════════════════════
    if "region_code" in df.columns:
        df["region"] = df["region_code"].map(REGION_LABELS).fillna("unknown")
        df["region_broad"] = df["region_code"].map(REGION_BROAD).fillna("unknown")
    else:
        df["region"] = "unknown"
        df["region_broad"] = "unknown"

    # ═══════════════════════════════════════════════════════════════════
    # Group 9: HRP age band
    # ═══════════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════════
    # Group 10: Household size
    # ═══════════════════════════════════════════════════════════════════
    # a049 = household size (total persons); NOT a040 which is children < 2
    if "household_size" in df.columns:
        df["hh_size"] = df["household_size"]
    elif "n_adults" in df.columns and "n_children" in df.columns:
        df["hh_size"] = df["n_adults"] + df["n_children"]
    else:
        df["hh_size"] = np.nan

    # ═══════════════════════════════════════════════════════════════════
    # COVID period flag (expanded: both 2019/20 and 2020/21)
    # ═══════════════════════════════════════════════════════════════════
    df["is_covid_year"] = df["year"].isin([2019, 2020])

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Quality assurance
# ═══════════════════════════════════════════════════════════════════════════

def qa_cell_sizes(df: pd.DataFrame) -> None:
    """
    Check that each archetype group x year has at least MIN_CELL_SIZE
    unweighted observations.  Print warnings for small cells.
    """
    group_vars = [
        ("tenure_type", None),
        ("is_pensioner", None),
        ("income_quintile", None),
        ("has_children", None),
        ("is_single_parent", None),
        ("is_disability", None),
        ("is_carer", None),
        ("employment_status", None),
        ("region_broad", None),
        ("hrp_age_band", None),
        ("hh_composition", None),
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


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("LCF Data Wrangling Pipeline (v2 - full archetypes)")
    print("=" * 60)

    years = sorted(LCF_DVHH.keys())

    # ── 1. Extract dvhh data ────────────────────────────────────────────
    print("\n[1/6] Extracting household expenditure data (dvhh)...")
    dvhh_frames = []
    for yr in years:
        frame = extract_dvhh(yr)
        print(f"  {yr}/{yr+1-2000:02d}: {len(frame):,} households, "
              f"{frame.shape[1]} vars")
        dvhh_frames.append(frame)
    dvhh = pd.concat(dvhh_frames, ignore_index=True)
    print(f"  TOTAL: {len(dvhh):,} household-year observations")

    # ── 2. Extract dvper data ───────────────────────────────────────────
    print("\n[2/6] Extracting person-level data (dvper)...")
    dvper_frames = []
    for yr in years:
        frame = extract_dvper(yr)
        print(f"  {yr}/{yr+1-2000:02d}: {len(frame):,} persons")
        dvper_frames.append(frame)
    dvper = pd.concat(dvper_frames, ignore_index=True)
    print(f"  TOTAL: {len(dvper):,} person-year observations")

    # ── 3. Save interim files ───────────────────────────────────────────
    print("\n[3/6] Saving interim files...")
    dvhh.to_parquet(INTERIM / "lcf_household.parquet", index=False)
    dvper.to_parquet(INTERIM / "lcf_person.parquet", index=False)
    print(f"  Saved: {INTERIM / 'lcf_household.parquet'}")
    print(f"  Saved: {INTERIM / 'lcf_person.parquet'}")

    # ── 4. Clean expenditure data ───────────────────────────────────────
    print("\n[4/6] Cleaning expenditure data...")
    cleaned = clean_expenditure(dvhh)

    # ── 5. Compute shares, Winsorise, add archetypes ────────────────────
    print("\n[5/6] Computing expenditure shares & adding archetypes...")
    analysis = compute_expenditure_shares(cleaned)
    analysis = winsorise_shares(analysis)
    analysis = add_lcf_archetypes(analysis, dvper)

    # Drop internal working columns
    internal_cols = [c for c in analysis.columns if c.startswith("_")]
    analysis = analysis.drop(columns=internal_cols, errors="ignore")

    analysis.to_parquet(
        PROCESSED / "lcf_expenditure_shares.parquet", index=False
    )
    print(f"  Saved: {PROCESSED / 'lcf_expenditure_shares.parquet'}")

    # ── 6. Quality assurance ────────────────────────────────────────────
    print("\n[6/6] Quality assurance...")
    qa_share_diagnostics(analysis)
    qa_cell_sizes(analysis)

    # ── Summary ─────────────────────────────────────────────────────────
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

    # ── Archetype distributions ──
    print(f"\n  Tenure distribution:")
    if "tenure_type" in analysis.columns:
        for t, n in analysis["tenure_type"].value_counts().items():
            pct = 100 * n / len(analysis)
            print(f"    {t}: {n:,} ({pct:.1f}%)")

    print(f"\n  Pensioner households: "
          f"{analysis['is_pensioner'].sum():,} "
          f"({100*analysis['is_pensioner'].mean():.1f}%)")

    print(f"\n  Disability households: "
          f"{analysis['is_disability'].sum():,} "
          f"({100*analysis['is_disability'].mean():.1f}%)")

    print(f"\n  Carer households: "
          f"{analysis['is_carer'].sum():,} "
          f"({100*analysis['is_carer'].mean():.1f}%)")

    print(f"\n  Single-parent households: "
          f"{analysis['is_single_parent'].sum():,} "
          f"({100*analysis['is_single_parent'].mean():.1f}%)")

    print(f"\n  Households with children: "
          f"{analysis['has_children'].sum():,} "
          f"({100*analysis['has_children'].mean():.1f}%)")

    print(f"\n  Household composition:")
    if "hh_composition" in analysis.columns:
        for t, n in analysis["hh_composition"].value_counts().items():
            pct = 100 * n / len(analysis)
            print(f"    {t}: {n:,} ({pct:.1f}%)")

    print(f"\n  Employment status:")
    if "employment_status" in analysis.columns:
        for t, n in analysis["employment_status"].value_counts().items():
            pct = 100 * n / len(analysis)
            print(f"    {t}: {n:,} ({pct:.1f}%)")

    print(f"\n  Region (broad):")
    if "region_broad" in analysis.columns:
        for t, n in analysis["region_broad"].value_counts().items():
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