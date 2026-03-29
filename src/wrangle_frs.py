"""
wrangle_frs.py
==============
Extract, harmonise and merge FRS household, adult, and benefit-unit data
across financial years 2015/16 – 2023/24.

Outputs
-------
- data/interim/frs_househol.parquet   – household-level panel (all 9 years)
- data/interim/frs_adult.parquet      – adult-level panel (all 9 years)
- data/interim/frs_benefits.parquet   – benefits-level panel (all 9 years)
- data/processed/frs_household_analysis.parquet – merged household file with
      adult-level aggregates (HRP demographics, disability flags) joined back,
      ready for archetype classification.

Raw data in data/raw/FRS/ is never modified.
"""

import pathlib
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FRS"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── File-path mapping (verified against actual directory structure) ──────────
FRS_PATHS = {
    2015: RAW / "FRS_2015" / "stata" / "stata11_se",
    2016: RAW / "FRS_2016" / "stata" / "stata13_se",
    2017: RAW / "FRS_2017" / "stata" / "stata13_se",
    2018: RAW / "FRS_2018" / "stata" / "stata13_se",
    2019: RAW / "FRS_2019" / "stata" / "stata13_se",
    2020: RAW / "FRS_2020" / "stata" / "stata13_se",
    2021: RAW / "FRS_2021" / "stata" / "stata13_se",
    2022: RAW / "FRS_2022" / "stata" / "stata13_se",
    2023: RAW / "FRS_2023" / "stata" / "stata13",
}

# ── Variable lists (all lowercase; verified present in every year) ───────────

HOUSEHOL_VARS = [
    "sernum",       # household ID
    "gross4",       # household grossing weight
    "hhinc",        # total household income (net, weekly)
    "hdhhinc",      # derived total household income
    "tenure",       # tenure type (broad)
    "ptentyp2",     # tenure type (detailed — see encoding note below)
    "hhcomps",      # household composition code
    "adulth",       # number of adults in household
    "depchldh",     # number of dependent children in household
    "gvtregn",      # Government Office Region
    "hhrent",       # weekly rent
    "mortint",      # weekly mortgage interest payments
    "ctband",       # council tax band
    "hhdisben",     # household total disability benefit amount (£/wk)
    "penage",       # HRP at state pension age (0/1)
    "hhagegr3",     # household age group (8 bands)
]

# NOTE on EUL data: exact AGE is suppressed (-1) in End User Licence files.
# Use AGE80 (top-coded at 80) for individual age, PENAGE for pension-age flag.

ADULT_VARS = [
    "sernum",       # household ID (link key)
    "benunit",      # benefit unit number within household
    "person",       # person number within benefit unit
    "age80",        # age, top-coded at 80 (EUL version)
    "iagegr3",      # age group (8 bands, available in EUL)
    "empstati",     # employment status: 1=FT emp, 6=retired, etc.
    "retire",       # retirement flag: 1=retired, 2=not retired, -1=N/A
    "inearns",      # gross individual earnings (£/wk)
    "seincam2",     # self-employment income (£/wk)
    "nindinc",      # net individual income (£/wk)
    "indisben",     # individual disability benefit amount (£/wk)
    "discora1",     # DDA core-activity disability
    "health1",      # long-standing illness/disability
]

BENEFITS_VARS = [
    "sernum",       # household ID (link key)
    "benunit",      # benefit unit number
    "benefit",      # benefit type code
    "benamt",       # benefit amount (weekly)
]


def _safe_cols(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Return the subset of *wanted* columns that actually exist in *df*."""
    available = {c.lower() for c in df.columns}
    return [c for c in wanted if c in available]


def load_stata(path: pathlib.Path) -> pd.DataFrame:
    """Read a Stata .dta file and lowercase all column names."""
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    return df


def extract_househol(year: int) -> pd.DataFrame:
    """Load household file for *year* and select target variables."""
    path = FRS_PATHS[year] / "househol.dta"
    df = load_stata(path)
    keep = _safe_cols(df, HOUSEHOL_VARS)
    df = df[keep].copy()
    df["year"] = year
    # Financial year label (e.g. "2015/16")
    df["fy"] = f"{year}/{str(year + 1)[-2:]}"
    return df


def extract_adult(year: int) -> pd.DataFrame:
    """Load adult file for *year* and select target variables."""
    path = FRS_PATHS[year] / "adult.dta"
    df = load_stata(path)
    keep = _safe_cols(df, ADULT_VARS)
    df = df[keep].copy()
    df["year"] = year
    return df


def extract_benefits(year: int) -> pd.DataFrame:
    """Load benefits file for *year* and select target variables."""
    path = FRS_PATHS[year] / "benefits.dta"
    df = load_stata(path)
    keep = _safe_cols(df, BENEFITS_VARS)
    df = df[keep].copy()
    df["year"] = year
    return df


# ── Household-archetype helper flags ────────────────────────────────────────

def add_archetype_flags(hh: pd.DataFrame, adult_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Add household-archetype indicator columns used for downstream
    classification and clustering.

    Parameters
    ----------
    hh : household-level DataFrame (already merged across years)
    adult_agg : household-level aggregates derived from adult file

    Returns
    -------
    DataFrame with extra boolean/categorical columns.
    """
    df = hh.merge(adult_agg, on=["sernum", "year"], how="left")

    # --- Tenure categories (from ptentyp2) ---
    # EUL encoding: 1=council rent, 2=HA rent, 3=private rent unfurnished,
    # 4=private rent furnished, 5=owned outright, 6=owned with mortgage
    tenure_map = {
        1: "social_rent",
        2: "social_rent",
        3: "private_rent",
        4: "private_rent",
        5: "own_outright",
        6: "own_mortgage",
    }
    if "ptentyp2" in df.columns:
        df["tenure_type"] = df["ptentyp2"].map(tenure_map).fillna("unknown")
    else:
        df["tenure_type"] = "unknown"

    # --- Pensioner household flag ---
    # Use PENAGE (HRP at state pension age) from househol,
    # OR HRP empstati==6 (retired), OR HRP age80 >= 66
    is_pen_hh = pd.Series(False, index=df.index)
    if "penage" in df.columns:
        is_pen_hh = is_pen_hh | (df["penage"] == 1)
    if "hrp_retired" in df.columns:
        is_pen_hh = is_pen_hh | df["hrp_retired"].fillna(False)
    if "hrp_age" in df.columns:
        is_pen_hh = is_pen_hh | (df["hrp_age"].fillna(0) >= 66)
    df["is_pensioner"] = is_pen_hh

    # --- Single-parent flag ---
    df["is_single_parent"] = (
        (df["adulth"].fillna(0) == 1) & (df["depchldh"].fillna(0) >= 1)
    )

    # --- Disability household flag ---
    # Any adult receives disability benefits (indisben > 0)
    # OR household disability benefit amount > 0
    is_dis = pd.Series(False, index=df.index)
    if "any_adult_disben" in df.columns:
        is_dis = is_dis | df["any_adult_disben"].fillna(False)
    if "hhdisben" in df.columns:
        is_dis = is_dis | (df["hhdisben"].fillna(0) > 0)
    df["is_disability"] = is_dis

    # --- Income quintile (within-year) ---
    # Use hhinc (total household net income), weighted by gross4
    quintiles = pd.Series(np.nan, index=df.index, dtype=float)
    for yr in df["year"].unique():
        mask = (df["year"] == yr) & df["hhinc"].notna()
        if mask.sum() > 0:
            q = pd.qcut(
                df.loc[mask, "hhinc"], q=5, labels=False, duplicates="drop"
            ) + 1  # 1-5 instead of 0-4
            quintiles.loc[mask] = q.astype(float)
    df["income_quintile"] = quintiles

    # --- COVID flag ---
    df["is_covid_year"] = df["year"] == 2020

    return df


def aggregate_adults(adults: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate adult-level data to household level.

    Returns one row per (sernum, year) with:
    - hrp_age: age of person==1 (Household Reference Person)
    - hrp_empstati: employment status of HRP
    - hrp_retired: whether HRP is retired
    - any_adult_disben: any adult in HH receiving disability benefit
    - hh_total_earnings: sum of individual earnings across adults
    - hh_total_nindinc: sum of net individual incomes
    - n_adults: count of adults
    """
    df = adults.copy()

    # HRP-level info (person == 1)
    hrp = df[df["person"] == 1].copy()
    rename = {}
    if "age80" in hrp.columns:
        rename["age80"] = "hrp_age"
    if "empstati" in hrp.columns:
        rename["empstati"] = "hrp_empstati"
    keep_cols = ["sernum", "year"] + list(rename.keys())
    hrp = hrp[keep_cols].rename(columns=rename)
    # empstati==6 means retired in FRS coding; retire==1 also indicates retired
    if "hrp_empstati" in hrp.columns:
        hrp["hrp_retired"] = hrp["hrp_empstati"] == 6
    else:
        hrp["hrp_retired"] = False

    # Household-level aggregates
    agg_dict = {}
    if "inearns" in df.columns:
        agg_dict["inearns"] = "sum"
    if "nindinc" in df.columns:
        agg_dict["nindinc"] = "sum"
    if "indisben" in df.columns:
        agg_dict["indisben"] = "max"  # 1 if any adult has disability benefit
    agg_dict["person"] = "count"

    hh_agg = df.groupby(["sernum", "year"]).agg(agg_dict).reset_index()
    rename_agg = {"person": "n_adults"}
    if "inearns" in hh_agg.columns:
        rename_agg["inearns"] = "hh_total_earnings"
    if "nindinc" in hh_agg.columns:
        rename_agg["nindinc"] = "hh_total_nindinc"
    if "indisben" in hh_agg.columns:
        rename_agg["indisben"] = "any_adult_disben"
    hh_agg = hh_agg.rename(columns=rename_agg)
    if "any_adult_disben" in hh_agg.columns:
        # indisben is £/wk amount; max > 0 means at least one adult receives it
        hh_agg["any_adult_disben"] = hh_agg["any_adult_disben"] > 0

    result = hrp.merge(hh_agg, on=["sernum", "year"], how="outer")
    return result


# ── Main pipeline ───────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("FRS Data Wrangling Pipeline")
    print("=" * 60)

    years = sorted(FRS_PATHS.keys())

    # ── 1. Extract household data ───────────────────────────────────────
    print("\n[1/5] Extracting household data...")
    hh_frames = []
    for yr in years:
        df = extract_househol(yr)
        print(f"  {yr}/{yr+1-2000:02d}: {len(df):,} households, {df.shape[1]} vars")
        hh_frames.append(df)
    househol = pd.concat(hh_frames, ignore_index=True)
    print(f"  TOTAL: {len(househol):,} household-year observations")

    # ── 2. Extract adult data ───────────────────────────────────────────
    print("\n[2/5] Extracting adult data...")
    adult_frames = []
    for yr in years:
        df = extract_adult(yr)
        print(f"  {yr}/{yr+1-2000:02d}: {len(df):,} adults")
        adult_frames.append(df)
    adults = pd.concat(adult_frames, ignore_index=True)
    print(f"  TOTAL: {len(adults):,} adult-year observations")

    # ── 3. Extract benefits data ────────────────────────────────────────
    print("\n[3/5] Extracting benefits data...")
    ben_frames = []
    for yr in years:
        df = extract_benefits(yr)
        print(f"  {yr}/{yr+1-2000:02d}: {len(df):,} benefit records")
        ben_frames.append(df)
    benefits = pd.concat(ben_frames, ignore_index=True)
    print(f"  TOTAL: {len(benefits):,} benefit-year records")

    # ── 4. Save interim files ───────────────────────────────────────────
    print("\n[4/5] Saving interim files...")
    househol.to_parquet(INTERIM / "frs_househol.parquet", index=False)
    adults.to_parquet(INTERIM / "frs_adult.parquet", index=False)
    benefits.to_parquet(INTERIM / "frs_benefits.parquet", index=False)
    print(f"  Saved: {INTERIM / 'frs_househol.parquet'}")
    print(f"  Saved: {INTERIM / 'frs_adult.parquet'}")
    print(f"  Saved: {INTERIM / 'frs_benefits.parquet'}")

    # ── 5. Build analysis-ready household file ──────────────────────────
    print("\n[5/5] Building analysis-ready household file...")
    adult_agg = aggregate_adults(adults)
    analysis = add_archetype_flags(househol, adult_agg)

    analysis.to_parquet(PROCESSED / "frs_household_analysis.parquet", index=False)
    print(f"  Saved: {PROCESSED / 'frs_household_analysis.parquet'}")

    # ── Summary statistics ──────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Years covered: {min(years)}/{min(years)+1-2000:02d} to {max(years)}/{max(years)+1-2000:02d}")
    print(f"  Total households: {len(analysis):,}")
    print(f"  Columns in analysis file: {analysis.shape[1]}")
    print(f"\n  Archetype breakdown (all years):")
    for col in ["is_pensioner", "is_single_parent", "is_disability"]:
        if col in analysis.columns:
            n = analysis[col].sum()
            pct = 100 * n / len(analysis)
            print(f"    {col}: {n:,.0f} ({pct:.1f}%)")
    print(f"\n  Tenure distribution:")
    if "tenure_type" in analysis.columns:
        for t, n in analysis["tenure_type"].value_counts().items():
            pct = 100 * n / len(analysis)
            print(f"    {t}: {n:,} ({pct:.1f}%)")
    print(f"\n  Households per year:")
    for yr in years:
        n = (analysis["year"] == yr).sum()
        print(f"    {yr}/{yr+1-2000:02d}: {n:,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
