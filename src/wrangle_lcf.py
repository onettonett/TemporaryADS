"""
wrangle_lcf.py
==============
Build a harmonised panel of LCF household expenditure shares and archetype
flags for financial years 2015/16 - 2023/24.

This is a linear top-to-bottom script: read each year's Stata files, combine,
clean, compute shares, tag archetypes, save one CSV.

Outputs one file:
    data/output/lcf_expenditure_shares.csv

Three archetype dimensions are produced:
    tenure_type, income_quintile, hrp_age_band
"""

import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LCF"
OUTPUT = ROOT / "data" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Stata file paths (one dvhh + one dvper per year) ──────────────────────
LCF_DVHH = {
    2015: RAW / "LCF_2015/stata/stata11_se/2015-16_dvhh_ukanon.dta",
    2016: RAW / "LCF_2016/stata/stata13_se/2016_17_dvhh_ukanon.dta",
    2017: RAW / "LCF_2017/stata/stata11_se/dvhh_ukanon_2017-18.dta",
    2018: RAW / "LCF_2018/stata/stata13/2018_dvhh_ukanon.dta",
    2019: RAW / "LCF_2019/stata/stata13/lcfs_2019_dvhh_ukanon.dta",
    2020: RAW / "LCF_2020/stata/stata13/lcfs_2020_dvhh_ukanon.dta",
    2021: RAW / "LCF_2021/stata/stata13_se/lcfs_2021_dvhh_ukanon.dta",
    2022: RAW / "LCF_2022/stata/stata13_se/dvhh_ukanon_2022.dta",
    2023: RAW / "LCF_2023/stata/stata13_se/dvhh_ukanon_v2_2023.dta",
}

LCF_DVPER = {
    2015: RAW / "LCF_2015/stata/stata11_se/2015-16_dvper_ukanon.dta",
    2016: RAW / "LCF_2016/stata/stata13_se/2016_17_dvper_ukanon.dta",
    2017: RAW / "LCF_2017/stata/stata11_se/dvper_ukanon_2017-18.dta",
    2018: RAW / "LCF_2018/stata/stata13/2018_dvper_ukanon201819.dta",
    2019: RAW / "LCF_2019/stata/stata13/lcfs_2019_dvper_ukanon201920.dta",
    2020: RAW / "LCF_2020/stata/stata13/lcfs_2020_dvper_ukanon202021.dta",
    2021: RAW / "LCF_2021/stata/stata13_se/lcfs_2021_dvper_ukanon202122.dta",
    2022: RAW / "LCF_2022/stata/stata13_se/dvper_ukanon_2022-23.dta",
    2023: RAW / "LCF_2023/stata/stata13_se/dvper_ukanon_202324_2023.dta",
}

# COICOP division expenditure columns (weekly £).  p600t is overall total.
COICOP_COLS = [f"p60{d}t" if d < 10 else f"p6{d}t" for d in range(1, 13)]
COICOP_LABELS = {
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

# Household demographic/identifier columns to keep + rename.
# p600t is the COICOP grand total (divisions 01-12 summed); we use it as the
# LCF's "official" total expenditure and rename to total_expenditure for
# readability.
DVHH_COLS = {
    "case": "household_id",
    "weighta": "household_weight",
    "a049": "household_size",
    "a121": "tenure_code",
    "p389p": "income_gross_weekly",
    "eqincdmp": "income_equivalised",
    "p600t": "total_expenditure",
    "b010": "rent_weekly",
}

TENURE_MAP = {
    1: "social_rent",    2: "social_rent",
    3: "private_rent",   4: "private_rent",
    5: "own_outright",
    6: "own_mortgage",   7: "own_mortgage",
    # 8 = rent free; left unmapped (too few, ~50/year)
}

AGE_BINS = [0, 30, 50, 65, 75, 200]
AGE_LABELS = ["under_30", "30_to_49", "50_to_64", "65_to_74", "75_plus"]

# Plausibility bounds on weekly household expenditure (£)
EXP_LOW, EXP_HIGH = 30.0, 3000.0

MIN_CELL_SIZE = 100  # warn if fewer than this per group-year


# ── 1. Load Stata files year by year ──────────────────────────────────────

def _read_stata(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    return df


print("=" * 60)
print("LCF wrangling pipeline")
print("=" * 60)

years = sorted(LCF_DVHH.keys())
dvhh_frames, dvper_frames = [], []

print("\n[1/5] Loading Stata files...")
for yr in years:
    # Household file: keep target + COICOP columns that exist
    hh = _read_stata(LCF_DVHH[yr])
    keep = [c for c in (list(DVHH_COLS.keys()) + COICOP_COLS)
            if c in hh.columns]
    hh = hh[keep].rename(columns=DVHH_COLS)
    hh["year"] = yr

    # Person file: HRP is person_id == 1; we only need age
    per = _read_stata(LCF_DVPER[yr])
    per = per[["case", "person", "a005p"]].rename(
        columns={"case": "household_id", "person": "person_id", "a005p": "hrp_age"}
    )
    per = per[per["person_id"] == 1][["household_id", "hrp_age"]]
    per["year"] = yr

    dvhh_frames.append(hh)
    dvper_frames.append(per)
    print(f"  {yr}/{(yr+1) % 100:02d}: {len(hh):,} households, {len(per):,} HRPs")

dvhh = pd.concat(dvhh_frames, ignore_index=True)
hrp = pd.concat(dvper_frames, ignore_index=True)
df = dvhh.merge(hrp, on=["household_id", "year"], how="left")
print(f"  TOTAL: {len(df):,} household-year observations")


# ── 2. Clean expenditure (negatives → NaN, apply plausibility filter) ─────

print("\n[2/5] Cleaning expenditure...")

# Negative totals and negative divisions become NaN
df.loc[df["total_expenditure"] < 0, "total_expenditure"] = np.nan
for col in COICOP_COLS:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

# Drop households outside plausible weekly-spend band (£30 - £3000)
n_before = len(df)
df = df[
    df["total_expenditure"].between(EXP_LOW, EXP_HIGH, inclusive="both")
].copy()
print(f"  Plausibility filter (£{EXP_LOW:.0f}-£{EXP_HIGH:.0f}/week): "
      f"dropped {n_before - len(df):,} of {n_before:,} rows")

# Denominator consistency: if the LCF total differs from the sum of COICOP
# divisions by >1%, use the division sum to keep shares internally consistent.
coicop_sum = df[COICOP_COLS].sum(axis=1)
both_pos = (df["total_expenditure"] > 0) & (coicop_sum > 0)
ratio = df.loc[both_pos, "total_expenditure"] / coicop_sum[both_pos]
n_bad = ((ratio < 0.99) | (ratio > 1.01)).sum()
if n_bad > 0:
    pct = 100 * n_bad / both_pos.sum()
    print(f"  Denominator check: {n_bad:,} rows ({pct:.1f}%) diverge >1% "
          f"→ using COICOP division sum as denominator")
    denom = coicop_sum.replace(0, np.nan)
else:
    denom = df["total_expenditure"].replace(0, np.nan)


# ── 3. Compute COICOP expenditure shares ──────────────────────────────────

print("\n[3/5] Computing expenditure shares...")
for col, label in COICOP_LABELS.items():
    df[f"share_{label}"] = df[col] / denom

# Split COICOP 04 into actual rent vs energy+other (needed for inflation calc)
rent = df["rent_weekly"].fillna(0).clip(lower=0)
p604 = df["p604t"].fillna(0).clip(lower=0)
rent_capped = rent.clip(upper=p604)
df["share_04_actual_rent"] = rent_capped / denom
df["share_04_energy_other"] = (p604 - rent_capped).clip(lower=0) / denom

# Domain-based household filter: incomplete/erroneous diaries
# (winsorisation would break compositional constraint)
exclude = (
    (df["share_01_food_non_alcoholic"] == 0) |
    (df["share_04_housing_fuel_power"] == 0)
)
print(f"  Dropping {exclude.sum():,} households (zero food / "
      f"zero housing) of {len(df):,}")
df = df[~exclude].copy()


# ── 4. Build archetypes ───────────────────────────────────────────────────

print("\n[4/5] Building archetype flags...")

# Tenure
df["tenure_type"] = df["tenure_code"].map(TENURE_MAP).fillna("unknown")

# Weighted income quintiles (modified-OECD equivalised income) within year
df.loc[df["income_equivalised"] <= 0, "income_equivalised"] = np.nan
df["income_quintile"] = np.nan
for yr in df["year"].unique():
    mask = (df["year"] == yr) & df["income_equivalised"].notna()
    sub = df.loc[mask, ["income_equivalised", "household_weight"]].sort_values(
        "income_equivalised"
    )
    cum_share = sub["household_weight"].fillna(1).cumsum() / \
        sub["household_weight"].fillna(1).sum()
    thresholds = np.arange(1, 6) / 5.0
    q = np.clip(np.searchsorted(thresholds, cum_share.values, side="left") + 1, 1, 5)
    df.loc[sub.index, "income_quintile"] = q.astype(float)

# HRP age band
df["hrp_age_band"] = pd.cut(
    df["hrp_age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
).astype(str).replace("nan", np.nan)


# ── 5. QA + save CSV ──────────────────────────────────────────────────────

print("\n[5/5] Quality assurance...")

main_12 = [c for c in df.columns
           if c.startswith("share_") and c not in
           ("share_04_actual_rent", "share_04_energy_other")]
share_sum = df[main_12].sum(axis=1)
print(f"  Share sum (12 main COICOP): mean={share_sum.mean():.4f}, "
      f"min={share_sum.min():.4f}, max={share_sum.max():.4f}")

# Cell-size warnings
for col in ("tenure_type", "income_quintile", "hrp_age_band"):
    counts = df.groupby(["year", col], observed=True).size()
    small = counts[counts < MIN_CELL_SIZE]
    if len(small) > 0:
        print(f"  WARNING: {len(small)} small cells in {col} (<{MIN_CELL_SIZE}):")
        for (yr, val), n in small.items():
            print(f"    {col}={val}, year={yr} → n={n}")

# Drop internal columns
internal_cols = COICOP_COLS + ["tenure_code", "rent_weekly"]
df = df.drop(columns=[c for c in internal_cols if c in df.columns])

out_path = OUTPUT / "lcf_expenditure_shares.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"  {len(df):,} households, {df.shape[1]} columns")

print("\nHouseholds per year:")
for yr in years:
    n = (df["year"] == yr).sum()
    print(f"  {yr}/{(yr+1) % 100:02d}: {n:,}")

print("\nTenure distribution:")
for t, n in df["tenure_type"].value_counts().items():
    print(f"  {t}: {n:,} ({100*n/len(df):.1f}%)")
