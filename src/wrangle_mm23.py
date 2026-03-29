"""
wrangle_mm23.py
===============
Extract and reshape CPIH price indices and weights from the ONS MM23 dataset.

The MM23 CSV is a wide file (~4000 columns) with metadata rows at the top.
Row 0 = titles, Row 1 = CDID series IDs, Rows 2-6 = metadata,
Row 7+ = data (annual, quarterly, then monthly).

We extract:
- CPIH index values at COICOP division level (01–12) + All Items (00)
- CPIH weights at COICOP division level
- Monthly frequency only (2015 JAN onwards), aligned to the study period

Outputs
-------
- data/interim/mm23_cpih_indices.parquet    – monthly CPIH indices, long format
- data/interim/mm23_cpih_weights.parquet    – annual CPIH weights, long format
- data/processed/cpih_monthly_indices.parquet – clean monthly index panel
      (date × COICOP division), ready for joining to LCF expenditure shares.

Raw data in data/raw/MM23/ is never modified.
"""

import pathlib
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "MM23" / "CSV_format" / "mm23.csv"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── CPIH INDEX series: COICOP division level (2015=100) ─────────────────────
# Verified against the actual MM23 column positions.

CPIH_INDEX_SERIES = {
    "L522": {"coicop": "00", "label": "all_items"},
    "L523": {"coicop": "01", "label": "food_non_alcoholic"},
    "L524": {"coicop": "02", "label": "alcohol_tobacco"},
    "L525": {"coicop": "03", "label": "clothing_footwear"},
    "L5PG": {"coicop": "04", "label": "housing_fuel_power"},
    "L527": {"coicop": "05", "label": "furnishings"},
    "L528": {"coicop": "06", "label": "health"},
    "L529": {"coicop": "07", "label": "transport"},
    "L52A": {"coicop": "08", "label": "communication"},
    "L52B": {"coicop": "09", "label": "recreation_culture"},
    "L52C": {"coicop": "10", "label": "education"},
    "L52D": {"coicop": "11", "label": "restaurants_hotels"},
    "L52E": {"coicop": "12", "label": "misc_goods_services"},
}

# Additional group-level series for finer-grained analysis
CPIH_INDEX_GROUPS = {
    "L5P5": {"coicop": "04.2", "label": "owner_occupier_housing"},
    "L52H": {"coicop": "01.1", "label": "food"},
    "L52I": {"coicop": "01.1.1", "label": "bread_cereals"},
    "L52J": {"coicop": "01.1.2", "label": "meat"},
}

# ── CPIH WEIGHTS series: COICOP division level (parts per 1000) ─────────────

CPIH_WEIGHT_SERIES = {
    "L5CY": {"coicop": "00", "label": "all_items"},
    "L5CZ": {"coicop": "01", "label": "food_non_alcoholic"},
    "L5D2": {"coicop": "02", "label": "alcohol_tobacco"},
    "L5D3": {"coicop": "03", "label": "clothing_footwear"},
    "L5D4": {"coicop": "04", "label": "housing_fuel_power"},
    "L5D5": {"coicop": "05", "label": "furnishings"},
    "L5D6": {"coicop": "06", "label": "health"},
    "L5D7": {"coicop": "07", "label": "transport"},
    "L5D8": {"coicop": "08", "label": "communication"},
    "L5D9": {"coicop": "09", "label": "recreation_culture"},
    "L5DA": {"coicop": "10", "label": "education"},
    "L5DB": {"coicop": "11", "label": "restaurants_hotels"},
    "L5DC": {"coicop": "12", "label": "misc_goods_services"},
}

# ── Month abbreviation to number mapping ────────────────────────────────────

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def load_mm23() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Load the MM23 CSV and return (titles, cdids, data_df).
    - titles: Series of column titles (row 0)
    - cdids: Series of CDID codes (row 1)
    - data_df: all data rows (row 7+), first column is the date string
    """
    titles = pd.read_csv(RAW, nrows=1, header=None, dtype=str).iloc[0]
    cdids = pd.read_csv(RAW, skiprows=1, nrows=1, header=None, dtype=str).iloc[0]
    data = pd.read_csv(RAW, skiprows=7, header=None, dtype=str)
    return titles, cdids, data


def find_cdid_col(cdids: pd.Series, target_cdid: str) -> int | None:
    """Find column index for a given CDID in the header."""
    matches = cdids[cdids == target_cdid].index.tolist()
    if matches:
        return matches[0]
    return None


def parse_monthly_date(date_str: str) -> pd.Timestamp | None:
    """Parse 'YYYY MMM' format to a Timestamp (first day of month)."""
    parts = str(date_str).strip().split()
    if len(parts) != 2:
        return None
    year_str, month_str = parts
    month = MONTH_MAP.get(month_str.upper())
    if month is None:
        return None
    try:
        return pd.Timestamp(year=int(year_str), month=month, day=1)
    except (ValueError, TypeError):
        return None


def extract_monthly_indices(cdids: pd.Series, data: pd.DataFrame,
                            series_map: dict) -> pd.DataFrame:
    """
    Extract monthly index/weight values for a set of CDID series.

    Returns a long-format DataFrame with columns:
    [date, cdid, coicop, label, value]
    """
    records = []

    for cdid, info in series_map.items():
        col_idx = find_cdid_col(cdids, cdid)
        if col_idx is None:
            print(f"  WARNING: CDID {cdid} ({info['label']}) not found in MM23")
            continue

        for _, row in data.iterrows():
            date = parse_monthly_date(row[0])
            if date is None:
                continue
            val = row[col_idx]
            try:
                val_float = float(val)
            except (ValueError, TypeError):
                continue
            records.append({
                "date": date,
                "cdid": cdid,
                "coicop": info["coicop"],
                "label": info["label"],
                "value": val_float,
            })

    return pd.DataFrame(records)


def extract_annual_weights(cdids: pd.Series, data: pd.DataFrame,
                           series_map: dict) -> pd.DataFrame:
    """
    Extract annual CPIH weights.
    Annual rows have a 4-digit year in column 0 (no space).
    """
    records = []

    for cdid, info in series_map.items():
        col_idx = find_cdid_col(cdids, cdid)
        if col_idx is None:
            print(f"  WARNING: CDID {cdid} ({info['label']}) not found in MM23")
            continue

        for _, row in data.iterrows():
            date_str = str(row[0]).strip()
            # Annual row: exactly 4 digits
            if len(date_str) == 4 and date_str.isdigit():
                year = int(date_str)
                if year < 2015 or year > 2026:
                    continue
                val = row[col_idx]
                try:
                    val_float = float(val)
                except (ValueError, TypeError):
                    continue
                records.append({
                    "year": year,
                    "cdid": cdid,
                    "coicop": info["coicop"],
                    "label": info["label"],
                    "weight_per_1000": val_float,
                })

    return pd.DataFrame(records)


def build_wide_monthly_panel(indices: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot monthly indices from long to wide format:
    rows = months, columns = COICOP divisions.
    Ready for merging with LCF expenditure shares.
    """
    pivot = indices.pivot_table(
        index="date", columns="label", values="value", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None

    # Add year and month columns for easy joins
    pivot["year"] = pivot["date"].dt.year
    pivot["month"] = pivot["date"].dt.month

    # Add financial year (April-March)
    pivot["fy_year"] = np.where(
        pivot["month"] >= 4,
        pivot["year"],
        pivot["year"] - 1
    )
    pivot["fy"] = pivot["fy_year"].astype(str) + "/" + (pivot["fy_year"] + 1).astype(str).str[-2:]

    return pivot.sort_values("date").reset_index(drop=True)


def compute_annual_average_indices(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute financial-year average indices for each COICOP division.
    Financial year = April to March (e.g., FY 2015/16 = Apr 2015 – Mar 2016).
    """
    label_cols = [c for c in monthly.columns
                  if c not in ("date", "year", "month", "fy_year", "fy")]

    annual = monthly.groupby(["fy_year", "fy"])[label_cols].mean().reset_index()
    annual = annual.rename(columns={"fy_year": "year"})
    return annual


# ── Main pipeline ───────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("MM23 / CPIH Data Wrangling Pipeline")
    print("=" * 60)

    # ── 1. Load MM23 ────────────────────────────────────────────────────
    print("\n[1/5] Loading MM23 CSV...")
    titles, cdids, data = load_mm23()
    print(f"  Loaded: {data.shape[0]} data rows, {data.shape[1]} columns")

    # ── 2. Extract monthly CPIH indices (division level) ────────────────
    print("\n[2/5] Extracting monthly CPIH indices (division level)...")
    all_index_series = {**CPIH_INDEX_SERIES, **CPIH_INDEX_GROUPS}
    indices = extract_monthly_indices(cdids, data, all_index_series)
    # Filter to study period: Jan 2015 – latest available
    indices = indices[indices["date"] >= "2015-01-01"].copy()
    print(f"  Extracted {len(indices):,} monthly index observations")
    print(f"  Series: {indices['label'].nunique()} CPIH index series")
    print(f"  Date range: {indices['date'].min():%Y-%m} to {indices['date'].max():%Y-%m}")

    # ── 3. Extract annual CPIH weights ──────────────────────────────────
    print("\n[3/5] Extracting annual CPIH weights...")
    weights = extract_annual_weights(cdids, data, CPIH_WEIGHT_SERIES)
    print(f"  Extracted {len(weights):,} weight observations")
    if len(weights) > 0:
        print(f"  Years: {weights['year'].min()} to {weights['year'].max()}")

    # ── 4. Save interim files ───────────────────────────────────────────
    print("\n[4/5] Saving interim files...")
    indices.to_parquet(INTERIM / "mm23_cpih_indices.parquet", index=False)
    weights.to_parquet(INTERIM / "mm23_cpih_weights.parquet", index=False)
    print(f"  Saved: {INTERIM / 'mm23_cpih_indices.parquet'}")
    print(f"  Saved: {INTERIM / 'mm23_cpih_weights.parquet'}")

    # ── 5. Build analysis-ready monthly panel ───────────────────────────
    print("\n[5/5] Building analysis-ready monthly index panel...")
    # Use division-level only for the wide panel
    div_indices = indices[indices["cdid"].isin(CPIH_INDEX_SERIES)].copy()
    monthly_wide = build_wide_monthly_panel(div_indices)
    annual_avg = compute_annual_average_indices(monthly_wide)

    monthly_wide.to_parquet(PROCESSED / "cpih_monthly_indices.parquet", index=False)
    annual_avg.to_parquet(PROCESSED / "cpih_annual_fy_indices.parquet", index=False)
    print(f"  Saved: {PROCESSED / 'cpih_monthly_indices.parquet'}")
    print(f"  Saved: {PROCESSED / 'cpih_annual_fy_indices.parquet'}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Monthly panel: {monthly_wide.shape[0]} months × {monthly_wide.shape[1]} columns")
    print(f"  Annual FY averages: {annual_avg.shape[0]} years")

    # Print latest annual average for each division
    if len(annual_avg) > 0:
        latest_yr = annual_avg["year"].max()
        latest = annual_avg[annual_avg["year"] == latest_yr]
        label_cols = [c for c in latest.columns if c not in ("year", "fy")]
        print(f"\n  CPIH Index (FY {latest_yr}/{latest_yr+1-2000:02d} average, 2015=100):")
        for col in sorted(label_cols):
            val = latest[col].values[0] if col in latest.columns else np.nan
            if pd.notna(val):
                print(f"    {col}: {val:.1f}")

    # Print weight distribution for latest year
    if len(weights) > 0:
        latest_wt_yr = weights["year"].max()
        wt_latest = weights[weights["year"] == latest_wt_yr]
        print(f"\n  CPIH Weights ({latest_wt_yr}, parts per 1000):")
        for _, row in wt_latest.sort_values("coicop").iterrows():
            print(f"    {row['coicop']} {row['label']}: {row['weight_per_1000']:.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
