"""
wrangle_hci.py
==============
Extract and reshape the ONS Household Costs Indices (HCI) reference tables.

The HCI provides inflation rates and price indices broken down by:
- Income decile (Tables 3-7)
- Tenure type (Tables 8-12)
- Retirement status (Tables 13-17)
- Children in household (Tables 18-22)
- All households (Tables 23-26)

These serve as validation benchmarks for the project's own calculations
of differential inflation by household type.

Outputs
-------
- data/interim/hci_summary_pctchange.parquet   – Table 1 (% change, all groups)
- data/interim/hci_tenure_index.parquet        – Table 9 (index, tenure)
- data/interim/hci_retirement_index.parquet    – Table 14 (index, retirement)
- data/interim/hci_income_index.parquet        – Table 4 (index, income decile)
- data/interim/hci_children_index.parquet      – Table 19 (index, children)
- data/processed/hci_validation.parquet        – combined all-items indices for
      all household groups, ready for comparison with project estimates.

Raw data in data/raw/HCI/ is never modified.
"""

import pathlib
import warnings
import pandas as pd
import numpy as np
import re

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "HCI" / "XLSX_format"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# The HCI file
HCI_FILE = list(RAW.glob("*.xlsx"))
if HCI_FILE:
    HCI_PATH = HCI_FILE[0]
else:
    HCI_PATH = RAW / "householdcostsindicesforukhouseholdgroupsocttodec25.xlsx"

# ── Sheet configurations ────────────────────────────────────────────────────
# Each table has:
# - header_rows: number of metadata rows before column headers
# - header_row_idx: which row contains column names (0-indexed)
# - grouping: what the table is grouped by

TABLE_CONFIG = {
    "Table 1": {
        "description": "All-items % change over 12 months, all subgroups",
        "header_row_idx": 2,
        "grouping": "summary",
        "metric": "pct_change",
    },
    "Table 4": {
        "description": "Index (2015=100) by income decile and COICOP division",
        "header_row_idx": 3,
        "grouping": "income_decile",
        "metric": "index",
    },
    "Table 9": {
        "description": "Index (2015=100) by tenure type and COICOP division",
        "header_row_idx": 3,
        "grouping": "tenure",
        "metric": "index",
    },
    "Table 14": {
        "description": "Index (2015=100) by retirement status and COICOP division",
        "header_row_idx": 3,
        "grouping": "retirement",
        "metric": "index",
    },
    "Table 19": {
        "description": "Index (2015=100) by children status and COICOP division",
        "header_row_idx": 3,
        "grouping": "children",
        "metric": "index",
    },
}


def parse_hci_date(date_str: str) -> pd.Timestamp | None:
    """Parse 'Mon-YYYY' format (e.g. 'Jan-2022') to Timestamp."""
    try:
        return pd.to_datetime(date_str, format="%b-%Y")
    except (ValueError, TypeError):
        return None


def parse_multiline_header(header_str: str) -> dict:
    """
    Parse multi-line HCI column headers like:
    'Mortgagor and other owner occupier\\n01\\nFood and non-alcoholic Beverages'

    Returns dict with keys: group, coicop_code, coicop_name
    """
    if pd.isna(header_str) or str(header_str).strip().lower() == "description:":
        return {"group": None, "coicop_code": None, "coicop_name": None}

    parts = str(header_str).split("\n")
    if len(parts) >= 3:
        return {
            "group": parts[0].strip(),
            "coicop_code": parts[1].strip(),
            "coicop_name": parts[2].strip(),
        }
    elif len(parts) == 1:
        # Simple header (Table 1 style)
        return {"group": parts[0].strip(), "coicop_code": "00", "coicop_name": "All items"}
    return {"group": str(header_str).strip(), "coicop_code": None, "coicop_name": None}


def load_hci_table(sheet_name: str, config: dict) -> pd.DataFrame:
    """
    Load an HCI table and convert to long format.

    Returns DataFrame with columns:
    [date, group, coicop_code, coicop_name, value, metric, grouping]
    """
    header_idx = config["header_row_idx"]

    # Read raw (all as string to handle [x] suppressed values)
    raw = pd.read_excel(
        HCI_PATH, sheet_name=sheet_name, header=None, dtype=str
    )

    # Parse column headers
    header_row = raw.iloc[header_idx]
    col_meta = [parse_multiline_header(h) for h in header_row]

    # Data rows start after header
    data = raw.iloc[header_idx + 1:].copy()
    data = data.reset_index(drop=True)

    # Build long-format records
    records = []
    for _, row in data.iterrows():
        date = parse_hci_date(row.iloc[0])
        if date is None:
            continue

        for col_idx in range(1, len(row)):
            if col_idx >= len(col_meta):
                break
            meta = col_meta[col_idx]
            if meta["group"] is None:
                continue

            val_str = str(row.iloc[col_idx]).strip()
            if val_str in ("[x]", "nan", "", ".."):
                continue
            try:
                value = float(val_str)
            except (ValueError, TypeError):
                continue

            records.append({
                "date": date,
                "group": meta["group"],
                "coicop_code": meta["coicop_code"],
                "coicop_name": meta["coicop_name"],
                "value": value,
                "metric": config["metric"],
                "grouping": config["grouping"],
            })

    return pd.DataFrame(records)


def build_validation_panel(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a combined validation panel containing all-items indices
    for all household groups, filtered to the study period (2015+).

    This is the primary file for comparing project estimates against HCI.
    """
    frames = []
    for name, df in tables.items():
        if df.empty:
            continue
        # Filter to all-items only (coicop 00 or 0) and index values
        all_items = df[
            (df["coicop_code"].isin(["0", "00"]))
            & (df["metric"] == "index")
        ].copy()
        if all_items.empty and df["metric"].iloc[0] == "pct_change":
            # Table 1 is % change, include it
            all_items = df.copy()
        frames.append(all_items)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["date"] >= "2015-01-01"].copy()

    # Add year, month, financial year
    combined["year"] = combined["date"].dt.year
    combined["month"] = combined["date"].dt.month
    combined["fy_year"] = np.where(
        combined["month"] >= 4,
        combined["year"],
        combined["year"] - 1
    )

    return combined.sort_values(["grouping", "group", "date"]).reset_index(drop=True)


# ── Main pipeline ───────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("HCI Validation Data Wrangling Pipeline")
    print("=" * 60)

    print(f"\n  Source: {HCI_PATH.name}")

    tables = {}

    # ── 1. Load each table ──────────────────────────────────────────────
    print("\n[1/3] Loading HCI tables...")
    for sheet_name, config in TABLE_CONFIG.items():
        print(f"  {sheet_name}: {config['description']}...")
        try:
            df = load_hci_table(sheet_name, config)
            tables[sheet_name] = df
            n_groups = df["group"].nunique() if not df.empty else 0
            date_range = (
                f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}"
                if not df.empty else "empty"
            )
            print(f"    {len(df):,} records, {n_groups} groups, {date_range}")
        except Exception as e:
            print(f"    ERROR: {e}")
            tables[sheet_name] = pd.DataFrame()

    # ── 2. Save interim files ───────────────────────────────────────────
    print("\n[2/3] Saving interim files...")
    file_map = {
        "Table 1": "hci_summary_pctchange.parquet",
        "Table 4": "hci_income_index.parquet",
        "Table 9": "hci_tenure_index.parquet",
        "Table 14": "hci_retirement_index.parquet",
        "Table 19": "hci_children_index.parquet",
    }
    for sheet_name, filename in file_map.items():
        if sheet_name in tables and not tables[sheet_name].empty:
            tables[sheet_name].to_parquet(INTERIM / filename, index=False)
            print(f"  Saved: {filename}")

    # ── 3. Build validation panel ───────────────────────────────────────
    print("\n[3/3] Building combined validation panel...")
    validation = build_validation_panel(tables)
    if not validation.empty:
        validation.to_parquet(PROCESSED / "hci_validation.parquet", index=False)
        print(f"  Saved: {PROCESSED / 'hci_validation.parquet'}")
    else:
        print("  WARNING: validation panel is empty")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────")
    if not validation.empty:
        print(f"  Validation panel: {len(validation):,} records")
        print(f"  Date range: {validation['date'].min():%Y-%m} to {validation['date'].max():%Y-%m}")
        print(f"\n  Groups available for validation:")
        for grouping in validation["grouping"].unique():
            groups = validation[validation["grouping"] == grouping]["group"].unique()
            print(f"    {grouping}: {', '.join(sorted(groups)[:5])}" +
                  (f" (+{len(groups)-5} more)" if len(groups) > 5 else ""))
    print("\nDone.")


if __name__ == "__main__":
    main()
