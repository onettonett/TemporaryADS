"""
compute_group_inflation.py
==========================
Calculate household-archetype-specific inflation rates by combining LCF
expenditure shares with ONS CPIH price indices.

Pipeline
--------
1. Load LCF expenditure shares (household-level) + archetypes
2. Load MM23 CPIH monthly price indices
3. Aggregate LCF shares to archetype-year level (weighted by household_weight)
4. Align COICOP categories between LCF and price indices (via concordance table)
5. Compute annual price changes (year-over-year) for each COICOP division
6. Calculate group-specific inflation rates: Σ_i [ weight_i,g,t × price_change_i,t ]
7. Output inflation rate panel (archetype × year) + detailed decomposition

Outputs
-------
data/processed/group_inflation_rates.parquet
    Panel of (archetype, year) → inflation_rate and component COICOP contributions.

data/processed/inflation_decomposition.parquet
    Detailed breakdown: (archetype, year, coicop) → weight × price_change contribution.

data/processed/archetype_inflation_summary.parquet
    Quintile comparisons and key statistics for each group.
"""

import pathlib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

ROOT = pathlib.Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

# ── COICOP Concordance ──────────────────────────────────────────────────────
# Maps LCF share column names to MM23 price index column names.
# LCF uses "share_NN_label" format; MM23 uses "label" format.
# This table handles the mapping and allows for 1:1 or 1:many relationships.

COICOP_CONCORDANCE = {
    # LCF share column                    MM23 price index column(s)  COICOP code
    "share_01_food_non_alcoholic":       ["food_non_alcoholic"],      # 01
    "share_02_alcohol_tobacco":          ["alcohol_tobacco"],         # 02
    "share_03_clothing_footwear":        ["clothing_footwear"],       # 03
    "share_04_housing_fuel_power":       ["housing_fuel_power"],      # 04
    "share_05_furnishings":              ["furnishings"],             # 05
    "share_06_health":                   ["health"],                  # 06
    "share_07_transport":                ["transport"],               # 07
    "share_08_communication":            ["communication"],           # 08
    "share_09_recreation_culture":       ["recreation_culture"],      # 09
    "share_10_education":                ["education"],               # 10
    "share_11_restaurants_hotels":       ["restaurants_hotels"],      # 11
    "share_12_misc_goods_services":      ["misc_goods_services"],     # 12
}

# If you want to include finer-grained price series (e.g., 04.2 for owner-occupier
# housing, or 01.1.1 for bread & cereals), add them here. For now, we stick to
# the 12 main COICOP divisions to match LCF granularity.


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def load_lcf_shares() -> pd.DataFrame:
    """Load LCF expenditure shares with archetypes."""
    path = PROCESSED / "lcf_expenditure_shares.parquet"
    df = pd.read_parquet(path)
    # Ensure required columns exist
    required = ["household_id", "year", "household_weight"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def load_price_indices() -> pd.DataFrame:
    """Load MM23 monthly CPIH indices."""
    path = PROCESSED / "cpih_monthly_indices.parquet"
    df = pd.read_parquet(path)
    # Ensure required columns
    if "date" not in df.columns:
        raise ValueError("MM23 file missing 'date' column")
    # Set date as datetime if it isn't already
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_shares_by_archetype(lcf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Aggregate household-level shares to archetype-year level.

    For each archetype dimension (tenure_type, is_pensioner, region_broad, etc.),
    compute weighted mean expenditure shares within each year.

    Returns a dict mapping archetype_name → DataFrame with:
        (archetype_value, year) → [share_01, share_02, ..., share_12]
    """
    # List of archetype columns in LCF
    archetype_cols = [
        "tenure_type", "is_pensioner", "income_quintile", "region_broad",
        "hrp_age_band", "hh_composition", "employment_status",
        "is_disability", "is_carer"
    ]

    share_cols = [c for c in lcf.columns if c.startswith("share_")]
    weight_col = "household_weight"

    results = {}

    for arch_col in archetype_cols:
        if arch_col not in lcf.columns:
            continue

        # Group by (archetype_value, year), compute weighted mean
        grouped = lcf.groupby([arch_col, "year"]).apply(
            lambda g: pd.Series({
                col: (g[col] * g[weight_col]).sum() / g[weight_col].sum()
                if g[weight_col].sum() > 0 else np.nan
                for col in share_cols
            }),
            include_groups=False
        ).reset_index()

        grouped = grouped.rename(columns={arch_col: "archetype_value"})
        grouped["archetype_name"] = arch_col
        grouped = grouped[["archetype_name", "archetype_value", "year"] + share_cols]

        results[arch_col] = grouped

    return results


def compute_annual_price_changes(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-over-year price changes for each COICOP division.

    Aggregates monthly indices to financial-year (April-March) averages,
    then computes YoY percentage changes.

    Returns DataFrame with (year, coicop_label) → price_change_pct
    """
    # Ensure fy column exists (should be from mm23 output)
    if "fy" not in prices.columns:
        prices["fy"] = (prices["year"].astype(str) + "/" +
                        (prices["year"] + 1).astype(str).str[-2:])

    # Get price column names (exclude date, year, month, fy_year, fy)
    price_cols = [c for c in prices.columns if c not in
                  ("date", "year", "month", "fy_year", "fy")]

    # Aggregate to financial year averages
    annual_prices = prices.groupby("fy_year")[price_cols].mean().reset_index()
    annual_prices = annual_prices.rename(columns={"fy_year": "year"})
    annual_prices = annual_prices.sort_values("year")

    # Compute YoY percentage changes
    pct_changes = annual_prices[price_cols].pct_change() * 100  # as percentage
    pct_changes["year"] = annual_prices["year"]

    # Reshape to long format: (year, coicop_label) → price_change_pct
    long = pct_changes.melt(id_vars=["year"], var_name="coicop_label",
                            value_name="price_change_pct")
    long = long.dropna(subset=["price_change_pct"])

    return long


def align_shares_to_prices(shares: pd.DataFrame,
                           price_changes: pd.DataFrame) -> pd.DataFrame:
    """
    Align LCF share column names to MM23 price index column names using concordance.

    IMPLEMENTS LASPEYRES INDEXING:
    Uses expenditure shares from year t-1 with price changes from year t.
    This ensures we're measuring inflation based on actual intended spending,
    not distorted by behavioral responses (substitution) to price changes.

    Reshape shares from wide (many share_* columns) to long format:
    (archetype_name, archetype_value, year, coicop_label) → share_value

    Then match to price_changes via: shares[year t-1] with prices[year t]
    """
    share_cols = [c for c in shares.columns if c.startswith("share_")]

    # Convert archetype_value to string EARLY to prevent pandas type coercion bugs
    # (where 1.0 gets converted to True during groupby operations)
    shares = shares.copy()
    shares["archetype_value"] = shares["archetype_value"].astype(str)

    # Melt shares to long format
    long_shares = shares.melt(
        id_vars=["archetype_name", "archetype_value", "year"],
        value_vars=share_cols,
        var_name="lcf_share_col",
        value_name="share"
    )

    # LASPEYRES: Rename year to year_base (the BASE PERIOD for expenditure weights)
    # We will merge this year_base with year_current in prices via year_current = year_base + 1
    long_shares = long_shares.rename(columns={"year": "year_base"})

    # Map LCF share columns to price index labels using concordance
    lcf_to_price = {}
    for lcf_col, price_cols in COICOP_CONCORDANCE.items():
        # For now, assume 1:1 mapping (first price col)
        if price_cols:
            lcf_to_price[lcf_col] = price_cols[0]

    long_shares["coicop_label"] = long_shares["lcf_share_col"].map(lcf_to_price)
    long_shares = long_shares.dropna(subset=["coicop_label"])

    # LASPEYRES: Create year column for merging with prices
    # year_base = t-1, so price year = t is: year_current = year_base + 1
    long_shares["year"] = long_shares["year_base"] + 1

    # Merge with price changes: year_base+1 prices paired with year_base weights
    # This gives us weight[t-1] × price_change[t], which is Laspeyres
    merged = long_shares.merge(
        price_changes,
        on=["year", "coicop_label"],
        how="inner"  # inner join ensures we only use years where we have both weights and prices
    )

    return merged


def compute_group_inflation(aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Laspeyres-weighted inflation rate for each (archetype, year).

    Laspeyres formula: inflation_rate = Σ_i [ weight[t-1]_i × price_change[t]_i ]

    Where:
    - weight[t-1] = expenditure share from PREVIOUS year
    - price_change[t] = percentage price change in CURRENT year
    - Avoids substitution bias by fixing the consumption basket at previous year levels

    Returns DataFrame: (archetype_name, archetype_value, year) → inflation_rate
    """
    # Compute contribution of each COICOP to inflation
    # share = expenditure weight from year_base (t-1)
    # price_change_pct = price change from year_base to year (t-1 to t)
    aligned["contribution"] = aligned["share"] * aligned["price_change_pct"]

    # Aggregate to archetype-year level
    # year = the inflation period year (t), computed from year_base (t-1) + 1
    inflation = aligned.groupby(
        ["archetype_name", "archetype_value", "year"]
    ).agg(
        inflation_rate=("contribution", "sum"),
        n_coicop=("coicop_label", "count"),
        mean_share=("share", "mean"),
        mean_price_change=("price_change_pct", "mean"),
    ).reset_index()

    inflation = inflation.sort_values(
        ["archetype_name", "archetype_value", "year"]
    )

    return inflation


def main() -> None:
    print("=" * 70)
    print("COMPUTING GROUP-SPECIFIC INFLATION RATES (LASPEYRES METHODOLOGY)")
    print("=" * 70)
    print()
    print("Methodology: Laspeyres Index")
    print("  inflation_rate[year t] = SUM[ weight[t-1]_i * price_change[t]_i ]")
    print()
    print("Key differences from contemporaneous index:")
    print("  - Uses PREVIOUS YEAR spending weights as fixed consumption basket")
    print("  - Avoids substitution bias: measures cost of original shopping pattern")
    print("  - When poor cut discretionary spending during crisis, Laspeyres still")
    print("    measures inflation on what they WOULD have bought at original rates")
    print("  - This reveals hidden inequality masked by behavioral responses")
    print()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading LCF expenditure shares and archetypes...")
    lcf = load_lcf_shares()
    print(f"  Loaded: {len(lcf):,} households across {lcf['year'].nunique()} years")
    print(f"  Archetype columns: {[c for c in lcf.columns if c.startswith(('is_', 'income_', 'tenure_', 'hh_', 'hrp_', 'region_', 'employment_', 'n_'))]}")

    print("\n[2/5] Loading MM23 CPIH monthly price indices...")
    prices = load_price_indices()
    print(f"  Loaded: {len(prices):,} monthly observations")
    print(f"  Date range: {prices['date'].min():%Y-%m} to {prices['date'].max():%Y-%m}")

    # ── 2. Aggregate shares by archetype ─────────────────────────────────────
    print("\n[3/5] Aggregating household shares to archetype level (weighted)...")
    print("  Note: Using Laspeyres methodology with lagged weights")
    archetype_shares = aggregate_shares_by_archetype(lcf)
    total_archetypes = sum(len(df) for df in archetype_shares.values())
    print(f"  Created {len(archetype_shares)} archetype dimensions")
    print(f"  Total archetype-year groups before lagging: {total_archetypes:,}")
    print(f"  (First year will be excluded - no prior year weights available)")

    # Combine all archetypes into one DataFrame
    all_shares = pd.concat(archetype_shares.values(), ignore_index=True)

    # ── 3. Compute annual price changes ──────────────────────────────────────
    print("\n[4/5] Computing annual price changes by COICOP...")
    price_changes = compute_annual_price_changes(prices)
    print(f"  Computed {len(price_changes):,} COICOP-year price changes")
    print(f"  Mean inflation across all COICOP: {price_changes['price_change_pct'].mean():.2f}%")

    # ── 4. Align and compute inflation ───────────────────────────────────────
    print("\n[5/5] Aligning shares to price indices and computing inflation...")
    aligned = align_shares_to_prices(all_shares, price_changes)
    print(f"  Aligned {len(aligned):,} archetype-COICOP-year combinations")

    inflation = compute_group_inflation(aligned)
    print(f"  Computed {len(inflation):,} group inflation rates")

    # ── 5. Save outputs ──────────────────────────────────────────────────────
    print("\n[6/6] Saving outputs...")

    # Note: archetype_value is already string type (converted early in align_shares_to_prices)

    # Main output: group inflation rates
    inflation.to_parquet(
        PROCESSED / "group_inflation_rates.parquet",
        index=False
    )
    print(f"  Saved: {PROCESSED / 'group_inflation_rates.parquet'}")

    # Detailed decomposition
    aligned.to_parquet(
        PROCESSED / "inflation_decomposition.parquet",
        index=False
    )
    print(f"  Saved: {PROCESSED / 'inflation_decomposition.parquet'}")

    # Summary statistics by archetype
    summary = inflation.groupby("archetype_name").agg(
        n_groups=("archetype_value", "nunique"),
        n_years=("year", "nunique"),
        mean_inflation=("inflation_rate", "mean"),
        std_inflation=("inflation_rate", "std"),
        min_inflation=("inflation_rate", "min"),
        max_inflation=("inflation_rate", "max"),
    ).reset_index()
    summary.to_parquet(
        PROCESSED / "archetype_inflation_summary.parquet",
        index=False
    )
    print(f"  Saved: {PROCESSED / 'archetype_inflation_summary.parquet'}")

    # ── Summary statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nInflation statistics by archetype:\n")
    print(summary.to_string(index=False))

    # Top and bottom groups by mean inflation
    print(f"\nTop 5 groups by mean inflation:")
    top5 = (
        inflation.groupby(["archetype_name", "archetype_value"])["inflation_rate"]
        .mean()
        .nlargest(5)
    )
    for (arch, val), rate in top5.items():
        print(f"  {arch}={val}: {rate:.2f}%")

    print(f"\nBottom 5 groups by mean inflation (lowest purchasing power loss):")
    bottom5 = (
        inflation.groupby(["archetype_name", "archetype_value"])["inflation_rate"]
        .mean()
        .nsmallest(5)
    )
    for (arch, val), rate in bottom5.items():
        print(f"  {arch}={val}: {rate:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
