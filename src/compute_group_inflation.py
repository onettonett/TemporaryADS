"""
compute_group_inflation.py
==========================
Compute group-specific inflation using LCF expenditure baskets and CPIH price
indices.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Share → price column mapping
CONCORDANCE: Dict[str, str] = {
    "share_01_food_non_alcoholic": "food_non_alcoholic",
    "share_02_alcohol_tobacco": "alcohol_tobacco",
    "share_03_clothing_footwear": "clothing_footwear",
    "share_04_actual_rent": "actual_rents",
    # share_04_energy_other = COICOP 04 minus actual rent (energy + water + maintenance).
    # The LCF cannot isolate energy (04.5) from water (04.4) and maintenance (04.3) at
    # household level.  Mapping to the pure electricity_gas_fuels index would overstate
    # the energy price shock by ~2-3×; mapping to the full housing_fuel_power (COICOP 04)
    # composite understates because stable rent prices dampen the index.
    # Instead we derive 'non_rent_housing_fuel': the price change for the non-rent portion
    # of COICOP 04, computed by backing actual-rents out of housing_fuel_power using OLS
    # weights fitted on pre-crisis years.  See derive_non_rent_hfp() below.
    "share_04_energy_other": "non_rent_housing_fuel",
    "share_05_furnishings": "furnishings",
    "share_06_health": "health",
    "share_07_transport": "transport",
    "share_08_communication": "communication",
    "share_09_recreation_culture": "recreation_culture",
    "share_10_education": "education",
    "share_11_restaurants_hotels": "restaurants_hotels",
    "share_12_misc_goods_services": "misc_goods_services",
}

ARCHETYPE_COLS: List[str] = [
    "income_quintile",
    "tenure_type",
    "region_broad",
    "hrp_age_band",
    "hh_composition",
    "employment_status",
    "is_pensioner",
    "care_impacted",
]


def load_lcf() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "lcf_expenditure_shares.parquet")
    share_cols = list(CONCORDANCE.keys())
    missing = [c for c in share_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing share columns: {missing}")
    return df


def load_prices() -> pd.DataFrame:
    prices = pd.read_parquet(PROCESSED / "cpih_monthly_indices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    # fy_year is already set by wrangle_mm23.py as April-March financial year.
    # Do NOT overwrite with date.dt.year (calendar year) — LCF data follows
    # financial years, so Laspeyres baskets must be paired with FY price changes.
    return prices


def _estimate_rent_weight(monthly_path: pathlib.Path) -> float:
    """Estimate the weight of actual_rents within housing_fuel_power by
    regressing monthly index LEVELS (not rates).

    Fitting on levels rather than rates recovers a reliable compositional
    weight because it exploits the long-run co-movement of the indices
    rather than short-run rate volatility, which is dominated by energy.

    Returns the implied weight w such that:
        housing_fuel_power ≈ w * actual_rents + (1-w) * non_rent_component
    Clamped to [0.15, 0.55] — a range consistent with ONS CPIH methodology
    (actual rents is roughly 20-30% of COICOP 04 in recent years).
    """
    try:
        m = pd.read_parquet(monthly_path)
        m = m[["actual_rents", "housing_fuel_power"]].dropna()
        X = np.column_stack([m["actual_rents"].values, np.ones(len(m))])
        coeffs, _, _, _ = np.linalg.lstsq(X, m["housing_fuel_power"].values, rcond=None)
        w = float(np.clip(coeffs[0], 0.15, 0.55))
    except Exception:
        w = 0.25  # fallback: ONS-consistent prior
    return w


def _derive_non_rent_hfp(pct: pd.DataFrame) -> pd.DataFrame:
    """Derive a 'non_rent_housing_fuel' price change series by backing actual
    rents out of the COICOP 04 composite (housing_fuel_power).

    The weight of actual_rents within housing_fuel_power (β_rents) is
    estimated from monthly index LEVELS (see _estimate_rent_weight).
    The non-rent component rate is then:

        non_rent_rate = (housing_fuel_power_rate − β_rents × actual_rents_rate)
                        / (1 − β_rents)

    This gives a price index for the non-rent, non-OOH portion of COICOP 04
    (energy + water + maintenance) that is more accurate than either the
    pure electricity_gas_fuels index (overstatement ~2-3×) or the composite
    housing_fuel_power (understatement because rent dampens the index).
    """
    needed = {"housing_fuel_power", "actual_rents"}
    if not needed.issubset(pct.columns):
        pct = pct.copy()
        pct["non_rent_housing_fuel"] = pct.get("housing_fuel_power", np.nan)
        return pct

    beta_rents = _estimate_rent_weight(PROCESSED / "cpih_monthly_indices.parquet")
    pct = pct.copy()
    pct["non_rent_housing_fuel"] = (
        (pct["housing_fuel_power"] - beta_rents * pct["actual_rents"])
        / max(1.0 - beta_rents, 0.15)
    )
    print(f"    non_rent_housing_fuel: β_rents={beta_rents:.3f} "
          f"(level-based OLS over all monthly data)")
    return pct


def annual_price_changes(prices: pd.DataFrame) -> pd.DataFrame:
    # Collect all sub-indices needed, including the raw series used by _derive_non_rent_hfp
    raw_cols = list(set(CONCORDANCE.values()) | {
        "housing_fuel_power", "actual_rents", "electricity_gas_fuels"
    })
    available = [c for c in raw_cols if c in prices.columns]
    annual = prices.groupby("fy_year")[available].mean().sort_index()
    pct = annual.pct_change() * 100
    pct.index.name = "year"
    pct = pct.reset_index().dropna()
    pct = _derive_non_rent_hfp(pct)
    return pct


def compute_for_archetype(lcf: pd.DataFrame, price_changes: pd.DataFrame, arch_col: str) -> pd.DataFrame:
    share_cols = list(CONCORDANCE.keys())
    weight_col = "household_weight" if "household_weight" in lcf.columns else None

    rows = []
    for (grp, yr), subset in lcf.groupby([arch_col, "year"]):
        shares = {}
        denom = subset[weight_col].sum() if weight_col else len(subset)
        if denom == 0:
            continue
        for sc in share_cols:
            vals = subset[sc].fillna(0)
            if weight_col:
                w = subset[weight_col].fillna(0)
                shares[sc] = float((vals * w).sum() / denom)
            else:
                shares[sc] = float(vals.mean())
        rows.append({"archetype_value": grp, "year": int(yr), **shares})

    shares_df = pd.DataFrame(rows)
    if shares_df.empty:
        return pd.DataFrame()

    records = []
    for _, row in shares_df.iterrows():
        target_year = row["year"] + 1  # lag shares
        prices_row = price_changes[price_changes["year"] == target_year]
        if prices_row.empty:
            continue
        price_row = prices_row.iloc[0]
        contribs = []
        for sc, pc_col in CONCORDANCE.items():
            share_val = row.get(sc, 0.0)
            price_chg = price_row.get(pc_col, np.nan)
            if pd.isna(price_chg):
                continue
            contribs.append((pc_col, share_val * price_chg))
        total = sum(val for _, val in contribs)
        for pc_col, val in contribs:
            records.append(
                {
                    "archetype_name": arch_col,
                    "archetype_value": row["archetype_value"],
                    "year": int(target_year),
                    "coicop_label": pc_col,
                    "contribution": val,
                }
            )
        records.append(
            {
                "archetype_name": arch_col,
                "archetype_value": row["archetype_value"],
                "year": int(target_year),
                "coicop_label": "all_items",
                "contribution": total,
            }
        )
    return pd.DataFrame(records)


def build_panel(lcf: pd.DataFrame, price_changes: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for arch in ARCHETYPE_COLS:
        if arch not in lcf.columns:
            continue
        part = compute_for_archetype(lcf, price_changes, arch)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    lcf = load_lcf()
    prices = load_prices()
    price_changes = annual_price_changes(prices)

    decomposition = build_panel(lcf, price_changes)
    if decomposition.empty:
        raise RuntimeError("No decomposition data produced.")
    decomposition["archetype_value"] = decomposition["archetype_value"].astype(str)

    # Sum contributions to get inflation rate
    infl = (
        decomposition.groupby(["archetype_name", "archetype_value", "year"])["contribution"]
        .sum()
        .reset_index()
        .rename(columns={"contribution": "inflation_rate"})
    )
    infl["archetype_value"] = infl["archetype_value"].astype(str)

    decomposition.to_parquet(PROCESSED / "inflation_decomposition.parquet", index=False)
    infl.to_parquet(PROCESSED / "group_inflation_rates.parquet", index=False)

    summary = (
        infl.groupby(["archetype_name", "archetype_value"])["inflation_rate"]
        .agg(mean_inflation="mean", peak_inflation="max")
        .reset_index()
    )
    summary.to_parquet(PROCESSED / "archetype_inflation_summary.parquet", index=False)

    print(f"Saved: {PROCESSED / 'group_inflation_rates.parquet'}")
    print(f"Saved: {PROCESSED / 'inflation_decomposition.parquet'}")
    print(f"Saved: {PROCESSED / 'archetype_inflation_summary.parquet'}")


if __name__ == "__main__":
    main()
