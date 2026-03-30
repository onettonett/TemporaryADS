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
    "share_04_energy_other": "electricity_gas_fuels",
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
    "tenure_type",
    "region_broad",
    "hrp_age_band",
    "hh_composition",
    "employment_status",
    "is_pensioner",
    "is_disability",
    "is_carer",
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
    prices["fy_year"] = prices["date"].dt.year
    return prices


def annual_price_changes(prices: pd.DataFrame) -> pd.DataFrame:
    price_cols = list(CONCORDANCE.values())
    annual = prices.groupby("fy_year")[price_cols].mean().sort_index()
    pct = annual.pct_change() * 100
    pct.index.name = "year"
    pct = pct.reset_index().dropna()
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
