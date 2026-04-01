"""
visualise_inflation.py
======================
Five-phase visualisation strategy for the inflation heterogeneity project.

  Phase 1 – Sample and Coverage
  Phase 2 – Expenditure Heterogeneity
  Phase 3 – The Price Environment
  Phase 4 – Group-Specific Inflation
  Phase 5 – Validation Against HCI

Charts saved as PNG at 150 dpi into phase subfolders under data/processed/charts/:
  01_data_quality/ · 02_expenditure_patterns/ · 03_price_environment/
  04_group_inflation/ · 05_hci_validation/
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHARTS    = PROCESSED / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# One subfolder per narrative phase
_P1 = CHARTS / "01_data_quality"
_P2 = CHARTS / "02_expenditure_patterns"
_P3 = CHARTS / "03_price_environment"
_P4 = CHARTS / "04_group_inflation"
_P5 = CHARTS / "05_hci_validation"
for _d in (_P1, _P2, _P3, _P4, _P5):
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# ── Constants ─────────────────────────────────────────────────────────────

MAIN_SHARE_COLS = [
    "share_01_food_non_alcoholic",
    "share_02_alcohol_tobacco",
    "share_03_clothing_footwear",
    "share_04_housing_fuel_power",
    "share_05_furnishings",
    "share_06_health",
    "share_07_transport",
    "share_08_communication",
    "share_09_recreation_culture",
    "share_10_education",
    "share_11_restaurants_hotels",
    "share_12_misc_goods_services",
]

COICOP_SHORT = {
    "share_01_food_non_alcoholic":  "Food & NA Bev.",
    "share_02_alcohol_tobacco":     "Alcohol & Tobacco",
    "share_03_clothing_footwear":   "Clothing",
    "share_04_housing_fuel_power":  "Housing & Fuel",
    "share_05_furnishings":         "Furnishings",
    "share_06_health":              "Health",
    "share_07_transport":           "Transport",
    "share_08_communication":       "Communication",
    "share_09_recreation_culture":  "Recreation & Culture",
    "share_10_education":           "Education",
    "share_11_restaurants_hotels":  "Restaurants & Hotels",
    "share_12_misc_goods_services": "Misc. Goods & Services",
}

PRICE_SHORT = {
    "food_non_alcoholic":    "Food & NA Bev.",
    "alcohol_tobacco":       "Alcohol & Tobacco",
    "clothing_footwear":     "Clothing",
    "actual_rents":           "Actual Rent",
    "non_rent_housing_fuel":  "Non-Rent Housing & Fuel",
    "housing_fuel_power":     "Housing, Fuel & Power (COICOP 04)",
    "electricity_gas_fuels":  "Energy (elec/gas only)",
    "furnishings":           "Furnishings",
    "health":                "Health",
    "transport":             "Transport",
    "communication":         "Communication",
    "recreation_culture":    "Recreation & Culture",
    "education":             "Education",
    "restaurants_hotels":    "Restaurants & Hotels",
    "misc_goods_services":   "Misc. Goods & Services",
}
# Columns used in the Laspeyres calculation (mirrors CONCORDANCE in compute_group_inflation.py)
PRICE_COLS = [
    "food_non_alcoholic", "alcohol_tobacco", "clothing_footwear",
    "actual_rents", "non_rent_housing_fuel", "furnishings", "health",
    "transport", "communication", "recreation_culture", "education",
    "restaurants_hotels", "misc_goods_services",
]
# All available sub-indices, used only for the descriptive Phase 3 charts
ALL_PRICE_COLS = list(PRICE_SHORT.keys())

QUINTILE_LABELS = {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4", "5.0": "Q5"}

ARCHETYPE_TITLES = {
    "income_quintile":   "Income Quintile",
    "tenure_type":       "Tenure Type",
    "is_pensioner":      "Pensioner Status",
    "hh_composition":    "Household Composition",
    "employment_status": "Employment Status",
    "care_impacted":     "Care-Impacted Status",
    "hrp_age_band":      "HRP Age Band",
    "region_broad":      "Broad Region",
}

CRISIS_YEARS = (2022, 2023)
_TAB20 = plt.get_cmap("tab20")
_TAB10 = plt.get_cmap("tab10")


# ── Helpers ───────────────────────────────────────────────────────────────

def _fy(yr: int) -> str:
    return f"{yr}/{str(yr + 1)[-2:]}"


def _save(fig: plt.Figure, name: str, subdir: pathlib.Path = CHARTS) -> None:
    path = subdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(CHARTS)}")


def _wmean(df: pd.DataFrame, cols: list, w_col: str | None) -> dict:
    """Weighted mean for each column in a group DataFrame."""
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        v = df[c].fillna(0)
        if w_col and w_col in df.columns:
            w = df[w_col].fillna(0)
            denom = w.sum()
            out[c] = (v * w).sum() / denom if denom > 0 else np.nan
        else:
            out[c] = v.mean()
    return out


def _chain_index(rates: pd.Series) -> pd.Series:
    """Chain-link annual % rates into an index where the base year = 100.

    result[yr] = cumulative price level AFTER applying yr's rate.
    A pre-base point at (first_year - 1) = 100 is prepended so the
    starting level is visible on the chart.
    """
    sorted_years = sorted(rates.dropna().index)
    if not sorted_years:
        return pd.Series(dtype=float)
    result = {sorted_years[0] - 1: 100.0}
    level = 100.0
    for yr in sorted_years:
        level = level * (1 + rates[yr] / 100)
        result[yr] = level
    return pd.Series(result)


def _cpih_headline(monthly: pd.DataFrame) -> pd.Series:
    """Annual CPIH all_items % change using financial-year (Apr–Mar) averages.
    Consistent with how compute_group_inflation.py aggregates prices after the FY fix."""
    annual = monthly.groupby("fy_year")["all_items"].mean()
    return (annual.pct_change() * 100).dropna().rename("CPIH Headline")


def _sort_arch_values(arch: str, vals) -> list:
    """Sensible display order for archetype values."""
    vals = list(vals)
    if arch == "income_quintile":
        return sorted(vals, key=lambda x: float(x))
    if arch == "hrp_age_band":
        order = ["under_30", "30_to_49", "50_to_64", "65_to_74", "75_plus"]
        return [v for v in order if v in vals] + [v for v in vals if v not in order]
    if arch in ("is_pensioner", "care_impacted"):
        return [v for v in ["True", "False"] if v in vals] + [
            v for v in vals if v not in ("True", "False")
        ]
    if arch == "tenure_type":
        order = ["social_rent", "private_rent", "own_outright", "own_mortgage", "rent_free"]
        return [v for v in order if v in vals] + [v for v in vals if v not in order]
    return sorted(vals)


def _label(arch: str, val: str) -> str:
    if arch == "income_quintile":
        return QUINTILE_LABELS.get(val, val)
    if arch == "is_pensioner":
        return "Pensioner" if val == "True" else "Non-Pensioner"
    if arch == "care_impacted":
        return "Care-Impacted" if val == "True" else "No Care Impact"
    return val.replace("_", " ").title()


def _shade_crisis(ax: plt.Axes) -> None:
    ax.axvspan(CRISIS_YEARS[0] - 0.5, CRISIS_YEARS[1] + 0.5,
               alpha=0.12, color="tomato", label="_nolegend_")


def _colour(i: int, n: int):
    cmap = _TAB10 if n <= 10 else _TAB20
    return cmap(i / max(n - 1, 1))


# ── Data loading ──────────────────────────────────────────────────────────

def load_data():
    infl    = pd.read_parquet(PROCESSED / "group_inflation_rates.parquet")
    decomp  = pd.read_parquet(PROCESSED / "inflation_decomposition.parquet")
    shares  = pd.read_parquet(PROCESSED / "lcf_expenditure_shares.parquet")
    monthly = pd.read_parquet(PROCESSED / "cpih_monthly_indices.parquet")
    monthly["date"] = pd.to_datetime(monthly["date"])
    fy_idx  = pd.read_parquet(PROCESSED / "cpih_annual_fy_indices.parquet")
    hci     = pd.read_parquet(PROCESSED / "hci_validation.parquet")
    hci["date"] = pd.to_datetime(hci["date"])
    # Normalise archetype_value to string throughout
    infl["archetype_value"]   = infl["archetype_value"].astype(str)
    decomp["archetype_value"] = decomp["archetype_value"].astype(str)
    return infl, decomp, shares, monthly, fy_idx, hci


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Sample and Coverage
# ═══════════════════════════════════════════════════════════════════════════

def p1_sample_size(shares: pd.DataFrame) -> None:
    print("\n[P1-1] LCF sample size by year")
    counts = shares.groupby("year").size().reset_index(name="n")
    years, ns = counts["year"].values, counts["n"].values
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(years, ns, color="steelblue", width=0.6)
    for bar, yr, n in zip(bars, years, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 20, f"{n:,}",
                ha="center", va="bottom", fontsize=7)
        if yr in (2019, 2020):
            ax.annotate("COVID\ndisruption",
                        xy=(yr, n), xytext=(yr + 0.05, n + 200),
                        ha="left", fontsize=7, color="tomato",
                        arrowprops=dict(arrowstyle="-", color="tomato", lw=0.8))
    ax.set_xlabel("LCF financial year (start year)")
    ax.set_ylabel("Number of households")
    ax.set_title("LCF Sample Size by Financial Year")
    ax.set_xticks(years)
    ax.set_xticklabels([_fy(y) for y in years], rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "p1_1_sample_size.png", _P1)


def p1_cell_sizes(shares: pd.DataFrame) -> None:
    print("\n[P1-2] Archetype cell sizes heatmap")
    arch_cols = [c for c in ARCHETYPE_TITLES if c in shares.columns and c != "region_broad"]
    years = sorted(shares["year"].unique())
    rows = []
    for arch in arch_cols:
        for val, grp in shares.groupby(arch):
            for yr, sub in grp.groupby("year"):
                rows.append({"dim": f"{_label(arch, str(val))}\n({arch})", "year": int(yr), "n": len(sub)})
    if not rows:
        return
    df = pd.DataFrame(rows).pivot(index="dim", columns="year", values="n").fillna(0)
    df = df[[c for c in years if c in df.columns]]
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns)), max(8, len(df) * 0.42)))
    im = ax.imshow(df.values, aspect="auto", cmap="YlOrRd")
    for i in range(len(df)):
        for j in range(len(df.columns)):
            v = int(df.iloc[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=6, color="white" if v < 80 else "black")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([_fy(c) for c in df.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=6.5)
    ax.set_title("LCF Cell Sizes by Archetype Group and Year\n(white = below 80 households)")
    fig.colorbar(im, ax=ax, label="Household count")
    fig.tight_layout()
    _save(fig, "p1_2_cell_sizes.png", _P1)


def p1_weight_distribution(shares: pd.DataFrame) -> None:
    print("\n[P1-3] Household weight distribution by year")
    if "household_weight" not in shares.columns:
        print("  No household_weight column, skipping.")
        return
    years = sorted(shares["year"].unique())
    data = [shares.loc[shares["year"] == yr, "household_weight"].dropna().values
            for yr in years]
    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot(data, positions=range(len(years)), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([_fy(y) for y in years], rotation=30, ha="right")
    ax.set_ylabel("Survey gross-up weight")
    ax.set_title("Household Weight Distribution by Financial Year")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "p1_3_weight_distribution.png", _P1)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Expenditure Heterogeneity
# ═══════════════════════════════════════════════════════════════════════════

def _draw_stacked_basket(ax: plt.Axes, group_labels: list, comp: dict, cols: list) -> None:
    """Draw stacked bars; comp[group_label][col] = mean share."""
    bottom = np.zeros(len(group_labels))
    x = np.arange(len(group_labels))
    for j, col in enumerate(cols):
        vals = np.array([comp.get(g, {}).get(col, 0.0) for g in group_labels])
        ax.bar(x, vals, bottom=bottom, color=_TAB20(j / len(cols)),
               label=COICOP_SHORT.get(col, col), width=0.65)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Expenditure share (weighted mean)")
    ax.legend(title="COICOP", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)


def p2_basket_by_quintile(shares: pd.DataFrame) -> None:
    print("\n[P2-4] Basket composition by income quintile")
    if "income_quintile" not in shares.columns:
        return
    w = "household_weight" if "household_weight" in shares.columns else None
    cols = [c for c in MAIN_SHARE_COLS if c in shares.columns]
    quintiles = sorted(shares["income_quintile"].dropna().unique(), key=float)
    labels = [QUINTILE_LABELS.get(str(q), str(q)) for q in quintiles]
    comp = {QUINTILE_LABELS.get(str(q), str(q)): _wmean(shares[shares["income_quintile"] == q], cols, w)
            for q in quintiles}
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_stacked_basket(ax, labels, comp, cols)
    ax.set_title("Mean Basket Composition by Income Quintile\n(Weighted expenditure shares, all years pooled)")
    ax.set_xlabel("Income quintile")
    fig.tight_layout()
    _save(fig, "p2_4_basket_income_quintile.png", _P2)


def p2_basket_by_tenure(shares: pd.DataFrame) -> None:
    print("\n[P2-5] Basket composition by tenure type")
    if "tenure_type" not in shares.columns:
        return
    w = "household_weight" if "household_weight" in shares.columns else None
    cols = [c for c in MAIN_SHARE_COLS if c in shares.columns]
    order = ["social_rent", "private_rent", "own_outright", "own_mortgage", "rent_free"]
    groups = [g for g in order if g in shares["tenure_type"].unique()]
    labels = [g.replace("_", " ").title() for g in groups]
    comp = {g.replace("_", " ").title(): _wmean(shares[shares["tenure_type"] == g], cols, w)
            for g in groups}
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_stacked_basket(ax, labels, comp, cols)
    ax.set_title("Mean Basket Composition by Tenure Type\n(Weighted expenditure shares, all years pooled)")
    ax.set_xlabel("Tenure type")
    fig.tight_layout()
    _save(fig, "p2_5_basket_tenure.png", _P2)


def p2_food_energy_density(shares: pd.DataFrame) -> None:
    print("\n[P2-6] Food + energy share density by income quintile")
    needed = {"income_quintile", "share_01_food_non_alcoholic", "share_04_energy_other"}
    if not needed.issubset(shares.columns):
        print("  Missing required columns, skipping.")
        return
    sub = shares.dropna(subset=list(needed)).copy()
    sub["food_energy"] = sub["share_01_food_non_alcoholic"] + sub["share_04_energy_other"]
    quintiles = sorted(sub["income_quintile"].unique(), key=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, q in enumerate(quintiles):
        vals = sub.loc[sub["income_quintile"] == q, "food_energy"].dropna().values
        counts, edges = np.histogram(vals, bins=80, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        # Gaussian smoothing so the line is smooth rather than a jagged histogram step
        kernel_width = max(1, len(counts) // 15)
        kernel = np.exp(-0.5 * (np.arange(-kernel_width * 2, kernel_width * 2 + 1) / kernel_width) ** 2)
        kernel /= kernel.sum()
        smoothed = np.convolve(counts, kernel, mode="same")
        ax.plot(centres, smoothed, label=QUINTILE_LABELS.get(str(q), str(q)),
                color=_colour(i, len(quintiles)), linewidth=1.8)
    ax.set_xlabel("Food + Non-Rent Housing (share_01 + share_04_energy_other)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Food & Non-Rent Housing Share by Income Quintile\n"
                 "(Q1 concentrated at structurally higher shares than Q5)")
    ax.legend(title="Quintile")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "p2_6_food_energy_density.png", _P2)


def p2_basket_shift(shares: pd.DataFrame) -> None:
    print("\n[P2-7] Basket shift over time — Q1 vs Q5 small multiples")
    if "income_quintile" not in shares.columns:
        return
    w = "household_weight" if "household_weight" in shares.columns else None
    cols = [c for c in MAIN_SHARE_COLS if c in shares.columns]
    years = sorted(shares["year"].unique())
    targets = {1.0: ("Q1", "steelblue"), 5.0: ("Q5", "tomato")}
    # Pre-compute (quintile, year) weighted mean shares
    records = []
    for q, (qlabel, _) in targets.items():
        for yr in years:
            sub = shares[(shares["income_quintile"] == q) & (shares["year"] == yr)]
            if sub.empty:
                continue
            row = _wmean(sub, cols, w)
            row.update({"year": yr, "quintile": qlabel})
            records.append(row)
    df = pd.DataFrame(records)
    if df.empty:
        return
    ncols_g, nrows_g = 4, (len(cols) + 3) // 4
    fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(14, nrows_g * 3), sharex=True)
    for idx, col in enumerate(cols):
        ax = axes.flat[idx]
        for q, (qlabel, colour) in targets.items():
            sub = df[df["quintile"] == qlabel][["year", col]].sort_values("year")
            if not sub.empty and col in sub.columns:
                ax.plot(sub["year"], sub[col], marker="o", color=colour,
                        label=qlabel, linewidth=1.5, markersize=3)
        ax.set_title(COICOP_SHORT.get(col, col), fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=7)
    for idx in range(len(cols), len(axes.flat)):
        axes.flat[idx].set_visible(False)
    handles = [plt.Line2D([0], [0], color=c, label=l, lw=1.5)
               for _, (l, c) in targets.items()]
    fig.legend(handles=handles, title="Quintile", loc="lower right", fontsize=8)
    fig.suptitle("Basket Shift Over Time: Q1 vs Q5 by COICOP Division (2015–2023)", y=1.01)
    fig.tight_layout()
    _save(fig, "p2_7_basket_shift.png", _P2)


def p2_share_correlation(shares: pd.DataFrame) -> None:
    print("\n[P2-8] Pairwise share correlation heatmap")
    cols = [c for c in MAIN_SHARE_COLS if c in shares.columns]
    corr = shares[cols].corr()
    short = [COICOP_SHORT.get(c, c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title("Pairwise Correlation of COICOP Expenditure Shares\n(All households, 2015–2023)")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    _save(fig, "p2_8_share_correlation.png", _P2)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — The Price Environment
# ═══════════════════════════════════════════════════════════════════════════

def p3_monthly_index(monthly: pd.DataFrame) -> None:
    print("\n[P3-9] CPIH sub-index time series rebased to Jan 2015 = 100")
    cols = [c for c in ALL_PRICE_COLS if c in monthly.columns]
    base_row = monthly[monthly["date"] == "2015-01-01"]
    if base_row.empty:
        base_row = monthly.sort_values("date").iloc[[0]]
    base_vals = {c: float(base_row[c].iloc[0]) for c in cols if base_row[c].iloc[0] != 0}
    ms = monthly.sort_values("date")
    highlight = {"food_non_alcoholic", "electricity_gas_fuels"}
    fig, ax = plt.subplots(figsize=(13, 6))
    for col in cols:
        if base_vals.get(col, 0) == 0:
            continue
        rebased = ms[col] / base_vals[col] * 100
        if col in highlight:
            ax.plot(ms["date"], rebased, linewidth=2.2,
                    label=PRICE_SHORT[col], zorder=3)
        else:
            ax.plot(ms["date"], rebased, linewidth=0.9,
                    color="lightgrey", alpha=0.8, zorder=1)
    grey_handle = plt.Line2D([0], [0], linewidth=0.9, color="lightgrey",
                             label="Other categories")
    highlight_handles = [h for h in ax.get_legend_handles_labels()[0]]
    ax.legend(handles=highlight_handles + [grey_handle], fontsize=8)
    ax.axhline(100, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylabel("Index (Jan 2015 = 100)")
    ax.set_title("CPIH Sub-Index Monthly Series (Jan 2015 = 100)\nFood & energy highlighted; other categories in grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "p3_9_monthly_index.png", _P3)


def p3_annual_price_change(fy_idx: pd.DataFrame) -> None:
    print("\n[P3-10] Annual price change by COICOP division (grouped bar)")
    cols = [c for c in ALL_PRICE_COLS if c in fy_idx.columns]
    pct = fy_idx.sort_values("year").set_index("year")[cols].pct_change() * 100
    pct = pct.dropna()
    years = pct.index.tolist()
    n_cols, n_years = len(cols), len(years)
    width = 0.7 / n_cols
    x = np.arange(n_years)
    fig, ax = plt.subplots(figsize=(14, 6))
    for j, col in enumerate(cols):
        offset = (j - n_cols / 2 + 0.5) * width
        ax.bar(x + offset, pct[col], width=width,
               color=_TAB20(j / n_cols), label=PRICE_SHORT.get(col, col))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_fy(y) for y in years], rotation=30, ha="right")
    ax.set_ylabel("Annual % change")
    ax.set_title("Annual Price Change by COICOP Sub-Index (FY averages)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    _save(fig, "p3_10_annual_price_change.png", _P3)


def p3_volatility_ranking(fy_idx: pd.DataFrame) -> None:
    print("\n[P3-11] Price volatility ranking")
    cols = [c for c in ALL_PRICE_COLS if c in fy_idx.columns]
    pct = fy_idx.sort_values("year").set_index("year")[cols].pct_change() * 100
    stds = pct.dropna().std().rename(index=PRICE_SHORT).sort_values()
    median_std = stds.median()
    colours = ["tomato" if s >= median_std else "steelblue" for s in stds.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(stds.index, stds.values, color=colours)
    ax.set_xlabel("Std dev of annual % changes (2016–2024)")
    ax.set_title("CPIH Sub-Index Price Volatility\n(Standard deviation of annual price changes; red = above median)")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "p3_11_volatility_ranking.png", _P3)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — Group-Specific Inflation
# ═══════════════════════════════════════════════════════════════════════════

def p4_all_group_series(infl: pd.DataFrame, headline: pd.Series) -> None:
    print("\n[P4] Group-specific inflation series (all archetypes)")
    for arch in [a for a in ARCHETYPE_TITLES if a in infl["archetype_name"].unique()]:
        sub = infl[infl["archetype_name"] == arch]
        vals = _sort_arch_values(arch, sub["archetype_value"].unique())
        years = sorted(sub["year"].unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, v in enumerate(vals):
            grp = sub[sub["archetype_value"] == v].sort_values("year")
            ax.plot(grp["year"], grp["inflation_rate"], marker="o",
                    color=_colour(i, len(vals)), linewidth=1.8,
                    label=_label(arch, v), zorder=3)
        common = [y for y in years if y in headline.index]
        ax.plot(common, headline.reindex(common), color="black", linewidth=1.5,
                linestyle="--", label="CPIH Headline", zorder=4)
        _shade_crisis(ax)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual inflation rate (%)")
        ax.set_title(f"Group-Specific Inflation: {ARCHETYPE_TITLES[arch]}\n"
                     "(Laspeyres, dashed = CPIH headline, shaded = crisis 2022–23)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(years)
        ax.set_xticklabels([_fy(y) for y in years], rotation=30, ha="right")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        _save(fig, f"p4_series_{arch}.png", _P4)


def _gap_bar(infl: pd.DataFrame, headline: pd.Series, arch: str,
             hi_grp: str, lo_grp: str) -> None:
    """Bar chart of (hi_grp − lo_grp) annual inflation gap."""
    sub = infl[infl["archetype_name"] == arch]
    years = sorted(sub["year"].unique())
    hi_r = sub[sub["archetype_value"] == hi_grp].set_index("year")["inflation_rate"]
    lo_r = sub[sub["archetype_value"] == lo_grp].set_index("year")["inflation_rate"]
    gaps = (hi_r - lo_r).reindex(years)
    fig, ax = plt.subplots(figsize=(10, 4))
    colours = ["tomato" if g >= 0 else "steelblue" for g in gaps.fillna(0)]
    ax.bar(years, gaps.values, color=colours, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    _shade_crisis(ax)
    ax.set_xticks(years)
    ax.set_xticklabels([_fy(y) for y in years], rotation=30, ha="right")
    ax.set_ylabel("Inflation gap (pp)")
    ax.set_title(f"Annual Inflation Gap: {_label(arch, hi_grp)} minus {_label(arch, lo_grp)}\n"
                 f"(red = {_label(arch, hi_grp)} pays more; shaded = crisis period)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, f"p4_gap_{arch}.png", _P4)


def _crisis_decomp(decomp: pd.DataFrame, arch: str, groups: list) -> None:
    """Paired horizontal stacked bar for peak crisis year."""
    sub = decomp[
        (decomp["archetype_name"] == arch) &
        (decomp["year"] == CRISIS_YEARS[0]) &
        (decomp["coicop_label"] != "all_items")
    ].copy()
    if sub.empty:
        return
    cats = sorted(sub["coicop_label"].unique())
    cat_colour = {c: _TAB20(i / max(len(cats) - 1, 1)) for i, c in enumerate(cats)}
    group_labels = [_label(arch, g) for g in groups]
    fig, ax = plt.subplots(figsize=(10, max(3, len(groups) * 0.9)))
    left = np.zeros(len(groups))
    for cat in cats:
        vals = []
        for g in groups:
            row = sub[(sub["archetype_value"] == g) & (sub["coicop_label"] == cat)]
            vals.append(float(row["contribution"].iloc[0]) if not row.empty else 0.0)
        ax.barh(group_labels, vals, left=left,
                color=cat_colour[cat], label=PRICE_SHORT.get(cat, cat), height=0.6)
        left += np.array(vals)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to inflation (pp)")
    ax.set_title(f"Crisis Decomposition {_fy(CRISIS_YEARS[0])}: COICOP Contributions\n"
                 f"{ARCHETYPE_TITLES.get(arch, arch)}")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    _save(fig, f"p4_decomp_{arch}.png", _P4)


def _cumulative_ppp(infl: pd.DataFrame, headline: pd.Series, arch: str) -> None:
    """Cumulative purchasing power index rebased to 100 before first year of data."""
    sub = infl[infl["archetype_name"] == arch]
    vals = _sort_arch_values(arch, sub["archetype_value"].unique())
    years = sorted(sub["year"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(vals):
        rates = sub[sub["archetype_value"] == v].set_index("year")["inflation_rate"].reindex(years)
        idx = _chain_index(rates.dropna())
        ax.plot(idx.index, idx.values, marker="o", color=_colour(i, len(vals) + 1),
                linewidth=1.8, label=_label(arch, v), zorder=3)
    hl_idx = _chain_index(headline.reindex(years).dropna())
    ax.plot(hl_idx.index, hl_idx.values, color="black", linewidth=1.8,
            linestyle="--", label="CPIH Headline", zorder=4)
    _shade_crisis(ax)
    ax.axhline(100, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative price level (base = 100)")
    base_yr = years[0] - 1
    ax.set_title(f"Cumulative Purchasing Power Loss: {ARCHETYPE_TITLES.get(arch, arch)}\n"
                 f"(base = 100 in {_fy(base_yr)}; gap compounds and does not close)")
    all_x = sorted(hl_idx.index.tolist())
    ax.set_xticks(all_x)
    ax.set_xticklabels([_fy(y) for y in all_x], rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, f"p4_cumulative_{arch}.png", _P4)


def _select_main_archetypes(infl: pd.DataFrame) -> list:
    """Main-body archetypes for the report.

    These three are fixed by policy relevance rather than auto-selected by gap size:
      - income_quintile : broadest distributional statement (the core thesis finding)
      - hrp_age_band    : largest absolute gap (~3pp Q1 vs Q5 energy exposure); intergenerational
      - is_pensioner    : triple-lock policy link; State Pension uprated by headline CPI

    Auto-selecting by max-gap across all groups is unreliable because small-cell outlier
    groups (rent_free ~50 hh/yr, has_unemployed) inflate the spread for tenure_type and
    employment_status without reflecting genuine structural inequality.
    """
    preferred = ["income_quintile", "hrp_age_band", "is_pensioner"]
    available = infl["archetype_name"].unique()
    return [a for a in preferred if a in available]


def p4_main_body_extras(infl: pd.DataFrame, decomp: pd.DataFrame,
                        headline: pd.Series) -> None:
    """Gap bar + crisis decomposition + cumulative PPP for top archetypes."""
    print("\n[P4] Main-body extras for top archetypes")
    main_archs = _select_main_archetypes(infl)
    print(f"  Selected: {main_archs}")
    for arch in main_archs:
        sub = infl[infl["archetype_name"] == arch]
        # Identify high-inflation and low-inflation groups by average rate
        avg = sub.groupby("archetype_value")["inflation_rate"].mean()
        hi_grp, lo_grp = avg.idxmax(), avg.idxmin()
        if hi_grp == lo_grp:
            continue
        _gap_bar(infl, headline, arch, hi_grp, lo_grp)
        _crisis_decomp(decomp, arch, [hi_grp, lo_grp])
        _cumulative_ppp(infl, headline, arch)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — Validation Against HCI
# ═══════════════════════════════════════════════════════════════════════════

def _hci_fy_rates(hci: pd.DataFrame, grouping: str) -> pd.DataFrame:
    """Annual FY inflation rates from HCI index data (pct change of FY mean index)."""
    sub = hci[
        (hci["grouping"] == grouping) &
        (hci["metric"] == "index") &
        (hci["coicop_code"].isin(["0", "00"]))
    ]
    if sub.empty:
        return pd.DataFrame()
    fy_mean = sub.groupby(["group", "fy_year"])["value"].mean().reset_index()
    fy_mean = fy_mean.sort_values(["group", "fy_year"])
    fy_mean["rate"] = fy_mean.groupby("group")["value"].pct_change() * 100
    return fy_mean.dropna(subset=["rate"])


def _plot_hci_comparison(ax: plt.Axes,
                         hci_rates: pd.DataFrame, hci_groups: list,
                         infl: pd.DataFrame, arch: str, my_map: dict,
                         headline: pd.Series) -> None:
    """Draw my estimates (solid) and HCI (dashed) on a shared axis."""
    palette = iter([f"C{i}" for i in range(8)])
    for my_val, label in my_map.items():
        c = next(palette)
        sub = infl[(infl["archetype_name"] == arch) &
                   (infl["archetype_value"] == my_val)].sort_values("year")
        ax.plot(sub["year"], sub["inflation_rate"], marker="o", color=c,
                linewidth=1.8, label=f"My: {label}", zorder=3)
    for hci_grp in hci_groups:
        c = next(palette)
        sub = hci_rates[hci_rates["group"] == hci_grp].sort_values("fy_year")
        # HCI fy_year=2015 = FY2015/16; rate applies from FY2015/16 → FY2016/17 → year=2016
        ax.plot(sub["fy_year"] + 1, sub["rate"], marker="s", color=c,
                linewidth=1.5, linestyle="--", label=f"HCI: {hci_grp}", alpha=0.85)
    ax.plot(headline.index, headline.values, color="black", linewidth=1.2,
            linestyle=":", label="CPIH Headline")
    _shade_crisis(ax)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual inflation rate (%)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")


def p5_income_comparison(infl: pd.DataFrame, hci: pd.DataFrame,
                         headline: pd.Series) -> None:
    print("\n[P5-19] Income comparison: Q1/Q5 vs HCI Decile 1/10")
    hci_rates = _hci_fy_rates(hci, "income_decile")
    if hci_rates.empty:
        print("  No HCI income_decile data, skipping.")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_hci_comparison(ax, hci_rates, ["Decile 1", "Decile 10"],
                         infl, "income_quintile",
                         {"1.0": "Q1 (Lowest)", "5.0": "Q5 (Highest)"},
                         headline)
    ax.set_title("Validation — Income: My Q1/Q5 vs HCI Decile 1/10\n"
                 "(solid = my estimates, dashed = HCI, dotted = CPIH headline)")
    fig.tight_layout()
    _save(fig, "p5_19_income_comparison.png", _P5)


def p5_tenure_comparison(infl: pd.DataFrame, hci: pd.DataFrame,
                         headline: pd.Series) -> None:
    print("\n[P5-20] Tenure comparison vs HCI")
    hci_rates = _hci_fy_rates(hci, "tenure")
    if hci_rates.empty:
        print("  No HCI tenure data, skipping.")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_hci_comparison(ax, hci_rates,
                         ["Private renter", "Outright owner occupier"],
                         infl, "tenure_type",
                         {"private_rent": "Private Rent", "own_outright": "Own Outright"},
                         headline)
    ax.set_title("Validation — Tenure: My Estimates vs HCI\n"
                 "(solid = my estimates, dashed = HCI, dotted = CPIH headline)")
    fig.tight_layout()
    _save(fig, "p5_20_tenure_comparison.png", _P5)


def p5_pensioner_comparison(infl: pd.DataFrame, hci: pd.DataFrame,
                             headline: pd.Series) -> None:
    print("\n[P5-21] Pensioner comparison vs HCI Retired/Non-Retired")
    hci_rates = _hci_fy_rates(hci, "retirement")
    if hci_rates.empty:
        print("  No HCI retirement data, skipping.")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_hci_comparison(ax, hci_rates, ["Retired", "Non-Retired"],
                         infl, "is_pensioner",
                         {"True": "Pensioner", "False": "Non-Pensioner"},
                         headline)
    ax.set_title("Validation — Pensioner Status: My Estimates vs HCI Retired/Non-Retired\n"
                 "(solid = my estimates, dashed = HCI, dotted = CPIH headline)")
    fig.tight_layout()
    _save(fig, "p5_21_pensioner_comparison.png", _P5)


def p5_residual_scatter(infl: pd.DataFrame, hci: pd.DataFrame) -> None:
    print("\n[P5-22] Residual scatter: my estimates vs HCI")
    mappings = [
        ("income_quintile", "income_decile",
         {"1.0": "Decile 1", "5.0": "Decile 10"}),
        ("tenure_type", "tenure", {
            "private_rent":  "Private renter",
            "own_outright":  "Outright owner occupier",
            "own_mortgage":  "Mortgagor and other owner occupier",
            "social_rent":   "Social and other renter",
        }),
        ("is_pensioner", "retirement",
         {"True": "Retired", "False": "Non-Retired"}),
    ]
    records = []
    for arch, hci_grouping, val_map in mappings:
        hci_rates = _hci_fy_rates(hci, hci_grouping)
        if hci_rates.empty:
            continue
        my = infl[infl["archetype_name"] == arch]
        for my_val, hci_grp in val_map.items():
            my_sub = my[my["archetype_value"] == my_val][["year", "inflation_rate"]]
            hci_sub = hci_rates[hci_rates["group"] == hci_grp][["fy_year", "rate"]].copy()
            hci_sub["year"] = hci_sub["fy_year"] + 1
            merged = my_sub.merge(hci_sub[["year", "rate"]], on="year", how="inner")
            merged["pair"] = f"{arch}={my_val}"
            records.append(merged)
    if not records:
        print("  No matched pairs, skipping.")
        return
    pts = pd.concat(records, ignore_index=True)
    if pts.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, (label, grp) in enumerate(pts.groupby("pair")):
        ax.scatter(grp["rate"], grp["inflation_rate"],
                   label=label, s=45, alpha=0.75, color=_colour(i, pts["pair"].nunique()))
    lims = [min(pts["rate"].min(), pts["inflation_rate"].min()) - 1,
            max(pts["rate"].max(), pts["inflation_rate"].max()) + 1]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="45° line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("HCI official rate (%)")
    ax.set_ylabel("My Laspeyres estimate (%)")
    ax.set_title("Validation Scatter: My Estimates vs HCI Official\n"
                 "(points on the 45° line = perfect agreement)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "p5_22_residual_scatter.png", _P5)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("Five-Phase Inflation Visualisation Pipeline")
    print("=" * 60)
    print(f"  Output directory: {CHARTS}")
    try:
        infl, decomp, shares, monthly, fy_idx, hci = load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}\nRun the pipeline first (wrangle_mm23 → wrangle_lcf → compute_group_inflation).")
        return
    print(f"  group_inflation_rates : {len(infl):,} rows, "
          f"{infl['archetype_name'].nunique()} archetypes")
    print(f"  lcf_expenditure_shares: {len(shares):,} rows")
    headline = _cpih_headline(monthly)

    print("\n── Phase 1: Sample and Coverage ──")
    p1_sample_size(shares)
    p1_cell_sizes(shares)
    p1_weight_distribution(shares)

    print("\n── Phase 2: Expenditure Heterogeneity ──")
    p2_basket_by_quintile(shares)
    p2_basket_by_tenure(shares)
    p2_food_energy_density(shares)
    p2_basket_shift(shares)
    p2_share_correlation(shares)

    print("\n── Phase 3: Price Environment ──")
    p3_monthly_index(monthly)
    p3_annual_price_change(fy_idx)
    p3_volatility_ranking(fy_idx)

    print("\n── Phase 4: Group-Specific Inflation ──")
    p4_all_group_series(infl, headline)
    p4_main_body_extras(infl, decomp, headline)

    print("\n── Phase 5: HCI Validation ──")
    p5_income_comparison(infl, hci, headline)
    p5_tenure_comparison(infl, hci, headline)
    p5_pensioner_comparison(infl, hci, headline)
    p5_residual_scatter(infl, hci)

    saved = sorted(CHARTS.rglob("*.png"))
    print(f"\nDone. {len(saved)} charts saved to {CHARTS}")


if __name__ == "__main__":
    main()
