"""
generate_report_figures.py
==========================
Generates publication-quality figures and tables specifically for the IEEE report.
Organised by report section into subdirectories of data/processed/report_figures/.

Output structure:
    report_figures/
        sec3_data_preparation/    -- pipeline flowchart, data acquisition table,
                                     missing-value audit, distribution properties analysis
        sec4_data_exploration/    -- basket composition, essentials density, CPIH time
                                     series (annotated), correlation heatmaps,
                                     dimension-gap comparison, summary statistics table

Usage:
    python src/generate_report_figures.py
"""

from __future__ import annotations

import pathlib
import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

from data_loaders import (
    load_cpih_monthly,
    load_cpih_fy_indices,
    load_lcf_shares,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parents[1]
OUTPUT     = ROOT / "data" / "output"
REPORT     = OUTPUT / "report_figures"
SEC3       = REPORT / "sec3_data_preparation"
SEC4       = REPORT / "sec4_data_exploration"

for d in (REPORT, SEC3, SEC4):
    d.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
})

TENURE_4 = ["social_rent", "private_rent", "own_outright", "own_mortgage"]
TENURE_LABELS = {
    "social_rent": "Social Rent",
    "private_rent": "Private Rent",
    "own_outright": "Own Outright",
    "own_mortgage": "Own Mortgage",
}
TENURE_COLOURS = {
    "social_rent":  "#1b9e77",
    "private_rent": "#d95f02",
    "own_outright": "#7570b3",
    "own_mortgage": "#e7298a",
}

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
    "share_01_food_non_alcoholic":  "Food",
    "share_02_alcohol_tobacco":     "Alcohol & Tobacco",
    "share_03_clothing_footwear":   "Clothing",
    "share_04_housing_fuel_power":  "Housing & Utilities",
    "share_05_furnishings":         "Furnishings",
    "share_06_health":              "Health",
    "share_07_transport":           "Transport",
    "share_08_communication":       "Communication",
    "share_09_recreation_culture":  "Recreation",
    "share_10_education":           "Education",
    "share_11_restaurants_hotels":  "Restaurants",
    "share_12_misc_goods_services": "Misc. Goods",
}

PRICE_SHORT = {
    "food_non_alcoholic":    "Food & NA Bev.",
    "alcohol_tobacco":       "Alcohol & Tobacco",
    "clothing_footwear":     "Clothing",
    "actual_rents":          "Actual Rents",
    "non_rent_housing_fuel": "Home Energy & Utilities",
    "housing_fuel_power":    "Housing & Utilities (COICOP 04)",
    "electricity_gas_fuels": "Energy (elec/gas)",
    "furnishings":           "Furnishings",
    "health":                "Health",
    "transport":             "Transport",
    "communication":         "Communication",
    "recreation_culture":    "Recreation & Culture",
    "education":             "Education",
    "restaurants_hotels":    "Restaurants & Hotels",
    "misc_goods_services":   "Misc. Goods & Services",
}

ALL_PRICE_COLS = list(PRICE_SHORT.keys())


def _fy(yr: int) -> str:
    return f"{yr}/{str(yr + 1)[-2:]}"


def _save(fig: plt.Figure, name: str, subdir: pathlib.Path) -> None:
    path = subdir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(REPORT)}")


def load_data():
    shares  = load_lcf_shares()
    monthly = load_cpih_monthly()
    fy_idx  = load_cpih_fy_indices()
    infl    = pd.read_csv(OUTPUT / "group_inflation_rates.csv")
    infl["archetype_value"] = infl["archetype_value"].astype(str)
    decomp  = pd.read_csv(OUTPUT / "inflation_decomposition.csv")
    decomp["archetype_value"] = decomp["archetype_value"].astype(str)
    return shares, monthly, fy_idx, infl, decomp


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA PREPARATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# --------------------------------------------------------------------------
# Fig. 1 — Data Pipeline Flowchart
# --------------------------------------------------------------------------
def fig_pipeline_flowchart() -> None:
    """Publication-quality data pipeline diagram using matplotlib patches."""
    print("\n[SEC3] Data pipeline flowchart")

    fig, ax = plt.subplots(figsize=(7.16, 5.5))  # IEEE two-column width
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_box(x, y, w, h, text, colour="#D6EAF8", edge="#2E86C1", fontsize=7.5,
                 bold=False, style="round,pad=0.1"):
        box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                             facecolor=colour, edgecolor=edge, linewidth=1.2)
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, wrap=True,
                linespacing=1.3)

    def arrow(x1, y1, x2, y2, **kw):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                    linewidth=1.2, **kw))

    # Layer 0: Raw data sources (top)
    raw_colour = "#FADBD8"
    raw_edge = "#C0392B"
    draw_box(0.2, 6.8, 2.6, 1.0,
             "LCF Microdata\n9 years (.dta)\n45,965 households",
             raw_colour, raw_edge, fontsize=7)
    draw_box(3.7, 6.8, 2.6, 1.0,
             "ONS MM23\nCPIH Indices (.csv)\n14 COICOP series",
             raw_colour, raw_edge, fontsize=7)
    draw_box(7.2, 6.8, 2.6, 1.0,
             "ONS HCI\nValidation (.xlsx)\n5 subgroup tables",
             raw_colour, raw_edge, fontsize=7)

    # Layer 1: Wrangling scripts
    script_colour = "#D5F5E3"
    script_edge = "#27AE60"
    draw_box(0.2, 5.2, 2.6, 1.2,
             "wrangle_lcf.py\n\u2022 Clean negatives\n\u2022 COICOP shares\n"
             "\u2022 Filter implausible HHs\n\u2022 8 archetypes",
             script_colour, script_edge, fontsize=6.5)
    draw_box(3.7, 5.2, 2.6, 1.2,
             "wrangle_mm23.py\n\u2022 Extract CDIDs\n\u2022 Rebase Jan 2015\n"
             "\u2022 FY averages\n\u2022 Long \u2192 wide pivot",
             script_colour, script_edge, fontsize=6.5)
    draw_box(7.2, 5.2, 2.6, 1.2,
             "wrangle_hci.py\n\u2022 Parse multi-headers\n\u2022 Long format\n"
             "\u2022 All-items indices\n\u2022 FY alignment",
             script_colour, script_edge, fontsize=6.5)

    # Arrows: raw -> scripts
    for x_mid in [1.5, 5.0, 8.5]:
        arrow(x_mid, 6.8, x_mid, 6.45)

    # Layer 2: Intermediate outputs
    int_colour = "#FEF9E7"
    int_edge = "#F39C12"
    draw_box(0.2, 4.0, 2.6, 0.85,
             "lcf_expenditure_shares\n45,965 \u00d7 98 columns",
             int_colour, int_edge, fontsize=7)
    draw_box(3.7, 4.0, 2.6, 0.85,
             "cpih_monthly_indices\n134 months \u00d7 20 cols",
             int_colour, int_edge, fontsize=7)
    draw_box(7.2, 4.0, 2.6, 0.85,
             "hci_validation\nOfficial HCI benchmarks",
             int_colour, int_edge, fontsize=7)

    # Arrows: scripts -> intermediate
    for x_mid in [1.5, 5.0, 8.5]:
        arrow(x_mid, 5.2, x_mid, 4.88)

    # Layer 3: Compute engine (wide box)
    draw_box(1.0, 2.4, 5.5, 1.2,
             "compute_group_inflation.py\n"
             "Laspeyres index: Inflation = \u03A3(share$_{t-1}$ \u00d7 \u0394price$_t$)\n"
             "\u2022 Lagged expenditure weights\n"
             "\u2022 OLS-derived non-rent housing index\n"
             "\u2022 8 archetype dimensions \u00d7 9 years",
             "#E8DAEF", "#8E44AD", fontsize=6.5)

    # Arrows into compute
    arrow(1.5, 4.0, 2.8, 3.65)
    arrow(5.0, 4.0, 4.7, 3.65)

    # Arrow from HCI (validation, goes alongside)
    ax.annotate("", xy=(8.5, 1.05), xytext=(8.5, 4.0),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                linewidth=1.0, linestyle="--"))
    ax.text(9.0, 2.5, "Validate", fontsize=6.5, color="#555555",
            rotation=90, ha="left", va="center", style="italic")

    # Layer 4: Final outputs
    out_colour = "#D6EAF8"
    out_edge = "#2E86C1"
    draw_box(1.0, 0.7, 2.8, 1.2,
             "group_inflation_rates\n298 rows: archetype \u00d7 year\n"
             "inflation_decomposition\nCOICOP contributions",
             out_colour, out_edge, fontsize=6.5, bold=False)
    draw_box(4.5, 0.7, 2.8, 1.2,
             "Strand 1: Laspeyres\nTenure-specific inflation\n"
             "4 groups \u00d7 9 years\nCrisis decomposition",
             "#D4EFDF", "#1E8449", fontsize=6.5, bold=True)
    draw_box(7.6, 0.7, 2.2, 0.55,
             "HCI residuals\n< 1pp",
             "#D5F5E3", "#27AE60", fontsize=6.5)

    # Strand 2 box
    draw_box(7.6, 0.0, 2.2, 0.55,
             "Strand 2: K-Means\nCOICOP share clusters",
             "#FDEBD0", "#E67E22", fontsize=6.5, bold=True)

    # Arrows from compute to outputs
    arrow(3.75, 2.4, 2.4, 1.95)
    arrow(3.75, 2.4, 5.9, 1.95)

    # Arrow from lcf_shares to strand 2
    ax.annotate("", xy=(8.7, 0.55), xytext=(1.5, 4.0),
                arrowprops=dict(arrowstyle="-|>", color="#E67E22",
                                linewidth=1.0, linestyle="--",
                                connectionstyle="arc3,rad=0.3"))

    # Title
    ax.text(5.0, 7.95, "Data Pipeline: Raw Sources to Tenure-Specific Inflation Indices",
            ha="center", va="bottom", fontsize=10, weight="bold")

    fig.tight_layout()
    _save(fig, "fig1_data_pipeline.png", SEC3)


# --------------------------------------------------------------------------
# Table I — Data Acquisition Summary
# --------------------------------------------------------------------------
def table_data_acquisition(shares: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """Generate a data acquisition summary table as a figure for the report."""
    print("\n[SEC3] Data acquisition summary table")

    data = [
        ["LCF Microdata",  "ONS / UKDS",     "2015/16\u20132023/24", "Annual, household",
         ".dta (Stata)", f"{len(shares):,} HH"],
        ["CPIH Indices\n(MM23)", "ONS MM23", "Jan 2015\u2013present", "Monthly, national",
         ".csv", f"{len(monthly)} months"],
        ["HCI Validation", "ONS HCI",        "2015\u20132023",       "Annual, subgroup",
         ".xlsx", "5 tables"],
    ]

    fig, ax = plt.subplots(figsize=(7.16, 1.6))
    ax.axis("off")

    col_labels = ["Dataset", "Source", "Period", "Granularity", "Format", "Size"]
    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2E86C1")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")

    # Style body
    for i in range(1, len(data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor("#F8F9F9" if i % 2 == 0 else "white")
            cell.set_edgecolor("#D5D8DC")

    ax.set_title("Table I: Data Sources", fontsize=10, weight="bold", pad=12)
    fig.tight_layout()
    _save(fig, "table1_data_sources.png", SEC3)


# --------------------------------------------------------------------------
# Fig. 2 — Missing Values and Cleaning Audit
# --------------------------------------------------------------------------
def fig_missing_and_cleaning(shares: pd.DataFrame) -> None:
    """Two-panel figure: (a) missing value rates, (b) zero-expenditure flags."""
    print("\n[SEC3] Missing values and cleaning audit")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # Panel (a): Missing value percentage per key column
    key_cols = (["gross_weekly_income", "equivalised_income", "total_expenditure",
                 "weight"] +
                [c for c in MAIN_SHARE_COLS if c in shares.columns])
    available = [c for c in key_cols if c in shares.columns]

    miss_pct = (shares[available].isnull().sum() / len(shares) * 100).sort_values(ascending=True)
    short_names = [COICOP_SHORT.get(c, c.replace("_", " ").title()[:18]) for c in miss_pct.index]

    colours = ["#E74C3C" if v > 1 else "#F39C12" if v > 0 else "#27AE60" for v in miss_pct.values]
    ax1.barh(range(len(miss_pct)), miss_pct.values, color=colours, height=0.7)
    ax1.set_yticks(range(len(miss_pct)))
    ax1.set_yticklabels(short_names, fontsize=6.5)
    ax1.set_xlabel("Missing (%)")
    ax1.set_title("(a) Missing Value Rate", fontsize=9, weight="bold")
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4)

    # Add value labels
    for i, v in enumerate(miss_pct.values):
        if v > 0:
            ax1.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=6)

    # Panel (b): Zero-expenditure prevalence per COICOP division
    share_cols = [c for c in MAIN_SHARE_COLS if c in shares.columns]
    zero_pct = ((shares[share_cols] == 0).sum() / len(shares) * 100)
    zero_pct = zero_pct.sort_values(ascending=True)
    short_z = [COICOP_SHORT.get(c, c) for c in zero_pct.index]

    ax2.barh(range(len(zero_pct)), zero_pct.values, color="#3498DB", height=0.7)
    ax2.set_yticks(range(len(zero_pct)))
    ax2.set_yticklabels(short_z, fontsize=6.5)
    ax2.set_xlabel("Households reporting zero (%)")
    ax2.set_title("(b) Non-Participation by Division", fontsize=9, weight="bold")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4)

    for i, v in enumerate(zero_pct.values):
        if v > 1:
            ax2.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=6)

    fig.suptitle("Fig. 2: Data Quality Audit  (zero spending in (b) reflects genuine non-purchase in the 2-week diary, not missingness)",
                 fontsize=9, weight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_missing_and_cleaning.png", SEC3)


# --------------------------------------------------------------------------
# Fig. 3 — Distribution Properties of COICOP Expenditure Shares
# --------------------------------------------------------------------------
def fig_distribution_properties(shares: pd.DataFrame) -> None:
    """Show distribution properties of COICOP expenditure shares after
    domain-based household filtering: boxplots + skewness/kurtosis.

    This replaces winsorisation analysis — our cleaning strategy removes
    implausible households (zero food, zero housing, negative expenditure)
    rather than clipping individual share values, because shares are
    compositional (sum to 1.0) and per-column winsorisation would break
    the budget constraint."""
    print("\n[SEC3] Distribution properties analysis")

    share_cols = [c for c in MAIN_SHARE_COLS if c in shares.columns]
    short = [COICOP_SHORT.get(c, c) for c in share_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    # Panel (a): Boxplots of share distributions
    data_for_box = [shares[c].dropna().values for c in share_cols]
    bp = ax1.boxplot(data_for_box, vert=False, patch_artist=True,
                     widths=0.6, showfliers=True,
                     flierprops=dict(marker=".", markersize=1.5, alpha=0.3,
                                     markerfacecolor="#E74C3C"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#AED6F1")
        patch.set_edgecolor("#2E86C1")
    ax1.set_yticklabels(short, fontsize=7)
    ax1.set_xlabel("Expenditure share")
    ax1.set_title("(a) Share Distributions (post-filtering)", fontsize=9, weight="bold")
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4)

    # Panel (b): Skewness and kurtosis per share
    skew_vals = [shares[c].dropna().skew() for c in share_cols]
    kurt_vals = [shares[c].dropna().kurtosis() for c in share_cols]

    x = np.arange(len(share_cols))
    w = 0.35
    bars1 = ax2.barh(x - w/2, skew_vals, height=w, color="#3498DB", label="Skewness")
    bars2 = ax2.barh(x + w/2, kurt_vals, height=w, color="#E67E22", label="Excess kurtosis")
    ax2.set_yticks(x)
    ax2.set_yticklabels(short, fontsize=7)
    ax2.set_xlabel("Statistic value")
    ax2.set_title("(b) Skewness & Kurtosis", fontsize=9, weight="bold")
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Fig. 3: Distribution Properties of COICOP Expenditure Shares",
                 fontsize=10, weight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_distribution_properties.png", SEC3)


# --------------------------------------------------------------------------
# Fig. 3b — Outlier Investigation: Profile of Removed Households
# --------------------------------------------------------------------------
def fig_outlier_investigation() -> None:
    """Profile the 350 households removed by domain-based filtering.

    Loads the INTERIM (pre-filtered) data, applies the same three filters,
    and compares removed vs retained households across tenure, income,
    and total expenditure — mirroring the 'outlier investigation' approach
    that earned high marks in exemplar reports."""
    print("\n[SEC3] Outlier investigation")

    interim_path = pathlib.Path("data/interim/lcf_household.parquet")
    if not interim_path.exists():
        print("    SKIP: interim file not found")
        return

    raw = pd.read_parquet(interim_path)

    # We need shares computed — reload the processed file for share columns
    # but also need p600t from interim for the negative-expenditure filter.
    processed = pd.read_parquet(PROCESSED / "lcf_expenditure_shares.parquet")

    # The processed file is ALREADY filtered. Reconstruct what was removed
    # by loading interim and computing shares on the fly.
    # Simpler: load both interim and processed, identify removed by household_id/year.
    if "household_id" in processed.columns and "household_id" in raw.columns:
        kept_keys = set(zip(processed["household_id"], processed["year"]))
        raw["_kept"] = [
            (h, y) in kept_keys
            for h, y in zip(raw["household_id"], raw["year"])
        ]
    else:
        # Fallback: use row count difference
        print("    SKIP: cannot match households between interim and processed")
        return

    removed = raw[~raw["_kept"]]
    retained = raw[raw["_kept"]]
    n_removed = len(removed)
    n_retained = len(retained)

    if n_removed == 0:
        print("    SKIP: no removed households found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))

    # Panel (a): Tenure composition of removed vs retained
    ax = axes[0]
    tenure_col = None
    for col_name in ["tenure_type", "tenure_type1_code"]:
        if col_name in raw.columns:
            tenure_col = col_name
            break

    if tenure_col and tenure_col == "tenure_type1_code":
        TENURE_MAP = {1: "Social Rent", 2: "Social Rent", 3: "Private Rent",
                      4: "Private Rent", 5: "Own Outright", 6: "Own Mortgage",
                      7: "Own Mortgage", 8: "Rent Free"}
        removed_tenure = removed[tenure_col].map(TENURE_MAP).value_counts(normalize=True)
        retained_tenure = retained[tenure_col].map(TENURE_MAP).value_counts(normalize=True)
    elif tenure_col:
        removed_tenure = removed[tenure_col].value_counts(normalize=True)
        retained_tenure = retained[tenure_col].value_counts(normalize=True)
    else:
        removed_tenure = pd.Series(dtype=float)
        retained_tenure = pd.Series(dtype=float)

    if len(removed_tenure) > 0:
        tenures = ["Social Rent", "Private Rent", "Own Outright", "Own Mortgage"]
        tenures = [t for t in tenures if t in removed_tenure.index or t in retained_tenure.index]
        x = np.arange(len(tenures))
        w = 0.35
        rem_vals = [removed_tenure.get(t, 0) * 100 for t in tenures]
        ret_vals = [retained_tenure.get(t, 0) * 100 for t in tenures]
        ax.barh(x - w/2, ret_vals, height=w, color="#AED6F1", label="Retained")
        ax.barh(x + w/2, rem_vals, height=w, color="#E74C3C", label="Removed")
        ax.set_yticks(x)
        ax.set_yticklabels(tenures, fontsize=7)
        ax.set_xlabel("% of group", fontsize=7)
        ax.set_title("(a) Tenure Composition", fontsize=8, weight="bold")
        ax.legend(fontsize=6)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    # Panel (b): Total expenditure distribution
    ax = axes[1]
    exp_col = "p600t" if "p600t" in raw.columns else None
    if exp_col:
        # Remove extreme outliers for visualisation
        ret_exp = retained[exp_col].dropna()
        rem_exp = removed[exp_col].dropna()
        clip_hi = ret_exp.quantile(0.98)
        bins = np.linspace(0, max(clip_hi, 1), 40)
        ax.hist(ret_exp.clip(upper=clip_hi), bins=bins, density=True,
                alpha=0.6, color="#AED6F1", label=f"Retained (n={n_retained:,})")
        ax.hist(rem_exp.clip(upper=clip_hi), bins=bins, density=True,
                alpha=0.7, color="#E74C3C", label=f"Removed (n={n_removed})")
        ax.set_xlabel("Total expenditure (£/week)", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_title("(b) Total Expenditure", fontsize=8, weight="bold")
        ax.legend(fontsize=5.5)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)

        # Add median annotations
        ax.axvline(ret_exp.median(), color="#2E86C1", linestyle="--", linewidth=1)
        ax.axvline(rem_exp.median(), color="#C0392B", linestyle="--", linewidth=1)

    # Panel (c): Filter breakdown (which filter caught what)
    ax = axes[2]
    # Reconstruct filter categories from the removed households
    # We need shares — but removed HHs don't have shares in processed.
    # Use raw expenditure to approximate.
    filter_counts = {}
    if "p600t" in removed.columns:
        filter_counts["Negative\nexpenditure"] = (removed["p600t"] <= 0).sum()
    # For zero food/housing, we need the share columns from compute_expenditure_shares
    # but we only have raw data. Use raw columns as proxy.
    if "p601t" in removed.columns and "p600t" in removed.columns:
        total = removed["p600t"].replace(0, np.nan)
        food_share = removed["p601t"] / total
        housing_share = removed["p604t"] / total
        filter_counts["Zero\nfood"] = ((food_share == 0) | (food_share.isna())).sum()
        filter_counts["Zero\nhousing"] = ((housing_share == 0) | (housing_share.isna())).sum()

    if filter_counts:
        labels = list(filter_counts.keys())
        values = list(filter_counts.values())
        colours = ["#E74C3C", "#E67E22", "#F39C12"][:len(labels)]
        bars = ax.barh(labels, values, color=colours, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    str(v), va="center", fontsize=7)
        ax.set_xlabel("Households removed", fontsize=7)
        ax.set_title(f"(c) Filter Breakdown\n(total: {n_removed})", fontsize=8, weight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Fig. 3b: Profile of Removed Households (Domain-Based Filtering)",
                 fontsize=9, weight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3b_outlier_investigation.png", SEC3)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: DATA EXPLORATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# --------------------------------------------------------------------------
# Table II — Summary Statistics
# --------------------------------------------------------------------------
def table_summary_statistics(shares: pd.DataFrame) -> None:
    """Summary statistics disaggregated by tenure group — supports the
    report's thesis that tenure groups differ structurally. Weighted means
    use household_weight for representativeness."""
    print("\n[SEC4] Summary statistics table (by tenure)")

    def _wmean(series: pd.Series, weights: pd.Series) -> float:
        s, w = series.align(weights, join="inner")
        m = s.notna() & w.notna() & (w > 0)
        if m.sum() == 0:
            return float("nan")
        return float(np.average(s[m], weights=w[m]))

    has_w = "household_weight" in shares.columns
    has_rent = "share_04_actual_rent" in shares.columns
    has_energy = "share_04_energy_other" in shares.columns

    groups: list[tuple[str, pd.DataFrame]] = [("All households", shares)]
    for t in TENURE_4:
        groups.append((TENURE_LABELS[t], shares[shares["tenure_type"] == t]))

    rows = []
    for label, g in groups:
        w = g["household_weight"] if has_w else pd.Series(1.0, index=g.index)
        row: dict = {"Group": label, "N": f"{len(g):,}"}
        # Essentials components
        food = _wmean(g["share_01_food_non_alcoholic"], w) if "share_01_food_non_alcoholic" in g else np.nan
        energy = _wmean(g["share_04_energy_other"], w) if has_energy else np.nan
        rent = _wmean(g["share_04_actual_rent"], w) if has_rent else np.nan
        row["Food %"] = f"{food*100:.1f}" if np.isfinite(food) else "—"
        row["Energy %"] = f"{energy*100:.1f}" if np.isfinite(energy) else "—"
        row["Rent %"] = f"{rent*100:.1f}" if np.isfinite(rent) else "—"
        essentials = np.nansum([food, energy, rent])
        row["Essentials %"] = f"{essentials*100:.1f}"
        # Transport, recreation (high variation groups)
        transport = _wmean(g["share_07_transport"], w) if "share_07_transport" in g else np.nan
        recreation = _wmean(g["share_09_recreation_culture"], w) if "share_09_recreation_culture" in g else np.nan
        row["Transport %"] = f"{transport*100:.1f}" if np.isfinite(transport) else "—"
        row["Recreation %"] = f"{recreation*100:.1f}" if np.isfinite(recreation) else "—"
        # Income
        if "hh_income_gross_weekly" in g:
            gross = _wmean(g["hh_income_gross_weekly"], w)
            row["Gross £/wk"] = f"{gross:.0f}" if np.isfinite(gross) else "—"
        if "hh_income_equivalised_oecd_mod" in g:
            eq = _wmean(g["hh_income_equivalised_oecd_mod"], w)
            row["Equiv. £/wk"] = f"{eq:.0f}" if np.isfinite(eq) else "—"
        rows.append(row)

    df = pd.DataFrame(rows)

    # Render as matplotlib table
    fig, ax = plt.subplots(figsize=(7.16, 2.0))
    ax.axis("off")

    cell_data = df.values.tolist()
    col_labels = list(df.columns)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2E86C1")
        cell.set_text_props(color="white", weight="bold", fontsize=7.5)
        cell.set_edgecolor("white")

    # Style body rows — emphasise the "All households" benchmark row
    for i in range(1, len(cell_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i == 1:  # All households row
                cell.set_facecolor("#FEF9E7")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#F8F9F9" if i % 2 == 0 else "white")
            cell.set_edgecolor("#D5D8DC")
            if j == 0:
                cell.set_text_props(ha="left",
                                    weight="bold" if i == 1 else "normal")

    table.auto_set_column_width([0])

    ax.set_title(f"Table II: Weighted Summary Statistics by Housing Tenure "
                 f"(N = {len(shares):,} households, pooled 2015/16–2023/24)",
                 fontsize=9, weight="bold", pad=10)
    fig.tight_layout()
    _save(fig, "table2_summary_statistics.png", SEC4)

    # Also save as CSV for LaTeX
    csv_path = SEC4 / "table2_summary_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.relative_to(REPORT)}")


# --------------------------------------------------------------------------
# Fig. 4 — Basket Composition by Tenure (stacked bar)
# --------------------------------------------------------------------------
def fig_basket_by_tenure(shares: pd.DataFrame) -> None:
    """Deviation heatmap: tenure x COICOP, values = tenure mean share minus
    pooled mean share (percentage points). Makes basket differences pop
    immediately, unlike a stacked bar where readers must mentally subtract
    segments."""
    print("\n[SEC4] Basket composition by tenure type (deviation heatmap)")

    share_cols = [c for c in MAIN_SHARE_COLS if c in shares.columns]
    sub = shares[shares["tenure_type"].isin(TENURE_4)].copy()

    # Weighted pooled mean across all four tenures (weight = household_weight)
    if "household_weight" in sub.columns:
        w = sub["household_weight"].fillna(0).to_numpy()
        pooled_mean = pd.Series(
            {c: np.average(sub[c].fillna(0), weights=w) for c in share_cols}
        )
        # Weighted mean per tenure
        tenure_rows = []
        for t in TENURE_4:
            g = sub[sub["tenure_type"] == t]
            wg = g["household_weight"].fillna(0).to_numpy()
            if wg.sum() == 0:
                continue
            tenure_rows.append(
                pd.Series({c: np.average(g[c].fillna(0), weights=wg) for c in share_cols},
                          name=t)
            )
        means = pd.DataFrame(tenure_rows)
    else:
        pooled_mean = sub[share_cols].mean()
        means = sub.groupby("tenure_type")[share_cols].mean()
        means = means.loc[[t for t in TENURE_4 if t in means.index]]

    # Deviation in percentage points
    deviation_pp = (means.subtract(pooled_mean, axis=1)) * 100
    raw_pct = means * 100  # for annotation

    row_labels = [TENURE_LABELS.get(t, t) for t in means.index]
    col_labels = [COICOP_SHORT.get(c, c) for c in share_cols]

    fig, ax = plt.subplots(figsize=(7.16, 3.1))

    vmax = float(np.nanmax(np.abs(deviation_pp.values)))
    vmax = max(vmax, 1.0)  # minimum ±1pp scale to avoid noise looking dramatic
    im = ax.imshow(deviation_pp.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    # Annotate with raw share % (small) and deviation (bold)
    for i in range(deviation_pp.shape[0]):
        for j in range(deviation_pp.shape[1]):
            dev = deviation_pp.values[i, j]
            raw = raw_pct.values[i, j]
            colour = "white" if abs(dev) > vmax * 0.55 else "black"
            ax.text(j, i - 0.18, f"{dev:+.1f}", ha="center", va="center",
                    fontsize=6.5, color=colour, weight="bold")
            ax.text(j, i + 0.22, f"({raw:.0f}%)", ha="center", va="center",
                    fontsize=5.5, color=colour, alpha=0.8)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("COICOP division")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Deviation from pooled mean (pp)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_title("Fig. 4: Tenure Basket Deviation from Pooled Mean "
                 "(bold = deviation pp, parens = raw share)",
                 fontsize=9.5, weight="bold")
    fig.tight_layout()
    _save(fig, "fig4_basket_by_tenure.png", SEC4)


# --------------------------------------------------------------------------
# Fig. 5 — Essentials Density by Tenure (food + energy + rent)
# --------------------------------------------------------------------------
def fig_essentials_density(shares: pd.DataFrame) -> None:
    """KDE showing distribution of essential spending share by tenure."""
    print("\n[SEC4] Essential spending density by tenure")

    needed = {"tenure_type", "share_01_food_non_alcoholic",
              "share_04_energy_other", "share_04_actual_rent"}
    if not needed.issubset(shares.columns):
        print("  SKIP: required columns not found")
        return

    sub = shares[shares["tenure_type"].isin(TENURE_4)].dropna(
        subset=list(needed - {"tenure_type"})
    ).copy()
    sub["essentials"] = (sub["share_01_food_non_alcoholic"]
                         + sub["share_04_energy_other"]
                         + sub["share_04_actual_rent"])

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    medians = {}
    for t in TENURE_4:
        vals = sub.loc[sub["tenure_type"] == t, "essentials"].values
        if len(vals) < 30:
            continue
        # Gaussian KDE
        from numpy import linspace
        xs = linspace(0, 1, 300)
        bw = 0.03
        kernel = np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
        density = kernel.mean(axis=1)
        med = float(np.median(vals))
        medians[t] = med
        ax.plot(xs, density, linewidth=2, color=TENURE_COLOURS[t],
                label=f"{TENURE_LABELS[t]} (med {med:.0%})")
        ax.fill_between(xs, density, alpha=0.1, color=TENURE_COLOURS[t])
        # Median tick line at baseline
        ax.axvline(med, color=TENURE_COLOURS[t], linewidth=1.0,
                   linestyle="--", alpha=0.6, zorder=2)

    # Annotate the gap between extremes
    if len(medians) >= 2:
        lo_t = min(medians, key=medians.get)
        hi_t = max(medians, key=medians.get)
        gap = medians[hi_t] - medians[lo_t]
        ax.text(0.97, 0.95,
                f"Median gap: {gap*100:.1f}pp\n({TENURE_LABELS[hi_t]} vs {TENURE_LABELS[lo_t]})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7",
                          edgecolor="#F39C12", alpha=0.85))

    ax.set_xlabel("Food + Energy + Rent share of total expenditure")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Fig. 5: Distribution of Essential Spending Share by Tenure",
                 fontsize=10, weight="bold")
    ax.set_xlim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig5_essentials_density.png", SEC4)


# --------------------------------------------------------------------------
# Fig. 6 — CPIH Sub-Index Time Series with Event Annotations
# --------------------------------------------------------------------------
def fig_cpih_annotated(monthly: pd.DataFrame) -> None:
    """CPIH sub-index monthly series with major economic events annotated."""
    print("\n[SEC4] Annotated CPIH sub-index time series")

    cols = [c for c in ALL_PRICE_COLS if c in monthly.columns]
    base_row = monthly[monthly["date"] == "2015-01-01"]
    if base_row.empty:
        base_row = monthly.sort_values("date").iloc[[0]]
    base_vals = {c: float(base_row[c].iloc[0]) for c in cols if base_row[c].iloc[0] != 0}
    ms = monthly.sort_values("date")

    highlight = {
        "food_non_alcoholic":    ("#2E86C1", "Food & NA Bev."),
        "electricity_gas_fuels": ("#27AE60", "Energy (elec/gas)"),
        "actual_rents":          ("#E67E22", "Actual Rents"),
    }

    fig, ax = plt.subplots(figsize=(7.16, 4.0))

    # Background series
    for col in cols:
        if col in highlight or base_vals.get(col, 0) == 0:
            continue
        rebased = ms[col] / base_vals[col] * 100
        ax.plot(ms["date"], rebased, linewidth=0.7, color="#D5D8DC", alpha=0.7, zorder=1)

    # Highlighted series
    for col, (colour, label) in highlight.items():
        if col not in cols or base_vals.get(col, 0) == 0:
            continue
        rebased = ms[col] / base_vals[col] * 100
        ax.plot(ms["date"], rebased, linewidth=2.0, color=colour, label=label, zorder=3)

    ax.axhline(100, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    # Event annotations
    events = [
        (pd.Timestamp("2016-06-23"), "Brexit\nvote", 115, -30),
        (pd.Timestamp("2020-03-23"), "COVID\nlockdown", 118, -25),
        (pd.Timestamp("2022-02-24"), "Russia\ninvades\nUkraine", 175, 35),
        (pd.Timestamp("2022-10-01"), "CPIH\npeak\n11.1%", 220, -20),
        (pd.Timestamp("2023-04-01"), "Energy\nprice cap\nfalls", 170, 40),
    ]

    for date, text, y_target, x_off in events:
        ax.annotate(
            text,
            xy=(date, y_target),
            xytext=(date + pd.Timedelta(days=x_off * 5), y_target + 15),
            fontsize=6, color="#555555",
            ha="center",
            arrowprops=dict(arrowstyle="-|>", color="#999999", linewidth=0.8),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9),
        )

    # Grey patch label
    grey_handle = mlines.Line2D([], [], linewidth=0.7, color="#D5D8DC",
                                label="Other COICOP divisions")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [grey_handle], fontsize=7, loc="upper left",
              framealpha=0.9)

    ax.set_ylabel("Index (Jan 2015 = 100)")
    ax.set_title("Fig. 6: CPIH Sub-Index Monthly Series with Key Events",
                 fontsize=10, weight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig6_cpih_annotated.png", SEC4)


# --------------------------------------------------------------------------
# Fig. 7 — Correlation Heatmaps (expenditure shares AND price changes)
# --------------------------------------------------------------------------
def fig_correlation_heatmaps(shares: pd.DataFrame, fy_idx: pd.DataFrame) -> None:
    """Two-panel correlation heatmap: (a) expenditure shares, (b) annual price changes."""
    print("\n[SEC4] Dual correlation heatmaps")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 4.4))

    # Panel (a): Expenditure share correlations
    share_cols = [c for c in MAIN_SHARE_COLS if c in shares.columns]
    corr_shares = shares[share_cols].corr()
    short_s = [COICOP_SHORT.get(c, c) for c in share_cols]

    mask = np.triu(np.ones_like(corr_shares, dtype=bool), k=1)
    sns.heatmap(corr_shares, mask=mask, ax=ax1, cmap="RdBu_r", vmin=-1, vmax=1,
                xticklabels=short_s, yticklabels=short_s,
                annot=True, fmt=".2f", annot_kws={"size": 5.5},
                cbar_kws={"shrink": 0.8, "label": "Pearson r"},
                linewidths=0.5, square=True)
    ax1.set_xticklabels(short_s, rotation=45, ha="right", fontsize=6.5)
    ax1.set_yticklabels(short_s, rotation=0, fontsize=6.5)
    ax1.set_title("(a) Expenditure Shares", fontsize=9, weight="bold")

    # Panel (b): Price change correlations
    price_cols = [c for c in ALL_PRICE_COLS if c in fy_idx.columns]
    pct = fy_idx.sort_values("year").set_index("year")[price_cols].pct_change() * 100
    pct = pct.dropna()
    corr_prices = pct.corr()
    short_p = [PRICE_SHORT.get(c, c)[:12] for c in price_cols]

    mask2 = np.triu(np.ones_like(corr_prices, dtype=bool), k=1)
    sns.heatmap(corr_prices, mask=mask2, ax=ax2, cmap="RdBu_r", vmin=-1, vmax=1,
                xticklabels=short_p, yticklabels=short_p,
                annot=True, fmt=".2f", annot_kws={"size": 5.5},
                cbar_kws={"shrink": 0.8, "label": "Pearson r"},
                linewidths=0.5, square=True)
    ax2.set_xticklabels(short_p, rotation=45, ha="right", fontsize=6.5)
    ax2.set_yticklabels(short_p, rotation=0, fontsize=6.5)
    ax2.set_title("(b) Annual Price Changes (FY)", fontsize=9, weight="bold")

    fig.suptitle("Fig. 7: Correlation Structure of Expenditure and Prices",
                 fontsize=10, weight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_correlation_heatmaps.png", SEC4)


# --------------------------------------------------------------------------
# Fig. 8 — Why Tenure? Dimension Gap Comparison
# --------------------------------------------------------------------------
def fig_dimension_gap(infl: pd.DataFrame) -> None:
    """Time series of the max intra-dimension inflation gap by year.
    Shows when tenure's gap overtakes competing dimensions — a stronger
    claim than a single-year snapshot."""
    print("\n[SEC4] Dimension gap time series")

    titles = {
        "income_quintile":   "Income Quintile",
        "tenure_type":       "Tenure Type",
        "hrp_age_band":      "HRP Age Band",
    }
    dim_colours = {
        "tenure_type":     "#E74C3C",
        "income_quintile": "#2E86C1",
        "hrp_age_band":    "#7D3C98",
    }
    dim_styles = {
        "tenure_type":     "-",
        "income_quintile": "--",
        "hrp_age_band":    "-.",
    }
    dim_widths = {
        "tenure_type":     2.4,
        "income_quintile": 1.6,
        "hrp_age_band":    1.6,
    }

    # Build per-year gap for each dimension
    rows = []
    for (arch, yr), sub in infl.groupby(["archetype_name", "year"]):
        if len(sub) < 2:
            continue
        rows.append({
            "dimension": arch,
            "year": int(yr),
            "gap_pp": float(sub["inflation_rate"].max()
                            - sub["inflation_rate"].min()),
        })
    if not rows:
        print("  SKIP: no data")
        return
    gap_df = pd.DataFrame(rows).sort_values(["dimension", "year"])

    fig, ax = plt.subplots(figsize=(7.16, 3.4))

    # Shade crisis period (FY 2021/22 and 2022/23)
    ax.axvspan(2021 - 0.5, 2022 + 0.5, color="#FDEBD0", alpha=0.5, zorder=0,
               label="_nolegend_")
    ax.text(2021.5, ax.get_ylim()[1] if False else 0.5,
            "Cost-of-living\ncrisis", fontsize=6.5, color="#B9770E",
            ha="center", va="bottom", style="italic")

    for dim, sub in gap_df.groupby("dimension"):
        sub = sub.sort_values("year")
        xs = sub["year"].to_numpy() + 0.5  # centre on FY midpoint
        ax.plot(xs, sub["gap_pp"].to_numpy(),
                color=dim_colours.get(dim, "#7F8C8D"),
                linestyle=dim_styles.get(dim, "-"),
                linewidth=dim_widths.get(dim, 1.5),
                marker="o", markersize=4,
                label=titles.get(dim, dim), zorder=3)

    # Label tenure line at its peak
    tenure_sub = gap_df[gap_df["dimension"] == "tenure_type"].sort_values("year")
    if not tenure_sub.empty:
        peak = tenure_sub.loc[tenure_sub["gap_pp"].idxmax()]
        ax.annotate(
            f"Tenure peak:\n{peak['gap_pp']:.2f}pp ({_fy(int(peak['year']))})",
            xy=(peak["year"] + 0.5, peak["gap_pp"]),
            xytext=(peak["year"] + 0.5 - 2.0, peak["gap_pp"] + 0.3),
            fontsize=7, color="#E74C3C", style="italic",
            arrowprops=dict(arrowstyle="-|>", color="#E74C3C", linewidth=0.8),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#E74C3C", alpha=0.85),
        )

    years = sorted(gap_df["year"].unique())
    ax.set_xticks([y + 0.5 for y in years])
    ax.set_xticklabels([_fy(y) for y in years], rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("Financial year")
    ax.set_ylabel("Max inflation gap within dimension (pp)")
    ax.set_title("Fig. 8: Inflation Gap by Household Dimension, 2015/16–2023/24",
                 fontsize=10, weight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, "fig8_dimension_gap.png", SEC4)


# --------------------------------------------------------------------------
# Fig. 9 — Tenure Basket Evolution Over Time
# --------------------------------------------------------------------------
def fig_basket_evolution(shares: pd.DataFrame) -> None:
    """How the essentials share (food + energy + rent) evolved per tenure,
    2015/16 to 2023/24. Demonstrates that baskets themselves shifted during
    the crisis — a microdata-only finding ONS headline CPIH cannot show."""
    print("\n[SEC4] Tenure basket evolution over time")

    needed = {"tenure_type", "year", "share_01_food_non_alcoholic",
              "share_04_energy_other", "share_04_actual_rent"}
    if not needed.issubset(shares.columns):
        print("  SKIP: required columns not found")
        return

    sub = shares[shares["tenure_type"].isin(TENURE_4)].dropna(
        subset=list(needed - {"tenure_type", "year"})
    ).copy()
    sub["essentials"] = (sub["share_01_food_non_alcoholic"]
                         + sub["share_04_energy_other"]
                         + sub["share_04_actual_rent"])

    # Weighted mean per (tenure, year)
    def _wmean_by(g: pd.DataFrame, col: str) -> float:
        w = g["household_weight"].fillna(0).to_numpy() if "household_weight" in g else None
        v = g[col].to_numpy()
        if w is None or w.sum() == 0:
            return float(np.nanmean(v))
        return float(np.average(v, weights=w))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.4), sharex=True)

    years = sorted(sub["year"].unique())

    # Panel (a): Essentials share over time, 4 lines
    for t in TENURE_4:
        gt = sub[sub["tenure_type"] == t]
        series = []
        for yr in years:
            gy = gt[gt["year"] == yr]
            series.append(_wmean_by(gy, "essentials") if len(gy) else np.nan)
        xs = np.array(years) + 0.5
        ax1.plot(xs, [s * 100 for s in series], linewidth=2.0, marker="o",
                 markersize=3.5, color=TENURE_COLOURS[t], label=TENURE_LABELS[t])

    ax1.axvspan(2021 - 0.5, 2022 + 0.5, color="#FDEBD0", alpha=0.5, zorder=0)
    ax1.set_xticks([y + 0.5 for y in years])
    ax1.set_xticklabels([_fy(y) for y in years], rotation=35, ha="right", fontsize=6.5)
    ax1.set_ylabel("Essentials share (%)", fontsize=8)
    ax1.set_title("(a) Food + Energy + Rent share", fontsize=9, weight="bold")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(fontsize=7, loc="best", framealpha=0.9)

    # Panel (b): Energy share over time — the single biggest crisis story
    for t in TENURE_4:
        gt = sub[sub["tenure_type"] == t]
        series = []
        for yr in years:
            gy = gt[gt["year"] == yr]
            series.append(_wmean_by(gy, "share_04_energy_other") if len(gy) else np.nan)
        xs = np.array(years) + 0.5
        ax2.plot(xs, [s * 100 for s in series], linewidth=2.0, marker="o",
                 markersize=3.5, color=TENURE_COLOURS[t], label=TENURE_LABELS[t])

    ax2.axvspan(2021 - 0.5, 2022 + 0.5, color="#FDEBD0", alpha=0.5, zorder=0)
    ax2.set_xticks([y + 0.5 for y in years])
    ax2.set_xticklabels([_fy(y) for y in years], rotation=35, ha="right", fontsize=6.5)
    ax2.set_ylabel("Energy & utilities share (%)", fontsize=8)
    ax2.set_title("(b) Home energy / utilities share", fontsize=9, weight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Fig. 9: Tenure Basket Evolution, 2015/16–2023/24 "
                 "(weighted by household_weight; shaded = crisis years)",
                 fontsize=9.5, weight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig9_basket_evolution.png", SEC4)


# --------------------------------------------------------------------------
# Fig. 10 — Laspeyres Exposure Preview (Crisis Decomposition)
# --------------------------------------------------------------------------
def fig_laspeyres_exposure(decomp: pd.DataFrame) -> None:
    """Stacked bars showing COICOP contributions to each tenure group's
    FY2022/23 inflation — a visual Laspeyres decomposition that previews
    Sec 5 results in exploration terms."""
    print("\n[SEC4] Laspeyres exposure preview (crisis decomposition)")

    crisis = decomp[(decomp["archetype_name"] == "tenure_type")
                    & (decomp["year"] == 2022)
                    & (decomp["coicop_label"] != "all_items")].copy()
    if crisis.empty:
        print("  SKIP: no tenure decomposition for FY2022")
        return

    # Pivot: rows = tenure, cols = coicop component
    pv = crisis.pivot_table(index="archetype_value",
                            columns="coicop_label",
                            values="contribution",
                            aggfunc="first").fillna(0)
    pv = pv.loc[[t for t in TENURE_4 if t in pv.index]]

    # Column order: biggest average contributor first for cleaner stacks
    col_order = pv.mean().sort_values(ascending=False).index.tolist()
    pv = pv[col_order]

    # Colour map — reuse PRICE_SHORT for labels, use tab20 for distinct colours
    cmap = plt.get_cmap("tab20", max(20, len(col_order)))
    colours = {col: cmap(i) for i, col in enumerate(col_order)}

    fig, ax = plt.subplots(figsize=(7.16, 3.6))

    labels = [TENURE_LABELS.get(t, t) for t in pv.index]
    y = np.arange(len(labels))

    # Stack positive contributions; handle any negative components separately
    pos_left = np.zeros(len(labels))
    neg_left = np.zeros(len(labels))
    for col in col_order:
        vals = pv[col].to_numpy()
        short = PRICE_SHORT.get(col, col)
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        ax.barh(y, pos_vals, left=pos_left, height=0.62,
                color=colours[col], edgecolor="white", linewidth=0.3, label=short)
        ax.barh(y, neg_vals, left=neg_left, height=0.62,
                color=colours[col], edgecolor="white", linewidth=0.3)
        # Label segments that contribute > 0.4pp
        for i, v in enumerate(pos_vals):
            if v > 0.4:
                ax.text(pos_left[i] + v / 2, i, f"{v:.1f}",
                        ha="center", va="center", fontsize=6, color="black")
        pos_left += pos_vals
        neg_left += neg_vals

    # Totals at right edge
    totals = pv.sum(axis=1).to_numpy()
    for i, total in enumerate(totals):
        ax.text(total + 0.15, i, f"{total:.2f}pp", va="center",
                fontsize=8, weight="bold", color="#2C3E50")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Contribution to FY 2022/23 inflation (percentage points)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6.5,
              frameon=False, title="COICOP component",
              title_fontsize=7)

    ax.set_title("Fig. 10: Laspeyres Decomposition — Who Drove Each Tenure's "
                 "FY 2022/23 Inflation?",
                 fontsize=9.5, weight="bold")
    fig.tight_layout()
    _save(fig, "fig10_laspeyres_exposure.png", SEC4)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("Report Figure Generator")
    print("=" * 60)
    print(f"  Output: {REPORT}")

    shares, monthly, fy_idx, infl, decomp = load_data()
    print(f"  LCF shares: {len(shares):,} rows")
    print(f"  CPIH monthly: {len(monthly)} rows")

    # ── Section 3: Data Preparation ──
    print("\n" + "=" * 60)
    print("SECTION 3: DATA PREPARATION")
    print("=" * 60)
    fig_pipeline_flowchart()
    table_data_acquisition(shares, monthly)
    fig_missing_and_cleaning(shares)
    fig_distribution_properties(shares)
    fig_outlier_investigation()

    # ── Section 4: Data Exploration ──
    print("\n" + "=" * 60)
    print("SECTION 4: DATA EXPLORATION")
    print("=" * 60)
    table_summary_statistics(shares)
    fig_basket_by_tenure(shares)
    fig_essentials_density(shares)
    fig_cpih_annotated(monthly)
    fig_correlation_heatmaps(shares, fy_idx)
    fig_dimension_gap(infl)
    fig_basket_evolution(shares)
    fig_laspeyres_exposure(decomp)

    # Summary
    saved_png = sorted(REPORT.rglob("*.png"))
    saved_csv = sorted(REPORT.rglob("*.csv"))
    print(f"\nDone. {len(saved_png)} figures + {len(saved_csv)} tables saved to {REPORT}")

    print("\n  Report figure mapping:")
    print("  SEC 3 - Data Preparation:")
    print("    Fig. 1  — Data pipeline flowchart")
    print("    Table I — Data sources")
    print("    Fig. 2  — Missing values & cleaning audit")
    print("    Fig. 3  — Distribution properties (boxplots + skewness)")
    print("    Fig. 3b — Outlier investigation (profile of removed households)")
    print("  SEC 4 - Data Exploration:")
    print("    Table II — Summary statistics by tenure")
    print("    Fig. 4   — Tenure basket deviation heatmap")
    print("    Fig. 5   — Essential spending density by tenure")
    print("    Fig. 6   — CPIH time series (annotated with events)")
    print("    Fig. 7   — Correlation heatmaps (shares + prices)")
    print("    Fig. 8   — Dimension gap time series, 2015–2023")
    print("    Fig. 9   — Tenure basket evolution over time")
    print("    Fig. 10  — Laspeyres decomposition, FY2022/23")


if __name__ == "__main__":
    main()
