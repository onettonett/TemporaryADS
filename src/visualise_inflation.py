"""
visualise_inflation.py
======================
Generate comprehensive visualisations for all archetype-group inflation
analyses.  Reads parquet outputs from compute_group_inflation.py and
lcf_expenditure_shares.parquet produced by wrangle_lcf.py.

All charts are saved as PNG at 150 dpi to data/processed/charts/.
"""

import pathlib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# matplotlib style setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_STYLES = ["seaborn-v0_8", "seaborn", "ggplot", "default"]
for _style in _STYLES:
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHARTS = PROCESSED / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# colour helpers

def _get_palette(n: int):
    """Return a list of n distinct colours from tab20/tab10."""
    cmap = plt.colormaps["tab20" if n > 10 else "tab10"]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# Data loaders

def load_data():
    """Load and return the three main parquet files."""
    infl = pd.read_parquet(PROCESSED / "group_inflation_rates.parquet")
    decomp = pd.read_parquet(PROCESSED / "inflation_decomposition.parquet")
    shares = pd.read_parquet(PROCESSED / "lcf_expenditure_shares.parquet")
    return infl, decomp, shares


# Utility

def _chain_cumulative(rates: pd.Series) -> float:
    """Chain-link annual rates (%) to a single cumulative % figure."""
    valid = rates.dropna()
    if len(valid) == 0:
        return np.nan
    return (np.prod(1 + valid / 100) - 1) * 100


def _weighted_means(grp: pd.DataFrame, share_cols: list[str],
                    weight_col: str | None) -> dict:
    """Compute weighted mean for each share column in a group DataFrame."""
    row = {}
    for sc in share_cols:
        if sc not in grp.columns:
            continue
        vals = grp[sc].fillna(0)
        if weight_col and weight_col in grp.columns:
            w = grp[weight_col].fillna(1)
            denom = w.sum()
            row[sc] = (vals * w).sum() / denom if denom > 0 else np.nan
        else:
            row[sc] = vals.mean()
    return row


def _save(fig, name: str) -> None:
    path = CHARTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Chart 1: Annual inflation line chart per archetype

def chart_annual_inflation(infl: pd.DataFrame) -> None:
    print("\n[Chart 1] Annual inflation line charts...")
    archetypes = infl["archetype_name"].unique()

    for arch in archetypes:
        sub = infl[infl["archetype_name"] == arch].copy()
        groups = sorted(sub["archetype_value"].unique())
        years = sorted(sub["year"].unique())

        fig, ax = plt.subplots(figsize=(10, 5))
        colours = _get_palette(len(groups))

        for g, col in zip(groups, colours):
            grp_data = sub[sub["archetype_value"] == g].sort_values("year")
            if len(grp_data) == 0:
                continue
            ax.plot(
                grp_data["year"],
                grp_data["inflation_rate"],
                marker="o",
                label=str(g),
                color=col,
                linewidth=1.8,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("%")
        ax.set_title(f"Annual Inflation Rate by {arch}")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title=arch, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        _save(fig, f"annual_inflation_{arch}.png")


# Chart 2: Cumulative inflation horizontal bar chart per archetype

def chart_cumulative_inflation(infl: pd.DataFrame) -> None:
    print("\n[Chart 2] Cumulative inflation bar charts...")
    archetypes = infl["archetype_name"].unique()

    for arch in archetypes:
        sub = infl[infl["archetype_name"] == arch].copy()
        groups = sub["archetype_value"].unique()

        cum_vals = {}
        for g in groups:
            rates = sub.loc[sub["archetype_value"] == g, "inflation_rate"]
            cum_vals[str(g)] = _chain_cumulative(rates)

        cum_series = pd.Series(cum_vals).dropna().sort_values(ascending=False)
        if len(cum_series) == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, max(4, len(cum_series) * 0.55)))
        colours = _get_palette(len(cum_series))
        bars = ax.barh(cum_series.index, cum_series.values, color=colours)

        for bar, val in zip(bars, cum_series.values):
            x_pos = val + (0.3 if val >= 0 else -0.3)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha=ha, fontsize=8)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Cumulative Inflation 2016–2024 (%)")
        ax.set_title(f"Cumulative Inflation by {arch}")
        fig.tight_layout()
        _save(fig, f"cumulative_inflation_{arch}.png")


# Chart 3: Crisis decomposition stacked bar for 2022

def chart_crisis_decomposition(decomp: pd.DataFrame) -> None:
    print("\n[Chart 3] Crisis decomposition charts (2022)...")
    archetypes = decomp["archetype_name"].unique()

    for arch in archetypes:
        sub = decomp[
            (decomp["archetype_name"] == arch) & (decomp["year"] == 2022)
        ].copy()

        if len(sub) == 0:
            print(f"    No 2022 data for {arch}, skipping.")
            continue

        coicop_cats = sub["coicop_label"].unique()
        groups = sorted(sub["archetype_value"].unique())
        colours = _get_palette(len(coicop_cats))
        coicop_colour = dict(zip(sorted(coicop_cats), colours))

        fig, ax = plt.subplots(figsize=(11, max(4, len(groups) * 0.6)))

        left_pos = np.zeros(len(groups))
        left_neg = np.zeros(len(groups))
        group_idx = {g: i for i, g in enumerate(groups)}

        for cat in sorted(coicop_cats):
            cat_sub = sub[sub["coicop_label"] == cat]
            contrib_vals = np.zeros(len(groups))
            for _, row in cat_sub.iterrows():
                idx = group_idx.get(row["archetype_value"])
                if idx is not None:
                    contrib_vals[idx] = row.get("contribution", 0)

            pos_vals = np.where(contrib_vals >= 0, contrib_vals, 0)
            neg_vals = np.where(contrib_vals < 0, contrib_vals, 0)

            ax.barh(groups, pos_vals, left=left_pos,
                    color=coicop_colour[cat], label=cat, height=0.7)
            ax.barh(groups, neg_vals, left=left_neg,
                    color=coicop_colour[cat], height=0.7)

            left_pos += pos_vals
            left_neg += neg_vals

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution (pp)")
        ax.set_title(f"2022 Inflation — COICOP Contributions by {arch}")
        ax.legend(title="COICOP", bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=7)
        fig.tight_layout()
        _save(fig, f"crisis_decomposition_{arch}.png")


# Chart 4: Basket composition stacked bar per archetype

def chart_basket_composition(shares: pd.DataFrame, infl: pd.DataFrame) -> None:
    print("\n[Chart 4] Basket composition charts...")
    share_cols = [c for c in shares.columns if c.startswith("share_")]
    # Filter to the 12 main COICOP division shares (exclude split sub-shares)
    main_share_cols = [
        c for c in share_cols
        if not any(c.endswith(s) for s in ["_actual_rent", "_energy_other"])
    ]
    if not main_share_cols:
        main_share_cols = share_cols

    arch_names = infl["archetype_name"].unique()
    colours = _get_palette(len(main_share_cols))
    share_colour = dict(zip(main_share_cols, colours))
    weight_col = "household_weight" if "household_weight" in shares.columns else None

    for arch in arch_names:
        if arch not in shares.columns:
            continue

        grp_shares = {
            str(grp_val): _weighted_means(grp_df, main_share_cols, weight_col)
            for grp_val, grp_df in shares.groupby(arch)
        }

        if not grp_shares:
            continue

        comp_df = pd.DataFrame(grp_shares).T
        comp_df = comp_df[[c for c in main_share_cols if c in comp_df.columns]]

        fig, ax = plt.subplots(figsize=(max(8, len(grp_shares) * 1.2), 6))
        bottom = np.zeros(len(comp_df))
        x = np.arange(len(comp_df))

        for sc in comp_df.columns:
            vals = comp_df[sc].fillna(0).values
            ax.bar(x, vals, bottom=bottom, color=share_colour.get(sc, "gray"),
                   label=sc.replace("share_", ""), width=0.7)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(comp_df.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Expenditure Share")
        ax.set_title(f"Basket Composition by {arch}")
        ax.legend(title="COICOP", bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=7)
        fig.tight_layout()
        _save(fig, f"basket_composition_{arch}.png")


# Chart 5: Cumulative crisis spread summary

def chart_cumulative_crisis_all_groups(infl: pd.DataFrame) -> None:
    print("\n[Chart 5] Cumulative crisis all-groups summary...")

    records = []
    for arch in infl["archetype_name"].unique():
        sub = infl[infl["archetype_name"] == arch]

        # Cumulative 2016-2024
        cum_by_group = {}
        for g in sub["archetype_value"].unique():
            rates = sub.loc[sub["archetype_value"] == g, "inflation_rate"]
            cum_by_group[g] = _chain_cumulative(rates)
        cum_vals = [v for v in cum_by_group.values() if pd.notna(v)]
        spread_cum = max(cum_vals) - min(cum_vals) if len(cum_vals) >= 2 else np.nan

        # Crisis: 2022 + 2023 combined
        crisis_by_group = {}
        for g in sub["archetype_value"].unique():
            crisis_rates = sub.loc[
                (sub["archetype_value"] == g) & (sub["year"].isin([2022, 2023])),
                "inflation_rate"
            ]
            crisis_by_group[g] = _chain_cumulative(crisis_rates)
        crisis_vals = [v for v in crisis_by_group.values() if pd.notna(v)]
        spread_crisis = (max(crisis_vals) - min(crisis_vals)
                         if len(crisis_vals) >= 2 else np.nan)

        records.append({
            "archetype": arch,
            "spread_cumulative": spread_cum,
            "spread_crisis": spread_crisis,
        })

    df_spread = pd.DataFrame(records).dropna(subset=["spread_cumulative"])
    df_spread = df_spread.sort_values("spread_cumulative", ascending=False)

    if len(df_spread) == 0:
        print("    No data for summary chart, skipping.")
        return

    x = np.arange(len(df_spread))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(df_spread) * 1.4), 6))
    bars1 = ax.bar(x - width / 2, df_spread["spread_cumulative"], width,
                   label="Spread: Cumulative 2016–2024", color="steelblue")
    bars2 = ax.bar(x + width / 2, df_spread["spread_crisis"].fillna(0), width,
                   label="Spread: Crisis 2022+2023", color="coral")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.1:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df_spread["archetype"], rotation=30, ha="right")
    ax.set_ylabel("Inflation Spread (pp, max−min across groups)")
    ax.set_title("Inflation Inequality by Archetype Dimension")
    ax.legend()
    fig.tight_layout()
    _save(fig, "cumulative_crisis_all_groups.png")


# Chart 6: Inequality over time for key archetypes

def chart_inequality_over_time(infl: pd.DataFrame) -> None:
    print("\n[Chart 6] Inequality over time...")
    KEY_ARCHETYPES = ["income_quintile", "tenure_type", "is_pensioner"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = _get_palette(len(KEY_ARCHETYPES))

    any_plotted = False
    for arch, col in zip(KEY_ARCHETYPES, colours):
        sub = infl[infl["archetype_name"] == arch]
        if len(sub) == 0:
            continue

        gap_by_year = []
        for yr in sorted(sub["year"].unique()):
            yr_sub = sub[sub["year"] == yr]["inflation_rate"].dropna()
            if len(yr_sub) >= 2:
                gap_by_year.append({"year": yr, "gap": yr_sub.max() - yr_sub.min()})

        if not gap_by_year:
            continue

        gap_df = pd.DataFrame(gap_by_year)
        ax.plot(gap_df["year"], gap_df["gap"], marker="o", label=arch,
                color=col, linewidth=1.8)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("    No data for inequality chart, skipping.")
        return

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation Gap (pp, max−min across groups)")
    ax.set_title("Inflation Inequality Over Time (Key Archetypes)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    _save(fig, "inequality_over_time.png")


# Chart 7: Basket shift heatmap

def chart_basket_shift_heatmap(shares: pd.DataFrame) -> None:
    print("\n[Chart 7] Basket shift heatmap...")
    share_cols = [c for c in shares.columns if c.startswith("share_")]
    # Main COICOP divisions only
    main_share_cols = [
        c for c in share_cols
        if not any(c.endswith(s) for s in ["_actual_rent", "_energy_other"])
    ]
    if not main_share_cols:
        main_share_cols = share_cols

    if "year" not in shares.columns:
        print("    No 'year' column in shares data, skipping heatmap.")
        return

    weight_col = "household_weight" if "household_weight" in shares.columns else None

    rows = [
        {"year": yr, **_weighted_means(grp, main_share_cols, weight_col)}
        for yr, grp in shares.groupby("year")
    ]

    heat_df = pd.DataFrame(rows).set_index("year")
    heat_df = heat_df[[c for c in main_share_cols if c in heat_df.columns]]
    heat_df.columns = [c.replace("share_", "") for c in heat_df.columns]

    if heat_df.empty:
        print("    Empty heatmap data, skipping.")
        return

    # Normalise each column to 0-1 for colour scaling within each COICOP
    heat_norm = heat_df.copy()
    for col in heat_norm.columns:
        col_min, col_max = heat_norm[col].min(), heat_norm[col].max()
        if col_max > col_min:
            heat_norm[col] = (heat_norm[col] - col_min) / (col_max - col_min)
        else:
            heat_norm[col] = 0.0

    fig, ax = plt.subplots(figsize=(max(14, len(heat_df.columns) * 1.1),
                                    max(5, len(heat_df) * 0.55)))
    cax = ax.imshow(heat_norm.values, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=1)

    # Annotate cells with raw share values
    for i in range(len(heat_df)):
        for j in range(len(heat_df.columns)):
            val = heat_df.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color="black")

    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(heat_df)))
    ax.set_yticklabels(heat_df.index.astype(str), fontsize=8)
    ax.set_title("National Mean Basket Composition Over Time\n"
                 "(colour scaled within each COICOP column)")
    fig.colorbar(cax, ax=ax, label="Relative share (within-column scaled)")
    fig.tight_layout()
    _save(fig, "basket_shift_heatmap.png")


# Main

def main() -> None:
    print("=" * 60)
    print("Inflation Visualisation Pipeline")
    print("=" * 60)
    print(f"  Output directory: {CHARTS}")

    print("\nLoading parquet files...")
    try:
        infl, decomp, shares = load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run the pipeline steps first (wrangle_mm23 -> wrangle_lcf -> compute_group_inflation).")
        return

    print(f"  group_inflation_rates: {len(infl):,} rows, "
          f"{infl['archetype_name'].nunique()} archetypes")
    print(f"  inflation_decomposition: {len(decomp):,} rows")
    print(f"  lcf_expenditure_shares: {len(shares):,} rows")

    chart_annual_inflation(infl)
    chart_cumulative_inflation(infl)
    chart_crisis_decomposition(decomp)
    chart_basket_composition(shares, infl)
    chart_cumulative_crisis_all_groups(infl)
    chart_inequality_over_time(infl)
    chart_basket_shift_heatmap(shares)

    saved = sorted(CHARTS.glob("*.png"))
    print(f"\nDone. {len(saved)} charts saved to {CHARTS}")


if __name__ == "__main__":
    main()
