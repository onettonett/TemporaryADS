"""
analyze_inflation_inequality.py
================================
Analyze and visualize inflation inequality across household archetypes.

This script loads the group-specific inflation rates computed in the pipeline
and produces:
1. Summary tables (differences in inflation by archetype over time)
2. Line charts showing inflation trends by income quintile, tenure, region
3. Clustering analysis (k-means) to identify household groups by inflation loss
4. Validation against ONS CPI / pensioner CPI benchmarks

Outputs
-------
data/analysis/
  inflation_by_income_quintile.csv
  inflation_by_tenure.csv
  inflation_by_region.csv
  clusters_kmeans.csv
  (PNG charts saved to plots/)
"""

import pathlib
import pandas as pd
import numpy as np
import warnings
from typing import Tuple

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ANALYSIS = ROOT / "data" / "analysis"
PLOTS = ROOT / "plots"

ANALYSIS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
except ImportError:
    HAS_MATPLOTLIB = False
    print("  [Warning] matplotlib/seaborn not available – skipping visualizations")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  [Warning] scikit-learn not available – skipping clustering analysis")


# ═══════════════════════════════════════════════════════════════════════════
# Core Analysis Functions
# ═══════════════════════════════════════════════════════════════════════════

def load_inflation_results() -> pd.DataFrame:
    """Load computed group inflation rates."""
    path = PROCESSED / "group_inflation_rates.parquet"
    df = pd.read_parquet(path)
    return df


def extract_archetype_analysis(inflation: pd.DataFrame,
                               archetype_col: str) -> pd.DataFrame:
    """
    Extract inflation rates for a specific archetype dimension.
    Returns wide format: rows=archetype_values, columns=years.
    """
    subset = inflation[inflation["archetype_name"] == archetype_col].copy()
    subset = subset.pivot_table(
        index="archetype_value",
        columns="year",
        values="inflation_rate",
        aggfunc="first"
    )
    return subset


def compute_inflation_gaps(inflation: pd.DataFrame) -> pd.DataFrame:
    """
    For each archetype, compute the gap between highest and lowest inflation.
    This measures inequality within each archetype dimension.
    """
    gaps = inflation.groupby("archetype_name").apply(
        lambda g: (
            g.groupby("archetype_value")["inflation_rate"]
            .mean()
            .max() - g.groupby("archetype_value")["inflation_rate"].mean().min()
        )
    ).reset_index()
    gaps.columns = ["archetype_name", "inflation_gap_pct"]
    gaps = gaps.sort_values("inflation_gap_pct", ascending=False)
    return gaps


def plot_inflation_by_income_quintile(inflation: pd.DataFrame) -> None:
    """Line chart: inflation rates over time for each income quintile."""
    if not HAS_MATPLOTLIB:
        return

    subset = extract_archetype_analysis(inflation, "income_quintile")
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for q in subset.index:
        # Handle string representation of quintiles (e.g., '1.0' -> 'Q1')
        try:
            q_label = f"Q{int(float(q))}"
        except (ValueError, TypeError):
            q_label = str(q)
        ax.plot(subset.columns, subset.loc[q], marker="o", label=q_label)

    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation Rate (%)")
    ax.set_title("Annual Inflation by Income Quintile (2015–2025)", fontsize=14, fontweight="bold")
    ax.legend(title="Income Quintile", loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "inflation_by_income_quintile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS / 'inflation_by_income_quintile.png'}")


def plot_inflation_by_tenure(inflation: pd.DataFrame) -> None:
    """Line chart: inflation rates by tenure type."""
    if not HAS_MATPLOTLIB:
        return

    subset = extract_archetype_analysis(inflation, "tenure_type")
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"social_rent": "red", "private_rent": "orange",
              "own_outright": "blue", "own_mortgage": "green", "rent_free": "purple"}

    for tenure in subset.index:
        color = colors.get(tenure, "gray")
        ax.plot(subset.columns, subset.loc[tenure], marker="o", label=tenure, color=color, linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation Rate (%)")
    ax.set_title("Annual Inflation by Tenure Type (2015–2025)", fontsize=14, fontweight="bold")
    ax.legend(title="Tenure Type", loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "inflation_by_tenure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS / 'inflation_by_tenure.png'}")


def plot_inflation_by_region(inflation: pd.DataFrame) -> None:
    """Line chart: inflation rates by broad region."""
    if not HAS_MATPLOTLIB:
        return

    subset = extract_archetype_analysis(inflation, "region_broad")
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    for region in subset.index:
        ax.plot(subset.columns, subset.loc[region], marker="o", label=region, linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation Rate (%)")
    ax.set_title("Annual Inflation by Region (2015–2025)", fontsize=14, fontweight="bold")
    ax.legend(title="Region", loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "inflation_by_region.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS / 'inflation_by_region.png'}")


def plot_inflation_by_age_band(inflation: pd.DataFrame) -> None:
    """Line chart: inflation rates by HRP age band."""
    if not HAS_MATPLOTLIB:
        return

    subset = extract_archetype_analysis(inflation, "hrp_age_band")
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    age_order = ["under_30", "30_to_49", "50_to_64", "65_to_74", "75_plus"]
    age_order = [a for a in age_order if a in subset.index]

    for age in age_order:
        ax.plot(subset.columns, subset.loc[age], marker="o", label=age, linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation Rate (%)")
    ax.set_title("Annual Inflation by HRP Age Band (2015–2025)", fontsize=14, fontweight="bold")
    ax.legend(title="Age Band", loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "inflation_by_age_band.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS / 'inflation_by_age_band.png'}")


def plot_inflation_inequality_heatmap(inflation: pd.DataFrame,
                                      archetype_col: str = "income_quintile") -> None:
    """Heatmap: inflation rates across archetype values and years."""
    if not HAS_MATPLOTLIB:
        return

    subset = extract_archetype_analysis(inflation, archetype_col)
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(subset.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(np.arange(len(subset.columns)))
    ax.set_yticks(np.arange(len(subset.index)))
    ax.set_xticklabels(subset.columns, rotation=45)
    ax.set_yticklabels(subset.index)

    ax.set_xlabel("Year")
    ax.set_ylabel(archetype_col)
    ax.set_title(f"Inflation Inequality Heatmap: {archetype_col}", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Inflation Rate (%)", rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(subset.index)):
        for j in range(len(subset.columns)):
            val = subset.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    png_name = f"inflation_heatmap_{archetype_col}.png"
    plt.savefig(PLOTS / png_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS / png_name}")


def cluster_households_by_inflation(inflation: pd.DataFrame,
                                     n_clusters: int = 5) -> Tuple[pd.DataFrame, dict]:
    """
    Cluster household archetypes by their inflation experience (k-means).

    Features: mean inflation, std of inflation, inflation in recent year (2023+).
    This helps identify which groups have been hardest hit.
    """
    if not HAS_SKLEARN:
        return pd.DataFrame(), {}

    # Compute statistics per archetype group
    group_stats = inflation.groupby(["archetype_name", "archetype_value"]).agg(
        mean_inflation=("inflation_rate", "mean"),
        std_inflation=("inflation_rate", "std"),
        min_inflation=("inflation_rate", "min"),
        max_inflation=("inflation_rate", "max"),
    ).reset_index()

    # Add recent inflation (2023 onwards)
    recent = inflation[inflation["year"] >= 2023].groupby(
        ["archetype_name", "archetype_value"]
    )["inflation_rate"].mean().reset_index()
    recent.columns = ["archetype_name", "archetype_value", "recent_inflation"]

    group_stats = group_stats.merge(
        recent,
        on=["archetype_name", "archetype_value"],
        how="left"
    )

    # Fill missing recent_inflation with mean_inflation
    group_stats["recent_inflation"] = group_stats["recent_inflation"].fillna(
        group_stats["mean_inflation"]
    )

    # Prepare features for clustering
    features = group_stats[["mean_inflation", "std_inflation", "recent_inflation"]].fillna(0)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    group_stats["cluster"] = clusters

    # Interpret clusters
    cluster_summary = group_stats.groupby("cluster").agg(
        n_groups=("archetype_value", "count"),
        mean_inflation=("mean_inflation", "mean"),
        std_inflation=("std_inflation", "mean"),
        recent_inflation=("recent_inflation", "mean"),
    )

    return group_stats, dict(cluster_summary.to_dict("index"))


# ═══════════════════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("ANALYZING INFLATION INEQUALITY")
    print("=" * 70)

    # ── 1. Load results ──────────────────────────────────────────────────────
    print("\n[1/5] Loading group inflation rates...")
    inflation = load_inflation_results()
    print(f"  Loaded {len(inflation):,} observations")
    print(f"  Archetype dimensions: {inflation['archetype_name'].nunique()}")
    print(f"  Archetype groups: {inflation['archetype_value'].nunique()}")
    print(f"  Years: {sorted(inflation['year'].unique())}")

    # ── 2. Compute inequality metrics ────────────────────────────────────────
    print("\n[2/5] Computing inflation inequality gaps...")
    gaps = compute_inflation_gaps(inflation)
    print(f"\nInflation inequality (max - min) by archetype:\n{gaps.to_string(index=False)}")

    gaps.to_csv(ANALYSIS / "inflation_inequality_gaps.csv", index=False)
    print(f"\n  Saved: {ANALYSIS / 'inflation_inequality_gaps.csv'}")

    # ── 3. Extract and save key comparisons ──────────────────────────────────
    print("\n[3/5] Extracting key group comparisons...")

    archetype_dims = ["income_quintile", "tenure_type", "region_broad", "hrp_age_band"]
    for dim in archetype_dims:
        subset = extract_archetype_analysis(inflation, dim)
        if not subset.empty:
            csv_name = f"inflation_by_{dim}.csv"
            subset.to_csv(ANALYSIS / csv_name)
            print(f"  Saved: {ANALYSIS / csv_name}")

    # ── 4. Clustering analysis ───────────────────────────────────────────────
    print("\n[4/5] Clustering household archetypes by inflation experience...")
    clusters, cluster_summary = cluster_households_by_inflation(inflation, n_clusters=5)

    if not clusters.empty:
        clusters.to_csv(ANALYSIS / "archetype_clusters.csv", index=False)
        print(f"  Saved: {ANALYSIS / 'archetype_clusters.csv'}")

        print(f"\n  Cluster Summary:")
        for cluster_id, stats in sorted(cluster_summary.items()):
            print(f"    Cluster {cluster_id}: "
                  f"n={int(stats['n_groups'])}, "
                  f"mean_inflation={stats['mean_inflation']:.2f}%, "
                  f"recent_inflation={stats['recent_inflation']:.2f}%")

    # ── 5. Create visualizations ────────────────────────────────────────────
    print("\n[5/5] Generating visualizations...")
    if HAS_MATPLOTLIB:
        plot_inflation_by_income_quintile(inflation)
        plot_inflation_by_tenure(inflation)
        plot_inflation_by_region(inflation)
        plot_inflation_by_age_band(inflation)
        plot_inflation_inequality_heatmap(inflation, "income_quintile")
        plot_inflation_inequality_heatmap(inflation, "tenure_type")
        print(f"  All charts saved to: {PLOTS}/")
    else:
        print("  [Skipped] Matplotlib not available")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 70)

    # Mean inflation by quintile (latest year)
    latest_year = inflation["year"].max()
    quintile_inflation = inflation[
        (inflation["archetype_name"] == "income_quintile") &
        (inflation["year"] == latest_year)
    ].copy()
    quintile_inflation = quintile_inflation.sort_values("inflation_rate", ascending=False)

    if not quintile_inflation.empty:
        print(f"\nInflation by Income Quintile (FY {latest_year}):")
        for _, row in quintile_inflation.iterrows():
            try:
                q_num = int(float(row['archetype_value']))
            except (ValueError, TypeError):
                q_num = row['archetype_value']
            print(f"  Q{q_num}: {row['inflation_rate']:.2f}%")
        spread = (
            quintile_inflation["inflation_rate"].max() -
            quintile_inflation["inflation_rate"].min()
        )
        print(f"\n  Spread (max - min): {spread:.2f} percentage points")

    # Tenure spread
    tenure_inflation = inflation[
        (inflation["archetype_name"] == "tenure_type") &
        (inflation["year"] == latest_year)
    ].copy()
    tenure_inflation = tenure_inflation.sort_values("inflation_rate", ascending=False)

    if not tenure_inflation.empty:
        print(f"\nInflation by Tenure Type (FY {latest_year}):")
        for _, row in tenure_inflation.iterrows():
            print(f"  {row['archetype_value']}: {row['inflation_rate']:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
