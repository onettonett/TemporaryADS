"""
HCI/CPIH exploration.
"""

from pathlib import Path
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots" / "exploration"
ANALYSIS = ROOT / "data" / "analysis" / "exploration"
PLOTS.mkdir(parents=True, exist_ok=True)
ANALYSIS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

HCI_FILES = {
    "tenure": ROOT / "data" / "interim" / "hci_tenure_index.parquet",
    "retirement": ROOT / "data" / "interim" / "hci_retirement_index.parquet",
    "children": ROOT / "data" / "interim" / "hci_children_index.parquet",
}


def load_hci():
    frames = {}
    for name, path in HCI_FILES.items():
        if path.exists():
            frames[name] = pd.read_parquet(path)
    if not frames:
        raise FileNotFoundError("No HCI files found")
    return frames


def load_cpih():
    df = pd.read_parquet(ROOT / "data" / "processed" / "cpih_monthly_indices.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def yoy(df, group_col, value_col):
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["coicop_code"].isin([0, "0", "00"])]
    data = data.sort_values(["group", "date"])
    data["yoy"] = data.groupby("group")[value_col].pct_change(12) * 100
    data["year"] = data["date"].dt.year
    return data.dropna(subset=["yoy"])


def save_fig(fig, name):
    fig.savefig(PLOTS / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def step_headline(hci_frames, cpih):
    frames = []
    for df in hci_frames.values():
        sub = df[df["coicop_code"].isin([0, "0", "00"])][["date", "group", "value"]]
        frames.append(sub)
    cpih_all = cpih[["date", "all_items"]].rename(columns={"all_items": "value"})
    cpih_all["group"] = "CPIH"
    frames.append(cpih_all)
    combined = pd.concat(frames)

    base = combined[combined["date"].dt.year == 2015].groupby("group")["value"].first()
    combined["rebased"] = combined["value"] / combined["group"].map(base) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for grp in sorted(combined["group"].unique()):
        gdf = combined[combined["group"] == grp]
        ax.plot(gdf["date"], gdf["rebased"], label=grp, linewidth=2)
    ax.set_title("HCI vs CPIH (rebased 2015=100)")
    ax.set_ylabel("Index")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "hci_headline")


def step_group_yoy(hci_frames):
    for name, df in hci_frames.items():
        y = yoy(df, "group", "value")
        annual = y.groupby(["group", "year"])["yoy"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        for grp in sorted(annual["group"].unique()):
            sub = annual[annual["group"] == grp]
            ax.plot(sub["year"], sub["yoy"], marker="o", label=grp)
        ax.set_title(f"HCI YoY by {name}")
        ax.set_ylabel("%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        save_fig(fig, f"hci_yoy_{name}")


def step_gap_vs_cpih(hci_frames, cpih):
    cpih_y = yoy(cpih.assign(group="CPIH", coicop_code="00"), "group", "all_items").rename(columns={"yoy": "baseline"})
    gaps = []
    for df in hci_frames.values():
        y = yoy(df, "group", "value")
        y = y.merge(cpih_y[["date", "baseline"]], on="date", how="inner")
        y["gap"] = y["yoy"] - y["baseline"]
        gaps.append(y)
    gdf = pd.concat(gaps)
    fig, ax = plt.subplots(figsize=(12, 6))
    for grp in sorted(gdf["group"].unique()):
        sub = gdf[gdf["group"] == grp]
        ax.plot(sub["date"], sub["gap"], label=grp)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Gap vs CPIH (YoY)")
    ax.set_ylabel("pp")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    save_fig(fig, "gap_vs_cpih")


def step_cpih_drivers(cpih):
    cats = ["electricity_gas_fuels", "food_non_alcoholic", "transport", "restaurants_hotels", "recreation_culture"]
    cpih = cpih.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in cats:
        yoy_series = cpih[col].pct_change(12) * 100
        ax.plot(cpih["date"], yoy_series, label=col, linewidth=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("CPIH drivers (YoY)")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, "cpih_drivers")


def main():
    hci_frames = load_hci()
    cpih = load_cpih()
    step_headline(hci_frames, cpih)
    step_group_yoy(hci_frames)
    step_gap_vs_cpih(hci_frames, cpih)
    step_cpih_drivers(cpih)
    print(f"Saved plots to {PLOTS} and summaries to {ANALYSIS}")


if __name__ == "__main__":
    main()
