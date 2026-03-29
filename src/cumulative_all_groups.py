"""
CUMULATIVE INFLATION BY ALL GROUPS (2015-2023)
===============================================

Shows the FULL story: which groups actually lost significantly different purchasing power?
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
inflation = pd.read_parquet(ROOT / "data" / "processed" / "group_inflation_rates.parquet")

print("=" * 100)
print("CUMULATIVE INFLATION 2015-2023: Which Groups Lost Most Purchasing Power?")
print("=" * 100)
print()

# Calculate cumulative for each archetype and value
results = []

for arch_name in sorted(inflation["archetype_name"].unique()):
    print(f"\n{'─' * 100}")
    print(f"{arch_name.upper()}")
    print(f"{'─' * 100}\n")

    arch_data = inflation[inflation["archetype_name"] == arch_name].copy()

    # For each archetype value, calculate cumulative
    group_cumulative = {}

    for arch_value in sorted(arch_data["archetype_value"].unique()):
        subset = arch_data[arch_data["archetype_value"] == arch_value]
        subset = subset.sort_values("year")

        if len(subset) > 0:
            rates = subset["inflation_rate"].values / 100
            cumulative = (np.prod(1 + rates) - 1) * 100
            group_cumulative[arch_value] = cumulative
            results.append({
                "Dimension": arch_name,
                "Group": arch_value,
                "Cumulative Loss (%)": cumulative
            })

    # Sort and display
    sorted_groups = sorted(group_cumulative.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Group':<40s} {'Cumulative Loss':<20s} {'Loss in GBP*':<20s}")
    print(f"{'─'*40} {'─'*20} {'─'*20}")

    for group, cumul in sorted_groups:
        # Estimate real income loss (varies by group)
        if "income" in arch_name:
            # Q1 = £20k, Q5 = £80k
            if str(group) in ["1.0", "1"]:
                est_income = 20000
            elif str(group) in ["5.0", "5"]:
                est_income = 80000
            else:
                est_income = 50000
        elif "tenure" in arch_name:
            # Renters typically £20k, owners £50k
            if "rent" in str(group).lower():
                est_income = 20000
            else:
                est_income = 50000
        else:
            est_income = 40000

        loss_gbp = est_income * (cumul / 100)

        print(f"{str(group):<40s} {cumul:>6.2f}%           {loss_gbp:>8,.0f}")

    # Show spread
    if len(group_cumulative) > 1:
        max_cumul = max(group_cumulative.values())
        min_cumul = min(group_cumulative.values())
        spread = max_cumul - min_cumul

        print(f"\n{'SPREAD (max - min)':<40s} {spread:>6.2f} pp")
        print(f"  → Highest loss: {max_cumul:.2f}%")
        print(f"  → Lowest loss:  {min_cumul:.2f}%")

# Summary table
print(f"\n\n{'=' * 100}")
print("SUMMARY: WHICH DIMENSIONS MATTER MOST?")
print(f"{'=' * 100}\n")

results_df = pd.DataFrame(results)
spread_by_dim = results_df.groupby("Dimension")["Cumulative Loss (%)"].agg(["min", "max", lambda x: x.max() - x.min()])
spread_by_dim.columns = ["Min Loss %", "Max Loss %", "Spread (pp)"]
spread_by_dim = spread_by_dim.sort_values("Spread (pp)", ascending=False)

print("Dimensions Ranked by Inequality (highest spread = most unequal):\n")
print(f"{'Dimension':<35s} {'Min':<12s} {'Max':<12s} {'Spread':<12s}")
print(f"{'─'*35} {'─'*12} {'─'*12} {'─'*12}")

for dim, row in spread_by_dim.iterrows():
    print(f"{dim:<35s} {row['Min Loss %']:>6.2f}%    {row['Max Loss %']:>6.2f}%    {row['Spread (pp)']:>6.2f} pp")

print(f"\n\nInterpretation:")
print(f"  • Spread < 0.5 pp: Not economically significant")
print(f"  • Spread 0.5-1.5 pp: Small but measurable differences")
print(f"  • Spread > 1.5 pp: Substantial inequality")

# Find the biggest losers
print(f"\n\n{'=' * 100}")
print("WHO LOST THE MOST PURCHASING POWER?")
print(f"{'=' * 100}\n")

# Get the worst affected groups
worst = results_df.nlargest(10, "Cumulative Loss (%)")
print(f"{'Dimension':<30s} {'Group':<30s} {'Loss %':<15s}")
print(f"{'─'*30} {'─'*30} {'─'*15}")

for _, row in worst.iterrows():
    print(f"{row['Dimension']:<30s} {str(row['Group']):<30s} {row['Cumulative Loss (%)']:>6.2f}%")

# Get the best protected groups
print(f"\n\nWho Lost the LEAST:")
best = results_df.nsmallest(10, "Cumulative Loss (%)")
print(f"{'Dimension':<30s} {'Group':<30s} {'Loss %':<15s}")
print(f"{'─'*30} {'─'*30} {'─'*15}")

for _, row in best.iterrows():
    print(f"{row['Dimension']:<30s} {str(row['Group']):<30s} {row['Cumulative Loss (%)']:>6.2f}%")

# The real question
print(f"\n\n{'=' * 100}")
print("IS THERE A SIGNIFICANT STORY?")
print(f"{'=' * 100}\n")

max_spread = spread_by_dim["Spread (pp)"].max()
max_dim = spread_by_dim["Spread (pp)"].idxmax()

print(f"Largest inequality: {max_dim}")
print(f"  → Maximum spread: {max_spread:.2f} percentage points")
print()

if max_spread < 0.5:
    print("❌ NO SIGNIFICANT STORY")
    print("   The differences are negligible (< 0.5 pp)")
    print("   Everyone lost roughly similar purchasing power")
elif max_spread < 1.5:
    print("⚠️  WEAK STORY")
    print(f"   Some groups lost ~{max_spread:.1f}% more than others")
    print("   This is real but small in absolute terms")
    print("   Matters mainly for extreme groups (top vs bottom)")
else:
    print("✅ SIGNIFICANT STORY")
    print(f"   Clear inequality: {max_spread:.1f} pp spread")
    print("   This is economically meaningful")

print("\n")
print("Practical example:")
print(f"  • If everyone lost 30% purchasing power on average")
print(f"  • With {max_spread:.2f} pp spread:")
print(f"    - Worst hit groups lost: {30 + max_spread/2:.1f}% purchasing power")
print(f"    - Best protected lost: {30 - max_spread/2:.1f}% purchasing power")
print(f"    - Difference: £{((max_spread/2) * 40000 / 100):,.0f} on a £40k income")
