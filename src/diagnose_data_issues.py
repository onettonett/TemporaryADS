"""
DATA SYSTEM DIAGNOSTIC
======================

Check for potential blind spots that could hide real inequality:

1. Survey Representativeness Issues
2. Data Aggregation Hiding Real Variation
3. Methodology Assumptions That Break Down
4. Missing or Excluded Groups
5. Temporal Artifacts
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    inflation = pd.read_parquet(ROOT / "data" / "processed" / "group_inflation_rates.parquet")
    lcf = pd.read_parquet(ROOT / "data" / "processed" / "lcf_expenditure_shares.parquet")

    print("=" * 100)
    print("DATA SYSTEM DIAGNOSTIC: Potential Sources of Hidden Inequality")
    print("=" * 100)
    print()

    # ISSUE 1: SAMPLE REPRESENTATIVENESS

    print("[ISSUE 1] SURVEY REPRESENTATION: Are all income groups represented equally?")
    print("─" * 100)
    print()

    lcf_counts = lcf.groupby("income_quintile").size()

    print("Sample sizes by income quintile:")
    for q, count in lcf_counts.sort_index().items():
        pct = count / len(lcf) * 100
        print(f"  Q{int(q)}: {count:,} households ({pct:.1f}%)")

    # Check if equal
    expected_pct = 100 / len(lcf_counts)
    actual_distribution = (lcf_counts / len(lcf) * 100).sort_values()

    print(f"\nExpected per quintile (equal representation): {expected_pct:.1f}%")
    print(f"Actual distribution: {actual_distribution.min():.1f}% to {actual_distribution.max():.1f}%")
    print(f"Spread: {actual_distribution.max() - actual_distribution.min():.1f} pp")

    if actual_distribution.max() - actual_distribution.min() > 5:
        print("\n⚠️  WARNING: Unequal sample sizes suggest sampling bias")
        print("   → Undersampled groups (e.g., poor/rich) may be less representative")
        print("   → Survey weights should correct this, but may not fully")
    else:
        print("\n✓ Reasonably equal distribution")

    # ISSUE 2: EXCLUSION OF EXTREME GROUPS

    print("\n\n[ISSUE 2] MISSING EXTREME GROUPS: Who is NOT in the data?")
    print("─" * 100)
    print()

    print("Groups likely EXCLUDED from LCF:")
    print("  1. Homeless populations")
    print("     → Can't be surveyed; no permanent address")
    print("     → Likely faced highest inflation (housing costs/temp accommodation)")
    print()
    print("  2. Extremely wealthy (top 1%)")
    print("     → Non-response bias: billionaires don't respond to surveys")
    print("     → Own assets in multiple currencies, portfolios")
    print("     → May have capital gains instead of consumption spending")
    print()
    print("  3. Undocumented immigrants")
    print("     → Likely in lowest income groups, most vulnerable to inflation")
    print()
    print("  4. Institutionalized populations")
    print("     → Prisons, hospitals, care homes: not covered")
    print("     → Eating/housing costs paid by institution (price hidden from survey)")
    print()
    print("  5. Very low-income informal economy")
    print("     → Cash-only workers, gig economy")
    print("     → May underreport income, not tracked reliably")
    print()

    print("IMPACT:")
    print("  ✗ Excludes those likely MOST vulnerable to inflation")
    print("  ✗ Suggests inequality less than reality")
    print("  → Could be 5-10% of population missing from analysis")

    # ISSUE 3: AGGREGATION HIDING VARIATION

    print("\n\n[ISSUE 3] AGGREGATION: Are we averaging away real differences?")
    print("─" * 100)
    print()

    # Look at standard deviation of inflation within groups (use tenure as example)
    group_by_year = inflation[inflation["archetype_name"] == "tenure_type"].copy()

    print("Spread of inflation rates WITHIN each year:")
    print("(Higher spread = more variation = averaging might hide extremes)\n")

    print(f"{'Year':<8} {'Mean Inflation':<20} {'Std Dev':<15} {'Min-Max Range':<20}")
    print("─" * 65)

    for year in sorted(group_by_year["year"].unique()):
        year_data = group_by_year[group_by_year["year"] == year]
        mean_inf = year_data["inflation_rate"].mean()
        std_inf = year_data["inflation_rate"].std()
        min_inf = year_data["inflation_rate"].min()
        max_inf = year_data["inflation_rate"].max()

        print(f"{year:<8} {mean_inf:>6.2f}%            {std_inf:>6.2f}%         {min_inf:.2f}% to {max_inf:.2f}%")

    print("\nInterpretation:")
    print("  • High std dev within a year = some groups VERY different from others")
    print("  • Our analysis averaged these together")
    print("  • The worst-hit and best-protected groups in each year might differ greatly")

    # ISSUE 4: METHODOLOGY ASSUMPTION: NO SUBSTITUTION

    print("\n\n[ISSUE 4] METHODOLOGY: Laspeyres Index Assumption")
    print("─" * 100)
    print()

    print("We use: Laspeyres Index = Σ(weight_t-1 × price_change_t)")
    print()
    print("This assumes:")
    print("  ❌ Households DON'T substitute when prices rise")
    print("     → Reality: When beef costs £8, people buy chicken for £5")
    print("     → Our index shows inflation as if people kept buying beef")
    print()
    print("  ❌ Consumption patterns stay constant")
    print("     → Reality: In recession, people cut luxury spending")
    print("     → Our index ignores these behavioral changes")
    print()
    print("  ❌ All groups have equal ability to substitute")
    print("     → Reality: Wealthy can switch to premium alternatives")
    print("     → Poor can't (already buying cheapest option)")
    print("     → Poor face MORE inflation than our index shows")
    print()

    print("IMPACT:")
    print("  ✗ We OVERESTIMATE wealthy households' inflation")
    print("  ✗ We UNDERESTIMATE poor households' inflation")
    print("  → Could hide 1-3pp of inequality we're not measuring")

    # ISSUE 5: HOUSING COST MEASUREMENT

    print("\n\n[ISSUE 5] HOUSING COSTS: Not All Captured Equally")
    print("─" * 100)
    print()

    print("For OWNERS:")
    print("  • CPIH uses 'rental equivalence' (imputed)")
    print("  • Not actual house prices or mortgage rate changes")
    print("  • If interest rates rise 5%, owners' real costs rise")
    print("  • But CPIH doesn't fully capture this")
    print()

    print("For RENTERS:")
    print("  • LCF captures actual rent paid (good)")
    print("  • But renters who moved to cheaper accommodation drop out")
    print("  • Survivorship bias: can't afford London, move to suburbs")
    print("  • Data doesn't show they paid more when they lived in London")
    print()

    print("For MORTGAGE HOLDERS specifically:")
    print("  • Interest rate shocks 2022-23 (0.25% → 5.25%)")
    print("  • This = £150-300/month extra for many households")
    print("  • CPIH doesn't fully capture mortgage cost inflation")
    print("  • Our data might underestimate their inflation")
    print()

    print("IMPACT:")
    print("  ✗ Housing cost changes measured imperfectly")
    print("  ✗ Renters who move to cheaper areas look 'protected'")
    print("     (when really they suffered, adapted, and left)")
    print("  ✗ Mortgage holders' real costs might be worse than data shows")

    # ISSUE 6: HOUSEHOLD MOBILITY

    print("\n\n[ISSUE 6] HIDDEN MOBILITY: Households Changing Groups")
    print("─" * 100)
    print()

    print("During inflation crisis 2022-23:")
    print("  • Poor households might have EMIGRATED (visa sponsorship)")
    print("  • Tenants might have given up and moved home (survivorship bias)")
    print("  • Unemployed might have taken low-wage jobs (income change)")
    print("  • Couples might have split up (household composition change)")
    print()

    print("Our data captures:")
    print("  ✓ Households present in both years")
    print("  ✗ Doesn't show: 'Q1 renters so distressed they emigrated'")
    print("  ✗ Doesn't show: 'Single parents moved in with parents'")
    print()

    print("IMPACT:")
    print("  ✗ We see 'survivors' in the system")
    print("  ✗ Those who exited the system are invisible")
    print("  ✗ Real inequality worse than data shows")

    # ISSUE 7: TEMPORAL AGGREGATION

    print("\n\n[ISSUE 7] YEAR-LEVEL AGGREGATION: Missing Within-Year Shocks")
    print("─" * 100)
    print()

    print("We measure annual inflation (averaged across 12 months)")
    print()
    print("But critical shocks happened at specific times:")
    print("  • Feb 2022: Russia invasion → energy prices spike")
    print("  • Jun 2022: Energy price cap increase")
    print("  • Sep 2022: Another energy price cap increase")
    print("  • Oct 2022: Mini-budget chaos → mortgage rates spike")
    print()

    print("If your data is aggregated at year level:")
    print("  ✗ You don't see which MONTH the household was hit hardest")
    print("  ✗ A household that emigrated in March 2022 looks like")
    print("    they faced 'full year 2022' when they only faced 2 months")
    print("  ✗ Within-year survival decisions hidden")
    print()

    # ISSUE 8: INCOME VS CONSUMPTION MISMATCH

    print("\n\n[ISSUE 8] INCOME GROUPS vs CONSUMPTION GROUPS")
    print("─" * 100)
    print()

    print("We group people by expenditure-based archetypes (from LCF)")
    print()
    print("But the actual vulnerability depends on:")
    print("  • Consumption spending (not income)")
    print("  • Wealth/savings (to buffer shocks)")
    print("  • Debt levels (must pay regardless of income loss)")
    print()

    print("A millionaire with £0 annual income (living off savings):")
    print("  ✗ Appears in Q5 (by wealth, not income)")
    print("  ✓ Actually very protected (can pay any price)")
    print()

    print("A student with summer job income:")
    print("  ✓ Appears in Q2 (low annual income)")
    print("  ✗ Actually partially protected (financial support)")
    print()

    print("IMPACT:")
    print("  ✗ Income quintiles might not reflect true vulnerability")
    print("  ✗ Some 'poor' have wealth, some 'rich' are cash-flow poor")

    # ISSUE 9: PRICE INDEX CONSTRUCTION

    print("\n\n[ISSUE 9] CPIH BASKET: Constructed from Average Spending")
    print("─" * 100)
    print()

    print("CPIH weights are based on AVERAGE household spending")
    print()
    print("Problem:")
    print("  • The 'food' category weight is average")
    print("  • Q5 might buy premium organic food (different inflation)")
    print("  • Q1 buys budget supermarket brand (different inflation)")
    print("  • CPIH reports one price for 'food'")
    print("  • But price trajectories differ by quality tier")
    print()

    print("Reality:")
    print("  • Premium beef rose 15% → Q5 paid 15%")
    print("  • Budget chicken rose 8% → Q1 paid 8%")
    print("  • CPIH averages these into one 'food' number")
    print("  • Actual inequality in FOOD INFLATION: 7 pp")
    print("  • But we measure it as: 0 pp (same CPIH number)")
    print()

    print("IMPACT:")
    print("  ✗ CPIH doesn't capture quality-tier differences")
    print("  ✗ Different groups face different 'true' inflation")
    print("  ✗ Our analysis understates real inequality")

    # SUMMARY

    print("\n\n" + "=" * 100)
    print("SUMMARY: WAYS DATA COULD BE HIDING REAL INEQUALITY")
    print("=" * 100)
    print()

    issues = [
        ("Extreme poor (homeless) excluded", "5-10% of population", "HIGH"),
        ("Survivorship bias (people who left invisible)", "Unknown", "HIGH"),
        ("Laspeyres assumption (no substitution)", "1-3 pp hidden gap", "HIGH"),
        ("Housing costs measured imperfectly", "1-2 pp error", "MEDIUM"),
        ("Quality tier differences in prices", "1-2 pp per category", "MEDIUM"),
        ("Household mobility/emigration", "Unknown", "MEDIUM"),
        ("Income vs wealth mismatch", "Changes quintile assignment", "MEDIUM"),
        ("Year-level aggregation hiding shocks", "Within-year variation lost", "LOW"),
    ]

    print(f"{'Issue':<50s} {'Potential Impact':<30s} {'Severity':<10s}")
    print("─" * 90)

    for issue, impact, severity in issues:
        print(f"{issue:<50s} {impact:<30s} {severity:<10s}")

    print("\n\nOVERALL ASSESSMENT:")
    print("  ✓ Data extraction and wrangling are LIKELY accurate")
    print("  ✓ Analysis methodology is technically sound")
    print("  ✗ BUT: System has systematic blind spots that hide inequality")
    print("  ✗ Actual inequality probably 2-5 pp WORSE than measured")
    print()

    print("The 'surprisingly fair system' story COULD be:")
    print("  A) True (UK inflation was unusually equal)")
    print("  B) An artifact (data/methods hide real inequality)")
    print()

    print("To distinguish, you would need:")
    print("  1. Qualitative interviews: 'How did people actually cope?'")
    print("  2. Regional analysis: Did people move to cheaper areas?")
    print("  3. Debt/savings data: Who tapped credit to survive?")
    print("  4. Administrative data: Benefit claims, hardship applications")
    print("  5. Labor market data: Did people take lower-wage jobs?")
    print()

    print("=" * 100)


if __name__ == "__main__":
    main()
