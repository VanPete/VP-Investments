"""Check news/macro success rate from latest factor monitoring."""
import json

# Load latest factor monitoring log
with open('logs/factor_monitoring_20251029_064320.json') as f:
    data = json.load(f)

print("\n" + "=" * 70)
print("NEWS/MACRO SUCCESS RATE - Latest Pipeline Run")
print("=" * 70)

# Overall stats
print(f"\nOverall Pipeline:")
print(f"  Total success rate: {data['overall_success_rate']*100:.1f}%")
print(f"  Total calculations: {data['total_calculations']}")

# News/Macro group
nm = data['group_summary']['news_macro']
print(f"\nNews/Macro Group:")
print(f"  Success rate: {nm['avg_success_rate']*100:.1f}%")
print(f"  Total factors: {nm['total_factors']}")
print(f"  Problematic count: {nm['problematic_count']}")

if nm['problematic_count'] > 0:
    print(f"\n  Problematic factors:")
    for factor in nm['problematic_factors'][:10]:
        print(f"    - {factor}")
else:
    print(f"\n  ✅ NO PROBLEMATIC FACTORS!")

# Compare to previous run
print(f"\n" + "=" * 70)
print("COMPARISON TO PREVIOUS RUN")
print("=" * 70)
print("  Before fix: 11.1% success rate (16/18 factors failing)")
print(f"  After fix:  {nm['avg_success_rate']*100:.1f}% success rate ({nm['problematic_count']}/{nm['total_factors']} factors failing)")
print(f"  Improvement: +{nm['avg_success_rate']*100 - 11.1:.1f} percentage points")
