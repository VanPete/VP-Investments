"""Check factor monitoring from latest pipeline run."""
import json

# Load latest factor monitoring log
with open('logs/factor_monitoring_20251029_064320.json') as f:
    data = json.load(f)

print("\n" + "=" * 60)
print("FACTOR MONITORING - Latest Run")
print("=" * 60)

# Overall stats
print(f"\nOverall:")
print(f"  Total factors calculated: {data.get('total_factors', 0)}")
print(f"  Successful: {data.get('successful_factors', 0)}")
print(f"  Failed: {data.get('failed_factors', 0)}")
print(f"  Success rate: {data.get('overall_success_rate', 0)*100:.1f}%")

# By group
print(f"\nBy Group:")
for group_name, stats in data.get('by_group', {}).items():
    print(f"\n  {group_name}:")
    print(f"    Total: {stats.get('total', 0)}")
    print(f"    Successful: {stats.get('successful', 0)}")
    print(f"    Failed: {stats.get('failed', 0)}")
    print(f"    Success rate: {stats.get('success_rate', 0)*100:.1f}%")

# Check news/macro factors specifically
nm = data.get('by_group', {}).get('news_macro', {})
if nm.get('failed', 0) > 0:
    print(f"\n{'='*60}")
    print("NEWS/MACRO FAILURES DETECTED!")
    print(f"{'='*60}")
    
    # Check individual factor failures
    print("\nFailed factors:")
    for factor_name, factor_stats in data.get('by_factor', {}).items():
        if 'news' in factor_name.lower() or 'macro' in factor_name.lower():
            if factor_stats.get('failed', 0) > 0:
                print(f"  - {factor_name}: {factor_stats.get('failed', 0)} failures")
