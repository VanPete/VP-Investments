"""Check how much historical data we have for all_time calculations"""
import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime, timedelta

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

print("\n" + "="*80)
print("HISTORICAL DATA ANALYSIS FOR ALL_TIME METRICS")
print("="*80)

# Count signals by age
result = supabase.table('signals').select('created_at').execute()

from datetime import timezone
today = datetime.now(timezone.utc)
age_buckets = {
    '< 1 day': 0,
    '1-3 days': 0,
    '3-7 days': 0,
    '7-14 days': 0,
    '14-30 days': 0,
    '30-90 days': 0,
    '> 90 days': 0
}

for signal in result.data:
    created_str = signal['created_at'].replace('Z', '+00:00')
    # Handle microseconds with variable length
    if '.' in created_str and '+' in created_str:
        parts = created_str.split('.')
        microseconds = parts[1].split('+')[0][:6].ljust(6, '0')  # Ensure 6 digits
        created_str = f"{parts[0]}.{microseconds}+{parts[1].split('+')[1]}"
    created = datetime.fromisoformat(created_str)
    age = (today - created).days
    
    if age < 1:
        age_buckets['< 1 day'] += 1
    elif age < 3:
        age_buckets['1-3 days'] += 1
    elif age < 7:
        age_buckets['3-7 days'] += 1
    elif age < 14:
        age_buckets['7-14 days'] += 1
    elif age < 30:
        age_buckets['14-30 days'] += 1
    elif age < 90:
        age_buckets['30-90 days'] += 1
    else:
        age_buckets['> 90 days'] += 1

print(f"\nSignal Age Distribution (Total: {len(result.data)}):")
for bucket, count in age_buckets.items():
    print(f"  {bucket:15s}: {count:4d} signals")

# Check how many performance records have data for each interval
perf_result = supabase.table('performance').select(
    'return_1d, return_3d, return_7d, return_10d, return_14d, return_30d, return_90d, '
    'spy_return_1d, spy_return_3d, spy_return_7d'
).execute()

interval_counts = {
    '1d': {'vp': 0, 'spy': 0},
    '3d': {'vp': 0, 'spy': 0},
    '7d': {'vp': 0, 'spy': 0},
    '10d': {'vp': 0, 'spy': 0},
    '14d': {'vp': 0, 'spy': 0},
    '30d': {'vp': 0, 'spy': 0},
    '90d': {'vp': 0, 'spy': 0}
}

for p in perf_result.data:
    for interval in ['1d', '3d', '7d', '10d', '14d', '30d', '90d']:
        if p.get(f'return_{interval}') is not None:
            interval_counts[interval]['vp'] += 1
        if p.get(f'spy_return_{interval}') is not None:
            interval_counts[interval]['spy'] += 1

print(f"\nInterval Data Availability (Total: {len(perf_result.data)} performance records):")
for interval, counts in interval_counts.items():
    print(f"  {interval:5s}: VP={counts['vp']:4d}, SPY={counts['spy']:4d}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Diagnose the issue
has_90d = interval_counts['90d']['vp'] > 0
has_30d = interval_counts['30d']['vp'] > 0
has_7d = interval_counts['7d']['vp'] > 0

if not has_90d:
    print("\n❌ NO 90-DAY DATA: Signals are too recent (< 90 days old)")
    print("   all_time metrics need signals that have completed all intervals")
if not has_30d:
    print("\n⚠️  LIMITED 30-DAY DATA: Most signals haven't reached 30-day mark")
if has_7d:
    print(f"\n✅ 7-DAY DATA EXISTS: {interval_counts['7d']['vp']} records have 7d returns")
    print("   This is good for basic analytics but not enough for all_time")

print(f"\nCONCLUSION:")
if has_90d:
    print("  ✅ System has mature signals - all_time should work")
else:
    print("  ❌ System is TOO NEW - need to wait for signals to mature")
    print("  ⏳ Come back in ~90 days for meaningful all_time metrics")

print("\n" + "="*80)
