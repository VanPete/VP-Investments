"""Debug why all_time benchmark metrics are NULL"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

print("\n" + "="*80)
print("DEBUGGING ALL_TIME BENCHMARK METRICS")
print("="*80)

# Check what's in performance table for benchmark returns
result = supabase.table('performance').select(
    'signal_id, baseline_date, '
    'return_1d, return_3d, return_7d, return_10d, return_14d, return_30d, return_90d, '
    'spy_return_1d, spy_return_3d, spy_return_7d, spy_return_10d, spy_return_14d, spy_return_30d, spy_return_90d'
).limit(5).execute()

print(f"\nSample from performance table ({len(result.data)} records):")
for i, row in enumerate(result.data, 1):
    print(f"\n{i}. {row['signal_id'][:8]}... @ {row['baseline_date']}")
    print(f"   VP Returns:  1d={row['return_1d']}, 3d={row['return_3d']}, 7d={row['return_7d']}")
    print(f"   SPY Returns: 1d={row['spy_return_1d']}, 3d={row['spy_return_3d']}, 7d={row['spy_return_7d']}")

# Count how many non-NULL benchmark returns exist
count_query = """
SELECT 
    COUNT(*) as total_rows,
    COUNT(spy_return_1d) as spy_1d_count,
    COUNT(spy_return_3d) as spy_3d_count,
    COUNT(spy_return_7d) as spy_7d_count,
    COUNT(qqq_return_1d) as qqq_1d_count,
    COUNT(qqq_return_3d) as qqq_3d_count,
    COUNT(qqq_return_7d) as qqq_7d_count
FROM performance
"""
result = supabase.rpc('execute_sql', {'query': count_query}).execute()
print(f"\nBenchmark return counts:")
print(result.data)

# Check analytics table to see what's being stored
analytics = supabase.table('analytics').select(
    'period_type, alpha_vs_spy, beta_vs_spy, alpha_vs_qqq, beta_vs_qqq'
).eq('period_type', 'all_time').execute()

print(f"\nAnalytics table (all_time):")
if analytics.data:
    row = analytics.data[0]
    print(f"  Alpha vs SPY: {row['alpha_vs_spy']}")
    print(f"  Beta vs SPY:  {row['beta_vs_spy']}")
    print(f"  Alpha vs QQQ: {row['alpha_vs_qqq']}")
    print(f"  Beta vs QQQ:  {row['beta_vs_qqq']}")
else:
    print("  No data found!")

print("\n" + "="*80)
