"""Check what's NULL in analytics table and why"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

print("\n" + "="*80)
print("ANALYTICS TABLE - NULL COLUMN INVESTIGATION")
print("="*80)

# Get all columns from analytics table
result = supabase.table('analytics').select('*').eq('period_type', '1d').execute()

if result.data:
    row = result.data[0]
    
    # Categorize columns
    null_columns = []
    non_null_columns = []
    
    for key, value in row.items():
        if value is None or value == 'NULL':
            null_columns.append(key)
        else:
            non_null_columns.append(key)
    
    print(f"\n✅ NON-NULL COLUMNS ({len(non_null_columns)}):")
    for col in sorted(non_null_columns)[:20]:  # Show first 20
        print(f"   {col}: {row[col]}")
    if len(non_null_columns) > 20:
        print(f"   ... and {len(non_null_columns) - 20} more")
    
    print(f"\n❌ NULL COLUMNS ({len(null_columns)}):")
    for col in sorted(null_columns):
        print(f"   {col}")
    
    # Check specific important columns
    print("\n" + "="*80)
    print("SPECIFIC COLUMN VALUES (1d interval):")
    print("="*80)
    
    important = [
        'cagr', 'volatility', 'sortino_ratio', 'calmar_ratio',
        'rolling_sharpe_30d', 'benchmark_correlations', 'signal_correlations',
        'top_positive_pairs', 'top_negative_pairs',
        'alpha_vs_spy', 'beta_vs_spy', 'alpha_vs_qqq', 'beta_vs_qqq'
    ]
    
    for col in important:
        if col in row:
            print(f"   {col:30s}: {row[col]}")
        else:
            print(f"   {col:30s}: [COLUMN DOESN'T EXIST]")

# Check all_time too
print("\n" + "="*80)
print("ALL_TIME INTERVAL CHECK:")
print("="*80)

result_all = supabase.table('analytics').select('*').eq('period_type', 'all_time').execute()
if result_all.data:
    row_all = result_all.data[0]
    
    null_count = sum(1 for v in row_all.values() if v is None or v == 'NULL')
    print(f"   NULL columns: {null_count}/{len(row_all)}")
    
    print("\n   Key metrics:")
    for col in ['cagr', 'volatility', 'alpha_vs_spy', 'beta_vs_spy']:
        if col in row_all:
            print(f"      {col:20s}: {row_all[col]}")

print("\n" + "="*80)
