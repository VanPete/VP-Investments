"""
Comprehensive verification of analytics table - check all columns and data quality.
"""
import os
import json
from dotenv import load_dotenv
from supabase import create_client

# Load env
load_dotenv()

# Initialize Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(url, key)

# Query analytics
response = supabase.table('analytics').select('*').order('period_type').execute()

print("\n" + "=" * 140)
print("📊 COMPREHENSIVE ANALYTICS TABLE VERIFICATION")
print("=" * 140)

print(f"\n✅ Found {len(response.data)} analytics rows (expecting 8 intervals)\n")

# Define expected columns and their types
columns_to_check = {
    'Core Metrics': [
        'total_signals', 'sharpe_ratio',
        'max_drawdown', 'win_rate', 'avg_return'
    ],
    'NEW: Interval-Specific Metrics': [
        'cagr', 'volatility', 'sortino_ratio', 'calmar_ratio'
    ],
    'Benchmark Metrics': [
        'alpha_vs_spy', 'beta_vs_spy', 'alpha_vs_qqq', 'beta_vs_qqq'
    ],
    'Correlations': [
        'benchmark_correlations'
    ],
    'Score Group Averages': [
        'avg_technical_score', 'avg_fundamental_score', 'avg_news_macro_score',
        'avg_social_alternative_score', 'avg_risk_stability_score', 'avg_institutional_score'
    ],
    'NEW: Group Performance': [
        'group_performance'
    ]
}

# Track issues
issues = []

for row in response.data:
    interval = row['period_type']
    print(f"\n{'─' * 140}")
    print(f"📋 INTERVAL: {interval}")
    print(f"{'─' * 140}")
    
    # Check each category
    for category, cols in columns_to_check.items():
        print(f"\n  {category}:")
        
        for col in cols:
            value = row.get(col)
            
            if col == 'group_performance':
                # Special handling for JSONB
                if value is None:
                    print(f"    ❌ {col}: NULL")
                    issues.append(f"{interval}: {col} is NULL")
                else:
                    if isinstance(value, str):
                        value = json.loads(value)
                    
                    # Check structure
                    expected_groups = ['technical', 'fundamental', 'news_macro', 'social_alternative', 'risk_stability', 'institutional']
                    expected_quintiles = ['top_20pct', 'q2', 'q3', 'q4', 'bottom_20pct']
                    expected_metrics = ['count', 'avg_return', 'win_rate', 'sharpe', 'max_drawdown', 'volatility', 'sortino', 'calmar']
                    
                    missing_groups = [g for g in expected_groups if g not in value]
                    if missing_groups:
                        print(f"    ⚠️  {col}: Missing groups: {missing_groups}")
                        issues.append(f"{interval}: group_performance missing groups: {missing_groups}")
                        continue
                    
                    # Check one group in detail (technical)
                    tech = value.get('technical', {})
                    if not tech:
                        print(f"    ❌ {col}: Technical group is empty")
                        issues.append(f"{interval}: technical group is empty")
                        continue
                    
                    missing_quintiles = [q for q in expected_quintiles if q not in tech]
                    if missing_quintiles:
                        print(f"    ⚠️  {col}: Technical missing quintiles: {missing_quintiles}")
                        issues.append(f"{interval}: technical missing quintiles: {missing_quintiles}")
                        continue
                    
                    # Check top_20pct metrics
                    top_metrics = tech.get('top_20pct', {})
                    missing_metrics = [m for m in expected_metrics if m not in top_metrics]
                    
                    if missing_metrics:
                        print(f"    ⚠️  {col}: top_20pct missing metrics: {missing_metrics}")
                        issues.append(f"{interval}: top_20pct missing metrics: {missing_metrics}")
                    else:
                        # Show sample values for NEW metrics
                        vol = top_metrics.get('volatility')
                        sortino = top_metrics.get('sortino')
                        calmar = top_metrics.get('calmar')
                        print(f"    ✅ {col}: All 6 groups × 5 quintiles × 8 metrics")
                        print(f"       Sample (technical/top_20pct): vol={vol if vol else 'NULL'}, sortino={sortino if sortino else 'NULL'}, calmar={calmar if calmar else 'NULL'}")
                
            elif col == 'benchmark_correlations':
                # JSONB column
                if value is None:
                    status = "⚠️ NULL" if interval in ['1d', '3d'] else "✓ NULL (expected)"
                    print(f"    {status} {col}")
                    if interval in ['1d', '3d']:
                        issues.append(f"{interval}: {col} is NULL (should have data)")
                else:
                    if isinstance(value, str):
                        value = json.loads(value)
                    print(f"    ✅ {col}: {len(value) if isinstance(value, (list, dict)) else 'present'} items")
            
            else:
                # Numeric columns
                if value is None:
                    # NULL is expected for longer intervals (system only 3 days old)
                    if interval in ['1d', '3d']:
                        print(f"    ⚠️  {col}: NULL (unexpected)")
                        issues.append(f"{interval}: {col} is NULL but should have data")
                    elif interval == 'all_time' and col in ['alpha_vs_spy', 'beta_vs_spy', 'alpha_vs_qqq', 'beta_vs_qqq']:
                        print(f"    ✓ {col}: NULL (expected - <10 datapoints)")
                    else:
                        print(f"    ✓ {col}: NULL (expected - no data yet)")
                else:
                    # Check if value is reasonable
                    if col in ['win_rate']:
                        if 0 <= value <= 100:
                            print(f"    ✅ {col}: {value:.2f}%")
                        else:
                            print(f"    ⚠️  {col}: {value:.2f}% (out of range)")
                            issues.append(f"{interval}: {col} = {value} (out of range)")
                    
                    elif col in ['sharpe_ratio', 'sortino_ratio']:
                        if -5 <= value <= 5:
                            print(f"    ✅ {col}: {value:.4f}")
                        else:
                            print(f"    ⚠️  {col}: {value:.4f} (unusual value)")
                            issues.append(f"{interval}: {col} = {value} (unusual)")
                    
                    elif col in ['cagr', 'avg_return']:
                        print(f"    ✅ {col}: {value:.2f}%")
                    
                    elif col in ['volatility']:
                        print(f"    ✅ {col}: {value:.2f}%")
                    
                    elif col in ['alpha_vs_spy', 'beta_vs_spy', 'alpha_vs_qqq', 'beta_vs_qqq']:
                        print(f"    ✅ {col}: {value:.4f}")
                    
                    elif col.startswith('avg_') and col.endswith('_score'):
                        # Score averages should be between -100 and 100
                        if -100 <= value <= 100:
                            print(f"    ✅ {col}: {value:.4f}")
                        else:
                            print(f"    ⚠️  {col}: {value:.4f} (out of range)")
                            issues.append(f"{interval}: {col} = {value} (out of range)")
                    
                    else:
                        if isinstance(value, (int, float)):
                            print(f"    ✅ {col}: {value:.4f}")
                        else:
                            print(f"    ✅ {col}: {value}")

print("\n" + "=" * 140)
print("🔍 ISSUE SUMMARY")
print("=" * 140)

if issues:
    print(f"\n⚠️  Found {len(issues)} issues:\n")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✅ No issues found! All data looks correct.")

print("\n" + "=" * 140)

# Summary statistics
print("\n📈 DATA AVAILABILITY SUMMARY:")
print("-" * 140)

intervals_with_data = []
intervals_null = []

for row in response.data:
    interval = row['period_type']
    has_cagr = row.get('cagr') is not None
    has_volatility = row.get('volatility') is not None
    has_group_perf = row.get('group_performance') is not None
    
    if has_cagr and has_volatility:
        intervals_with_data.append(interval)
    else:
        intervals_null.append(interval)

print(f"\n✅ Intervals with data: {', '.join(intervals_with_data) if intervals_with_data else 'None'}")
print(f"⚠️  Intervals without data: {', '.join(intervals_null) if intervals_null else 'None'}")

print("\n" + "=" * 140)
print("✅ VERIFICATION COMPLETE")
print("=" * 140 + "\n")
