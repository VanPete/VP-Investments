"""Comprehensive verification of all Phase 7 improvements"""
import os
from dotenv import load_dotenv
from supabase import create_client
import json

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

print("\n" + "="*80)
print("PHASE 7 IMPROVEMENTS - COMPREHENSIVE VERIFICATION")
print("="*80)

# Get all_time and 1d intervals
result = supabase.table('analytics').select(
    'period_type, '
    'avg_social_alternative_score, '
    'alpha_vs_spy, beta_vs_spy, '
    'alpha_vs_qqq, beta_vs_qqq, '
    'group_performance'
).in_('period_type', ['all_time', '1d']).execute()

for row in result.data:
    interval = row['period_type']
    print(f"\n{'='*80}")
    print(f"INTERVAL: {interval.upper()}")
    print(f"{'='*80}")
    
    # 1. Benchmark Metrics
    print("\n1. BENCHMARK METRICS:")
    print(f"   Alpha vs SPY: {row['alpha_vs_spy']:.4f}" if row['alpha_vs_spy'] else "   Alpha vs SPY: NULL")
    print(f"   Beta vs SPY:  {row['beta_vs_spy']:.4f}" if row['beta_vs_spy'] else "   Beta vs SPY:  NULL")
    print(f"   Alpha vs QQQ: {row['alpha_vs_qqq']:.4f}" if row['alpha_vs_qqq'] else "   Alpha vs QQQ: NULL")
    print(f"   Beta vs QQQ:  {row['beta_vs_qqq']:.4f}" if row['beta_vs_qqq'] else "   Beta vs QQQ:  NULL")
    
    if row['alpha_vs_spy'] != 0 or row['beta_vs_spy'] != 0:
        print("   ✅ PASSED: Non-zero benchmark metrics!")
    else:
        print("   ❌ FAILED: Still showing 0")
    
    # 2. Social Score
    print("\n2. SOCIAL ALTERNATIVE SCORE:")
    print(f"   Value: {row['avg_social_alternative_score']}")
    if row['avg_social_alternative_score'] != 0:
        print("   ✅ PASSED: Non-zero value!")
    else:
        print("   ⚠️  WARNING: Still showing 0")
    
    # 3. Group Performance
    print("\n3. GROUP PERFORMANCE (Quintile Analysis):")
    gp = row['group_performance']
    if gp and isinstance(gp, dict) and len(gp) > 0:
        print(f"   ✅ PASSED: {len(gp)} groups found!")
        
        # Show details for first group
        groups = list(gp.keys())
        if groups:
            first_group = groups[0]
            print(f"\n   Sample: {first_group}")
            if first_group in gp and isinstance(gp[first_group], dict):
                for quintile, metrics in list(gp[first_group].items())[:2]:  # Show first 2 quintiles
                    print(f"      {quintile}:")
                    print(f"         Count:       {metrics.get('count', 0)}")
                    print(f"         Avg Return:  {metrics.get('avg_return', 0):.2f}%")
                    print(f"         Win Rate:    {metrics.get('win_rate', 0):.1f}%")
                    print(f"         Sharpe:      {metrics.get('sharpe', 0):.2f}")
    else:
        print("   ❌ FAILED: No group performance data")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Final summary
all_time = [r for r in result.data if r['period_type'] == 'all_time'][0]
one_d = [r for r in result.data if r['period_type'] == '1d'][0]

issues = []
successes = []

if all_time['alpha_vs_spy'] != 0 or all_time['beta_vs_spy'] != 0:
    successes.append("✅ all_time benchmark metrics FIXED!")
else:
    issues.append("❌ all_time benchmark metrics still 0")

if one_d['alpha_vs_spy'] != 0 or one_d['beta_vs_spy'] != 0:
    successes.append("✅ 1d benchmark metrics working")
else:
    issues.append("❌ 1d benchmark metrics = 0")

if all_time.get('group_performance') and len(all_time.get('group_performance', {})) > 0:
    successes.append("✅ all_time group_performance populated!")
else:
    issues.append("❌ all_time group_performance empty")

if one_d.get('group_performance') and len(one_d.get('group_performance', {})) > 0:
    successes.append("✅ 1d group_performance populated!")
else:
    issues.append("❌ 1d group_performance empty")

if all_time['avg_social_alternative_score'] != 0:
    successes.append("✅ all_time social score non-zero!")
else:
    issues.append("⚠️  all_time social score = 0")

print("\nSUCCESSES:")
for s in successes:
    print(f"  {s}")

if issues:
    print("\nISSUES:")
    for i in issues:
        print(f"  {i}")

print("\n" + "="*80)
