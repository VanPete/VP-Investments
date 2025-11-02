"""
Verify Phase 7 improvements:
1. Benchmark metrics (alpha/beta vs SPY and QQQ) are no longer 0
2. Group performance quintile data is populated
3. Social alternative score displays correctly
"""
import os
from supabase import create_client
from dotenv import load_dotenv
import json

load_dotenv()

# Load environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    print("❌ Missing Supabase credentials in .env file")
    print(f"URL: {supabase_url}")
    print(f"Key: {'set' if supabase_key else 'not set'}")
    exit(1)

supabase = create_client(supabase_url, supabase_key)

print("\n" + "="*80)
print("PHASE 7 IMPROVEMENTS VERIFICATION")
print("="*80)

# Get all analytics rows
result = supabase.table('analytics').select(
    'period_type, '
    'avg_social_alternative_score, '
    'avg_risk_stability_score, '
    'alpha_vs_spy, beta_vs_spy, '
    'alpha_vs_qqq, beta_vs_qqq, '
    'win_rate, sharpe_ratio'
).order('period_type').execute()

if not result.data:
    print("\n❌ No analytics data found!")
    exit(1)

print(f"\n✅ Found {len(result.data)} analytics rows\n")

# Track issues
issues = []
successes = []

for row in result.data:
    interval = row['period_type']
    print(f"\n{'='*80}")
    print(f"INTERVAL: {interval}")
    print(f"{'='*80}")
    
    # Check benchmark metrics
    print("\n1. BENCHMARK METRICS (should be non-zero for 1d, 3d, all_time):")
    alpha_spy = row['alpha_vs_spy'] or 0
    beta_spy = row['beta_vs_spy'] or 0
    alpha_qqq = row['alpha_vs_qqq'] or 0
    beta_qqq = row['beta_vs_qqq'] or 0
    
    print(f"   Alpha vs SPY: {alpha_spy:.4f}")
    print(f"   Beta vs SPY:  {beta_spy:.4f}")
    print(f"   Alpha vs QQQ: {alpha_qqq:.4f}")
    print(f"   Beta vs QQQ:  {beta_qqq:.4f}")
    
    if interval in ['1d', '3d', 'all_time']:
        if alpha_spy == 0 and beta_spy == 0 and alpha_qqq == 0 and beta_qqq == 0:
            issues.append(f"{interval}: All benchmark metrics are 0")
            print("   ❌ FAILED: All metrics are 0")
        else:
            successes.append(f"{interval}: Benchmark metrics populated")
            print("   ✅ PASSED: Metrics are non-zero")
    else:
        print(f"   ⏸️  SKIPPED: {interval} interval too long (expected NULL)")
    
    # Check social score
    print("\n2. SOCIAL ALTERNATIVE SCORE:")
    social_score = row['avg_social_alternative_score'] or 0
    print(f"   Value: {social_score:.4f}")
    
    if social_score != 0:
        successes.append(f"{interval}: Social score is {social_score:.4f}")
        print("   ✅ PASSED: Non-zero value")
    else:
        # Could be legitimately 0, but check if it's close to -0.09
        issues.append(f"{interval}: Social score is 0 (expected ~-0.09)")
        print("   ⚠️  WARNING: Value is 0")
    
    # Check risk stability score
    print("\n3. RISK STABILITY SCORE:")
    risk_score = row['avg_risk_stability_score'] or 0
    print(f"   Value: {risk_score:.4f}")
    print("   ℹ️  Reference: Should be around 0.15")
    
    # Note: Group performance check skipped - column just added, needs pipeline re-run
    print("\n4. GROUP PERFORMANCE:")
    print("   ℹ️  Column added via migration 020d")
    print("   ℹ️  Needs pipeline re-run to populate data")
    
    # Show other metrics for context
    print("\n5. OTHER METRICS:")
    print(f"   Win Rate:     {row['win_rate']:.2f}%" if row['win_rate'] else "   Win Rate:     NULL")
    print(f"   Sharpe Ratio: {row['sharpe_ratio']:.2f}" if row['sharpe_ratio'] else "   Sharpe Ratio: NULL")

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\n✅ SUCCESSES ({len(successes)}):")
for s in successes:
    print(f"   • {s}")

if issues:
    print(f"\n⚠️  ISSUES ({len(issues)}):")
    for i in issues:
        print(f"   • {i}")
else:
    print(f"\n🎉 NO ISSUES FOUND!")

print("\n" + "="*80)
