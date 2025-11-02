"""Check what Phase 7 would calculate for social scores"""
import os
from dotenv import load_dotenv
from supabase import create_client
import numpy as np

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

print("\n" + "="*80)
print("SIMULATING PHASE 7 SOCIAL SCORE CALCULATION")
print("="*80)

# Fetch performance data the way Phase 7 does
from datetime import datetime, timezone

period_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
period_end = datetime.now(timezone.utc)

result = supabase.table('performance').select('''
    *,
    signals!inner(
        ticker,
        overall_score,
        technical_score,
        fundamental_score,
        news_macro_score,
        social_alternative_score,
        risk_stability_score,
        institutional_smart_money_score,
        created_at
    )
''').gte('baseline_date', period_start.isoformat()).lte('baseline_date', period_end.isoformat()).execute()

performance_data = result.data

print(f"\nPerformance records fetched: {len(performance_data)}")

# Extract signals the way Phase 7 does
signals = [p['signals'] for p in performance_data]
print(f"Signals extracted: {len(signals)}")

# Get social scores the way Phase 7 does
social_scores = [s['social_alternative_score'] for s in signals]
print(f"\nSocial scores list length: {len(social_scores)}")

# Filter None values (like _safe_avg does)
valid_social = [v for v in social_scores if v is not None]
print(f"Non-NULL social scores: {len(valid_social)}")

if valid_social:
    # Calculate average (like _safe_avg does)
    avg = np.mean(valid_social)
    print(f"\nRaw average: {avg}")
    print(f"Rounded to 2 decimals: {round(avg, 2)}")
    
    # Check distribution
    print(f"\nDistribution:")
    print(f"  Min: {min(valid_social):.6f}")
    print(f"  Max: {max(valid_social):.6f}")
    print(f"  Median: {np.median(valid_social):.6f}")
    print(f"  Std Dev: {np.std(valid_social):.6f}")
    
    # Sample values
    print(f"\nFirst 10 values:")
    for i, val in enumerate(valid_social[:10], 1):
        print(f"  {i}. {val:.6f}")
else:
    print("\n❌ NO VALID SOCIAL SCORES!")

print("\n" + "="*80)
print("COMPARISON WITH STORED VALUE")
print("="*80)

# Check what's in analytics
analytics = supabase.table('analytics').select('avg_social_alternative_score').eq('period_type', '1d').execute()
if analytics.data:
    stored = analytics.data[0]['avg_social_alternative_score']
    print(f"\nStored in analytics (1d): {stored}")
    
    if valid_social:
        expected = round(np.mean(valid_social), 2)
        print(f"Expected from calculation: {expected}")
        
        if stored != expected:
            print(f"\n⚠️  MISMATCH! Stored {stored} but expected {expected}")
        elif stored == 0:
            print(f"\n⚠️  Both are 0 - average might be very close to 0!")
            print(f"   Raw average: {np.mean(valid_social):.6f}")
            print(f"   This rounds to: {round(np.mean(valid_social), 2)}")
        else:
            print(f"\n✅ Match! Calculation is correct.")

print("\n" + "="*80)
