"""Check what's actually stored for social scores in analytics table"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

print("\n" + "="*80)
print("SOCIAL SCORE STORAGE INVESTIGATION")
print("="*80)

# Check analytics table
result = supabase.table('analytics').select(
    'period_type, '
    'avg_social_alternative_score, '
    'avg_technical_score, '
    'avg_fundamental_score, '
    'avg_news_macro_score, '
    'avg_risk_stability_score'
).execute()

print(f"\nAnalytics Table ({len(result.data)} rows):")
for row in result.data:
    print(f"\n  {row['period_type']:10s}:")
    print(f"    Technical:   {row['avg_technical_score']}")
    print(f"    Fundamental: {row['avg_fundamental_score']}")
    print(f"    News/Macro:  {row['avg_news_macro_score']}")
    print(f"    Social:      {row['avg_social_alternative_score']} ⚠️")
    print(f"    Risk:        {row['avg_risk_stability_score']}")

# Check what signals have for social scores
signals = supabase.table('signals').select(
    'social_alternative_score'
).limit(50).execute()

social_scores = [s['social_alternative_score'] for s in signals.data if s['social_alternative_score'] is not None]
print(f"\n\nSignals Table (sample of 50):")
print(f"  Non-NULL scores: {len(social_scores)}")
if social_scores:
    avg = sum(social_scores) / len(social_scores)
    print(f"  Average: {avg:.6f}")
    print(f"  Range: {min(social_scores):.6f} to {max(social_scores):.6f}")
    print(f"  Sample values: {social_scores[:10]}")
else:
    print("  ❌ NO NON-NULL SOCIAL SCORES FOUND")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if result.data:
    any_social_non_zero = any(row['avg_social_alternative_score'] != 0 for row in result.data)
    
    if not any_social_non_zero:
        print("\n❌ All analytics rows have social_score = 0")
        
        if social_scores:
            print("✅ But signals DO have non-zero social scores")
            print("\nPOSSIBLE CAUSES:")
            print("1. Signals being used don't have social scores yet (timing issue)")
            print("2. JOIN in Phase 7 not finding correct signals")
            print("3. Calculation logic has bug")
            print("4. Column type truncating small values")
        else:
            print("❌ Signals ALSO have no social scores")
            print("\nROOT CAUSE: Social scores not being calculated in earlier phases!")
    else:
        print("\n✅ Some intervals have non-zero social scores!")
        for row in result.data:
            if row['avg_social_alternative_score'] != 0:
                print(f"  {row['period_type']}: {row['avg_social_alternative_score']}")

print("\n" + "="*80)
