from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()
s = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
r = s.table('analytics').select('period_type, avg_social_alternative_score, alpha_vs_spy, beta_vs_spy, alpha_vs_qqq, beta_vs_qqq').eq('period_type', 'all_time').execute()
row = r.data[0]

print("\n" + "="*60)
print("ALL_TIME INTERVAL - QUICK CHECK")
print("="*60)
print(f"Social Score:      {row['avg_social_alternative_score']}")
print(f"Alpha vs SPY:      {row['alpha_vs_spy']}")
print(f"Beta vs SPY:       {row['beta_vs_spy']}")
print(f"Alpha vs QQQ:      {row['alpha_vs_qqq']}")
print(f"Beta vs QQQ:       {row['beta_vs_qqq']}")

if row['alpha_vs_spy'] != 0 or row['beta_vs_spy'] != 0:
    print("\n✅ Benchmark metrics FIXED!")
else:
    print("\n❌ Benchmark metrics still 0")

if row['avg_social_alternative_score'] != 0:
    print("✅ Social score non-zero!")
else:
    print("⚠️  Social score still 0")
print("="*60)
