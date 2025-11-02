"""Debug social score in performance analytics calculation"""
import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime, timedelta

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

# Get the most recent analytics period
analytics_result = supabase.table('analytics').select('period_start, period_end, run_id').order('created_at', desc=True).limit(1).execute()

if not analytics_result.data:
    print("No analytics data found")
    exit(1)

period_start = analytics_result.data[0]['period_start']
period_end = analytics_result.data[0]['period_end']
run_id = analytics_result.data[0]['run_id']

print(f"Period: {period_start} to {period_end}")
print(f"Run ID: {run_id}")

# Fetch performance data the same way Phase 7 does
query = supabase.table('performance').select('''
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
        created_at,
        run_id
    )
''').gte('baseline_date', period_start).lte('baseline_date', period_end)

if run_id:
    query = query.eq('signals.run_id', run_id)

result = query.execute()

print(f"\nTotal performance records: {len(result.data)}")

if result.data:
    print("\nFirst 5 records:")
    for i, p in enumerate(result.data[:5]):
        signals = p.get('signals', {})
        print(f"\n{i+1}. Ticker: {signals.get('ticker', 'N/A')}")
        print(f"   Social Score: {signals.get('social_alternative_score', 'N/A')}")
        print(f"   Risk Score: {signals.get('risk_stability_score', 'N/A')}")
        print(f"   Overall Score: {signals.get('overall_score', 'N/A')}")
    
    # Calculate average like Phase 7 does
    import numpy as np
    signals_list = [p['signals'] for p in result.data]
    social_scores = [s.get('social_alternative_score') for s in signals_list if s.get('social_alternative_score') is not None]
    
    print(f"\n\nSocial scores extracted: {len(social_scores)} out of {len(signals_list)} signals")
    if social_scores:
        print(f"Average (raw): {np.mean(social_scores)}")
        print(f"Average (rounded to 2): {round(np.mean(social_scores), 2)}")
        print(f"Min: {min(social_scores)}")
        print(f"Max: {max(social_scores)}")
    else:
        print("No social scores found!")
