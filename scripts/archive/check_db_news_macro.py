"""Check news/macro factor values in database."""
import os
from supabase import create_client

# Get Supabase credentials
url = os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
key = os.environ.get('NEXT_PUBLIC_SUPABASE_ANON_KEY')

if not url or not key:
    print("ERROR: Missing Supabase credentials!")
    print("Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY")
    exit(1)

supabase = create_client(url, key)

# Get latest signal run
response = supabase.table('signal_runs').select('*').order('created_at', desc=True).limit(1).execute()

if not response.data:
    print("No signal runs found!")
    exit(1)

run_id = response.data[0]['id']
print(f"\nLatest run: {run_id}")
print(f"Created: {response.data[0]['created_at']}")
print(f"Signals: {response.data[0]['total_signals']}")

# Get news/macro factors for NVDA (top ranked)
print(f"\n{'='*70}")
print("NVDA News/Macro Factors (Raw Values)")
print(f"{'='*70}")

response = supabase.table('factor_news_macro').select('*').eq('run_id', run_id).eq('ticker', 'NVDA').execute()

if response.data:
    factors = response.data[0]
    print(f"\nAll factors for NVDA:")
    for key, value in sorted(factors.items()):
        if key not in ['id', 'ticker', 'run_id', 'created_at']:
            print(f"  {key:35} = {value}")
else:
    print("No news/macro factors found for NVDA!")

# Check a few more tickers
print(f"\n{'='*70}")
print("News Sentiment Values Across Tickers")
print(f"{'='*70}\n")

response = supabase.table('factor_news_macro').select('ticker, news_sentiment, news_sentiment_consensus, vix_level, treasury_yield_10y').eq('run_id', run_id).limit(10).execute()

if response.data:
    for row in response.data:
        ticker = row['ticker']
        ns = row.get('news_sentiment')
        nsc = row.get('news_sentiment_consensus')
        vix = row.get('vix_level')
        treasury = row.get('treasury_yield_10y')
        print(f"{ticker:6} | news_sent={ns} | consensus={nsc} | vix={vix} | 10y={treasury}")
else:
    print("No data!")
