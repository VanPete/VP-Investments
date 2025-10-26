"""Quick script to check if backtest data exists in Supabase"""
import os
from supabase import create_client

# Load from .env
from dotenv import load_dotenv
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

client = create_client(url, key)

# Check signal_runs
print("=== SIGNAL RUNS ===")
runs = client.table('signal_runs').select('id, run_timestamp, total_tickers, status').order('run_timestamp', desc=True).limit(10).execute()
print(f"Found {len(runs.data)} runs:")
for run in runs.data:
    print(f"  - {run['run_timestamp']}: {run['total_tickers']} tickers, status={run['status']}")

print("\n=== SIGNALS WITH BACKTEST DATA ===")
# Check if any signals have backtest data
signals = client.table('signals').select('ticker, return_1d, return_7d, return_30d, spy_return_1d, backtest_status').limit(10).execute()
print(f"Sample of {len(signals.data)} signals:")
for sig in signals.data:
    print(f"  - {sig['ticker']}: 1D={sig['return_1d']}, 7D={sig['return_7d']}, 30D={sig['return_30d']}, SPY_1D={sig['spy_return_1d']}, status={sig['backtest_status']}")

# Check total count with backtest data
count_query = client.table('signals').select('id', count='exact').not_.is_('return_1d', 'null').execute()
print(f"\nTotal signals with return_1d data: {count_query.count}")
