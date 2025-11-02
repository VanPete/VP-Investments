"""Check performance record status."""
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))

# Get latest run
run = client.table('signal_runs').select('id').order('run_timestamp', desc=True).limit(1).execute()
run_id = run.data[0]['id']
print(f"Latest run: {run_id}\n")

# Get performance records
perf = client.table('performance').select(
    'status, baseline_date, intervals_completed, signals!inner(ticker, run_id)'
).eq('signals.run_id', run_id).limit(10).execute()

print(f"Found {len(perf.data)} performance records:\n")
for p in perf.data:
    ticker = p['signals']['ticker']
    status = p.get('status', 'N/A')
    completed = p.get('intervals_completed') or []
    print(f"  {ticker:6s} | Status: {status:15s} | Completed: {len(completed)}/7 intervals")
