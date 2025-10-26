import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")

if not url or not key:
    raise ValueError("Missing Supabase credentials")

client: Client = create_client(url, key)

print("=== CHECKING ALL SIGNAL RUNS IN SUPABASE ===\n")

# Query all runs
response = client.table('signal_runs').select('*').order('run_timestamp', desc=True).execute()

if response.data:
    print(f"Found {len(response.data)} runs in database:\n")
    for run in response.data:
        print(f"ID: {run['id']}")
        print(f"Timestamp: {run['run_timestamp']}")
        print(f"Status: {run['status']}")
        print(f"Total Tickers: {run.get('total_tickers', 'N/A')}")
        
        # Count signals for this run
        signals_count = client.table('signals').select('id', count='exact').eq('run_id', run['id']).execute()
        print(f"Signals Count: {signals_count.count if hasattr(signals_count, 'count') else 'N/A'}")
        print("-" * 50)
else:
    print("No runs found in database")

print("\n=== CHECKING PIPELINE JSON FILES ===\n")

# Check what JSON files exist
results_dir = 'frontend/public/results'
if os.path.exists(results_dir):
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files:\n")
    for f in sorted(json_files, reverse=True):
        print(f"  - {f}")
else:
    print(f"Directory {results_dir} does not exist")
