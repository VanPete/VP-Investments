import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")

if not url or not key:
    raise ValueError("Missing Supabase credentials")

client: Client = create_client(url, key)

print("=== CHECKING signal_runs TABLE SCHEMA ===\n")

# Try to get one row to see the columns
response = client.table('signal_runs').select('*').limit(1).execute()

if response.data and len(response.data) > 0:
    print("Current columns in signal_runs table:")
    for key in response.data[0].keys():
        print(f"  - {key}")
else:
    print("Table is empty, checking with dummy query...")
    
print("\n=== SAMPLE ROW ===")
print(response.data[0] if response.data else "No data")
