"""
Quick script to check what run_type values are valid in the database.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment
load_dotenv()

supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')

# Connect to Supabase
supabase = create_client(supabase_url, supabase_key)

# Query existing runs to see what run_types are accepted
print("Checking existing runs table for valid run_type values...")
print("=" * 60)

try:
    # Get distinct run_type values from existing runs
    result = supabase.table('runs').select('run_type').execute()
    
    if result.data:
        unique_types = set(row.get('run_type') for row in result.data if row.get('run_type'))
        print(f"Found {len(result.data)} existing runs")
        print(f"\nUnique run_type values:")
        for run_type in sorted(unique_types):
            print(f"  - {run_type}")
    else:
        print("No existing runs found")
        
except Exception as e:
    print(f"Error querying runs: {e}")

print("\n" + "=" * 60)
print("\nTrying to find the constraint definition...")

# Try to query the constraint directly
try:
    # This might not work with Supabase client, but worth a try
    result = supabase.rpc('exec_sql', {
        'sql': """
        SELECT conname, consrc 
        FROM pg_constraint 
        WHERE conname = 'runs_run_type_check'
        """
    }).execute()
    print(f"Constraint info: {result.data}")
except Exception as e:
    print(f"Could not query constraint (expected): {e}")
