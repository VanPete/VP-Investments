"""Quick script to check market_cap_category constraint"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

# Query constraint definition
query = """
SELECT 
    conname AS constraint_name,
    pg_get_constraintdef(c.oid) AS constraint_definition
FROM pg_constraint c
JOIN pg_namespace n ON n.oid = c.connamespace
JOIN pg_class cl ON cl.oid = c.conrelid
WHERE 
    n.nspname = 'public' 
    AND cl.relname = 'signals'
    AND conname LIKE '%market_cap%';
"""

result = supabase.rpc('exec_sql', {'query': query}).execute()
print("Market Cap Category Constraint:")
print(result.data)
