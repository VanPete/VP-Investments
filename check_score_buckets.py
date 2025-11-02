"""
Check the exact structure of score_buckets data
"""
import os
import json
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Get Supabase credentials
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")

if not url or not key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_ANON_KEY environment variable")

supabase = create_client(url, key)

# Get analytics for 1d period - first check what columns exist
result = supabase.table("analytics").select("*").eq("period_type", "1d").execute()

if result.data and len(result.data) > 0:
    print("Available columns:")
    print(list(result.data[0].keys()))
    
    # Look for score bucket related data
    data = result.data[0]
    score_related = [k for k in data.keys() if 'score' in k.lower() or 'bucket' in k.lower()]
    
    print(f"\n\nScore/bucket related columns: {score_related}")
    
    for col in score_related:
        print(f"\n\n{col}:")
        value = data[col]
        print(f"Type: {type(value)}")
        if isinstance(value, dict):
            print(f"Keys: {list(value.keys())}")
            print(json.dumps(value, indent=2)[:500])  # First 500 chars
        else:
            print(f"Value: {value}")
else:
    print("No analytics data found for 1d period")
