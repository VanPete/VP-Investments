"""
Check the actual avg_return values in score buckets
"""
import os
import json
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")

if not url or not key:
    raise ValueError("Missing credentials")

supabase = create_client(url, key)

# Get 1d analytics
result = supabase.table("analytics").select("score_bucket_performance").eq("period_type", "1d").execute()

if result.data and len(result.data) > 0:
    buckets = result.data[0]["score_bucket_performance"]
    
    print("Score Bucket Raw Values (1d interval):")
    print("=" * 60)
    for bucket_name, bucket_data in buckets.items():
        print(f"\n{bucket_name.upper()}:")
        print(f"  count: {bucket_data['count']}")
        print(f"  avg_return: {bucket_data['avg_return']}")
        print(f"  avg_return * 100: {bucket_data['avg_return'] * 100}")
        print(f"  win_rate: {bucket_data['win_rate']}")
        print(f"  win_rate * 100: {bucket_data['win_rate'] * 100}")
