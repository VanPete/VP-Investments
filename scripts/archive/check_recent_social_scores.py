"""Check social scores in signals table"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

# Get recent signals
result = supabase.table('signals').select(
    'ticker, social_alternative_score'
).order('created_at', desc=True).limit(10).execute()

print("\nRecent signals social_alternative_score:")
print("-" * 50)
for row in result.data:
    print(f"{row['ticker']:6s}: {row['social_alternative_score']}")

# Check if any are non-zero
all_scores = [row['social_alternative_score'] for row in result.data if row['social_alternative_score'] is not None]
if all_scores:
    import numpy as np
    print(f"\nAverage: {np.mean(all_scores):.4f}")
    print(f"Min: {min(all_scores):.4f}")
    print(f"Max: {max(all_scores):.4f}")
    print(f"Non-zero count: {len([s for s in all_scores if s != 0])}")
