"""
Quick check of recent signals data to verify improvements
"""
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

# Try multiple key names
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_ANON_KEY') or os.getenv('supabase.anon_key')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: SUPABASE_URL or SUPABASE_KEY not found in .env")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Get 10 most recent signals
recent = supabase.table('signals').select(
    'ticker,beta,upvotes,macd_line,bollinger_upper,weighted_score,financial_score,created_at'
).order('created_at', desc=True).limit(10).execute()

print("="*80)
print("RECENT SIGNALS DATA CHECK (Last 10)")
print("="*80)
print("\nChecking if recent signals have the data that older ones are missing:\n")

for i, signal in enumerate(recent.data, 1):
    print(f"{i}. {signal['ticker']} (created: {signal['created_at'][:10]})")
    print(f"   Beta: {signal['beta']}")
    print(f"   Upvotes: {signal['upvotes']}")
    print(f"   MACD Line: {signal['macd_line']}")
    print(f"   Bollinger Upper: {signal['bollinger_upper']}")
    print(f"   Weighted Score: {signal['weighted_score']}")
    print(f"   Financial Score: {signal['financial_score']}")
    print()

# Check a sample from older signals
print("="*80)
print("OLDER SIGNALS DATA CHECK (Sample from beginning)")
print("="*80)

old = supabase.table('signals').select(
    'ticker,beta,upvotes,macd_line,bollinger_upper,created_at'
).order('created_at', asc=True).limit(5).execute()

print("\nOlder signals (first 5 in database):\n")
for i, signal in enumerate(old.data, 1):
    print(f"{i}. {signal['ticker']} (created: {signal['created_at'][:10]})")
    print(f"   Beta: {signal['beta']}")
    print(f"   Upvotes: {signal['upvotes']}")
    print(f"   MACD Line: {signal['macd_line']}")
    print(f"   Bollinger Upper: {signal['bollinger_upper']}")
    print()

print("="*80)
print("\nCONCLUSION:")
print("If recent signals have data and old ones don't, the calculations are working!")
print("The 100% NULL columns are just not yet implemented features.")
print("="*80)
