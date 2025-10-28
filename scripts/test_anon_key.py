"""Test if ANON key can access Supabase tables (RLS check)"""
import os
from supabase import create_client, Client

# Use ANON key like the frontend does
SUPABASE_URL = "https://rdkxwoqevjicupmefbem.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJka3h3b3FldmppY3VwbWVmYmVtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2NDA4NzIsImV4cCI6MjA3NjIxNjg3Mn0._vl2RNIunavY_egYndewij_DksgOUbXrQwhNTXdTDiE"

print("Testing ANON key access (like frontend)...")
print("=" * 60)

# Create client with ANON key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Test 1: Query signal_runs
print("\n1. Testing signal_runs...")
try:
    result = supabase.table('signal_runs').select('id, run_timestamp').limit(1).execute()
    print(f"✅ SUCCESS: Found {len(result.data)} signal_runs")
    if result.data:
        print(f"   Sample: {result.data[0]['id']}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: Query signals
print("\n2. Testing signals...")
try:
    result = supabase.table('signals').select('id, ticker, sector').limit(3).execute()
    print(f"✅ SUCCESS: Found {len(result.data)} signals")
    for signal in result.data[:3]:
        print(f"   - {signal['ticker']}: {signal.get('sector', 'NO SECTOR')}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Query performance
print("\n3. Testing performance...")
try:
    result = supabase.table('performance').select('id, signal_id, baseline_price').limit(3).execute()
    print(f"✅ SUCCESS: Found {len(result.data)} performance records")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 4: Query signals with performance JOIN (like frontend)
print("\n4. Testing signals WITH performance JOIN (exact frontend query)...")
try:
    result = supabase.table('signals').select('''
        id,
        ticker,
        sector,
        current_price,
        overall_score,
        performance (
            baseline_price,
            return_1d
        )
    ''').limit(3).execute()
    print(f"✅ SUCCESS: Found {len(result.data)} signals with JOIN")
    for signal in result.data[:3]:
        perf = signal.get('performance')
        print(f"   - {signal['ticker']}: performance = {perf}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n" + "=" * 60)
print("If any test failed, RLS policies need to be added in Supabase!")
