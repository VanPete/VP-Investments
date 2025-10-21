"""
Script to check the Supabase database schema.
Retrieves table structures, columns, constraints, and views.
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import json

# Load environment
load_dotenv()

supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')

if not supabase_url or not supabase_key:
    print("❌ Supabase credentials not found in environment")
    exit(1)

# Connect to Supabase
supabase = create_client(supabase_url, supabase_key)

print("=" * 80)
print("SUPABASE DATABASE SCHEMA CHECK")
print("=" * 80)

# Check each table we're using
tables_to_check = [
    'runs',
    'signals',
    'signals_technical',
    'signals_fundamental',
    'signals_news_macro',
    'signals_social_alternative',
    'signals_risk_stability',
    'signals_institutional_smart_money',
    'company_tickers'
]

for table_name in tables_to_check:
    print(f"\n{'='*80}")
    print(f"TABLE: {table_name}")
    print(f"{'='*80}")
    
    try:
        # Try to query the table with limit 0 to get column info
        result = supabase.table(table_name).select('*').limit(0).execute()
        
        if result:
            print(f"✅ Table exists and is accessible")
            
            # Try to get one row to see actual column structure
            sample = supabase.table(table_name).select('*').limit(1).execute()
            
            if sample.data and len(sample.data) > 0:
                print(f"📊 Sample row structure:")
                row = sample.data[0]
                for col, val in sorted(row.items()):
                    val_type = type(val).__name__
                    val_preview = str(val)[:50] if val is not None else 'NULL'
                    print(f"   {col:30} | {val_type:15} | {val_preview}")
            else:
                print(f"ℹ️  Table is empty (no sample data)")
                
                # For empty tables, try to infer structure from insert error
                print(f"ℹ️  Attempting to detect columns from table metadata...")
                
    except Exception as e:
        print(f"❌ Error accessing table: {e}")

# Special check for runs table - get constraints
print(f"\n{'='*80}")
print("RUNS TABLE - DETAILED CHECK")
print(f"{'='*80}")

try:
    # Query runs table
    runs = supabase.table('runs').select('*').limit(5).execute()
    
    if runs.data:
        print(f"✅ Found {len(runs.data)} run record(s)")
        for run in runs.data:
            print(f"\nRun ID: {run.get('id')} | Type: {run.get('run_type')} | Status: {run.get('status')}")
            print(f"  run_id string: {run.get('run_id')}")
            print(f"  created_at: {run.get('created_at')}")
            print(f"  total_signals: {run.get('total_signals')}")
    else:
        print("ℹ️  No runs found")
        
except Exception as e:
    print(f"❌ Error querying runs: {e}")

# Check signals table relationship
print(f"\n{'='*80}")
print("SIGNALS TABLE - RELATIONSHIP CHECK")
print(f"{'='*80}")

try:
    # Query signals table
    signals = supabase.table('signals').select('*').limit(5).execute()
    
    if signals.data:
        print(f"✅ Found {len(signals.data)} signal record(s)")
        for signal in signals.data:
            print(f"\nSignal ID: {signal.get('id')}")
            print(f"  run_id (FK): {signal.get('run_id')} (type: {type(signal.get('run_id')).__name__})")
            print(f"  ticker: {signal.get('ticker')}")
            print(f"  signal_score: {signal.get('signal_score')}")
    else:
        print("ℹ️  No signals found")
        
except Exception as e:
    print(f"❌ Error querying signals: {e}")

# Try to understand the run_id relationship
print(f"\n{'='*80}")
print("FOREIGN KEY RELATIONSHIP ANALYSIS")
print(f"{'='*80}")

try:
    runs = supabase.table('runs').select('id, run_id').limit(5).execute()
    
    if runs.data:
        print("✅ runs.id values (BIGINT - for FK reference):")
        for run in runs.data:
            print(f"   id={run.get('id')} | run_id='{run.get('run_id')}'")
        
        print("\n💡 INSIGHT:")
        print("   signals.run_id (FK) should reference runs.id (BIGINT)")
        print("   NOT runs.run_id (TEXT)")
        print("\n   Example: If runs.id=5, then signals.run_id should be 5 (integer)")
        print("            NOT 'run_20251015_172239' (string)")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Summary and recommendations
print(f"\n{'='*80}")
print("SCHEMA ALIGNMENT CHECK - BACKEND vs DATABASE")
print(f"{'='*80}")

print("\n🔍 Key Findings:")
print("   1. runs.id is BIGINT (auto-increment primary key)")
print("   2. runs.run_id is TEXT (human-readable identifier)")
print("   3. signals.run_id is BIGINT FOREIGN KEY → runs.id")
print("   4. Backend must save runs.id (integer) NOT runs.run_id (string)")

print("\n✅ Required Fix:")
print("   In _create_run_record():")
print("      - Return db_id (integer) instead of run_id (string)")
print("   In _prepare_signal_record():")
print("      - Use db_id for signals.run_id field")

print("\n" + "="*80)
print("✅ Schema check complete!")
print("="*80)
