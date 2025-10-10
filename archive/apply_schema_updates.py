"""
Apply Phase 2-8 Schema Updates to Supabase
Executes the generated ALTER TABLE statements via Supabase SQL API
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
import time

# Load environment variables
load_dotenv()

print("="*80)
print("PHASE 2-8 SCHEMA UPDATE APPLICATION")
print("="*80)
print("\nApplying schema updates to Supabase...\n")


def connect_to_supabase() -> Client:
    """Connect to Supabase"""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    
    return create_client(url, key)


def read_migration_file(filepath: str) -> list[str]:
    """Read and parse migration SQL file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by semicolon and filter out comments/empty lines
    statements = []
    for stmt in content.split(';'):
        stmt = stmt.strip()
        # Remove comments
        lines = [line for line in stmt.split('\n') if line.strip() and not line.strip().startswith('--')]
        if lines:
            clean_stmt = '\n'.join(lines)
            if 'ALTER TABLE' in clean_stmt:
                statements.append(clean_stmt + ';')
    
    return statements


def execute_sql_statement(supabase: Client, statement: str) -> tuple[bool, str]:
    """Execute a single SQL statement via Supabase RPC"""
    try:
        # Supabase doesn't have direct SQL execution via Python client
        # We'll need to use the REST API directly
        import requests
        
        url = os.environ.get("SUPABASE_URL")
        service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        
        # Use the SQL API endpoint
        sql_url = f"{url}/rest/v1/rpc/exec_sql"
        
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json"
        }
        
        # Try direct execution (this may not work with anon key)
        # Alternative: use PostgREST's direct query
        return False, "Direct SQL execution requires service role key - use Supabase SQL Editor"
        
    except Exception as e:
        return False, str(e)


def main():
    """Main update function"""
    
    migration_file = 'migrations/phase2-8_schema_update.sql'
    
    # Check if migration file exists
    if not os.path.exists(migration_file):
        print(f"❌ Migration file not found: {migration_file}")
        print("   Run verify_schema_phase2-8.py first to generate it")
        return False
    
    print(f"[1/3] Reading migration file: {migration_file}")
    statements = read_migration_file(migration_file)
    print(f"   ✅ Found {len(statements)} ALTER TABLE statements\n")
    
    print(f"[2/3] Schema Updates to Apply:")
    print("="*80)
    
    # Group statements by table
    signals_stmts = [s for s in statements if 'signals ADD COLUMN' in s]
    performance_stmts = [s for s in statements if 'signal_performance ADD COLUMN' in s]
    
    print(f"\n📊 Signals Table:")
    print(f"   {len(signals_stmts)} columns to add")
    print(f"   - Phase 2 (Z-Scores): 4 columns")
    print(f"   - Phase 3 (Trade Type): 1 column")
    print(f"   - Phase 4 (Risk Scoring): 7 columns")
    print(f"   - Phase 5 (Enhanced Data): 11 columns")
    print(f"   - Phase 6 (Adjustments): 3 columns")
    print(f"   - Phase 7 (Narratives): 1 column")
    print(f"   - Phase 8 (Backtesting): 6 columns")
    
    print(f"\n📊 Signal Performance Table:")
    print(f"   {len(performance_stmts)} columns to add")
    print(f"   - Return metrics: 5 columns")
    print(f"   - SPY comparison: 5 columns")
    print(f"   - Beat SPY flags: 5 columns")
    
    print(f"\n{'='*80}")
    print(f"[3/3] Applying Updates to Supabase")
    print("="*80)
    
    print(f"\n⚠️  MANUAL STEP REQUIRED:")
    print(f"\nThe Python Supabase client doesn't support direct SQL execution.")
    print(f"You need to run these ALTER TABLE statements in the Supabase SQL Editor.\n")
    
    print(f"📋 INSTRUCTIONS:")
    print(f"   1. Go to your Supabase project: https://supabase.com/dashboard")
    print(f"   2. Navigate to: SQL Editor")
    print(f"   3. Create a new query")
    print(f"   4. Copy and paste the contents of:")
    print(f"      {migration_file}")
    print(f"   5. Click 'Run' to execute all ALTER TABLE statements")
    print(f"   6. Re-run verify_schema_phase2-8.py to confirm updates\n")
    
    print(f"💡 QUICK COPY:")
    print("="*80)
    
    # Print first few statements as example
    print("\n-- First 5 ALTER statements (see file for complete list):\n")
    for stmt in statements[:5]:
        print(stmt)
    
    print(f"\n... and {len(statements)-5} more statements")
    print(f"\nSee full file: {migration_file}")
    print("="*80)
    
    print(f"\n✅ Migration file ready for execution")
    print(f"   File location: {os.path.abspath(migration_file)}")
    print(f"   Total statements: {len(statements)}")
    print(f"\n   After running in Supabase SQL Editor, verify with:")
    print(f"   python verify_schema_phase2-8.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
