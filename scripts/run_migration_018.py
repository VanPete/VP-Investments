"""Run SQL migration to add Phase 6 performance indexes."""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

def main():
    # Load environment
    load_dotenv()
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials")
        return
    
    # Read migration file
    migration_file = Path(__file__).parent.parent / 'migrations' / '018_optimize_phase6_queries.sql'
    
    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        return
    
    with open(migration_file, 'r') as f:
        sql = f.read()
    
    print("=" * 80)
    print("RUNNING MIGRATION: 018_optimize_phase6_queries.sql")
    print("=" * 80)
    print("\nThis will create the following indexes:")
    print("  1. idx_performance_status_baseline (composite: status + baseline_date)")
    print("  2. idx_performance_baseline_date (single: baseline_date)")
    print("  3. idx_performance_active_signals (partial: baseline_date WHERE status IN pending/in_progress)")
    print("\n" + "=" * 80)
    
    # Connect to Supabase
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        # Execute SQL (using RPC or direct SQL execution)
        # Note: Supabase Python client doesn't directly support SQL execution
        # We'll use the PostgREST API to execute via a stored procedure
        
        print("\n⚠️  Note: This migration should be run directly in Supabase SQL Editor")
        print("    Or using psql with the DATABASE_URL")
        print("\nSQL to execute:")
        print("-" * 80)
        print(sql)
        print("-" * 80)
        
        # Alternative: Use psycopg2 if available
        try:
            import psycopg2
            database_url = os.getenv('SUPABASE_DATABASE_URL')
            
            if database_url:
                print("\n🔄 Attempting to run via psycopg2...")
                conn = psycopg2.connect(database_url)
                cur = conn.cursor()
                
                # Split by semicolon and execute each statement
                statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
                
                for stmt in statements:
                    if 'SELECT' in stmt.upper():
                        print(f"\n📊 Executing query: {stmt[:100]}...")
                        cur.execute(stmt)
                        results = cur.fetchall()
                        for row in results:
                            print(f"  {row}")
                    else:
                        print(f"\n✅ Executing: {stmt[:100]}...")
                        cur.execute(stmt)
                
                conn.commit()
                cur.close()
                conn.close()
                
                print("\n" + "=" * 80)
                print("✅ MIGRATION COMPLETED SUCCESSFULLY")
                print("=" * 80)
            else:
                print("\n❌ SUPABASE_DATABASE_URL not found in .env")
                
        except ImportError:
            print("\n❌ psycopg2 not installed - please run migration manually in Supabase SQL Editor")
        
    except Exception as e:
        print(f"\n❌ Error running migration: {e}")

if __name__ == '__main__':
    main()
