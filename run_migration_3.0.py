"""
Quick script to execute the 3.0 nuclear migration using psycopg2
"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def run_migration():
    # Get database URL
    database_url = os.getenv('SUPABASE_DATABASE_URL')
    
    if not database_url:
        print("❌ SUPABASE_DATABASE_URL not found in environment")
        return
    
    print("⚠️  EXECUTING NUCLEAR MIGRATION - 3.0 SCHEMA")
    print("=" * 70)
    print("This will DELETE all tables except company_tickers and guardrails_config")
    print("=" * 70)
    
    # Connect to database
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Read migration file
    with open('migrations/001_nuclear_reset_v3.sql', 'r', encoding='utf-8') as f:
        migration_sql = f.read()
    
    try:
        print("\n📝 Executing migration...")
        cursor.execute(migration_sql)
        
        print("\n✅ Migration executed successfully!")
        print("\n📊 Verifying tables...")
        
        # Verify tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print(f"\n✅ Found {len(tables)} tables:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"   - {table[0]}: {count} rows")
        
        print("\n" + "=" * 70)
        print("🎉 3.0 SCHEMA MIGRATION COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_migration()
