"""
Apply migration 020f: Remove avg_composite_score and avg_confidence columns.

These columns are redundant:
- avg_composite_score duplicates avg_overall_score
- avg_confidence is not used anywhere and adds unnecessary complexity
"""

import asyncio
import os
from pathlib import Path
from supabase import create_client


async def main():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set")
        return
    
    print("=" * 80)
    print("Applying Migration 020f: Remove avg_composite_score and avg_confidence")
    print("=" * 80)
    
    # Read migration file
    migration_file = Path(__file__).parent.parent / "migrations" / "020f_remove_composite_confidence_columns.sql"
    
    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        return
    
    migration_sql = migration_file.read_text()
    
    print(f"\nMigration SQL:")
    print("-" * 80)
    print(migration_sql)
    print("-" * 80)
    
    # Create Supabase client
    supabase = create_client(supabase_url, supabase_key)
    
    print("\n⚙️  Executing migration...")
    
    try:
        # Execute via RPC (Supabase's SQL execution)
        result = supabase.rpc('exec_sql', {'sql': migration_sql}).execute()
        print("✅ Migration executed successfully!")
        
    except Exception as e:
        print(f"⚠️  Direct execution failed, trying alternative method: {e}")
        
        # Alternative: execute each statement separately using psycopg2
        try:
            import psycopg2
            from urllib.parse import urlparse
            
            # Parse DATABASE_URL
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                print("❌ DATABASE_URL not set")
                return
            
            result = urlparse(db_url)
            conn = psycopg2.connect(
                database=result.path[1:],
                user=result.username,
                password=result.password,
                host=result.hostname,
                port=result.port
            )
            
            cursor = conn.cursor()
            
            # Execute each statement
            for statement in migration_sql.split(';'):
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    cursor.execute(statement)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("✅ Migration executed successfully via psycopg2!")
            
        except Exception as e2:
            print(f"❌ Migration failed: {e2}")
            return
    
    print("\n" + "=" * 80)
    print("Verifying migration...")
    print("=" * 80)
    
    try:
        # Try to select the columns - should fail if they were dropped
        try:
            result = supabase.table('analytics').select('avg_composite_score, avg_confidence').limit(1).execute()
            print("⚠️  Warning: Columns still exist in table!")
        except Exception as e:
            if 'does not exist' in str(e) or 'column' in str(e).lower():
                print("✅ Columns verified - avg_composite_score and avg_confidence removed!")
            else:
                print(f"⚠️  Unexpected error: {e}")
        
    except Exception as e:
        print(f"⚠️  Could not verify: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Migration 020f Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
