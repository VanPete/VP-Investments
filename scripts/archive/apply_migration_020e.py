"""
Apply migration 020e: Add avg_composite_score and avg_confidence columns.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

def apply_migration():
    """Apply migration 020e."""
    # Initialize Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    supabase = create_client(url, key)
    
    print("=" * 80)
    print("Applying Migration 020e: Add avg_composite_score and avg_confidence")
    print("=" * 80)
    
    # Read migration file
    migration_file = Path(__file__).parent.parent / "migrations" / "020e_add_composite_confidence_columns.sql"
    
    with open(migration_file, 'r') as f:
        sql = f.read()
    
    print("\nMigration SQL:")
    print("-" * 80)
    print(sql)
    print("-" * 80)
    
    try:
        # Execute migration using RPC
        result = supabase.rpc('exec_sql', {'sql': sql}).execute()
        print("\n✅ Migration applied successfully!")
        
    except Exception as e:
        # Try direct execution
        print(f"\n⚠️ RPC method not available, trying direct execution...")
        try:
            # For direct execution, we'll need to use the database URL
            import psycopg2
            db_url = os.environ.get("SUPABASE_DATABASE_URL")
            
            if not db_url:
                print("❌ SUPABASE_DATABASE_URL not found in environment")
                return
            
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            cur.close()
            conn.close()
            
            print("✅ Migration applied successfully using direct connection!")
            
        except Exception as e2:
            print(f"❌ Error applying migration: {e2}")
            print("\nPlease apply manually using the SQL above.")
    
    print("\n" + "=" * 80)
    print("Verifying columns exist...")
    print("=" * 80)
    
    try:
        # Query to check if columns exist
        result = supabase.table('analytics').select('avg_composite_score, avg_confidence').limit(1).execute()
        print("✅ Columns verified - avg_composite_score and avg_confidence exist!")
    except Exception as e:
        print(f"⚠️ Could not verify columns: {e}")
    
    print("\n" + "=" * 80)
    print("Migration 020e Complete")
    print("=" * 80)

if __name__ == "__main__":
    apply_migration()
