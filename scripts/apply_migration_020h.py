"""
Apply migration 020h: Add factor_return_correlations column to analytics table.

This migration adds a JSONB column to store factor-return correlations
for ML feature importance analysis.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def apply_migration():
    """Apply migration 020h to add factor_return_correlations column."""
    
    # Get database connection details
    db_url = os.getenv('SUPABASE_DATABASE_URL')
    if not db_url:
        print("❌ Error: SUPABASE_DATABASE_URL not found in environment variables")
        return False
    
    print("=" * 80)
    print("Migration 020h: Add factor_return_correlations column")
    print("=" * 80)
    
    try:
        # Connect to database
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Read migration file
        migration_file = project_root / 'migrations' / '020h_add_factor_return_correlations.sql'
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        # Execute migration
        print("\n⚙️  Adding factor_return_correlations column...")
        cur.execute(migration_sql)
        conn.commit()
        print("✅ Column added successfully!")
        
        # Verify column exists
        print("\n⚙️  Verifying column...")
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'analytics' 
            AND column_name = 'factor_return_correlations'
        """)
        result = cur.fetchone()
        
        if result:
            print(f"✅ Verified: {result[0]} ({result[1]})")
        else:
            print("⚠️  Warning: Column not found in verification")
        
        # Close connection
        cur.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ Migration 020h Complete!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error applying migration: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    success = apply_migration()
    sys.exit(0 if success else 1)
