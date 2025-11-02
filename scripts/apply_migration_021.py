"""
Apply Migration 021: Add CASCADE DELETE to foreign key constraints

This script applies the migration to add CASCADE DELETE to all foreign key constraints,
allowing runs to be deleted without manual cleanup of related records.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import get_supabase_database


async def apply_migration():
    """Apply migration 021 to add CASCADE DELETE constraints."""
    
    print("=" * 80)
    print("MIGRATION 021: Add CASCADE DELETE to Foreign Key Constraints")
    print("=" * 80)
    print()
    
    # Get database connection
    db = await get_supabase_database()
    
    # Read migration file
    migration_file = Path(__file__).parent.parent / 'migrations' / '021_add_cascade_delete.sql'
    
    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        return False
    
    print(f"📄 Reading migration file: {migration_file.name}")
    migration_sql = migration_file.read_text()
    
    # Split into individual statements
    statements = [s.strip() for s in migration_sql.split(';') if s.strip() and not s.strip().startswith('--')]
    
    print(f"📝 Found {len(statements)} SQL statements to execute")
    print()
    
    # Execute each statement
    success_count = 0
    for i, statement in enumerate(statements, 1):
        # Extract table name from statement for display
        table_name = "unknown"
        if "ALTER TABLE public." in statement:
            parts = statement.split("ALTER TABLE public.")[1].split()[0]
            table_name = parts
        
        print(f"[{i}/{len(statements)}] Updating {table_name}...", end=" ")
        
        try:
            # Execute via RPC or direct SQL
            # Note: Supabase Python client doesn't support DDL directly
            # We'll use the REST API to execute raw SQL
            result = db.client.rpc('exec_sql', {'query': statement}).execute()
            print("✅")
            success_count += 1
        except Exception as e:
            # Check if it's a "function does not exist" error
            if "function public.exec_sql(query => text) does not exist" in str(e):
                print("⚠️  (exec_sql function not available)")
                print()
                print("=" * 80)
                print("MANUAL MIGRATION REQUIRED")
                print("=" * 80)
                print()
                print("Please apply this migration manually in Supabase SQL Editor:")
                print()
                print("1. Go to: https://supabase.com/dashboard/project/YOUR_PROJECT/sql")
                print("2. Copy and paste the SQL from: migrations/021_add_cascade_delete.sql")
                print("3. Click 'Run' to execute")
                print()
                print("Or use the Supabase CLI:")
                print("  supabase db push")
                print()
                return False
            else:
                print(f"❌ Error: {e}")
    
    print()
    print("=" * 80)
    print(f"✅ Migration Complete: {success_count}/{len(statements)} statements executed")
    print("=" * 80)
    print()
    print("Verification:")
    print("Run: python scripts/verify_migration_021.py")
    print()
    
    return True


async def verify_migration():
    """Verify that CASCADE DELETE constraints were applied."""
    
    print("=" * 80)
    print("VERIFYING MIGRATION 021")
    print("=" * 80)
    print()
    
    db = await get_supabase_database()
    
    # Read verification SQL
    verify_file = Path(__file__).parent.parent / 'migrations' / 'verify_migration_021.sql'
    
    if not verify_file.exists():
        print("⚠️  Verification file not found")
        return
    
    verify_sql = verify_file.read_text()
    
    try:
        result = db.client.rpc('exec_sql', {'query': verify_sql}).execute()
        
        print("Foreign Key Constraints:")
        print()
        for row in result.data:
            delete_rule = row.get('delete_rule', 'UNKNOWN')
            status = "✅" if delete_rule == "CASCADE" else "❌"
            print(f"{status} {row['table_name']}.{row['constraint_name']}: {delete_rule}")
        
        print()
        
    except Exception as e:
        print(f"⚠️  Could not verify automatically: {e}")
        print()
        print("Please verify manually in Supabase SQL Editor:")
        print()
        print(verify_sql)
        print()


if __name__ == '__main__':
    success = asyncio.run(apply_migration())
    
    if success:
        asyncio.run(verify_migration())
    else:
        print()
        print("📋 MIGRATION SQL:")
        print("=" * 80)
        migration_file = Path(__file__).parent.parent / 'migrations' / '021_add_cascade_delete.sql'
        print(migration_file.read_text())
        print("=" * 80)
