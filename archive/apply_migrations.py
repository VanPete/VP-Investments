"""
Apply Phase 1.4 Database Migrations

This script applies migrations 001 and 002:
1. Remove 11 dead columns from signals table
2. Document Phase 2-4 placeholder columns

Run with: python apply_migrations.py
"""

import asyncio
import os
from pathlib import Path
from backend.storage.database import SupabaseInterface

async def apply_migration(db: SupabaseInterface, migration_file: str) -> bool:
    """Apply a single migration file."""
    try:
        migration_path = Path(__file__).parent / 'migrations' / migration_file
        
        if not migration_path.exists():
            print(f"❌ Migration file not found: {migration_file}")
            return False
        
        print(f"\n{'='*80}")
        print(f"Applying migration: {migration_file}")
        print(f"{'='*80}\n")
        
        # Read migration SQL
        with open(migration_path, 'r') as f:
            sql_content = f.read()
        
        # Note: Supabase Python client doesn't support raw SQL execution
        # You need to run these migrations directly in Supabase SQL Editor
        # or use psycopg2 connection
        
        print(f"⚠️  IMPORTANT: This migration must be run in Supabase SQL Editor")
        print(f"\n📋 Migration SQL:")
        print(f"{'-'*80}")
        print(sql_content[:500] + "...\n[truncated, see full file]")
        print(f"{'-'*80}\n")
        
        response = input(f"Have you run this migration in Supabase? (yes/no): ")
        
        if response.lower() == 'yes':
            print(f"✅ Migration {migration_file} marked as applied")
            return True
        else:
            print(f"⏸️  Migration {migration_file} skipped")
            return False
            
    except Exception as e:
        print(f"❌ Error applying migration {migration_file}: {e}")
        return False

async def verify_migration_001(db: SupabaseInterface) -> bool:
    """Verify that dead columns were removed."""
    try:
        print("\n🔍 Verifying migration 001 (dead columns removal)...")
        
        # Try to select a dead column - should fail if removed
        dead_columns = [
            'commentary_metadata',
            'score_components', 
            'scoring_version',
            'ai_commentary_version',
            'rowid',
            'ml_confidence_score',
            'prediction_confidence',
            'pattern_match_score',
            'signal_duration',
            'option_chain_data',
            'option_volume_ratio'
        ]
        
        # Get all columns in signals table
        result = db.supabase.from_('signals').select('*').limit(1).execute()
        
        if result.data and len(result.data) > 0:
            existing_columns = set(result.data[0].keys())
            
            removed_count = 0
            still_exist = []
            
            for col in dead_columns:
                if col not in existing_columns:
                    removed_count += 1
                else:
                    still_exist.append(col)
            
            print(f"✅ {removed_count}/{len(dead_columns)} dead columns removed")
            
            if still_exist:
                print(f"⚠️  These columns still exist: {', '.join(still_exist)}")
                return False
            
            print(f"✅ Migration 001 verified successfully!")
            return True
        else:
            print("⚠️  Could not verify - no signals in database")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

async def verify_migration_002(db: SupabaseInterface) -> bool:
    """Verify that Phase 2-4 placeholders are documented."""
    try:
        print("\n🔍 Verifying migration 002 (placeholder documentation)...")
        
        # Check if phase_implementation_tracker table exists
        try:
            result = db.supabase.from_('phase_implementation_tracker').select('phase, status').execute()
            
            if result.data:
                phases = {row['phase']: row['status'] for row in result.data}
                print(f"✅ Found {len(phases)} phases tracked:")
                for phase, status in sorted(phases.items()):
                    status_emoji = {
                        'Complete': '✅',
                        'In Progress': '⏳',
                        'Planned': '📋',
                        'Deferred': '⏸️'
                    }.get(status, '❓')
                    print(f"   {status_emoji} {phase}: {status}")
                
                print(f"\n✅ Migration 002 verified successfully!")
                return True
            else:
                print("⚠️  Phase tracker table exists but is empty")
                return False
                
        except Exception as e:
            print(f"⚠️  Phase tracker table not found: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

async def main():
    """Main migration workflow."""
    print("\n" + "="*80)
    print("PHASE 1.4.1 & 1.4.2: DATABASE MIGRATIONS")
    print("="*80)
    
    db = SupabaseInterface()
    await db.connect()
    
    print("\n📋 Migrations to apply:")
    print("   1. 001_remove_dead_columns.sql")
    print("   2. 002_document_phase_placeholders.sql")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("   • These migrations modify the signals table schema")
    print("   • Backup your data before proceeding")
    print("   • Migrations must be run in Supabase SQL Editor")
    print("   • This script will verify the migrations after you run them")
    
    response = input("\nReady to proceed? (yes/no): ")
    
    if response.lower() != 'yes':
        print("\n❌ Migration cancelled")
        return
    
    # Apply migrations
    print("\n" + "="*80)
    print("STEP 1: Apply Migration Files")
    print("="*80)
    
    migration_001_ok = await apply_migration(db, '001_remove_dead_columns.sql')
    migration_002_ok = await apply_migration(db, '002_document_phase_placeholders.sql')
    
    # Verify migrations
    if migration_001_ok:
        print("\n" + "="*80)
        print("STEP 2: Verify Migrations")
        print("="*80)
        
        verify_001_ok = await verify_migration_001(db)
        verify_002_ok = await verify_migration_002(db)
        
        # Summary
        print("\n" + "="*80)
        print("MIGRATION SUMMARY")
        print("="*80)
        
        print(f"\n001_remove_dead_columns.sql: {'✅ SUCCESS' if verify_001_ok else '❌ FAILED'}")
        print(f"002_document_phase_placeholders.sql: {'✅ SUCCESS' if verify_002_ok else '❌ FAILED'}")
        
        if verify_001_ok and verify_002_ok:
            print("\n🎉 All migrations completed successfully!")
            print("\n✅ Phase 1.4.1 Complete: Dead columns removed")
            print("✅ Phase 1.4.2 Complete: Placeholders documented")
            print("\n➡️  Next steps:")
            print("   • Phase 1.4.3: Fix Phase 1 timing issue")
            print("   • Phase 1.4.4: Enhance financial_score with Phase 1 metrics")
        else:
            print("\n⚠️  Some migrations failed. Please review and fix before proceeding.")
    else:
        print("\n⏸️  Migrations not applied. Please run them in Supabase SQL Editor first.")

if __name__ == '__main__':
    asyncio.run(main())
