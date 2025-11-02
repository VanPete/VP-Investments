"""Drop the analytics_run_id_unique constraint to allow multi-window analytics."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def drop_constraint():
    db = SupabaseInterface()
    await db.connect()
    
    try:
        print(f"\n{'='*80}")
        print("DROPPING analytics_run_id_unique CONSTRAINT")
        print(f"{'='*80}\n")
        
        print("This constraint blocks Phase 7 from creating 3 analytics rows (all_time, 90d, 30d)")
        print("because all 3 rows reference the same run_id.\n")
        
        # Drop constraint
        await db.execute_query("""
            ALTER TABLE analytics DROP CONSTRAINT IF EXISTS analytics_run_id_unique
        """)
        print("✓ Dropped constraint: analytics_run_id_unique")
        
        # Drop associated index
        await db.execute_query("""
            DROP INDEX IF EXISTS analytics_run_id_unique
        """)
        print("✓ Dropped index: analytics_run_id_unique\n")
        
        # Verify
        result = await db.execute_query("""
            SELECT conname
            FROM pg_constraint
            WHERE conrelid = 'analytics'::regclass
            AND conname = 'analytics_run_id_unique'
        """)
        
        if not result:
            print("✅ SUCCESS: Constraint successfully removed!")
            print("\nNow you can run the pipeline and Phase 7 will create 3 analytics rows:\n")
            print("  - all_time (all historical data)")
            print("  - 90d (last 90 days)")
            print("  - 30d (last 30 days)\n")
        else:
            print("⚠️  WARNING: Constraint still exists after drop attempt")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(drop_constraint())
