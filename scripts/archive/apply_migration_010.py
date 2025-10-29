"""Apply Migration 010: Add Analytics Table"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def apply_migration_010():
    db = SupabaseInterface()
    await db.connect()
    
    print("=" * 80)
    print("APPLYING MIGRATION 010: ADD ANALYTICS TABLE")
    print("=" * 80)
    
    migration_sql = Path(__file__).parent.parent / 'migrations' / '010_add_analytics_table.sql'
    
    with open(migration_sql, 'r') as f:
        sql = f.read()
    
    print("\nExecuting migration...")
    
    try:
        await db.execute_non_query(sql)
        print("\n✅ Migration 010 applied successfully!")
        print("\nAnalytics table created with:")
        print("  - 71 columns for comprehensive metrics")
        print("  - Win rates for all 7 intervals")
        print("  - Sharpe ratios for all 7 intervals")
        print("  - Max drawdowns for all 7 intervals")
        print("  - Sector performance tracking")
        print("  - Factor analysis support")
        
    except Exception as e:
        print(f"\n❌ Error applying migration: {e}")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(apply_migration_010())
