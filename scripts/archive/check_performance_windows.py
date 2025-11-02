"""Check if we have performance data for recent time windows."""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def check_performance_data():
    db = SupabaseInterface()
    await db.connect()
    
    try:
        now = datetime.now(timezone.utc)
        windows = [
            {'name': 'All-time', 'start': datetime(2020, 1, 1, tzinfo=timezone.utc), 'end': now},
            {'name': '90 days', 'start': now - timedelta(days=90), 'end': now},
            {'name': '30 days', 'start': now - timedelta(days=30), 'end': now},
        ]
        
        print(f"\n{'='*80}")
        print("PERFORMANCE DATA CHECK")
        print(f"{'='*80}\n")
        
        for window in windows:
            result = await db.execute_query("""
                SELECT COUNT(*) as count
                FROM performance
                WHERE baseline_date >= $1 AND baseline_date <= $2
            """, [window['start'], window['end']])
            
            count = result[0]['count'] if result else 0
            print(f"{window['name']}: {count} performance records")
            print(f"  Period: {window['start'].date()} to {window['end'].date()}")
            
            if count == 0:
                print(f"  ⚠️  NO DATA - Phase 7 will skip this window!\n")
            else:
                print()
                
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check_performance_data())
