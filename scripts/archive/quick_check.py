"""Simple check - how many analytics rows."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    await db.connect()
    
    try:
        result = await db.execute_query("SELECT period_type, total_signals, performance_records_used FROM analytics ORDER BY period_type")
        print(f"\n{len(result)} analytics rows:\n")
        for row in result:
            print(f"  {row['period_type']}: {row['total_signals']} signals, {row['performance_records_used']} records")
    finally:
        await db.disconnect()

asyncio.run(main())
