import asyncio
from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    await db.connect()
    result = await db.pool.fetch("""
        SELECT backtest_status, COUNT(*) as count
        FROM signals
        GROUP BY backtest_status
        ORDER BY count DESC
    """)
    print("\nBacktest Status Counts:")
    for r in result:
        print(f"  {r['backtest_status'] or 'NULL'}: {r['count']}")
    await db.disconnect()

asyncio.run(main())
