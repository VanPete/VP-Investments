#!/usr/bin/env python
"""Quick verification of backfill results."""
import sys
sys.path.insert(0, '.')

import asyncio
from backend.storage.database import get_supabase_database


async def main():
    db = await get_supabase_database()
    
    # Count NULL benchmarks
    result = await db.fetch_one(
        "SELECT COUNT(*) as cnt FROM performance WHERE return_1d IS NOT NULL AND spy_return_1d IS NULL"
    )
    null_count = result['cnt']
    
    print(f"\n📊 Database Status:")
    print(f"Records with ticker returns but NULL SPY benchmarks: {null_count}")
    
    if null_count == 0:
        print("✅ SUCCESS: All records with ticker returns now have benchmarks!\n")
    else:
        print(f"⚠️ WARNING: {null_count} records still have NULL benchmarks\n")
    
    # Sample Oct 28 records
    records = await db.fetch_all("""
        SELECT 
            s.ticker,
            p.intervals_completed,
            p.return_1d,
            p.spy_return_1d,
            p.qqq_return_1d,
            p.sector_return_1d
        FROM performance p
        INNER JOIN signals s ON p.signal_id = s.id
        WHERE p.baseline_date >= '2025-10-28'
          AND p.baseline_date < '2025-10-29'
        ORDER BY s.ticker
        LIMIT 5
    """)
    
    print(f"Sample Oct 28 records ({len(records)} shown):")
    for r in records:
        spy_status = "✓" if r['spy_return_1d'] is not None else "✗ NULL"
        qqq_status = "✓" if r['qqq_return_1d'] is not None else "✗ NULL"
        print(f"  {r['ticker']}: intervals={r['intervals_completed']}")
        print(f"    return_1d={r['return_1d']:.4f}, SPY={r['spy_return_1d']:.4f if r['spy_return_1d'] else 'NULL'} {spy_status}, QQQ={r['qqq_return_1d']:.4f if r['qqq_return_1d'] else 'NULL'} {qqq_status}")


if __name__ == "__main__":
    asyncio.run(main())
