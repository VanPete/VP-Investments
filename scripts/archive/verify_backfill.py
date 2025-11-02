"""Verify benchmark backfill results."""
import asyncio
from backend.storage.supabase import SupabaseClient


async def main():
    """Query and display benchmark data to verify backfill."""
    client = SupabaseClient()
    
    # Query Oct 28 records
    query = """
        SELECT 
            signals.ticker,
            performance.baseline_date,
            performance.intervals_completed,
            performance.return_1d,
            performance.spy_return_1d,
            performance.qqq_return_1d,
            performance.sector_return_1d,
            performance.return_3d,
            performance.spy_return_3d,
            performance.qqq_return_3d,
            performance.sector_return_3d
        FROM performance
        INNER JOIN signals ON performance.signal_id = signals.id
        WHERE baseline_date >= '2025-10-28'
          AND baseline_date < '2025-10-29'
        ORDER BY signals.ticker
        LIMIT 5
    """
    
    result = await client.query(query)
    
    print(f"\n✅ Found {len(result)} Oct 28 records:\n")
    
    for r in result:
        print(f"{r['ticker']}:")
        print(f"  Intervals: {r['intervals_completed']}")
        
        # Check 1d data
        if r['return_1d'] is not None:
            spy_status = "✓" if r['spy_return_1d'] is not None else "✗ NULL"
            qqq_status = "✓" if r['qqq_return_1d'] is not None else "✗ NULL"
            sector_status = "✓" if r['sector_return_1d'] is not None else "✗ NULL (OK if no sector)"
            
            print(f"  1d: return={r['return_1d']:.4f}")
            print(f"      SPY={r['spy_return_1d']:.4f if r['spy_return_1d'] else 'NULL'} {spy_status}")
            print(f"      QQQ={r['qqq_return_1d']:.4f if r['qqq_return_1d'] else 'NULL'} {qqq_status}")
            print(f"      Sector={r['sector_return_1d']:.4f if r['sector_return_1d'] else 'NULL'} {sector_status}")
        
        # Check 3d data
        if r['return_3d'] is not None:
            spy_status = "✓" if r['spy_return_3d'] is not None else "✗ NULL"
            qqq_status = "✓" if r['qqq_return_3d'] is not None else "✗ NULL"
            sector_status = "✓" if r['sector_return_3d'] is not None else "✗ NULL (OK if no sector)"
            
            print(f"  3d: return={r['return_3d']:.4f}")
            print(f"      SPY={r['spy_return_3d']:.4f if r['spy_return_3d'] else 'NULL'} {spy_status}")
            print(f"      QQQ={r['qqq_return_3d']:.4f if r['qqq_return_3d'] else 'NULL'} {qqq_status}")
            print(f"      Sector={r['sector_return_3d']:.4f if r['sector_return_3d'] else 'NULL'} {sector_status}")
        
        print()
    
    # Count records with NULL benchmarks
    null_check_query = """
        SELECT COUNT(*) as count
        FROM performance
        WHERE return_1d IS NOT NULL
          AND spy_return_1d IS NULL
    """
    
    null_result = await client.query(null_check_query)
    null_count = null_result[0]['count'] if null_result else 0
    
    print(f"\n📊 Records with ticker returns but NULL SPY benchmarks: {null_count}")
    
    if null_count == 0:
        print("✅ SUCCESS: All records with ticker returns now have benchmarks!")
    else:
        print(f"⚠️ WARNING: {null_count} records still have NULL benchmarks")


if __name__ == "__main__":
    asyncio.run(main())
