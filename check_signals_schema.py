"""
Quick check of signals table schema and current backtest data state
"""
import asyncio
from datetime import datetime
from backend.storage.database import SupabaseInterface

async def check_schema():
    """Check signals table schema and data."""
    
    # Connect to database
    db = SupabaseInterface()
    await db.connect()
    
    if not db.pool:
        print("Error: Database pool not initialized")
        return
    
    print("\n" + "="*80)
    print("SIGNALS TABLE SCHEMA & DATA CHECK")
    print("="*80 + "\n")
    
    # 1. Get all backtest-related columns
    async with db.pool.acquire() as conn:
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'signals' 
            AND (column_name LIKE 'backtest%' OR column_name LIKE '%return%' OR column_name LIKE 'spy%')
            ORDER BY column_name
        """)
        
        print(f"Performance Tracking Columns ({len(columns)}):")
        print("-"*80)
        for col in columns:
            print(f"  {col['column_name']:30} {col['data_type']:15} NULL: {col['is_nullable']}")
        
        # 2. Count total signals
        total = await conn.fetchval("SELECT COUNT(*) FROM signals")
        print(f"\nTotal Signals: {total}")
        
        # 3. Check date range
        date_range = await conn.fetchrow("""
            SELECT 
                MIN(created_at) as oldest,
                MAX(created_at) as newest,
                COUNT(*) as total
            FROM signals
        """)
        print(f"Date Range: {date_range['oldest'].date()} to {date_range['newest'].date()}")
        
        # 4. Check backtest data coverage
        coverage = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_signals,
                COUNT(backtest_baseline_price) as has_baseline,
                COUNT(backtest_baseline_date) as has_baseline_date,
                COUNT(return_1d) as has_1d,
                COUNT(return_3d) as has_3d,
                COUNT(return_7d) as has_7d,
                COUNT(return_30d) as has_30d,
                COUNT(spy_return_1d) as has_spy_1d
            FROM signals
            WHERE created_at >= '2025-10-17'
        """)
        
        print(f"\nBacktest Data Coverage (Oct 17+):")
        print("-"*80)
        print(f"  Total Signals:     {coverage['total_signals']}")
        print(f"  Has Baseline:      {coverage['has_baseline']} ({coverage['has_baseline']/coverage['total_signals']*100:.1f}%)")
        print(f"  Has Baseline Date: {coverage['has_baseline_date']} ({coverage['has_baseline_date']/coverage['total_signals']*100:.1f}%)")
        print(f"  Has 1d Return:     {coverage['has_1d']} ({coverage['has_1d']/coverage['total_signals']*100:.1f}%)")
        print(f"  Has 3d Return:     {coverage['has_3d']} ({coverage['has_3d']/coverage['total_signals']*100:.1f}%)")
        print(f"  Has 7d Return:     {coverage['has_7d']} ({coverage['has_7d']/coverage['total_signals']*100:.1f}%)")
        print(f"  Has 30d Return:    {coverage['has_30d']} ({coverage['has_30d']/coverage['total_signals']*100:.1f}%)")
        print(f"  Has SPY 1d:        {coverage['has_spy_1d']} ({coverage['has_spy_1d']/coverage['total_signals']*100:.1f}%)")
        
        # 5. Count signals missing baseline
        missing = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM signals 
            WHERE created_at >= '2025-10-17' 
            AND backtest_baseline_price IS NULL
        """)
        print(f"\n⚠️  Signals Missing Baseline: {missing}")
        
        # 6. Sample signals with/without data
        print(f"\nSample Signals WITH Backtest Data:")
        print("-"*80)
        with_data = await conn.fetch("""
            SELECT ticker, created_at, backtest_baseline_price, return_1d, return_3d, return_7d
            FROM signals 
            WHERE backtest_baseline_price IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        for sig in with_data:
            age = (datetime.now() - sig['created_at'].replace(tzinfo=None)).days
            print(f"  {sig['ticker']:6} | {sig['created_at'].date()} (Age: {age}d) | "
                  f"Baseline: ${sig['backtest_baseline_price'] or 0:7.2f} | "
                  f"1d: {sig['return_1d'] or 0:+6.2f}% | 3d: {sig['return_3d'] or 0:+6.2f}%")
        
        print(f"\nSample Signals WITHOUT Backtest Data:")
        print("-"*80)
        without_data = await conn.fetch("""
            SELECT ticker, created_at, backtest_baseline_price
            FROM signals 
            WHERE backtest_baseline_price IS NULL
            ORDER BY created_at DESC
            LIMIT 10
        """)
        
        for sig in without_data:
            age = (datetime.now() - sig['created_at'].replace(tzinfo=None)).days
            print(f"  {sig['ticker']:6} | {sig['created_at'].date()} (Age: {age}d) | Baseline: NULL")
    
    print("\n" + "="*80 + "\n")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check_schema())
