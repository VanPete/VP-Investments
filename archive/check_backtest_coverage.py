"""
Check backtest data coverage for signals.
"""
import asyncio
from backend.storage.database import SupabaseInterface

async def check_backtest_data():
    db = SupabaseInterface()
    await db.connect()
    
    # Check signals with backtest data
    query = """
    SELECT 
        COUNT(*) as total_signals,
        COUNT(return_1d) as has_1d,
        COUNT(return_7d) as has_7d,
        COUNT(return_30d) as has_30d,
        COUNT(return_90d) as has_90d
    FROM signals
    """
    
    result = await db.execute_query(query)
    stats = result[0]
    
    print("=" * 80)
    print("BACKTEST DATA COVERAGE")
    print("=" * 80)
    print(f"Total signals: {stats['total_signals']}")
    print(f"Has 1D returns:  {stats['has_1d']} ({stats['has_1d']/stats['total_signals']*100:.1f}%)")
    print(f"Has 7D returns:  {stats['has_7d']} ({stats['has_7d']/stats['total_signals']*100:.1f}%)")
    print(f"Has 30D returns: {stats['has_30d']} ({stats['has_30d']/stats['total_signals']*100:.1f}%)")
    print(f"Has 90D returns: {stats['has_90d']} ({stats['has_90d']/stats['total_signals']*100:.1f}%)")
    
    # Get sample of signals without backtest data
    query2 = """
    SELECT ticker, created_at, return_1d, return_7d, return_30d
    FROM signals
    WHERE return_1d IS NULL
    LIMIT 5
    """
    
    no_backtest = await db.execute_query(query2)
    
    print("\n" + "=" * 80)
    print("SAMPLE SIGNALS WITHOUT BACKTEST DATA:")
    print("=" * 80)
    for sig in no_backtest:
        print(f"Ticker: {sig['ticker']}, Created: {sig['created_at']}")
    
    # Check oldest signal without backtest
    query3 = """
    SELECT ticker, created_at, run_id
    FROM signals
    WHERE return_1d IS NULL
    ORDER BY created_at ASC
    LIMIT 1
    """
    
    oldest = await db.execute_query(query3)
    if oldest:
        print(f"\nOldest signal without backtest: {oldest[0]['ticker']} from {oldest[0]['created_at']}")
    
    # Check newest signal with backtest
    query4 = """
    SELECT ticker, created_at, return_1d, return_7d, return_30d
    FROM signals
    WHERE return_1d IS NOT NULL
    ORDER BY created_at DESC
    LIMIT 1
    """
    
    newest = await db.execute_query(query4)
    if newest:
        print(f"Newest signal WITH backtest: {newest[0]['ticker']} from {newest[0]['created_at']}")
        print(f"  Returns: 1D={newest[0]['return_1d']:.2f}%, 7D={newest[0].get('return_7d')}, 30D={newest[0].get('return_30d')}")

if __name__ == "__main__":
    asyncio.run(check_backtest_data())
