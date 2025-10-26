"""
Check what signal runs exist in database and their details.
"""
import asyncio
from backend.storage.database import SupabaseInterface

async def check_runs():
    db = SupabaseInterface()
    await db.connect()
    
    # Get all runs
    query = """
    SELECT id, run_timestamp, total_tickers, successful_tickers, 
           status, pipeline_version
    FROM signal_runs
    ORDER BY run_timestamp DESC
    """
    
    runs = await db.execute_query(query)
    
    print("=" * 80)
    print(f"SIGNAL RUNS IN DATABASE: {len(runs)}")
    print("=" * 80)
    
    for i, run in enumerate(runs, 1):
        print(f"\n{i}. Run ID: {run['id']}")
        print(f"   Timestamp: {run['run_timestamp']}")
        print(f"   Tickers: {run['successful_tickers']}/{run['total_tickers']}")
        print(f"   Status: {run['status']}")
        print(f"   Version: {run.get('pipeline_version', 'N/A')}")
        
        # Count signals for this run
        count_query = "SELECT COUNT(*) as count FROM signals WHERE run_id = $1"
        count_result = await db.execute_query(count_query, [run['id']])
        signal_count = count_result[0]['count'] if count_result else 0
        print(f"   Signals: {signal_count}")

if __name__ == "__main__":
    asyncio.run(check_runs())
