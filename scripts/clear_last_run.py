"""Clear the most recent signal run from Supabase"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backend.storage.database import SupabaseInterface

async def clear_last_run():
    db = SupabaseInterface()
    await db.connect()
    
    print("=" * 80)
    print("CLEARING MOST RECENT SIGNAL RUN")
    print("=" * 80)
    
    # Get the most recent run
    query = """
    SELECT id, created_at, total_tickers, successful_tickers
    FROM signal_runs
    ORDER BY created_at DESC
    LIMIT 1
    """
    
    runs = await db.execute_query(query)
    
    if not runs:
        print("\n❌ No signal runs found in database")
        await db.disconnect()
        return
    
    run = runs[0]
    run_id = run['id']
    created_at = run['created_at']
    total_tickers = run.get('total_tickers', 0)
    
    print(f"\nMost recent run:")
    print(f"  ID: {run_id}")
    print(f"  Created: {created_at}")
    print(f"  Tickers: {total_tickers}")
    
    # Count signals in this run
    count_query = "SELECT COUNT(*) as count FROM signals WHERE run_id = $1"
    count_result = await db.execute_query(count_query, [run_id])
    signal_count = count_result[0]['count']
    
    print(f"  Signals: {signal_count}")
    
    # Delete detail tables first (foreign key constraints)
    print("\nDeleting signal detail tables...")
    detail_tables = [
        'signals_technical',
        'signals_fundamental', 
        'signals_news_macro',
        'signals_social_alternative',
        'signals_risk_stability',
        'signals_institutional_smart_money'
    ]
    
    for table in detail_tables:
        delete_query = f"""
        DELETE FROM {table}
        WHERE signal_id IN (
            SELECT id FROM signals WHERE run_id = $1
        )
        """
        await db.execute_non_query(delete_query, [run_id])
        print(f"  ✓ Cleared {table}")
    
    # Delete signals
    print("\nDeleting signals...")
    delete_signals = "DELETE FROM signals WHERE run_id = $1"
    await db.execute_non_query(delete_signals, [run_id])
    print(f"  ✓ Deleted {signal_count} signals")
    
    # Delete the run
    print("\nDeleting signal run...")
    delete_run = "DELETE FROM signal_runs WHERE id = $1"
    await db.execute_non_query(delete_run, [run_id])
    print(f"  ✓ Deleted run {run_id}")
    
    # Verify
    print("\nVerifying cleanup...")
    remaining_runs = await db.execute_query(
        "SELECT COUNT(*) as count FROM signal_runs"
    )
    remaining_signals = await db.execute_query(
        "SELECT COUNT(*) as count FROM signals"
    )
    
    print(f"  Remaining runs: {remaining_runs[0]['count']}")
    print(f"  Remaining signals: {remaining_signals[0]['count']}")
    
    print("\n" + "=" * 80)
    print("✅ LAST RUN CLEARED SUCCESSFULLY")
    print("=" * 80)
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(clear_last_run())
