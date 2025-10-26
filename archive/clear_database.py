"""Clear all data from the database to start fresh"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backend.storage.database import SupabaseInterface

async def clear_database():
    db = SupabaseInterface()
    await db.connect()
    
    print("=" * 80)
    print("CLEARING ALL DATABASE DATA")
    print("=" * 80)
    
    # Get current counts
    print("\nCurrent data:")
    
    runs_count = await db.execute_query("SELECT COUNT(*) as count FROM signal_runs")
    signals_count = await db.execute_query("SELECT COUNT(*) as count FROM signals")
    print(f"  Signal runs: {runs_count[0]['count']}")
    print(f"  Signals: {signals_count[0]['count']}")
    
    # Delete detail tables first (foreign key constraints)
    print("\nDeleting detail tables...")
    detail_tables = [
        'signals_technical',
        'signals_fundamental', 
        'signals_news_macro',
        'signals_social_alternative',
        'signals_risk_stability',
        'signals_institutional_smart_money'
    ]
    
    for table in detail_tables:
        result = await db.execute_non_query(f"DELETE FROM {table}")
        print(f"  ✓ Cleared {table}")
    
    # Delete signals
    print("\nDeleting signals...")
    await db.execute_non_query("DELETE FROM signals")
    print("  ✓ Cleared signals")
    
    # Delete runs
    print("\nDeleting signal_runs...")
    await db.execute_non_query("DELETE FROM signal_runs")
    print("  ✓ Cleared signal_runs")
    
    # Verify
    print("\nVerifying cleanup:")
    runs_count = await db.execute_query("SELECT COUNT(*) as count FROM signal_runs")
    signals_count = await db.execute_query("SELECT COUNT(*) as count FROM signals")
    print(f"  Signal runs: {runs_count[0]['count']}")
    print(f"  Signals: {signals_count[0]['count']}")
    
    print("\n" + "=" * 80)
    print("DATABASE CLEARED SUCCESSFULLY")
    print("=" * 80)
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(clear_database())
