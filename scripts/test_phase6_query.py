"""Test Phase 6 performance updater with direct debugging"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from backend.storage.database import get_supabase_database

async def test_phase6_query():
    """Test the exact query Phase 6 uses"""
    db = await get_supabase_database()
    
    print("Testing Phase 6 Query")
    print("=" * 100)
    
    # Test the exact query Phase 6 uses
    result = db.client.table('performance').select(
        'id, signal_id, baseline_price, baseline_date, intervals_completed, sector, sector_etf, signals!inner(ticker, created_at)'
    ).in_(
        'status', ['pending', 'in_progress']
    ).order('created_at', desc=False).limit(5).execute()
    
    print(f"\nFound {len(result.data)} records")
    
    for i, perf in enumerate(result.data, 1):
        print(f"\nRecord {i}:")
        print(f"  ID: {perf['id']}")
        print(f"  Signal ID: {perf['signal_id']}")
        print(f"  Baseline Price: {perf['baseline_price']}")
        print(f"  Baseline Date: {perf['baseline_date']}")
        print(f"  Sector: {perf.get('sector')}")
        print(f"  Sector ETF: {perf.get('sector_etf')}")
        print(f"  Intervals Completed: {perf.get('intervals_completed')}")
        print(f"  Signals join: {perf.get('signals')}")
        
        # Try to access ticker
        if 'signals' in perf:
            print(f"  Ticker: {perf['signals'].get('ticker') if isinstance(perf['signals'], dict) else 'ERROR: signals not a dict'}")
        else:
            print(f"  Ticker: ERROR - No 'signals' key")

if __name__ == "__main__":
    asyncio.run(test_phase6_query())
