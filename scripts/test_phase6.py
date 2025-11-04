"""Test Phase 6 execution directly."""

import asyncio
from backend.storage.database import get_database
from backend.phases.phase6_performance import PerformanceUpdater

async def main():
    db = get_database()
    
    print("=" * 80)
    print("TESTING PHASE 6 EXECUTION")
    print("=" * 80)
    
    # Create Phase 6 updater
    updater = PerformanceUpdater(db)
    
    # Try to run with a small limit
    print("\nRunning Phase 6 with limit=10...")
    result = await updater.update_pending_performance(limit=10, benchmark_cache={})
    
    print("\n" + "=" * 80)
    print("PHASE 6 RESULTS")
    print("=" * 80)
    print(f"Processed: {result.get('processed', 0)}")
    print(f"Updated: {result.get('updated', 0)}")
    print(f"Skipped: {result.get('skipped', 0)}")
    print(f"Failed: {result.get('failed', 0)}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    asyncio.run(main())
