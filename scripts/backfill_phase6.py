"""
Phase 6 Backfill - Process Pending Performance Records
======================================================

Runs ONLY Phase 6 performance tracking without full pipeline.
Useful for catching up on backlog without fetching new data.

Usage:
    python scripts/backfill_phase6.py [--limit 1000]
"""

import asyncio
import logging
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.phases.phase6_performance import PerformanceUpdater
from backend.storage.database import get_supabase_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run Phase 6 backfill."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Backfill Phase 6 performance data')
    parser.add_argument('--limit', type=int, default=1000, 
                       help='Maximum records to process (default: 1000)')
    args = parser.parse_args()
    
    print('='*100)
    print(f'PHASE 6 BACKFILL - Processing up to {args.limit} pending records')
    print('='*100)
    
    try:
        # Initialize database
        db = await get_supabase_database()
        logger.info('✓ Database connected')
        
        # Check current status
        result = db.client.table('performance').select('status').execute()
        from collections import Counter
        statuses = Counter(r['status'] for r in result.data)
        total = len(result.data)
        
        print(f'\n📊 BEFORE Backfill:')
        print(f'  Total records: {total}')
        print(f'  Pending:       {statuses.get("pending", 0):4d} ({statuses.get("pending", 0)/total*100:5.1f}%)')
        print(f'  In Progress:   {statuses.get("in_progress", 0):4d} ({statuses.get("in_progress", 0)/total*100:5.1f}%)')
        print(f'  Completed:     {statuses.get("completed", 0):4d} ({statuses.get("completed", 0)/total*100:5.1f}%)')
        
        # Initialize Phase 6
        logger.info('\n🔄 Initializing Phase 6...')
        updater = PerformanceUpdater(db=db)
        
        # Run backfill (no benchmark cache - will fetch as needed)
        print(f'\n🚀 Running Phase 6 backfill (limit={args.limit})...')
        print('   Note: No benchmark cache - yfinance will fetch SPY/QQQ/sector data')
        print('   This may take a few minutes...\n')
        
        stats = await updater.update_pending_performance(
            limit=args.limit,
            benchmark_cache=None  # Fetch benchmarks as needed
        )
        
        print('\n' + '='*100)
        print('PHASE 6 BACKFILL RESULTS')
        print('='*100)
        print(f'  Processed: {stats["processed"]}')
        print(f'  Updated:   {stats["updated"]}')
        print(f'  Failed:    {stats["failed"]}')
        
        # Check status after
        result_after = db.client.table('performance').select('status').execute()
        statuses_after = Counter(r['status'] for r in result_after.data)
        
        print(f'\n📊 AFTER Backfill:')
        print(f'  Total records: {len(result_after.data)}')
        print(f'  Pending:       {statuses_after.get("pending", 0):4d} ({statuses_after.get("pending", 0)/len(result_after.data)*100:5.1f}%) '
              f'[{statuses_after.get("pending", 0) - statuses.get("pending", 0):+d}]')
        print(f'  In Progress:   {statuses_after.get("in_progress", 0):4d} ({statuses_after.get("in_progress", 0)/len(result_after.data)*100:5.1f}%) '
              f'[{statuses_after.get("in_progress", 0) - statuses.get("in_progress", 0):+d}]')
        print(f'  Completed:     {statuses_after.get("completed", 0):4d} ({statuses_after.get("completed", 0)/len(result_after.data)*100:5.1f}%) '
              f'[{statuses_after.get("completed", 0) - statuses.get("completed", 0):+d}]')
        
        print('\n' + '='*100)
        
        if statuses_after.get("pending", 0) > 0:
            print(f'⏳ Still {statuses_after.get("pending", 0)} pending records remaining')
            print(f'   Run backfill again to continue processing')
        else:
            print('✅ All pending records processed!')
        
        print('='*100)
        
        await db.disconnect()
        return 0
        
    except Exception as e:
        logger.error(f'❌ Backfill failed: {e}', exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
