"""
Test Phase 6 Performance Tracking Manually
==========================================

Manually triggers Phase 6 update_pending_performance() to debug
why it's processing 0 records during pipeline execution.

This script:
1. Fetches Oct 28 performance records directly
2. Calls Phase 6's update_pending_performance() method
3. Shows detailed logs for debugging
4. Verifies benchmark data is populated
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.phases.phase6_performance import PerformanceUpdater
from backend.storage.database import get_supabase_database

# Configure logging to see all details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run Phase 6 manually and show results."""
    
    print("=" * 100)
    print("MANUAL PHASE 6 TEST")
    print("=" * 100)
    
    try:
        # Initialize database
        db = await get_supabase_database()
        logger.info("✓ Database connected")
        
        # Check current performance records
        result = db.client.table('performance').select(
            'id, signal_id, baseline_date, intervals_completed, status, signals!inner(ticker)'
        ).in_('status', ['pending', 'in_progress']).limit(10).execute()
        
        logger.info(f"\n📊 Found {len(result.data)} performance records with pending/in_progress status")
        
        if result.data:
            for i, rec in enumerate(result.data[:5], 1):
                logger.info(
                    f"  {i}. {rec['signals']['ticker']}: "
                    f"baseline={rec['baseline_date']}, "
                    f"intervals={rec.get('intervals_completed', [])}, "
                    f"status={rec.get('status')}"
                )
        
        # Initialize Phase 6
        logger.info("\n🔄 Initializing Phase 6...")
        updater = PerformanceUpdater(db=db)
        
        # Run update (without benchmark cache to test fallback)
        logger.info("\n🚀 Running Phase 6 update_pending_performance()...")
        logger.info("   (No benchmark cache - will test fallback to direct yfinance fetch)")
        
        stats = await updater.update_pending_performance(
            limit=50,
            benchmark_cache=None  # Force fallback to yfinance
        )
        
        logger.info("\n" + "=" * 100)
        logger.info("PHASE 6 RESULTS")
        logger.info("=" * 100)
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Updated:   {stats['updated']}")
        logger.info(f"  Failed:    {stats['failed']}")
        
        # Check if benchmarks were populated
        logger.info("\n🔍 Checking if benchmarks were populated...")
        
        result = db.client.table('performance').select(
            'baseline_date, intervals_completed, return_1d, spy_return_1d, qqq_return_1d, sector_return_1d, '
            'return_3d, spy_return_3d, qqq_return_3d, sector_return_3d, '
            'signals!inner(ticker)'
        ).eq('baseline_date', '2025-10-28T15:43:38.172366+00:00').limit(3).execute()
        
        if result.data:
            logger.info(f"\n📈 Sample of Oct 28 records after Phase 6:")
            for rec in result.data[:3]:
                ticker = rec['signals']['ticker']
                intervals = rec.get('intervals_completed', [])
                
                logger.info(f"\n  {ticker}:")
                logger.info(f"    Intervals completed: {intervals}")
                
                if 1 in intervals:
                    logger.info(
                        f"    1d: return={rec.get('return_1d')}, "
                        f"spy={rec.get('spy_return_1d')}, "
                        f"qqq={rec.get('qqq_return_1d')}, "
                        f"sector={rec.get('sector_return_1d')}"
                    )
                
                if 3 in intervals:
                    logger.info(
                        f"    3d: return={rec.get('return_3d')}, "
                        f"spy={rec.get('spy_return_3d')}, "
                        f"qqq={rec.get('qqq_return_3d')}, "
                        f"sector={rec.get('sector_return_3d')}"
                    )
        
        logger.info("\n" + "=" * 100)
        logger.info("✅ TEST COMPLETE")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
