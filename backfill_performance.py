"""
Backfill Performance Tracking for Signals
==========================================

Backfills performance data (1d, 3d, 7d, 10d, 14d, 30d, 90d returns)
for signals created from October 17, 2025 onwards.

Usage:
    python backfill_performance.py

This will:
1. Set baseline prices (next day open) for all signals
2. Calculate returns for all eligible intervals based on signal age
3. Add SPY benchmark comparisons
4. Update signals table with performance data
"""

import asyncio
import logging
from datetime import datetime
from backend.phases.phase6_backtest import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s'
)
logger = logging.getLogger(__name__)


async def backfill_performance():
    """Backfill performance tracking for signals from Oct 17, 2025."""
    logger.info("=" * 80)
    logger.info("PERFORMANCE TRACKING BACKFILL")
    logger.info("=" * 80)
    print()
    
    # Initialize tracker
    tracker = PerformanceTracker()
    await tracker.set_database()
    
    # Start date: October 17, 2025 (first day with real signals)
    start_date = datetime(2025, 10, 17)
    
    logger.info(f"Backfilling signals from {start_date.date()} onwards...")
    print()
    
    # Run backfill
    stats = await tracker.backfill_signals(
        start_date=start_date,
        limit=100  # Process up to 100 signals
    )
    
    logger.info("=" * 80)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 80)
    print()
    
    logger.info(f"Signals Processed: {stats['processed']}")
    logger.info(f"Signals Updated: {stats['updated']}")
    logger.info(f"Signals Failed: {stats['failed']}")
    
    if stats['updated'] > 0:
        success_rate = (stats['updated'] / stats['processed']) * 100 if stats['processed'] > 0 else 0
        logger.info(f"Success Rate: {success_rate:.1f}%")
    
    print()
    logger.info("✅ Performance tracking backfill complete!")
    print()
    
    # Show sample of updated signals
    logger.info("Fetching sample of updated signals...")
    result = tracker.db.client.table('signals').select(
        'ticker, created_at, backtest_baseline_price, return_1d, return_3d, return_7d, spy_return_7d, backtest_status'
    ).not_.is_('return_1d', 'null').order('created_at', desc=True).limit(5).execute()
    
    if result.data:
        print()
        logger.info("Sample Updated Signals:")
        logger.info("-" * 80)
        for signal in result.data:
            ticker = signal['ticker']
            baseline = signal.get('backtest_baseline_price', 0)
            r1d = signal.get('return_1d', 0)
            r3d = signal.get('return_3d', 0)
            r7d = signal.get('return_7d', 0)
            spy_r7d = signal.get('spy_return_7d', 0)
            status = signal.get('backtest_status', 'unknown')
            
            beat_spy = ''
            if r7d is not None and spy_r7d is not None:
                if r7d > spy_r7d:
                    beat_spy = '📈 Beats SPY'
                else:
                    beat_spy = '📉 Below SPY'
            
            logger.info(
                f"  {ticker:6} | Baseline: ${baseline:7.2f} | "
                f"1d: {r1d:+6.2f}% | 3d: {r3d:+6.2f}% | 7d: {r7d:+6.2f}% | "
                f"SPY 7d: {spy_r7d:+6.2f}% | {beat_spy} | Status: {status}"
            )


if __name__ == "__main__":
    asyncio.run(backfill_performance())
