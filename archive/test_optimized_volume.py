"""
Quick Volume Test with Optimizations Enabled

Tests both Phase 1 and Phase 5 optimizations with 50 tickers.
"""

import asyncio
import os
import time

# Enable both optimizations
os.environ['ENABLE_PHASE1_OPTIMIZATION'] = 'true'
os.environ['ENABLE_PHASE5_OPTIMIZATION'] = 'true'

from backend.pipeline import run_pipeline
from backend.utils.logger import setup_logging, get_logger

setup_logging(log_level="INFO", log_dir="logs", console_output=True)
logger = get_logger(__name__)


async def main():
    # Test with 50 tickers
    test_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B', 'V', 'MA', 'AXP',
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY', 'ABT', 'DHR', 'BMY',
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
        'WMT', 'HD', 'COST', 'TGT', 'LOW', 'TJX', 'DG', 'ROST', 'BBY', 'DLTR'
    ]
    
    logger.info("=" * 80)
    logger.info("VOLUME TEST: 50 Tickers with Optimizations Enabled")
    logger.info("=" * 80)
    logger.info(f"Phase 1 Optimization: ENABLED")
    logger.info(f"Phase 5 Optimization: ENABLED")
    logger.info("")
    
    start_time = time.time()
    
    try:
        results = await run_pipeline(tickers=test_tickers)
        duration = time.time() - start_time
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("VOLUME TEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total Duration: {duration:.2f}s")
        logger.info(f"Tickers Processed: {len(results)}")
        logger.info(f"Avg Time per Ticker: {duration / len(results):.2f}s")
        logger.info("")
        logger.info("Baseline (Phase 5.5): 336.3s for 50 tickers (6.73s per ticker)")
        logger.info(f"Current Run: {duration:.2f}s ({duration / len(results):.2f}s per ticker)")
        
        if duration < 336.3:
            improvement = ((336.3 - duration) / 336.3) * 100
            logger.info(f"🚀 IMPROVEMENT: {improvement:.1f}% faster!")
        
    except Exception as e:
        logger.error(f"Volume test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
