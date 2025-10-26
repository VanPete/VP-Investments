"""
Quick Test: Phase 5 Bulk INSERT Optimization

Validates that the optimized Phase 5 persistence with bulk INSERT operations
works correctly and is faster than the original sequential implementation.

This test will:
1. Run a small pipeline (10 tickers) with ORIGINAL Phase 5
2. Run same pipeline with OPTIMIZED Phase 5  
3. Compare timing and validate results match
"""

import asyncio
import os
import sys
import time

# Set environment variables BEFORE importing pipeline
os.environ['ENABLE_PHASE1_OPTIMIZATION'] = 'false'  # Disable Phase 1 optimization for pure Phase 5 test
os.environ['ENABLE_PHASE5_OPTIMIZATION'] = 'false'  # Start with original

from backend.pipeline import run_pipeline
from backend.storage.database import get_supabase_database
from backend.utils.logger import setup_logging, get_logger

setup_logging(log_level="INFO", log_dir="logs", console_output=True)
logger = get_logger(__name__)


async def clear_test_data():
    """Clear test data from database before running tests."""
    logger.info("🧹 Clearing old test data from database...")
    db = await get_supabase_database()
    await db.connect()
    
    try:
        # Delete old test runs (keep only latest 5 runs)
        await db.execute_non_query("""
            DELETE FROM signal_runs
            WHERE id NOT IN (
                SELECT id FROM signal_runs
                ORDER BY run_timestamp DESC
                LIMIT 5
            )
        """, [])
        logger.info("✅ Test data cleared")
    except Exception as e:
        logger.warning(f"⚠️ Could not clear test data: {e}")
    finally:
        await db.disconnect()


async def get_run_details(run_id: str):
    """Get details about a specific run."""
    db = await get_supabase_database()
    await db.connect()
    
    try:
        # Get run metadata
        run_info = await db.execute_query(
            "SELECT * FROM signal_runs WHERE id = $1",
            [run_id]
        )
        
        # Get signal count
        signal_count = await db.execute_query(
            "SELECT COUNT(*) as count FROM signals WHERE run_id = $1",
            [run_id]
        )
        
        # Get factor counts
        factor_counts = {}
        for table in ['technical', 'fundamental', 'news_macro', 'social_alternative', 'risk_stability', 'institutional_smart_money']:
            count = await db.execute_query(
                f"SELECT COUNT(*) as count FROM signals_{table} WHERE signal_id IN (SELECT id FROM signals WHERE run_id = $1)",
                [run_id]
            )
            factor_counts[table] = count[0]['count'] if count else 0
        
        return {
            'run_info': run_info[0] if run_info else None,
            'signal_count': signal_count[0]['count'] if signal_count else 0,
            'factor_counts': factor_counts
        }
    finally:
        await db.disconnect()


async def main():
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC']
    
    logger.info("=" * 80)
    logger.info("QUICK TEST: Phase 5 Bulk INSERT Optimization")
    logger.info("=" * 80)
    logger.info(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")
    logger.info("")
    
    # Clear old test data
    await clear_test_data()
    
    # Test 1: Original Phase 5 (Sequential INSERTs)
    logger.info("=" * 80)
    logger.info("TEST 1: Original Phase 5 Persistence (Sequential INSERTs)")
    logger.info("=" * 80)
    os.environ['ENABLE_PHASE5_OPTIMIZATION'] = 'false'
    
    start_time = time.time()
    try:
        result1 = await run_pipeline(tickers=test_tickers)
        duration1 = time.time() - start_time
        
        # Get run ID from database (latest run)
        db = await get_supabase_database()
        await db.connect()
        run1_info = await db.execute_query(
            "SELECT id FROM signal_runs ORDER BY run_timestamp DESC LIMIT 1",
            []
        )
        run1_id = run1_info[0]['id'] if run1_info else None
        await db.disconnect()
        
        logger.info(f"✅ Original Phase 5: {duration1:.2f}s")
        logger.info(f"   Run ID: {run1_id}")
        logger.info("")
        
        # Get detailed stats
        if run1_id:
            details1 = await get_run_details(run1_id)
            logger.info(f"   Signals: {details1['signal_count']}")
            logger.info(f"   Factor records:")
            for table, count in details1['factor_counts'].items():
                logger.info(f"      {table}: {count}")
        
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        duration1 = time.time() - start_time
        run1_id = None
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Waiting 3 seconds before Test 2...")
    logger.info("=" * 80)
    await asyncio.sleep(3)
    
    # Test 2: Optimized Phase 5 (Bulk INSERTs)
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 2: Optimized Phase 5 Persistence (Bulk INSERTs + Parallel)")
    logger.info("=" * 80)
    os.environ['ENABLE_PHASE5_OPTIMIZATION'] = 'true'
    
    start_time = time.time()
    try:
        result2 = await run_pipeline(tickers=test_tickers)
        duration2 = time.time() - start_time
        
        # Get run ID from database (latest run)
        db = await get_supabase_database()
        await db.connect()
        run2_info = await db.execute_query(
            "SELECT id FROM signal_runs ORDER BY run_timestamp DESC LIMIT 1",
            []
        )
        run2_id = run2_info[0]['id'] if run2_info else None
        await db.disconnect()
        
        logger.info(f"✅ Optimized Phase 5: {duration2:.2f}s")
        logger.info(f"   Run ID: {run2_id}")
        logger.info("")
        
        # Get detailed stats
        if run2_id:
            details2 = await get_run_details(run2_id)
            logger.info(f"   Signals: {details2['signal_count']}")
            logger.info(f"   Factor records:")
            for table, count in details2['factor_counts'].items():
                logger.info(f"      {table}: {count}")
        
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        duration2 = time.time() - start_time
        run2_id = None
    
    # Compare results
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON")
    logger.info("=" * 80)
    
    if duration1 and duration2:
        speedup = ((duration1 - duration2) / duration1) * 100
        time_saved = duration1 - duration2
        
        logger.info(f"Original:  {duration1:.2f}s")
        logger.info(f"Optimized: {duration2:.2f}s")
        logger.info(f"")
        logger.info(f"{'🚀 SPEEDUP' if speedup > 0 else '⚠️ SLOWER'}: {speedup:+.1f}%")
        logger.info(f"Time saved: {time_saved:.2f}s")
        
        if run1_id and run2_id:
            details1 = await get_run_details(run1_id)
            details2 = await get_run_details(run2_id)
            
            logger.info("")
            logger.info("Data validation:")
            if details1['signal_count'] == details2['signal_count']:
                logger.info(f"✅ Signal count matches: {details1['signal_count']}")
            else:
                logger.warning(f"⚠️ Signal count mismatch: {details1['signal_count']} vs {details2['signal_count']}")
            
            all_match = True
            for table in details1['factor_counts'].keys():
                if details1['factor_counts'][table] == details2['factor_counts'][table]:
                    logger.info(f"✅ {table}: {details1['factor_counts'][table]} records")
                else:
                    logger.warning(f"⚠️ {table} mismatch: {details1['factor_counts'][table]} vs {details2['factor_counts'][table]}")
                    all_match = False
            
            if all_match:
                logger.info("")
                logger.info("✅ All data validation checks passed!")
    
    logger.info("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
