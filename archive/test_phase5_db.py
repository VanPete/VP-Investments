"""
Test Phase 5 Database Methods

Verifies that Phase 5 persistence methods work correctly with the new schema.
"""

import asyncio
import logging
from datetime import datetime

# Import Phase 5 methods (auto-adds to SupabaseInterface)
from backend.phases.phase5_persist import add_phase5_methods_to_supabase_interface
from backend.storage.database import SupabaseInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_phase5_methods():
    """Test Phase 5 database methods."""
    
    logger.info("="*80)
    logger.info("TESTING PHASE 5 DATABASE METHODS")
    logger.info("="*80)
    
    # Initialize database
    db = SupabaseInterface()
    await db.connect()
    
    logger.info("\n✅ Database connected")
    
    try:
        # Test 1: Create signal run
        logger.info("\n📝 Test 1: Create signal run")
        run_config = {
            'total_tickers': 2,
            'successful_tickers': 0,
            'failed_tickers': 0,
            'pipeline_version': '2.0-test',
            'status': 'running'
        }
        
        run_id = await db.create_signal_run(run_config)
        logger.info(f"   ✅ Created run_id: {run_id}")
        
        # Test 2: Insert test signals
        logger.info("\n📝 Test 2: Insert signal batch")
        test_signals = [
            {
                'ticker': 'AAPL',
                'rank': 1,
                'overall_score': 0.95,
                'total_coverage': 0.76,  # Average of all coverages
                'technical_score': 0.90,
                'technical_coverage': 0.85,
                'fundamental_score': 0.88,
                'fundamental_coverage': 0.80,
                'news_macro_score': 0.75,
                'news_macro_coverage': 0.70,
                'social_alternative_score': 0.82,
                'social_alternative_coverage': 0.65,
                'risk_stability_score': 0.78,
                'risk_stability_coverage': 0.75,
                'institutional_smart_money_score': 0.91,
                'institutional_smart_money_coverage': 0.82
            },
            {
                'ticker': 'MSFT',
                'rank': 2,
                'overall_score': 0.92,
                'total_coverage': 0.76,  # Average of all coverages
                'technical_score': 0.88,
                'technical_coverage': 0.80,
                'fundamental_score': 0.90,
                'fundamental_coverage': 0.85,
                'news_macro_score': 0.72,
                'news_macro_coverage': 0.68,
                'social_alternative_score': 0.79,
                'social_alternative_coverage': 0.62,
                'risk_stability_score': 0.81,
                'risk_stability_coverage': 0.78,
                'institutional_smart_money_score': 0.89,
                'institutional_smart_money_coverage': 0.80
            }
        ]
        
        signal_ids = await db.insert_signals_batch(run_id, test_signals)
        logger.info(f"   ✅ Inserted {len(signal_ids)} signals")
        logger.info(f"   Signal IDs: {signal_ids}")
        
        # Test 3: Insert factor details for first signal
        if signal_ids:
            logger.info("\n📝 Test 3: Insert factor details")
            signal_id = signal_ids[0]
            
            # Technical factors
            technical_factors = {
                "rsi_14": {"raw": 65.2, "normalized": 0.75, "percentile": 0.82},
                "macd": {"raw": 1.2, "normalized": 0.60, "percentile": 0.65},
                "volume_ratio": {"raw": 1.5, "normalized": 0.68, "percentile": 0.71}
            }
            await db.insert_technical_factors(signal_id, technical_factors)
            logger.info(f"   ✅ Inserted technical factors")
            
            # Fundamental factors
            fundamental_factors = {
                "pe_ratio": {"raw": 25.3, "normalized": 0.55, "percentile": 0.60},
                "profit_margin": {"raw": 0.21, "normalized": 0.82, "percentile": 0.85},
                "roe": {"raw": 0.35, "normalized": 0.88, "percentile": 0.90}
            }
            await db.insert_fundamental_factors(signal_id, fundamental_factors)
            logger.info(f"   ✅ Inserted fundamental factors")
        
        # Test 4: Query signals
        logger.info("\n📝 Test 4: Query signals by run")
        signals = await db.get_signals_by_run_id(run_id, limit=10)
        logger.info(f"   ✅ Retrieved {len(signals)} signals")
        for sig in signals:
            logger.info(f"   - {sig['ticker']}: score={sig['overall_score']:.2f}, rank={sig['rank']}")
        
        # Test 5: Get signal with factors
        if signal_ids:
            logger.info("\n📝 Test 5: Get complete signal with factors")
            complete_signal = await db.get_signal_with_factors(signal_ids[0])
            if complete_signal:
                logger.info(f"   ✅ Retrieved complete signal for {complete_signal['ticker']}")
                logger.info(f"   Technical factors: {len(complete_signal.get('technical_factors', {}))} factors")
                logger.info(f"   Fundamental factors: {len(complete_signal.get('fundamental_factors', {}))} factors")
            else:
                logger.warning(f"   ⚠️  No complete signal found")
        
        # Test 6: Update run status
        logger.info("\n📝 Test 6: Update run status")
        success = await db.update_signal_run(run_id, {
            'status': 'completed',
            'successful_tickers': len(test_signals),
            'failed_tickers': 0
        })
        logger.info(f"   ✅ Updated run status: {success}")
        
        # Test 7: Get run statistics
        logger.info("\n📝 Test 7: Get signal statistics")
        stats = await db.get_signal_statistics(run_id)
        logger.info(f"   ✅ Statistics:")
        logger.info(f"   - Total signals: {stats.get('total_signals')}")
        logger.info(f"   - Avg score: {stats.get('avg_score', 0):.3f}")
        logger.info(f"   - Top ticker: {stats.get('top_ticker')}")
        logger.info(f"   - Top score: {stats.get('top_score', 0):.3f}")
        
        # Test 8: Get recent runs
        logger.info("\n📝 Test 8: Get recent signal runs")
        recent_runs = await db.get_recent_signal_runs(limit=5)
        logger.info(f"   ✅ Retrieved {len(recent_runs)} recent runs")
        for run in recent_runs:
            logger.info(f"   - Run {run['id']}: {run['status']}, {run.get('successful_tickers', 0)} successful, {run.get('failed_tickers', 0)} failed")
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL PHASE 5 TESTS PASSED!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        logger.exception("Full error details:")
        
    finally:
        await db.disconnect()
        logger.info("\n✅ Database disconnected")


if __name__ == "__main__":
    asyncio.run(test_phase5_methods())
