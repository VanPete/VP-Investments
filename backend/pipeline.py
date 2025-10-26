"""VP Investments Pipeline v3.2 - With Performance Analytics"""
import asyncio
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict
warnings.filterwarnings('ignore', category=DeprecationWarning)
from backend.utils.logger import setup_logging, get_logger
setup_logging(log_level="INFO", log_dir="logs", console_output=True)
logger = get_logger(__name__)

from backend.phases.phase1_fetch import Phase1Fetcher
from backend.phases.phase2_calculate import Phase2Calculator  
from backend.phases.phase3_normalize import Phase3Normalizer
from backend.phases.phase4_score_assemble import Phase4ScoreAssembler, ScoreResult
from backend.storage.database import get_supabase_database
from backend.phases.phase5_persist import Phase5Persist

# Phase 5.6 Optimizations - Enable via environment variable
ENABLE_PHASE1_OPTIMIZATION = os.getenv('ENABLE_PHASE1_OPTIMIZATION', 'true').lower() == 'true'
PHASE1_MAX_CONCURRENT = int(os.getenv('PHASE1_MAX_CONCURRENT', '10'))
ENABLE_PHASE5_OPTIMIZATION = os.getenv('ENABLE_PHASE5_OPTIMIZATION', 'true').lower() == 'true'

async def run_pipeline(tickers=None):
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info(f"VP INVESTMENTS PIPELINE v3.2 {'(OPTIMIZED)' if ENABLE_PHASE1_OPTIMIZATION else ''}")
    logger.info("=" * 80)
    
        # Phase 1: Fetch & Cache
    if ENABLE_PHASE1_OPTIMIZATION:
        try:
            from backend.phases.phase1_fetch import get_optimized_phase1_fetcher
            p1 = get_optimized_phase1_fetcher()
            logger.info(f"🚀 [OPTIMIZATION] Using optimized Phase 1 fetcher (parallel batch processing)")
        except Exception as e:
            logger.warning(f"[WARNING] Could not load optimized Phase 1, falling back to standard: {e}")
            from backend.phases.phase1_fetch import Phase1Fetcher
            p1 = Phase1Fetcher()
    else:
        from backend.phases.phase1_fetch import Phase1Fetcher
        p1 = Phase1Fetcher()
    p2 = Phase2Calculator()
    p3 = Phase3Normalizer()
    p4 = Phase4ScoreAssembler()
    
    # Phase 1: Fetch
    phase1_start = datetime.now()
    if tickers:
        phase1_results = await p1.fetch_all_data(tickers=tickers)
    else:
        phase1_results = await p1.fetch_all_data()
    phase1_duration = (datetime.now() - phase1_start).total_seconds()
    
    # Phase 2-4: Calculate, Normalize, Score
    raw_cache = phase1_results.get('raw_cache_by_ticker', {})
    reddit_data = phase1_results.get('reddit_data', {})
    news_data = phase1_results.get('news_data', {})
    market_data = phase1_results.get('market_data')  # NEW: market-wide data
    
    phase2_start = datetime.now()
    phase2_results = p2.calculate_batch(raw_cache, reddit_data=reddit_data, news_data_by_ticker=news_data, market_data=market_data)
    phase2_duration = (datetime.now() - phase2_start).total_seconds()
    
    phase3_start = datetime.now()
    phase3_results = p3.normalize_batch(phase2_results)
    phase3_duration = (datetime.now() - phase3_start).total_seconds()
    
    phase4_start = datetime.now()
    phase4_results = p4.score_all_tickers(phase3_results)
    phase4_duration = (datetime.now() - phase4_start).total_seconds()
    
    # ============================================================================
    # PHASE 5: DATABASE PERSISTENCE
    # ============================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 5: DATABASE PERSISTENCE")
    logger.info("=" * 80)
    
    phase5_start = datetime.now()
    phase5_duration = 0
    
    try:
        # Connect to database
        db = await get_supabase_database()
        await db.connect()
        
        # Phase 5.6: Use optimized Phase 5 persistence if enabled
        if ENABLE_PHASE5_OPTIMIZATION:
            try:
                from backend.phases.phase5_persist import get_optimized_phase5_persist
                p5 = get_optimized_phase5_persist(db)
                logger.info(f"🚀 [OPTIMIZATION] Using optimized Phase 5 persistence (bulk INSERT)")
            except Exception as e:
                logger.warning(f"[WARNING] Could not load optimized Phase 5: {e}. Falling back to standard persistence.")
                p5 = Phase5Persist(db)
        else:
            p5 = Phase5Persist(db)
        
        # Transform phase4_results (Dict[str, ScoreResult]) to list format for Phase 5
        phase4_list = []
        for ticker, result in phase4_results.items():
            # Get normalized factor data from phase3_results (NormalizedGroupFactors)
            ticker_norm = phase3_results.get(ticker)
            if not ticker_norm:
                logger.warning(f"Skipping {ticker}: No normalized factors found")
                continue
            
            phase4_list.append({
                'ticker': ticker,
                'rank': None,  # Will be set based on sort order
                'overall_score': result.overall_score,
                'total_coverage': result.total_coverage,
                'technical_score': result.technical.score,
                'technical_coverage': result.technical.coverage,
                'fundamental_score': result.fundamental.score,
                'fundamental_coverage': result.fundamental.coverage,
                'news_macro_score': result.news_macro.score,
                'news_macro_coverage': result.news_macro.coverage,
                'social_score': result.social_alternative.score,
                'social_coverage': result.social_alternative.coverage,
                'risk_score': result.risk_stability.score,
                'risk_coverage': result.risk_stability.coverage,
                'institutional_score': result.institutional_smart_money.score,
                'institutional_coverage': result.institutional_smart_money.coverage,
                'technical_data': ticker_norm.technical,
                'fundamental_data': ticker_norm.fundamental,
                'news_macro_data': ticker_norm.news_macro,
                'social_data': ticker_norm.social_alternative,
                'risk_data': ticker_norm.risk_stability,
                'institutional_data': ticker_norm.institutional_smart_money
            })
        
        # Sort by overall_score descending and assign ranks
        phase4_list.sort(key=lambda x: x['overall_score'], reverse=True)
        for rank, item in enumerate(phase4_list, 1):
            item['rank'] = rank
        
        # Persist complete pipeline run to database
        logger.info(f"[STATS] Persisting {len(phase4_list)} signals to database...")
        run_id = await p5.persist_pipeline_run(
            phase4_results=phase4_list,
            phase1_cache=raw_cache
        )
        
        phase5_duration = (datetime.now() - phase5_start).total_seconds()
        
        logger.info(f"[SUCCESS] Phase 5 complete in {phase5_duration:.2f}s")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Signals persisted: {len(phase4_list)}")
        logger.info("=" * 80)
        
        # PHASE 6: PERFORMANCE TRACKING & ANALYTICS
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 6: PERFORMANCE TRACKING & ANALYTICS")
        logger.info("=" * 80)
        
        phase6_start = datetime.now()
        
        try:
            from backend.phases.phase6_performance import get_performance_updater
            from backend.phases.phase7_analytics import get_performance_analytics
            
            # Part 1: Update performance intervals for pending signals
            p6_tracker = get_performance_updater(db)
            perf_stats = await p6_tracker.update_pending_performance(limit=100)
            
            logger.info(f"[TRACKING] Performance records processed: {perf_stats['processed']}")
            logger.info(f"[TRACKING] Performance records updated: {perf_stats['updated']}")
            if perf_stats['failed'] > 0:
                logger.warning(f"[TRACKING] Performance records failed: {perf_stats['failed']}")
            
            # Part 2: Calculate risk metrics and analytics
            p6_analytics = get_performance_analytics(db)
            analytics_7d = await p6_analytics.calculate_all_metrics(interval='7d', min_signals=3)
            
            phase6_duration = (datetime.now() - phase6_start).total_seconds()
            
            if analytics_7d.get('total_signals', 0) >= 3:
                logger.info(f"[ANALYTICS] Sharpe Ratio (7d): {analytics_7d['sharpe_ratio']:.2f}")
                logger.info(f"[ANALYTICS] Sortino Ratio (7d): {analytics_7d['sortino_ratio']:.2f}")
                logger.info(f"[ANALYTICS] Win Rate (7d): {analytics_7d['win_rate_pct']:.1f}%")
                logger.info(f"[ANALYTICS] Profit Factor (7d): {analytics_7d['profit_factor']:.2f}x")
                logger.info(f"[ANALYTICS] Max Drawdown: {analytics_7d['max_drawdown_pct']:.1f}%")
                logger.info(f"[ANALYTICS] Signals analyzed: {analytics_7d['total_signals']}")
            else:
                logger.info(f"[ANALYTICS] Skipped - insufficient data ({analytics_7d.get('total_signals', 0)} signals)")
            
            logger.info(f"[SUCCESS] Phase 6 complete in {phase6_duration:.2f}s")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.warning(f"[WARNING] Phase 6 failed: {e}")
            logger.warning("   This is non-critical - continuing...")
            phase6_duration = (datetime.now() - phase6_start).total_seconds()
        
        # Disconnect database
        await db.disconnect()
        
    except Exception as e:
        logger.error(f"[ERROR] Phase 5 failed: {e}")
        logger.error(f"   Continuing with export to JSON...")
        logger.exception(e)  # Log full traceback for debugging
        phase5_duration = (datetime.now() - phase5_start).total_seconds()
        phase6_duration = 0
    
    # Print results
    total_duration = (datetime.now() - start_time).total_seconds()
    phase_timings = {
        'phase1': phase1_duration,
        'phase2': phase2_duration,
        'phase3': phase3_duration,
        'phase4': phase4_duration,
        'phase5': phase5_duration,
        'phase6': phase6_duration,  # Now includes both tracking and analytics
    }
    _print_results(phase4_results, total_duration, phase_timings)
    _export_json(phase4_results, phase1_results)
    
    return phase4_results

def _print_results(results: Dict[str, ScoreResult], duration: float, phase_timings: Dict[str, float]) -> None:
    """Display top 10 results with timing breakdown."""
    sorted_results = sorted(results.items(), key=lambda x: x[1].overall_score, reverse=True)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("TOP 10 RESULTS")
    logger.info("=" * 80)
    for idx, (ticker, result) in enumerate(sorted_results[:10], 1):
        logger.info(
            f"{idx:2}. {ticker:6}  Score: {result.overall_score:7.4f}"
        )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("TIMING BREAKDOWN")
    logger.info("=" * 80)
    logger.info(f"  Phase 1 (Fetch):       {phase_timings['phase1']:6.1f}s  ({phase_timings['phase1']/duration*100:5.1f}%)")
    logger.info(f"  Phase 2 (Calculate):   {phase_timings['phase2']:6.1f}s  ({phase_timings['phase2']/duration*100:5.1f}%)")
    logger.info(f"  Phase 3 (Normalize):   {phase_timings['phase3']:6.1f}s  ({phase_timings['phase3']/duration*100:5.1f}%)")
    logger.info(f"  Phase 4 (Score):       {phase_timings['phase4']:6.1f}s  ({phase_timings['phase4']/duration*100:5.1f}%)")
    logger.info(f"  Phase 5 (Persist):     {phase_timings['phase5']:6.1f}s  ({phase_timings['phase5']/duration*100:5.1f}%)")
    logger.info(f"  Phase 6 (Tracking+Analytics): {phase_timings['phase6']:6.1f}s  ({phase_timings['phase6']/duration*100:5.1f}%)")
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"[SUCCESS] COMPLETE - {len(results)} tickers analyzed in {duration:.1f}s")
    logger.info("=" * 80)

def _export_json(results: dict, phase1_results: dict) -> None:
    """Export results to JSON with enhanced metadata."""
    try:
        # Save directly to frontend public directory
        results_dir = Path("frontend/public/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract discovery info from phase1
        reddit_tickers = phase1_results.get('discovered_tickers', [])
        news_tickers = phase1_results.get('news_discovered_tickers', [])
        all_tickers = phase1_results.get('all_tickers', [])
        
        # Build JSON structure
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tickers": len(results),
                "source": "manual" if reddit_tickers and news_tickers else "auto-discovery",
                "discovery": {
                    "reddit_tickers": len(reddit_tickers),
                    "news_tickers": len(news_tickers),
                    "total_universe": len(all_tickers)
                }
            },
            "rankings": []
        }
        
        # Sort by score
        sorted_results = sorted(results.items(), key=lambda x: x[1].overall_score, reverse=True)
        
        for rank, (ticker, result) in enumerate(sorted_results, 1):
            # Extract group scores from individual attributes
            group_scores = {
                "technical": result.technical.score,
                "fundamental": result.fundamental.score,
                "news_macro": result.news_macro.score,
                "social_alternative": result.social_alternative.score,
                "risk_stability": result.risk_stability.score,
                "institutional_smart_money": result.institutional_smart_money.score
            }
            
            output["rankings"].append({
                "rank": rank,
                "ticker": ticker,
                "overall_score": round(result.overall_score, 4),
                "group_scores": {k: round(v, 4) for k, v in group_scores.items()}
            })
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"pipeline_results_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")

if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(run_pipeline(tickers))
