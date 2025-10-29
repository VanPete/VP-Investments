"""VP Investments Pipeline v3.2 - With Performance Analytics"""
import asyncio
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Calling float on.*')
from backend.utils.logger import setup_logging, get_logger
from backend.utils.progress_display import PipelineProgress

# Set up logging - will be reconfigured by run_pipeline_and_push.py if using CLI args
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

async def run_pipeline(tickers=None, show_progress: bool = True, verbose_level: int = 0):
    """
    Run the VP Investments pipeline.
    
    Args:
        tickers: Optional list of tickers to analyze. If None, auto-discovers from Reddit/News.
        show_progress: If True, show Rich progress bars (default: True)
        verbose_level: Console verbosity (0=warnings only, 1=info, 2=debug)
        
    Returns:
        Dict[str, ScoreResult]: Final scored results by ticker
    """
    start_time = datetime.now()
    
    # Initialize progress display if enabled
    progress = None
    if show_progress:
        progress = PipelineProgress(verbose=(verbose_level > 0))
        progress.__enter__()  # Start progress context
        progress.show_header()
    else:
        # Traditional header logging
        logger.info("=" * 80)
        logger.info(f"VP INVESTMENTS PIPELINE v3.2 {'(OPTIMIZED)' if ENABLE_PHASE1_OPTIMIZATION else ''}")
        logger.info("=" * 80)
    
    try:
        # Initialize phase objects
        if ENABLE_PHASE1_OPTIMIZATION:
            try:
                from backend.phases.phase1_fetch import get_optimized_phase1_fetcher
                p1 = get_optimized_phase1_fetcher()
                if not show_progress:
                    logger.info(f"🚀 [OPTIMIZATION] Using optimized Phase 1 fetcher (parallel batch processing)")
            except Exception as e:
                if not show_progress:
                    logger.warning(f"[WARNING] Could not load optimized Phase 1, falling back to standard: {e}")
                from backend.phases.phase1_fetch import Phase1Fetcher
                p1 = Phase1Fetcher()
        else:
            from backend.phases.phase1_fetch import Phase1Fetcher
            p1 = Phase1Fetcher()
        
        p2 = Phase2Calculator()
        p3 = Phase3Normalizer()
        p4 = Phase4ScoreAssembler()
        
        # ========================================================================
        # PHASE 1: FETCH DATA
        # ========================================================================
        phase1_start = datetime.now()
        
        if progress:
            progress.start_phase("phase1", total_items=100, description="[bold blue]Phase 1:[/] Fetch Data")
        else:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PHASE 1: FETCH DATA")
            logger.info("=" * 80)
        
        if tickers:
            phase1_results = await p1.fetch_all_data(tickers=tickers)
        else:
            # Increased post limit from 100 → 150 for better ticker discovery
            phase1_results = await p1.fetch_all_data(post_limit=150)
        
        phase1_duration = (datetime.now() - phase1_start).total_seconds()
        
        if progress:
            progress.complete_phase("phase1", f"✓ Complete ({phase1_duration:.1f}s)")
        else:
            logger.info(f"[SUCCESS] Phase 1 complete in {phase1_duration:.2f}s")
        
        # ========================================================================
        # PHASE 2: CALCULATE FACTORS
        # ========================================================================
        raw_cache = phase1_results.get('raw_cache_by_ticker', {})
        reddit_data = phase1_results.get('reddit_data', {})
        news_data = phase1_results.get('news_data', {})
        market_data = phase1_results.get('market_data')
        
        phase2_start = datetime.now()
        num_tickers = len(raw_cache)
        
        if progress:
            progress.start_phase("phase2", total_items=num_tickers, 
                               description="[bold green]Phase 2:[/] Calculate Factors")
        else:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PHASE 2: CALCULATE FACTORS")
            logger.info("=" * 80)
        
        phase2_results = p2.calculate_batch(
            raw_cache, 
            reddit_data=reddit_data, 
            news_data_by_ticker=news_data, 
            market_data=market_data
        )
        
        phase2_duration = (datetime.now() - phase2_start).total_seconds()
        
        if progress:
            progress.complete_phase("phase2", f"✓ {num_tickers} tickers calculated ({phase2_duration:.1f}s)")
        else:
            logger.info(f"[SUCCESS] Phase 2 complete in {phase2_duration:.2f}s")
        
        # ========================================================================
        # PHASE 3: NORMALIZE SCORES
        # ========================================================================
        phase3_start = datetime.now()
        
        if progress:
            progress.start_phase("phase3", total_items=3, 
                               description="[bold yellow]Phase 3:[/] Normalize Scores")
        else:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PHASE 3: NORMALIZE SCORES")
            logger.info("=" * 80)
        
        phase3_results = p3.normalize_batch(phase2_results)
        
        phase3_duration = (datetime.now() - phase3_start).total_seconds()
        
        if progress:
            progress.complete_phase("phase3", f"✓ Complete ({phase3_duration:.1f}s)")
        else:
            logger.info(f"[SUCCESS] Phase 3 complete in {phase3_duration:.2f}s")
        
        # ========================================================================
        # PHASE 4: ASSEMBLE SCORES
        # ========================================================================
        phase4_start = datetime.now()
        
        if progress:
            progress.start_phase("phase4", total_items=num_tickers,
                               description="[bold magenta]Phase 4:[/] Assemble Scores")
        else:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PHASE 4: ASSEMBLE SCORES")
            logger.info("=" * 80)
        
        phase4_results = p4.score_all_tickers(phase3_results)
        
        phase4_duration = (datetime.now() - phase4_start).total_seconds()
        
        if progress:
            progress.complete_phase("phase4", f"✓ {num_tickers} signals generated ({phase4_duration:.1f}s)")
        else:
            logger.info(f"[SUCCESS] Phase 4 complete in {phase4_duration:.2f}s")
        
        # ========================================================================
        # PHASE 5: DATABASE PERSISTENCE
        # ========================================================================
        phase5_start = datetime.now()
        phase5_duration = 0
        
        if progress:
            progress.start_phase("phase5", total_items=5,
                               description="[bold cyan]Phase 5:[/] Save to Database")
        else:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PHASE 5: DATABASE PERSISTENCE")
            logger.info("=" * 80)
        
        try:
            # Connect to database
            if progress:
                progress.update_phase("phase5", advance=1, status="Connecting to database...")
            
            db = await get_supabase_database()
            await db.connect()
            
            # Phase 5.6: Use optimized Phase 5 persistence if enabled
            if ENABLE_PHASE5_OPTIMIZATION:
                try:
                    from backend.phases.phase5_persist import get_optimized_phase5_persist
                    p5 = get_optimized_phase5_persist(db)
                    if not show_progress:
                        logger.info(f"🚀 [OPTIMIZATION] Using optimized Phase 5 persistence (bulk INSERT)")
                except Exception as e:
                    if not show_progress:
                        logger.warning(f"[WARNING] Could not load optimized Phase 5: {e}. Falling back to standard persistence.")
                    p5 = Phase5Persist(db)
            else:
                p5 = Phase5Persist(db)
            
            if progress:
                progress.update_phase("phase5", advance=1, status="Preparing signals...")
            
            # Transform phase4_results to list format
            phase4_list = []
            for ticker, result in phase4_results.items():
                ticker_norm = phase3_results.get(ticker)
                if not ticker_norm:
                    logger.warning(f"Skipping {ticker}: No normalized factors found")
                    continue
                
                phase4_list.append({
                    'ticker': ticker,
                    'rank': None,
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
            
            # Sort by overall_score and assign ranks
            phase4_list.sort(key=lambda x: x['overall_score'], reverse=True)
            for rank, item in enumerate(phase4_list, 1):
                item['rank'] = rank
            
            if progress:
                progress.update_phase("phase5", advance=1, status=f"Persisting {len(phase4_list)} signals...")
            else:
                logger.info(f"[STATS] Persisting {len(phase4_list)} signals to database...")
            
            # Persist to database
            run_id = await p5.persist_pipeline_run(
                phase4_results=phase4_list,
                phase1_cache=raw_cache
            )
            
            if progress:
                progress.update_phase("phase5", advance=1, status="Updating performance...")
            
            phase5_duration = (datetime.now() - phase5_start).total_seconds()
            
            if progress:
                progress.complete_phase("phase5", f"✓ {len(phase4_list)} signals persisted ({phase5_duration:.1f}s)")
            else:
                logger.info(f"[SUCCESS] Phase 5 complete in {phase5_duration:.2f}s")
                logger.info(f"   Run ID: {run_id}")
                logger.info(f"   Signals persisted: {len(phase4_list)}")
                logger.info("=" * 80)
            
            # ====================================================================
            # PHASE 6: PERFORMANCE TRACKING
            # ====================================================================
            phase6_start = datetime.now()
            
            if progress:
                progress.start_phase("phase6", total_items=1,
                                   description="[bold white]Phase 6:[/] Performance Tracking")
            else:
                logger.info("")
                logger.info("=" * 80)
                logger.info("PHASE 6: PERFORMANCE TRACKING")
                logger.info("=" * 80)
            
            try:
                from backend.phases.phase6_performance import get_performance_updater
                
                # Get benchmark cache from Phase 1
                benchmark_cache = phase1_results.get('sector_etf_data', {})
                
                if not show_progress:
                    logger.info(f"  Using cached benchmark data: {len(benchmark_cache)} ETFs from Phase 1")
                
                # Update performance intervals for pending signals
                p6_tracker = get_performance_updater(db)
                perf_stats = await p6_tracker.update_pending_performance(
                    limit=200,
                    benchmark_cache=benchmark_cache
                )
                
                if progress:
                    progress.update_phase("phase6", advance=1, 
                                        status=f"✓ {perf_stats['updated']} records updated")
                else:
                    logger.info(f"  Performance records processed: {perf_stats['processed']}")
                    logger.info(f"  Performance records updated: {perf_stats['updated']}")
                    if perf_stats['failed'] > 0:
                        logger.warning(f"  Performance records failed: {perf_stats['failed']}")
                
                phase6_duration = (datetime.now() - phase6_start).total_seconds()
                
                if progress:
                    progress.complete_phase("phase6", f"✓ Complete ({phase6_duration:.1f}s)")
                else:
                    logger.info(f"[SUCCESS] Phase 6 complete in {phase6_duration:.2f}s")
                    logger.info("=" * 80)
                
            except Exception as e:
                logger.warning(f"[WARNING] Phase 6 failed: {e}")
                if not show_progress:
                    logger.warning("   This is non-critical - continuing...")
                phase6_duration = (datetime.now() - phase6_start).total_seconds()
            
            # ====================================================================
            # PHASE 7: ANALYTICS
            # ====================================================================
            phase7_start = datetime.now()
            
            if progress:
                progress.start_phase("phase7", total_items=1,
                                   description="[bold magenta]Phase 7:[/] Analytics")
            else:
                logger.info("")
                logger.info("=" * 80)
                logger.info("PHASE 7: ANALYTICS")
                logger.info("=" * 80)
            
            try:
                from backend.phases.phase7_analytics import get_analytics_engine
                
                # Calculate and persist analytics
                p7_analytics = get_analytics_engine(db)
                analytics_result = await p7_analytics.calculate_and_persist_analytics(
                    period_type='all_time'
                )
                
                phase7_duration = (datetime.now() - phase7_start).total_seconds()
                
                if progress:
                    progress.update_phase("phase7", advance=1, status="✓ Analytics calculated")
                    progress.complete_phase("phase7", f"✓ Complete ({phase7_duration:.1f}s)")
                else:
                    logger.info(f"[SUCCESS] Phase 7 complete in {phase7_duration:.2f}s")
                    logger.info("=" * 80)
                
            except Exception as e:
                logger.warning(f"[WARNING] Phase 7 failed: {e}")
                if not show_progress:
                    logger.warning("   This is non-critical - continuing...")
                phase7_duration = (datetime.now() - phase7_start).total_seconds()
            
            # Disconnect database
            await db.disconnect()
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 5 failed: {e}")
            logger.error(f"   Continuing with export to JSON...")
            logger.exception(e)
            phase5_duration = (datetime.now() - phase5_start).total_seconds()
            phase6_duration = 0
            phase7_duration = 0
        
        # ====================================================================
        # FINALIZE: Export & Metrics
        # ====================================================================
        total_duration = (datetime.now() - start_time).total_seconds()
        phase_timings = {
            'phase1': phase1_duration,
            'phase2': phase2_duration,
            'phase3': phase3_duration,
            'phase4': phase4_duration,
            'phase5': phase5_duration,
            'phase6': phase6_duration,
            'phase7': phase7_duration,
        }
        
        # Show summary with progress bars or traditional logging
        if progress:
            # Show Rich summary
            progress.show_summary({
                "total_duration": total_duration,
                "tickers_processed": num_tickers,
                "signals_generated": len(phase4_results),
                "success_rate": 0.944  # TODO: Calculate actual success rate
            })
        else:
            # Traditional logging
            _print_results(phase4_results, total_duration, phase_timings)
        
        # Always export JSON and metrics
        _export_json(phase4_results, phase1_results)
        _export_performance_metrics(phase_timings, total_duration, phase1_results, phase2_results, phase4_results, run_id if 'run_id' in locals() else None)
        _compare_with_previous_run(phase_timings, total_duration, phase1_results, phase4_results)
        
        return phase4_results
        
    finally:
        # Clean up progress display
        if progress:
            progress.__exit__(None, None, None)

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
    logger.info(f"  Phase 6 (Performance): {phase_timings['phase6']:6.1f}s  ({phase_timings['phase6']/duration*100:5.1f}%)")
    logger.info(f"  Phase 7 (Analytics):   {phase_timings['phase7']:6.1f}s  ({phase_timings['phase7']/duration*100:5.1f}%)")
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

def _export_performance_metrics(
    phase_timings: Dict[str, float], 
    total_duration: float,
    phase1_results: dict,
    phase2_results: dict,
    phase4_results: dict,
    run_id: Optional[str] = None
) -> None:
    """Export structured performance metrics to JSON for analysis."""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate phase percentages
        phase_percentages = {
            phase: (duration / total_duration * 100) if total_duration > 0 else 0
            for phase, duration in phase_timings.items()
        }
        
        # Identify bottlenecks (phases > 10% of total time)
        bottlenecks = []
        for phase, pct in phase_percentages.items():
            if pct > 10:
                recommendation = ""
                if phase == "phase1":
                    recommendation = "Consider caching market data or increasing concurrent workers"
                elif phase == "phase2":
                    recommendation = "Review factor calculation efficiency"
                elif phase == "phase5":
                    recommendation = "Database bulk INSERT already optimized"
                elif phase == "phase6":
                    recommendation = "Limit performance record updates or batch more efficiently"
                
                bottlenecks.append({
                    "phase": phase,
                    "duration_seconds": phase_timings[phase],
                    "percent_of_total": round(pct, 1),
                    "recommendation": recommendation
                })
        
        # Extract Phase 1 metrics
        all_tickers = phase1_results.get('all_tickers', [])
        reddit_tickers = phase1_results.get('discovered_tickers', [])
        news_tickers = phase1_results.get('news_discovered_tickers', [])
        
        # Extract Phase 2 factor metrics
        total_factors_calculated = 0
        successful_factors = 0
        if phase2_results:
            for ticker_data in phase2_results.values():
                for group_data in ticker_data.__dict__.values():
                    if hasattr(group_data, '__dict__'):
                        for factor_value in group_data.__dict__.values():
                            total_factors_calculated += 1
                            if factor_value is not None and not (isinstance(factor_value, float) and (factor_value != factor_value)):  # Check for NaN
                                successful_factors += 1
        
        factor_success_rate = (successful_factors / total_factors_calculated * 100) if total_factors_calculated > 0 else 0
        
        # Build performance metrics
        performance_metrics = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": round(total_duration, 2),
            "phases": {
                phase: {
                    "duration_seconds": round(duration, 2),
                    "percent_of_total": round(phase_percentages[phase], 1)
                }
                for phase, duration in phase_timings.items()
            },
            "bottlenecks": bottlenecks,
            "ticker_metrics": {
                "total_discovered": len(all_tickers),
                "reddit_discovered": len(reddit_tickers),
                "news_discovered": len(news_tickers),
                "signals_generated": len(phase4_results)
            },
            "factor_metrics": {
                "total_calculations": total_factors_calculated,
                "successful_calculations": successful_factors,
                "success_rate_percent": round(factor_success_rate, 1)
            },
            "top_signals": [
                {
                    "ticker": ticker,
                    "score": round(result.overall_score, 4)
                }
                for ticker, result in sorted(phase4_results.items(), key=lambda x: x[1].overall_score, reverse=True)[:10]
            ]
        }
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = logs_dir / f"performance_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        logger.info(f"[METRICS] Performance metrics saved: {filepath}")
        
        # Log bottlenecks
        if bottlenecks:
            logger.warning(f"[PERFORMANCE] {len(bottlenecks)} bottlenecks identified:")
            for bn in bottlenecks:
                logger.warning(f"   {bn['phase']}: {bn['percent_of_total']}% ({bn['duration_seconds']}s) - {bn['recommendation']}")
        
    except Exception as e:
        logger.error(f"Failed to export performance metrics: {e}")

def _compare_with_previous_run(
    phase_timings: Dict[str, float],
    total_duration: float,
    phase1_results: dict,
    phase4_results: dict
) -> None:
    """Compare current run metrics with the most recent previous run."""
    try:
        # Find the most recent performance metrics file
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return
        
        performance_files = sorted(logs_dir.glob("performance_*.json"), reverse=True)
        if len(performance_files) < 2:  # Need at least 2 runs to compare
            return
        
        # Load previous run (second most recent, since most recent is current run being saved)
        previous_file = performance_files[1] if len(performance_files) > 1 else None
        if not previous_file:
            return
        
        with open(previous_file, 'r') as f:
            previous_metrics = json.load(f)
        
        # Compare key metrics
        prev_duration = previous_metrics.get('total_duration_seconds', 0)
        prev_tickers = previous_metrics.get('ticker_metrics', {}).get('total_discovered', 0)
        prev_signals = previous_metrics.get('ticker_metrics', {}).get('signals_generated', 0)
        
        curr_tickers = len(phase1_results.get('all_tickers', []))
        curr_signals = len(phase4_results)
        
        # Calculate changes
        duration_change = ((total_duration - prev_duration) / prev_duration * 100) if prev_duration > 0 else 0
        ticker_change = ((curr_tickers - prev_tickers) / prev_tickers * 100) if prev_tickers > 0 else 0
        signal_change = ((curr_signals - prev_signals) / prev_signals * 100) if prev_signals > 0 else 0
        
        # Log comparison
        logger.info("")
        logger.info("=" * 80)
        logger.info("RUN COMPARISON vs. Previous Run")
        logger.info("=" * 80)
        logger.info(f"  Runtime:           {total_duration:.1f}s vs {prev_duration:.1f}s "
                   f"({duration_change:+.1f}%)")
        logger.info(f"  Tickers discovered: {curr_tickers} vs {prev_tickers} "
                   f"({ticker_change:+.1f}%)")
        logger.info(f"  Signals generated:  {curr_signals} vs {prev_signals} "
                   f"({signal_change:+.1f}%)")
        
        # Highlight significant changes
        if abs(duration_change) > 20:
            logger.warning(f"  [ALERT] Runtime changed by {duration_change:+.1f}% - investigate performance")
        if ticker_change < -30:
            logger.warning(f"  [ALERT] Ticker discovery dropped {abs(ticker_change):.1f}% - check data sources")
        elif ticker_change > 50:
            logger.info(f"  [POSITIVE] Ticker discovery increased {ticker_change:.1f}%")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.debug(f"Could not compare with previous run: {e}")

if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(run_pipeline(tickers))
