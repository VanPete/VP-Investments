"""VP Investments Pipeline v3.1 - Working Version"""
import asyncio
import json
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

async def run_pipeline(tickers=None):
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("VP INVESTMENTS PIPELINE v3.1")
    logger.info("=" * 80)
    
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
    
    # Print results
    total_duration = (datetime.now() - start_time).total_seconds()
    phase_timings = {
        'phase1': phase1_duration,
        'phase2': phase2_duration,
        'phase3': phase3_duration,
        'phase4': phase4_duration,
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
            f"{idx:2}. {ticker:6}  Score: {result.overall_score:7.4f}  "
            f"Coverage: {result.total_coverage * 100:5.2f}%"
        )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("TIMING BREAKDOWN")
    logger.info("=" * 80)
    logger.info(f"  Phase 1 (Fetch):     {phase_timings['phase1']:6.1f}s  ({phase_timings['phase1']/duration*100:5.1f}%)")
    logger.info(f"  Phase 2 (Calculate): {phase_timings['phase2']:6.1f}s  ({phase_timings['phase2']/duration*100:5.1f}%)")
    logger.info(f"  Phase 3 (Normalize): {phase_timings['phase3']:6.1f}s  ({phase_timings['phase3']/duration*100:5.1f}%)")
    logger.info(f"  Phase 4 (Score):     {phase_timings['phase4']:6.1f}s  ({phase_timings['phase4']/duration*100:5.1f}%)")
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
            
            # Extract group coverages
            group_coverages = {
                "technical": result.technical.coverage,
                "fundamental": result.fundamental.coverage,
                "news_macro": result.news_macro.coverage,
                "social_alternative": result.social_alternative.coverage,
                "risk_stability": result.risk_stability.coverage,
                "institutional_smart_money": result.institutional_smart_money.coverage
            }
            
            output["rankings"].append({
                "rank": rank,
                "ticker": ticker,
                "overall_score": round(result.overall_score, 4),
                "total_coverage": round(result.total_coverage, 4),
                "group_scores": {k: round(v, 4) for k, v in group_scores.items()},
                "group_coverages": {k: round(v, 4) for k, v in group_coverages.items()}
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
