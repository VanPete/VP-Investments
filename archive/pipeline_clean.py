"""
VP Investments Unified Pipeline 3.0
====================================

Pure orchestration pipeline that delegates ALL work to phase modules.
This is the central coordinator for the 6-phase signal generation system.

Architecture:
- Phase 1: Data Fetching (Reddit, Yahoo Finance, News) - Phase1Fetcher
- Phase 2: Data Normalization (Standardize signals) - Phase2Normalizer  
- Phase 3: Signal Scoring (6 group scores via SignalScorer) - SignalScorer
- Phase 4: Score Assembly (Combine group scores) - Phase4Assembler
- Phase 5: Database Persistence (Save signals) - Phase5Persister
- Phase 6: Post-Operations (AI strategies, cleanup) - Phase6PostOps

3.0 Signal Groups (6 groups):
- technical (20%)
- fundamental (25%)
- news_macro (15%)
- social_alternative (10%)
- risk_stability (15%)
- institutional_smart_money (15%)

Design Principles:
- Pure orchestration - NO business logic in pipeline
- All data fetching in Phase 1 ONLY
- All scoring logic in SignalScorer
- All persistence in Phase 5
- Clean separation of concerns
"""

import os
import sys
import logging
import asyncio
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
from backend.utils.logger import setup_logging, get_logger

setup_logging(
    log_level="INFO",
    log_dir="logs",
    console_output=True,
    structured_logging=False
)
logger = get_logger(__name__)

# Import phase modules
from backend.phases.phase1_fetch import Phase1Fetcher
from archive.phase2_normalize import Phase2Normalizer
from archive.phase4_assemble import Phase4Assembler
from backend.phases.phase5_persist import Phase5Persister
from backend.phases.phase6_post_ops import Phase6PostOps

# Import SignalScorer for Phase 3
from backend.core.signals import SignalScorer


class Config:
    """Configuration class for pipeline settings."""
    
    def __init__(self):
        self.reddit_post_limit = 100
        self.min_mentions = 1
        self.max_signals = 50
    
    def get(self, key, default=None):
        """Get configuration value with environment variable support."""
        if key == 'scoring.weights':
            from dotenv import load_dotenv
            load_dotenv()
            return {
                'technical': float(os.getenv('SCORE_WEIGHT_TECHNICAL', '0.20')),
                'fundamental': float(os.getenv('SCORE_WEIGHT_FUNDAMENTAL', '0.25')),
                'news_macro': float(os.getenv('SCORE_WEIGHT_NEWS_MACRO', '0.15')),
                'social_alternative': float(os.getenv('SCORE_WEIGHT_SOCIAL_ALTERNATIVE', '0.10')),
                'risk_stability': float(os.getenv('SCORE_WEIGHT_RISK_STABILITY', '0.15')),
                'institutional_smart_money': float(os.getenv('SCORE_WEIGHT_INSTITUTIONAL_SMART_MONEY', '0.15'))
            }
        return default


class UnifiedPipeline:
    """
    VP Investments 3.0 Pipeline - Pure Orchestration
    
    This pipeline coordinates the 6-phase signal generation system by delegating
    ALL work to specialized phase modules. It contains NO business logic.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline with all phase module instances."""
        self.config = config or Config()
        self.logger = logger
        
        # Initialize all phase modules
        self.phase1 = Phase1Fetcher()
        self.phase2 = Phase2Normalizer()
        self.signal_scorer = SignalScorer()  # Phase 3
        self.phase4 = Phase4Assembler()
        self.phase5 = Phase5Persister()
        self.phase6 = Phase6PostOps()
        
        self.logger.info("Pipeline 3.0 initialized with phase modules")
        self.logger.info(f"  Phase 1: {self.phase1.__class__.__name__}")
        self.logger.info(f"  Phase 2: {self.phase2.__class__.__name__}")
        self.logger.info(f"  Phase 3: {self.signal_scorer.__class__.__name__}")
        self.logger.info(f"  Phase 4: {self.phase4.__class__.__name__}")
        self.logger.info(f"  Phase 5: {self.phase5.__class__.__name__}")
        self.logger.info(f"  Phase 6: {self.phase6.__class__.__name__}")
    
    async def run_pipeline(self, 
                          subreddits: List[str] = None,
                          post_limit: int = 100,
                          min_mentions: int = 1,
                          max_signals: int = 50,
                          test_mode: bool = False) -> Dict[str, Any]:
        """
        Run the complete 6-phase pipeline.
        
        Args:
            subreddits: Subreddits to scrape
            post_limit: Posts per subreddit
            min_mentions: Minimum mentions required
            max_signals: Maximum signals to return
            test_mode: If True, use minimal settings for testing
            
        Returns:
            Dict with pipeline execution results
        """
        # Test mode override
        if test_mode:
            subreddits = ['wallstreetbets']
            post_limit = 10
            min_mentions = 1
            max_signals = 5
            self.logger.info("🧪 TEST MODE - Using minimal settings")
        
        pipeline_start = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("VP INVESTMENTS PIPELINE 3.0 - STARTING")
        self.logger.info("=" * 60)
        
        try:
            # PHASE 1: Data Fetching
            self.logger.info("📥 PHASE 1: Fetching all data...")
            phase1_data = await self.phase1.fetch_all_data(
                subreddits=subreddits or ['wallstreetbets', 'stocks', 'investing'],
                post_limit=post_limit,
                min_mentions=min_mentions
            )
            
            if not phase1_data or not phase1_data.get('ticker_mentions'):
                raise ValueError("Phase 1 failed - no tickers found")
            
            ticker_mentions = phase1_data['ticker_mentions']
            ticker_data_cache = phase1_data.get('ticker_data', {})
            all_tickers = list(ticker_mentions.keys())
            
            self.logger.info(f"✅ Phase 1: {len(all_tickers)} tickers")
            
            # PHASE 2: Signal Normalization
            self.logger.info("🔄 PHASE 2: Normalizing signals...")
            phase2_signals = await self.phase2.normalize_all_signals(
                ticker_mentions=ticker_mentions,
                ticker_data_cache=ticker_data_cache,
                all_tickers=all_tickers
            )
            
            self.logger.info(f"✅ Phase 2: {len(phase2_signals)} signals")
            
            # PHASE 3: Signal Scoring (6 groups)
            self.logger.info("📊 PHASE 3: Scoring signals...")
            phase3_scored = []
            
            for signal in phase2_signals:
                ticker = signal['ticker']
                ticker_data = ticker_data_cache.get(ticker, {})
                
                # Calculate 6 group scores
                group_scores = self.signal_scorer.score_ticker(ticker, ticker_data)
                signal.update(group_scores)
                phase3_scored.append(signal)
            
            self.logger.info(f"✅ Phase 3: {len(phase3_scored)} scored")
            
            # PHASE 4: Score Assembly
            self.logger.info("🎯 PHASE 4: Assembling final scores...")
            phase4_final = await self.phase4.assemble_final_scores(
                scored_signals=phase3_scored,
                ticker_data_cache=ticker_data_cache
            )
            
            phase4_final = phase4_final[:max_signals]
            self.logger.info(f"✅ Phase 4: {len(phase4_final)} final")
            
            # PHASE 5: Database Persistence
            self.logger.info("💾 PHASE 5: Saving to database...")
            phase5_result = await self.phase5.save_signals(
                signals=phase4_final,
                run_metadata={
                    'subreddits': subreddits,
                    'post_limit': post_limit,
                    'test_mode': test_mode
                }
            )
            
            if not phase5_result.get('success'):
                raise ValueError("Phase 5 failed")
            
            run_id = phase5_result.get('run_id')
            self.logger.info(f"✅ Phase 5: Saved (run_id: {run_id})")
            
            # PHASE 6: Post-Operations
            self.logger.info("🔧 PHASE 6: Post-operations...")
            phase6_result = await self.phase6.run_post_operations(run_id=run_id)
            
            self.logger.info(f"✅ Phase 6: Complete")
            
            # Pipeline complete
            execution_time = (datetime.now() - pipeline_start).total_seconds()
            
            results = {
                'success': True,
                'execution_time_seconds': execution_time,
                'signals_generated': len(phase4_final),
                'run_id': run_id,
                'phase_results': {
                    'phase1': {'tickers': len(all_tickers)},
                    'phase2': {'signals': len(phase2_signals)},
                    'phase3': {'scored': len(phase3_scored)},
                    'phase4': {'final': len(phase4_final)},
                    'phase5': phase5_result,
                    'phase6': phase6_result
                },
                'top_signals': phase4_final[:10]
            }
            
            self.logger.info("=" * 60)
            self.logger.info(f"✅ PIPELINE COMPLETE - {len(phase4_final)} signals in {execution_time:.2f}s")
            self.logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': (datetime.now() - pipeline_start).total_seconds(),
                'signals_generated': 0
            }
    
    async def generate_single_signal(self, 
                                     ticker: str, 
                                     include_reddit: bool = False) -> Optional[Dict[str, Any]]:
        """
        Generate a signal for a single ticker (for frontend manual requests).
        
        This method uses the same 6-phase architecture but optimized for single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            include_reddit: Whether to fetch Reddit data (slower, requires full scraping)
            
        Returns:
            Complete signal dict or None if failed
        """
        try:
            self.logger.info(f"🎯 Generating single signal for {ticker}...")
            start_time = datetime.now()
            
            # Validate ticker
            ticker = ticker.upper().strip()
            if not ticker or len(ticker) > 10:
                raise ValueError(f"Invalid ticker: {ticker}")
            
            # PHASE 1: Fetch single ticker data
            self.logger.info(f"Phase 1: Fetching data for {ticker}...")
            
            # Use Phase1Fetcher's internal method for single ticker
            ticker_data = self.phase1._fetch_ticker_data_sync(ticker)
            
            if not ticker_data:
                self.logger.error(f"Failed to fetch data for {ticker}")
                return None
            
            # Create minimal phase1_data structure
            phase1_data = {
                'ticker_mentions': {ticker: {'mention_count': 1, 'avg_sentiment': 0, 'mentions': []}},
                'ticker_data': {ticker: ticker_data}
            }
            
            # PHASE 2: Normalize
            self.logger.info(f"Phase 2: Normalizing...")
            phase2_signals = await self.phase2.normalize_all_signals(
                ticker_mentions=phase1_data['ticker_mentions'],
                ticker_data_cache=phase1_data['ticker_data'],
                all_tickers=[ticker]
            )
            
            if not phase2_signals:
                self.logger.error(f"Phase 2 normalization failed")
                return None
            
            signal = phase2_signals[0]
            
            # PHASE 3: Score
            self.logger.info(f"Phase 3: Scoring...")
            group_scores = self.signal_scorer.score_ticker(ticker, ticker_data)
            signal.update(group_scores)
            
            # PHASE 4: Assemble
            self.logger.info(f"Phase 4: Assembling...")
            phase4_signals = await self.phase4.assemble_final_scores(
                scored_signals=[signal],
                ticker_data_cache=phase1_data['ticker_data']
            )
            
            if not phase4_signals:
                self.logger.error(f"Phase 4 assembly failed")
                return None
            
            final_signal = phase4_signals[0]
            
            # PHASE 5: Save
            self.logger.info(f"Phase 5: Saving...")
            save_result = await self.phase5.save_signals(
                signals=[final_signal],
                run_metadata={'source': 'manual_single_ticker', 'ticker': ticker}
            )
            
            if not save_result.get('success'):
                self.logger.warning(f"Database save failed")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Signal generated for {ticker} in {elapsed:.2f}s")
            self.logger.info(f"   Score: {final_signal.get('signal_score', 'N/A')}")
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate signal for {ticker}: {e}")
            self.logger.error(traceback.format_exc())
            return None


async def main():
    """Main execution function."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        pipeline = UnifiedPipeline()
        
        results = await pipeline.run_pipeline(
            subreddits=['stocks', 'investing', 'wallstreetbets'],
            post_limit=100,
            min_mentions=1,
            max_signals=50
        )
        
        if results['success']:
            print(f"\n✅ Pipeline complete!")
            print(f"   Signals: {results['signals_generated']}")
            print(f"   Time: {results['execution_time_seconds']:.2f}s")
            print(f"   Run ID: {results.get('run_id')}")
            
            if 'top_signals' in results:
                print(f"\nTop 5 signals:")
                for i, signal in enumerate(results['top_signals'][:5], 1):
                    print(f"  {i}. {signal['ticker']}: {signal['signal_score']:.3f}")
        else:
            print(f"\n❌ Pipeline failed: {results.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)
