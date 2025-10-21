"""
Phase 6: Post-Operations
=========================

Post-processing operations after signal persistence:
- AI strategy generation
- Backtesting
- Cleanup and validation
- Report generation

This phase runs AFTER signals are saved to the database.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class Phase6PostOps:
    """
    Phase 6: Post-Operations
    
    Handles post-processing after signals are persisted:
    - AI strategy generation for top signals
    - Backtesting (future)
    - Data cleanup
    - Report generation
    """
    
    def __init__(self):
        """Initialize Phase 6 post-operations module."""
        self.logger = logger
        self._init_ai_generator()
    
    def _init_ai_generator(self):
        """Initialize AI strategy generator if available."""
        try:
            from backend.integrations.ai_strategy_generator import AIStrategyGenerator
            self.AIStrategyGenerator = AIStrategyGenerator
            self.ai_available = True
            self.logger.info("AI strategy generator available")
        except ImportError:
            self.AIStrategyGenerator = None
            self.ai_available = False
            self.logger.warning("AI strategy generator not available")
    
    async def run_post_operations(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run all post-operations after signal persistence.
        
        Args:
            run_id: Pipeline run ID for tracking
            
        Returns:
            Dict with operation results
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 6: POST-OPERATIONS")
        self.logger.info("=" * 60)
        
        results = {
            'ai_strategies': await self.generate_ai_strategies(run_id),
            'backtests': {'success': True, 'message': 'Backtesting not yet implemented'},
            'cleanup': self._run_cleanup(),
        }
        
        self.logger.info(f"✅ Phase 6 complete")
        return results
    
    async def generate_ai_strategies(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate AI strategies for top signals.
        
        This delegates to the AIStrategyGenerator integration to create
        trading strategies based on the top-scoring signals from this run.
        
        Args:
            run_id: Pipeline run ID to generate strategies for
            
        Returns:
            Dict with strategy generation results:
                - success: bool
                - strategies_count: int
                - tickers_count: int
                - strategy_summary: List[str]
                - message: str
        """
        try:
            # Check if AI strategies are enabled
            ai_enabled = os.getenv('AI_STRATEGY_ENABLED', 'false').lower() == 'true'
            
            if not ai_enabled:
                self.logger.info("AI strategy generation disabled, skipping")
                return {'success': True, 'strategies_count': 0, 'message': 'AI strategies disabled'}
            
            # Check if AI generator is available
            if not self.ai_available or not self.AIStrategyGenerator:
                self.logger.warning("AI strategy generator not available")
                return {'success': False, 'strategies_count': 0, 'message': 'AI generator not available'}
            
            # Initialize and run AI strategy generator with run_id
            generator = self.AIStrategyGenerator(run_id=run_id)
            
            if not generator.ai_enabled:
                self.logger.warning("AI strategy generator not properly initialized")
                return {'success': False, 'strategies_count': 0, 'message': 'AI generator not initialized'}
            
            # Generate strategies for top signals
            self.logger.info(f"Generating AI strategies for top {generator.top_signals_limit} signals...")
            strategies = await generator.generate_strategies_for_top_signals()
            
            if strategies:
                total_strategies = sum(len(s) for s in strategies.values())
                self.logger.info(f"[SUCCESS] Generated {total_strategies} AI strategies for {len(strategies)} tickers")
                
                # Log strategy summary
                strategy_summary = []
                for ticker, ticker_strategies in strategies.items():
                    strategy_types = [s.strategy_type for s in ticker_strategies]
                    strategy_summary.append(f"{ticker}: {len(ticker_strategies)} ({', '.join(strategy_types)})")
                    self.logger.info(f"   [STATS] {ticker}: {len(ticker_strategies)} strategies")
                
                return {
                    'success': True, 
                    'strategies_count': total_strategies,
                    'tickers_count': len(strategies),
                    'strategy_summary': strategy_summary,
                    'message': f'Generated {total_strategies} strategies for {len(strategies)} tickers'
                }
            else:
                self.logger.warning("No AI strategies were generated")
                return {'success': False, 'strategies_count': 0, 'message': 'No strategies generated'}
                
        except Exception as e:
            self.logger.error(f"AI strategy generation failed: {e}")
            return {'success': False, 'strategies_count': 0, 'message': f'Error: {str(e)}'}
    
    def _run_cleanup(self) -> Dict[str, Any]:
        """
        Run cleanup operations.
        
        Future: Could include cache cleanup, old data removal, etc.
        
        Returns:
            Dict with cleanup results
        """
        self.logger.info("Running post-processing cleanup...")
        
        # Placeholder for future cleanup operations
        return {
            'success': True,
            'message': 'Cleanup complete'
        }
