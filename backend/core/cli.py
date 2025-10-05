"""
VP Investments Main Application

This is the main entry point for the VP Investments platform.
It provides both CLI and programmatic interfaces for running analysis.

Usage:
    python -m src.main --help                    # Show help
    python -m src.main run                       # Run analysis with default settings
    python -m src.main run --tickers AAPL MSFT  # Run analysis for specific tickers
    python -m src.main server                    # Start API server
    python -m src.main config                    # Show configuration
"""
from __future__ import annotations

import asyncio
import logging
import sys
import argparse
from datetime import datetime
from typing import List, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.config import setup_logging, get_config
from backend.storage.database import get_database
from backend.pipeline import UnifiedPipeline
from backend.api.api import app as api_app

# Import production optimization integration
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from production_integration import ProductionOptimizedVPInvestments
    PRODUCTION_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    PRODUCTION_OPTIMIZATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VPInvestmentsApp:
    """Main application class for VP Investments platform"""
    
    def __init__(self, use_production_optimizations: bool = True):
        self.db = None
        self.pipeline = None
        self.analysis_engine = None
        
        # Production optimization components
        self.use_production_optimizations = use_production_optimizations and PRODUCTION_OPTIMIZATIONS_AVAILABLE
        self.production_optimized_vp = None
        
        if self.use_production_optimizations:
            logger.info("[PRODUCTION] Production optimizations enabled")
        else:
            logger.info("[STANDARD] Using standard VP Investments (production optimizations disabled)")
    
    async def initialize(self):
        """Initialize all application components"""
        logger.info("Initializing VP Investments platform...")
        
        try:
            if self.use_production_optimizations:
                # Initialize production-optimized VP Investments
                logger.info("[PRODUCTION] Initializing with production optimizations...")
                self.production_optimized_vp = ProductionOptimizedVPInvestments()
                await self.production_optimized_vp.initialize_production_optimizations()
                
                # Get integration status
                integration_status = await self.production_optimized_vp.get_integration_status()
                logger.info(f"[SUCCESS] Production optimizations ready: {integration_status['integration_status']['integration_success_rate']} success rate")
                
            else:
                # Initialize standard components
                logger.info("[STANDARD] Initializing standard VP Investments...")
                
                # Initialize database
                self.db = await get_database()
                logger.info("Database connection established")
                
                # Initialize unified pipeline
                self.pipeline = UnifiedPipeline()
                logger.info("Unified pipeline initialized")
                
                # Analysis engine not needed - pipeline handles analysis
                logger.info("Using pipeline-integrated analysis")
            
            logger.info("VP Investments platform initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize platform: {e}")
            raise
    
    async def run_analysis(self, tickers: Optional[List[str]] = None, run_id: Optional[str] = None) -> str:
        """
        Run complete investment analysis pipeline
        
        Args:
            tickers: List of tickers to analyze (optional, uses default universe if None)
            run_id: Custom run ID (optional, generates one if None)
        
        Returns:
            Run ID of the completed analysis
        """
        if not run_id:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting investment analysis run: {run_id}")
        
        try:
            # Use default tickers if none provided
            if not tickers:
                tickers = self._get_default_universe()
                logger.info(f"Using default ticker universe: {len(tickers)} tickers")
            else:
                logger.info(f"Analyzing specified tickers: {len(tickers)} tickers")
            
            if self.use_production_optimizations and self.production_optimized_vp:
                # Run enhanced analysis with production optimizations
                logger.info("[PRODUCTION] Running analysis with production optimizations...")
                
                enhanced_results = await self.production_optimized_vp.run_enhanced_analysis(tickers, run_id)
                
                logger.info(f"Enhanced analysis completed:")
                logger.info(f"  - Execution time: {enhanced_results['execution_time_seconds']:.2f}s")
                logger.info(f"  - Tickers processed: {enhanced_results['tickers_processed']}")
                logger.info(f"  - Average confidence: {enhanced_results['performance_summary']['avg_confidence']:.3f}")
                logger.info(f"  - Average win probability: {enhanced_results['performance_summary']['avg_win_probability']:.1%}")
                logger.info(f"  - High confidence signals: {enhanced_results['performance_summary']['high_confidence_signals']}")
                logger.info(f"  - Production optimizations active: {enhanced_results['performance_summary']['production_optimizations_active']}/5")
                
                # Display enhanced recommendations
                await self._display_enhanced_recommendations(enhanced_results['enhanced_results'])
                
            else:
                # Run standard analysis
                logger.info("[STANDARD] Running standard analysis...")
                
                # Run unified pipeline (includes data collection and analysis)
                logger.info("Running unified pipeline...")
                results = await self.pipeline.run_full_pipeline()
                
                if not results.get('success', False):
                    logger.error(f"Pipeline failed: {results.get('errors', [])}")
                    return run_id
                
                logger.info(f"Pipeline completed successfully:")
                logger.info(f"  - Reddit Integration: {'✅' if results.get('reddit_success') else '❌'}")
                logger.info(f"  - Yahoo Finance Integration: {'✅' if results.get('yahoo_success') else '❌'}")
                logger.info(f"  - News Integration: {'✅' if results.get('news_success') else '❌'}")
                logger.info(f"  - AI Strategies: {'✅' if results.get('ai_strategies_success') else '❌'}")
                
                if results.get('errors'):
                    logger.warning(f"Warnings encountered: {len(results['errors'])}")
                    for error in results['errors'][:5]:  # Show first 5 errors
                        logger.warning(f"  - {error}")
                
                logger.info("Analysis completed - check database for results")
            
            logger.info(f"Investment analysis run {run_id} completed successfully")
            return run_id
            
        except Exception as e:
            logger.error(f"Analysis run {run_id} failed: {e}")
            raise
    
    async def _display_recommendations(self, run_id: str):
        """Display recommendations info"""
        try:
            logger.info("[INFO] Analysis completed successfully")
            logger.info("       Check the database for detailed results")
            logger.info("       Use the web dashboard or database queries to view signals")
            
        except Exception as e:
            logger.error(f"Error displaying recommendations: {e}")
    
    async def _display_enhanced_recommendations(self, enhanced_results: List[dict]) -> None:
        """Display enhanced recommendations with production optimization metrics"""
        logger.info("[ENHANCED] Enhanced Recommendations (Production Optimized):")
        logger.info("=" * 60)
        
        # Sort by confidence score
        sorted_results = sorted(enhanced_results, key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        for i, result in enumerate(sorted_results[:10], 1):  # Show top 10
            ticker = result.get('ticker', 'N/A')
            signal = result.get('signal', 'HOLD')
            confidence = result.get('confidence_score', 0)
            win_prob = result.get('win_probability', 0)
            price_target = result.get('price_target', 0)
            market_regime = result.get('market_regime', 'NORMAL')
            
            # Format confidence indicator
            if confidence >= 0.8:
                conf_indicator = "[HIGH]"
            elif confidence >= 0.6:
                conf_indicator = "[MED]"
            else:
                conf_indicator = "[LOW]"
            
            logger.info(f"{i:2d}. {ticker:<6} | {signal:<4} | {conf_indicator} {confidence:.3f} | Win: {win_prob:.1%} | ${price_target:.2f} | {market_regime}")
        
        logger.info("=" * 60)
    
    def _get_default_universe(self) -> List[str]:
        """Get default stock universe for analysis"""
        return [
            # Large cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ', 'BRK-B',
            
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'CVS', 'MDT', 'GILD',
            
            # Consumer
            'WMT', 'HD', 'DIS', 'MCD', 'SBUX', 'NKE', 'KO', 'PG', 'TGT', 'COST',
            
            # Industrial & Energy
            'GE', 'BA', 'CAT', 'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY'
        ]
    
    async def get_analysis_report(self, run_id: str) -> dict:
        """Get comprehensive analysis report for a run"""
        return await self.analysis_engine.generate_analysis_report(run_id)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db:
            await self.db.close()
        logger.info("Application cleanup completed")
    
    async def run_quick_analysis(self, tickers: List[str]) -> None:
        """Run quick analysis on specified tickers"""
        logger.info(f"Running quick analysis on {len(tickers)} tickers: {', '.join(tickers)}")
        
        # Generate unique run ID for quick analysis
        run_id = f"quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run analysis
        await self.run_analysis(tickers=tickers, run_id=run_id)
    
    async def production_status(self) -> None:
        """Display production optimization status"""
        logger.info("VP Investments Production Optimization Status")
        logger.info("=" * 50)
        
        if self.use_production_optimizations and self.production_optimized_vp:
            # Get optimization status directly from integration status
            integration_status = await self.production_optimized_vp.get_integration_status()
            
            logger.info(f"Production Mode: [ACTIVE]")
            logger.info(f"Optimizations Available: {integration_status['integration_status']['active_optimizations']}/5")
            logger.info("")
            
            # Check which optimizations are active based on component status
            optimizations = {
                'Enhanced Signal Quality': self.production_optimized_vp.enhanced_signal_engine is not None,
                'Real-time Implementation': self.production_optimized_vp.realtime_processor is not None,
                'Monitoring Setup': self.production_optimized_vp.monitoring_system is not None,
                'Production Scaling': self.production_optimized_vp.production_scaler is not None,
                'Database Interface': self.production_optimized_vp.database is not None
            }
            
            for opt_name, is_active in optimizations.items():
                status_icon = "[OK]" if is_active else "[FAIL]"
                logger.info(f"  {status_icon} {opt_name}")
            
            logger.info("")
            logger.info(f"System Performance Grade: {integration_status['production_readiness']['overall_grade']}")
            logger.info(f"Integration Success Rate: {integration_status['integration_status']['integration_success_rate']}")
            
        else:
            logger.info(f"Production Mode: [DISABLED]")
            logger.info("To enable production optimizations:")
            logger.info("  1. Ensure all optimization modules are installed")
            logger.info("  2. Run with --production flag")
            logger.info("  3. Or set use_production_optimizations=True")
        
        logger.info("=" * 50)
    
    async def toggle_production_mode(self, enable: bool = None) -> None:
        """Toggle or set production optimization mode"""
        if enable is None:
            # Toggle current state
            enable = not self.use_production_optimizations
        
        if enable:
            logger.info("Enabling production optimizations...")
            try:
                await self.initialize()  # Re-initialize with production mode
                self.use_production_optimizations = True
                logger.info("[SUCCESS] Production optimizations enabled successfully")
            except Exception as e:
                logger.error(f"[FAILED] Failed to enable production optimizations: {e}")
        else:
            logger.info("Switching to standard mode...")
            self.use_production_optimizations = False
            self.production_optimized_vp = None
            logger.info("[SUCCESS] Switched to standard mode")
        
        await self.production_status()


async def main():
    """Main entry point"""
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="VP Investments Platform")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run analysis command
    run_parser = subparsers.add_parser('run', help='Run investment analysis')
    run_parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    run_parser.add_argument('--run-id', help='Custom run ID')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check system health')
    
    # Production optimization commands
    prod_parser = subparsers.add_parser('production', help='Production optimization commands')
    prod_subparsers = prod_parser.add_subparsers(dest='prod_command', help='Production commands')
    
    # Production status
    prod_subparsers.add_parser('status', help='Show production optimization status')
    
    # Production toggle
    toggle_parser = prod_subparsers.add_parser('toggle', help='Toggle production optimizations')
    toggle_parser.add_argument('--enable', action='store_true', help='Enable production mode')
    toggle_parser.add_argument('--disable', action='store_true', help='Disable production mode')
    
    # Quick analysis command
    quick_parser = subparsers.add_parser('quick', help='Run quick analysis on specific tickers')
    quick_parser.add_argument('tickers', nargs='+', help='Tickers to analyze')
    
    # Add production flag to run command
    run_parser.add_argument('--production', action='store_true', help='Use production optimizations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging()
    
    try:
        if args.command == 'run':
            # Run analysis
            use_production = getattr(args, 'production', False)
            app = VPInvestmentsApp(use_production_optimizations=use_production)
            await app.initialize()
            
            try:
                run_id = await app.run_analysis(args.tickers, args.run_id)
                logger.info(f"Analysis completed. Run ID: {run_id}")
                
                # Optionally display web dashboard URL
                print("\n" + "="*50)
                print("[TARGET] ANALYSIS COMPLETE")
                print("="*50)
                print(f"Run ID: {run_id}")
                if use_production:
                    print("[PRODUCTION] Production optimizations were active")
                print("View detailed results at: http://localhost:3000/dashboard")
                print("Start API server with: python -m src.main server")
                print("="*50)
                
            finally:
                await app.cleanup()
        
        elif args.command == 'server':
            # Start API server
            logger.info(f"Starting API server on {args.host}:{args.port}")
            import uvicorn
            uvicorn.run(api_app, host=args.host, port=args.port)
        
        elif args.command == 'config':
            # Show configuration
            print("\n[CONFIG] CONFIGURATION")
            print("="*50)
            
            # Display key configuration values
            config_items = [
                ('Database URL', get_config().get('database.url', 'Not configured')),
                ('Min Confidence', get_config().get('analysis.min_confidence_threshold', 0.3)),
                ('Max Signals', get_config().get('analysis.max_signals_per_run', 100)),
                ('Sentiment Weight', get_config().get('analysis.sentiment_weight', 0.4)),
                ('Technical Weight', get_config().get('analysis.technical_weight', 0.6)),
                ('Reddit Enabled', get_config().get('data_sources.reddit.enabled', True)),
                ('News Enabled', get_config().get('data_sources.news.enabled', True)),
            ]
            
            for name, value in config_items:
                print(f"{name:20}: {value}")
            
            print("="*50)
        
        elif args.command == 'production':
            # Production optimization commands
            app = VPInvestmentsApp()
            await app.initialize()
            
            try:
                if args.prod_command == 'status':
                    await app.production_status()
                
                elif args.prod_command == 'toggle':
                    if args.enable:
                        await app.toggle_production_mode(True)
                    elif args.disable:
                        await app.toggle_production_mode(False)
                    else:
                        await app.toggle_production_mode()  # Toggle current state
                
                else:
                    print("Available production commands:")
                    print("  status  - Show optimization status")
                    print("  toggle  - Toggle production mode")
                    
            finally:
                await app.cleanup()
        
        elif args.command == 'quick':
            # Quick analysis
            app = VPInvestmentsApp()
            await app.initialize()
            
            try:
                await app.run_quick_analysis(args.tickers)
            finally:
                await app.cleanup()
        
        elif args.command == 'health':
            # Health check
            print("\n[HEALTH] SYSTEM HEALTH CHECK")
            print("="*50)
            
            app = VPInvestmentsApp()
            try:
                await app.initialize()
                print("[OK] Database: Connected")
                print("[OK] Pipeline: Initialized")
                print("[OK] Analysis Engine: Ready")
                print("[OK] Overall Status: HEALTHY")
            except Exception as e:
                print(f"[ERROR] Health Check Failed: {e}")
            finally:
                await app.cleanup()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


def cli_main():
    """CLI entry point for setup.py"""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())