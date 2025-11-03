"""
Run Pipeline and Push to GitHub

Executes the VP Investments pipeline and pushes results to GitHub.

Usage:
    python run_pipeline_and_push.py              # Default: clean progress bars
    python run_pipeline_and_push.py --quiet      # Only errors and final result
    python run_pipeline_and_push.py -v           # Verbose: show INFO logs
    python run_pipeline_and_push.py -vv          # Very verbose: show DEBUG logs
"""

import asyncio
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Suppress ALL warnings globally BEFORE importing any other modules
# This is necessary because yfinance issues DeprecationWarnings at runtime
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter('ignore')  # Catch-all for any remaining warnings

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.pipeline import run_pipeline
from backend.utils.log_config import configure_pipeline_logging
from backend.utils.progress_display import PipelineProgress, show_error_summary
from backend.utils.error_buffer import ErrorBuffer

# Will be configured based on CLI args
logger = None


async def main(args):
    """
    Run pipeline with default tickers.
    
    Args:
        args: Parsed command-line arguments
    """
    global logger
    
    # Create error buffer for consolidating errors
    error_buffer = ErrorBuffer()
    
    # Configure logging based on verbosity
    # Pass error_buffer to capture ERROR logs for end-of-run summary
    verbose_level = args.verbose if not args.quiet else -1
    logger = configure_pipeline_logging(
        verbose=verbose_level, 
        quiet=args.quiet,
        error_buffer=error_buffer if not args.quiet else None  # Only capture errors in normal mode
    )
    
    # Use a small set of high-quality tickers for testing
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    start_time = datetime.now()
    
    # Header will be shown by PipelineProgress when pipeline starts
    
    try:
        # Run pipeline with progress display
        results = await run_pipeline(
            tickers=tickers,
            show_progress=not args.quiet,
            verbose_level=verbose_level
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Show error summary if any errors were captured
        if not args.quiet and error_buffer.has_errors():
            show_error_summary(error_buffer)
        
        # Show summary
        if not args.quiet:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            
            # Get actual ticker count from results
            actual_tickers = len(results) if results else len(tickers)
            
            console.print()
            console.print(Panel(
                f"[bold green]✅ Pipeline Complete[/]\n\n"
                f"Duration: {duration:.1f}s\n"
                f"Tickers Processed: {actual_tickers}\n"
                f"Signals Generated: {len(results) if results else 0}\n"
                f"Results: frontend/public/results/latest.json",
                style="bold green",
                title="Success"
            ))
        else:
            # Quiet mode: just print success
            print(f"✅ Pipeline complete ({duration:.1f}s)")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=(verbose_level > 0))
        
        if not args.quiet:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            
            console.print()
            console.print(Panel(
                f"[bold red]❌ Pipeline Failed[/]\n\n{str(e)}",
                style="bold red",
                title="Error"
            ))
        
        raise
    
    finally:
        # Clean up database connections
        try:
            from backend.storage.database import get_database
            db = get_database()
            if hasattr(db, 'close'):
                db.close()
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run VP Investments Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline_and_push.py              # Clean progress bars (recommended)
  python run_pipeline_and_push.py --quiet      # Only show errors and result
  python run_pipeline_and_push.py -v           # Show INFO logs
  python run_pipeline_and_push.py -vv          # Show DEBUG logs (very verbose)
        """
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors and final result'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (-v for INFO, -vv for DEBUG)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline with proper cleanup
    try:
        # Use asyncio.run() which handles cleanup automatically in Python 3.10+
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
