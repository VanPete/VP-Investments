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
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.pipeline import run_pipeline
from backend.utils.log_config import configure_pipeline_logging
from backend.utils.progress_display import PipelineProgress

# Will be configured based on CLI args
logger = None


async def main(args):
    """
    Run pipeline with default tickers.
    
    Args:
        args: Parsed command-line arguments
    """
    global logger
    
    # Configure logging based on verbosity
    verbose_level = args.verbose if not args.quiet else -1
    logger = configure_pipeline_logging(verbose=verbose_level, quiet=args.quiet)
    
    # Use a small set of high-quality tickers for testing
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    start_time = datetime.now()
    
    # Show clean header (unless quiet)
    if not args.quiet:
        with PipelineProgress(verbose=(verbose_level > 0)) as progress:
            progress.show_header()
            print(f"Running pipeline for {len(tickers)} tickers...\n")
    
    try:
        # Run pipeline
        # TODO: In next step, we'll modify the pipeline itself to use progress bars
        results = await run_pipeline(tickers=tickers)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Show summary
        if not args.quiet:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            
            console.print()
            console.print(Panel(
                f"[bold green]✅ Pipeline Complete[/]\n\n"
                f"Duration: {duration:.1f}s\n"
                f"Tickers: {len(tickers)}\n"
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
    
    # Run pipeline
    asyncio.run(main(args))
