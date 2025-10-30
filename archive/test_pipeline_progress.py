"""
Quick Test of Pipeline with Progress Bars

Tests the new progress display with a single ticker.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.pipeline import run_pipeline
from backend.utils.log_config import configure_pipeline_logging


async def main():
    """Test pipeline with progress bars."""
    
    # Configure clean logging (warnings only to console)
    logger = configure_pipeline_logging(verbose=0, quiet=False)
    
    print("\n" + "=" * 80)
    print("TESTING: Pipeline with Progress Bars (Single Ticker)")
    print("=" * 80 + "\n")
    
    # Run pipeline with single ticker and progress bars enabled
    results = await run_pipeline(
        tickers=['AAPL'],
        show_progress=True,
        verbose_level=0
    )
    
    print("\n" + "=" * 80)
    print(f"✅ Test Complete! {len(results)} ticker(s) processed")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
