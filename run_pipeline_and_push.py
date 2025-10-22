"""
Run Pipeline and Push to GitHub

Executes the VP Investments pipeline and pushes results to GitHub.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.pipeline import run_pipeline
from backend.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Run pipeline with default tickers."""
    
    # Use a small set of high-quality tickers for testing
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    logger.info("=" * 80)
    logger.info(f"Running pipeline for {len(tickers)} tickers...")
    logger.info("=" * 80)
    
    try:
        # Run pipeline
        results = await run_pipeline(tickers=tickers)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"✅ Results saved to frontend/public/results/latest.json")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
