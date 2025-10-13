"""
Run the full production pipeline with all 7 subreddits.
This will take 10-15 minutes to complete.
"""
import asyncio
from backend.pipeline import UnifiedPipeline

async def run_production_pipeline():
    """Run full pipeline with all subreddits."""
    print("=" * 80)
    print("RUNNING FULL PRODUCTION PIPELINE")
    print("=" * 80)
    print("\n📊 Configuration:")
    print("   - Subreddits: 6 (wallstreetbets, stocks, investing, SecurityAnalysis,")
    print("                    ValueInvesting, StockMarket)")
    print("   - Post limit: 100 per subreddit")
    print("   - Max signals: 50")
    print("   - Mode: PRODUCTION (all features enabled)")
    print("\n⏱️  Estimated time: 10-15 minutes")
    print("\n" + "=" * 80)
    print()
    
    pipeline = UnifiedPipeline()
    
    # Run with production settings (test_mode=False means all 7 subreddits)
    result = await pipeline.run_pipeline(
        test_mode=False,      # Use all 7 subreddits
        post_limit=100,       # 100 posts per subreddit
        max_signals=50,       # Up to 50 signals
        min_mentions=2        # At least 2 mentions
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Signals generated: {result.get('signals_created', 0)}")
    print(f"📈 Tickers analyzed: {result.get('total_tickers', 0)}")
    print(f"⏱️  Duration: {result.get('duration', 'N/A')}")
    print("\n✅ Ready to analyze fundamental field coverage!\n")
    
    return result

if __name__ == "__main__":
    asyncio.run(run_production_pipeline())
