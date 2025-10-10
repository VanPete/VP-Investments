"""
Test the new generate_single_signal() feature
This demonstrates the user-requested signal generation capability.
"""
import asyncio
from backend.pipeline import UnifiedPipeline
from backend.core.config import setup_logging

setup_logging()

async def test_single_signal_generation():
    """Test generating a single signal on-demand"""
    pipeline = UnifiedPipeline()
    
    # Test Case 1: With Reddit data
    print("\n" + "="*80)
    print("TEST 1: Generate signal with Reddit data")
    print("="*80)
    
    signal = await pipeline.generate_single_signal('AAPL', include_reddit=True)
    
    if signal:
        print(f"\n[SUCCESS] Signal generated successfully!")
        print(f"   Ticker: {signal.get('ticker')}")
        print(f"   Weighted Score: {signal.get('weighted_score')}")
        print(f"   Financial Score: {signal.get('financial_score')}")
        print(f"   Reddit Score: {signal.get('reddit_score')}")
        print(f"   Beta: {signal.get('beta')}")
        print(f"   MACD Line: {signal.get('macd_line')}")
        print(f"   Bollinger Upper: {signal.get('bollinger_upper')}")
        print(f"   Upvotes: {signal.get('upvotes')}")
        print(f"   Mention Count: {signal.get('mention_count')}")
    else:
        print(f"\n[FAILED] Failed to generate signal")
    
    # Test Case 2: Without Reddit data (faster)
    print("\n" + "="*80)
    print("TEST 2: Generate signal without Reddit data (faster)")
    print("="*80)
    
    signal2 = await pipeline.generate_single_signal('TSLA', include_reddit=False)
    
    if signal2:
        print(f"\n[SUCCESS] Signal generated successfully!")
        print(f"   Ticker: {signal2.get('ticker')}")
        print(f"   Weighted Score: {signal2.get('weighted_score')}")
        print(f"   Financial Score: {signal2.get('financial_score')}")
        print(f"   Beta: {signal2.get('beta')}")
        print(f"   MACD Line: {signal2.get('macd_line')}")
        print(f"   Upvotes: {signal2.get('upvotes')} (should be 0)")
    else:
        print(f"\n[FAILED] Failed to generate signal")
    
    print("\n" + "="*80)
    print("FRONTEND INTEGRATION READY")
    print("="*80)
    print("\nUsage from frontend:")
    print("  POST /api/signals/generate")
    print("  Body: { ticker: 'AAPL', include_reddit: true }")
    print("\nResponse:")
    print("  { ticker, weighted_score, financial_score, reddit_score, beta, macd_line, upvotes, ... }")

if __name__ == "__main__":
    asyncio.run(test_single_signal_generation())
