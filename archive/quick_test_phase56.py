"""
Quick Test: Phase 5.6 Yfinance Optimization Only
=================================================

Tests only the yfinance fetching optimization without Reddit/News.
"""

import asyncio
import time
from backend.phases.phase1_fetch import Phase1Fetcher
from backend.phases.phase1_fetch_optimized import get_optimized_phase1_fetcher

async def quick_test():
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("\n" + "=" * 60)
    print("QUICK TEST: YFinance Fetch Optimization")
    print("=" * 60)
    print(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}\n")
    
    # Test 1: Original
    print("Test 1: Original Fetcher")
    p1 = Phase1Fetcher()
    start = time.time()
    try:
        result1 = await p1._fetch_comprehensive_yfinance_data(test_tickers)
        time1 = time.time() - start
        print(f"✅ Original: {time1:.1f}s - {len(result1)} tickers fetched\n")
    except Exception as e:
        print(f"❌ Original failed: {e}\n")
        time1 = None
    
    # Test 2: Optimized
    print("Test 2: Optimized Fetcher (3 concurrent)")
    p1_opt = get_optimized_phase1_fetcher(max_concurrent=3)
    start = time.time()
    try:
        result2 = await p1_opt._fetch_comprehensive_yfinance_data(test_tickers)
        time2 = time.time() - start
        print(f"✅ Optimized: {time2:.1f}s - {len(result2)} tickers fetched\n")
    except Exception as e:
        print(f"❌ Optimized failed: {e}\n")
        time2 = None
    
    # Compare
    if time1 and time2:
        speedup = (time1 / time2 - 1) * 100
        print("=" * 60)
        print(f"RESULT: {speedup:+.1f}% {'faster' if speedup > 0 else 'slower'}")
        print(f"Saved: {time1 - time2:.1f}s")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(quick_test())
