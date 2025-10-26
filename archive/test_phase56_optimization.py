"""
Test Phase 5.6 Optimization - Phase 1 Parallel Fetching
========================================================

Quick test to validate Phase 1 optimization works correctly.
"""

import asyncio
import time
from datetime import datetime

async def test_phase1_optimization():
    """Test optimized Phase 1 fetcher with small ticker set."""
    print("=" * 80)
    print("PHASE 5.6 OPTIMIZATION TEST - Phase 1 Parallel Fetching")
    print("=" * 80)
    
    # Import both fetchers
    from backend.phases.phase1_fetch import Phase1Fetcher
    from backend.phases.phase1_fetch_optimized import get_optimized_phase1_fetcher
    
    # Test with 10 tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                    'NVDA', 'META', 'NFLX', 'AMD', 'INTC']
    
    print(f"\n📊 Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")
    
    # Test 1: Original fetcher
    print("\n" + "=" * 80)
    print("TEST 1: Original Phase 1 Fetcher (Sequential)")
    print("=" * 80)
    
    fetcher_original = Phase1Fetcher()
    start_time = time.time()
    
    try:
        results_original = await fetcher_original.fetch_all_data(tickers=test_tickers)
        duration_original = time.time() - start_time
        
        raw_cache_original = results_original.get('raw_cache_by_ticker', {})
        success_count_original = len(raw_cache_original)
        
        print(f"\n✅ Original fetcher complete:")
        print(f"   Duration: {duration_original:.2f}s")
        print(f"   Success: {success_count_original}/{len(test_tickers)} tickers")
        print(f"   Per-ticker: {duration_original / len(test_tickers):.2f}s")
    except Exception as e:
        print(f"❌ Original fetcher failed: {e}")
        duration_original = None
        success_count_original = 0
    
    # Test 2: Optimized fetcher
    print("\n" + "=" * 80)
    print("TEST 2: Optimized Phase 1 Fetcher (Parallel, 5 concurrent)")
    print("=" * 80)
    
    fetcher_optimized = get_optimized_phase1_fetcher(max_concurrent=5)
    start_time = time.time()
    
    try:
        results_optimized = await fetcher_optimized.fetch_all_data(tickers=test_tickers)
        duration_optimized = time.time() - start_time
        
        raw_cache_optimized = results_optimized.get('raw_cache_by_ticker', {})
        success_count_optimized = len(raw_cache_optimized)
        
        print(f"\n✅ Optimized fetcher complete:")
        print(f"   Duration: {duration_optimized:.2f}s")
        print(f"   Success: {success_count_optimized}/{len(test_tickers)} tickers")
        print(f"   Per-ticker: {duration_optimized / len(test_tickers):.2f}s")
    except Exception as e:
        print(f"❌ Optimized fetcher failed: {e}")
        duration_optimized = None
        success_count_optimized = 0
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    if duration_original and duration_optimized:
        speedup = (duration_original / duration_optimized - 1) * 100
        time_saved = duration_original - duration_optimized
        
        print(f"Original:  {duration_original:.2f}s ({success_count_original} tickers)")
        print(f"Optimized: {duration_optimized:.2f}s ({success_count_optimized} tickers)")
        print(f"Speedup:   {speedup:.1f}% faster")
        print(f"Time saved: {time_saved:.2f}s")
        
        if speedup > 0:
            print(f"\n✅ SUCCESS: Optimization working! {speedup:.1f}% faster")
        else:
            print(f"\n⚠️  WARNING: Optimization slower by {-speedup:.1f}%")
            print("   (This can happen with small batches due to overhead)")
    else:
        print("❌ Could not compare - one or both tests failed")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_phase1_optimization())
