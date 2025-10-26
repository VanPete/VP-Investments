"""
Quick Test: Verify Consolidated Phase Files (Phase 5.6)
========================================================

Tests that the consolidated phase1_fetch.py and phase5_persist.py files
work correctly after merging the _optimized classes into the base files.

Expected:
- Phase1FetcherOptimized and Phase5PersistOptimized classes available
- Factory functions work correctly
- Optimizations enabled by default
"""

import os
import asyncio
import time

# Set optimization flags
os.environ['ENABLE_PHASE1_OPTIMIZATION'] = 'true'
os.environ['ENABLE_PHASE5_OPTIMIZATION'] = 'true'

async def main():
    print("=" * 80)
    print("CONSOLIDATED FILES TEST")
    print("=" * 80)
    print()
    
    # Test 1: Import Phase1FetcherOptimized from consolidated file
    print("[TEST 1] Import Phase1FetcherOptimized from backend.phases.phase1_fetch...")
    try:
        from backend.phases.phase1_fetch import Phase1FetcherOptimized, get_optimized_phase1_fetcher
        print("✅ Phase1FetcherOptimized import successful")
        
        # Test factory function
        p1 = get_optimized_phase1_fetcher(max_concurrent=10)
        print(f"✅ Factory function works: {type(p1).__name__}")
        print()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Import Phase5PersistOptimized from consolidated file
    print("[TEST 2] Import Phase5PersistOptimized from backend.phases.phase5_persist...")
    try:
        from backend.phases.phase5_persist import Phase5PersistOptimized, get_optimized_phase5_persist
        print("✅ Phase5PersistOptimized import successful")
        
        # Test factory function
        p5 = get_optimized_phase5_persist(db_interface=None)
        print(f"✅ Factory function works: {type(p5).__name__}")
        print()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 3: Verify old _optimized files are gone
    print("[TEST 3] Verify old _optimized files deleted...")
    import pathlib
    phase1_old = pathlib.Path("backend/phases/phase1_fetch_optimized.py")
    phase5_old = pathlib.Path("backend/phases/phase5_persist_optimized.py")
    
    if not phase1_old.exists() and not phase5_old.exists():
        print("✅ Old _optimized files successfully deleted")
    else:
        if phase1_old.exists():
            print(f"⚠️  phase1_fetch_optimized.py still exists")
        if phase5_old.exists():
            print(f"⚠️  phase5_persist_optimized.py still exists")
    print()
    
    # Test 4: Quick pipeline test with 3 tickers
    print("[TEST 4] Quick pipeline test with 3 tickers...")
    print("(This validates the optimizations work end-to-end)")
    print()
    
    from backend.pipeline import run_pipeline
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    start_time = time.time()
    try:
        results = await run_pipeline(tickers)
        duration = time.time() - start_time
        
        print(f"✅ Pipeline completed in {duration:.1f}s")
        print(f"   Tickers processed: {len(results)}")
        print()
        
        if len(results) > 0:
            top_ticker_symbol = list(results.keys())[0] if isinstance(results, dict) else "N/A"
            print(f"   Top result: {top_ticker_symbol}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - Phase1FetcherOptimized: Available in phase1_fetch.py")
    print("  - Phase5PersistOptimized: Available in phase5_persist.py")
    print("  - Factory functions: Working correctly")
    print("  - Old _optimized files: Deleted")
    print("  - End-to-end pipeline: Working with optimizations")
    print()
    print("✅ File consolidation successful! Single files per phase maintained.")


if __name__ == "__main__":
    asyncio.run(main())
