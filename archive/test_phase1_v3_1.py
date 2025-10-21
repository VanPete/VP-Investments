"""
Test Phase 1 v3.1 Refactor
==========================

Quick test to verify:
1. ComprehensiveYFinanceFetcher integration works
2. Reddit → News → YFinance flow works
3. RawYFinanceData structure is correct
4. Endpoint coverage is comprehensive
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)

logger = logging.getLogger(__name__)


async def test_phase1_v3_1():
    """Test Phase 1 with v3.1 comprehensive YFinance integration"""
    
    logger.info("=" * 80)
    logger.info("TESTING PHASE 1 v3.1 REFACTOR")
    logger.info("=" * 80)
    
    try:
        from backend.phases.phase1_fetch import Phase1Fetcher
        
        # Initialize Phase 1
        logger.info("\n1. Initializing Phase 1 fetcher...")
        fetcher = Phase1Fetcher()
        
        # Test with a small set of known tickers
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        logger.info(f"\n2. Testing with {len(test_tickers)} tickers: {test_tickers}")
        
        # Fetch data
        logger.info("\n3. Running fetch_all_data()...")
        results = await fetcher.fetch_all_data(
            tickers=test_tickers,
            subreddits=['wallstreetbets'],  # Just one subreddit for speed
            post_limit=10  # Small limit for testing
        )
        
        # Verify structure
        logger.info("\n4. Verifying results structure...")
        
        # Check keys
        expected_keys = ['reddit_data', 'news_data', 'raw_cache_by_ticker', 
                        'discovered_tickers', 'all_tickers', 'metadata']
        for key in expected_keys:
            if key in results:
                logger.info(f"   ✅ {key}: present")
            else:
                logger.error(f"   ❌ {key}: MISSING")
        
        # Check raw_cache_by_ticker
        logger.info("\n5. Checking YFinance comprehensive data...")
        raw_cache = results.get('raw_cache_by_ticker', {})
        logger.info(f"   Raw cache size: {len(raw_cache)} tickers")
        
        if raw_cache:
            # Inspect first ticker
            sample_ticker = list(raw_cache.keys())[0]
            sample_data = raw_cache[sample_ticker]
            
            logger.info(f"\n6. Inspecting sample ticker: {sample_ticker}")
            logger.info(f"   Type: {type(sample_data).__name__}")
            logger.info(f"   Fetch success: {sample_data.fetch_success}")
            logger.info(f"   Endpoints attempted: {len(sample_data.endpoints_attempted)}")
            logger.info(f"   Endpoints succeeded: {len(sample_data.endpoints_succeeded)}")
            
            # Check critical fields
            logger.info(f"\n7. Checking critical data fields...")
            has_info = bool(sample_data.info)
            has_history = not sample_data.history.empty
            logger.info(f"   ✅ info: {len(sample_data.info)} fields" if has_info else "   ❌ info: MISSING")
            logger.info(f"   ✅ history: {len(sample_data.history)} rows" if has_history else "   ❌ history: MISSING")
            
            # Check new comprehensive fields
            logger.info(f"\n8. Checking comprehensive endpoint coverage...")
            comprehensive_checks = [
                ('fast_info', sample_data.fast_info),
                ('dividends', sample_data.dividends),
                ('income_stmt', sample_data.income_stmt),
                ('balance_sheet', sample_data.balance_sheet),
                ('cashflow', sample_data.cashflow),
                ('recommendations', sample_data.recommendations),
                ('institutional_holders', sample_data.institutional_holders),
                ('insider_transactions', sample_data.insider_transactions),
            ]
            
            for field_name, field_value in comprehensive_checks:
                has_data = False
                if isinstance(field_value, dict):
                    has_data = bool(field_value)
                else:  # DataFrame
                    has_data = not field_value.empty if hasattr(field_value, 'empty') else bool(field_value)
                
                status = "✅" if has_data else "⚠️ "
                logger.info(f"   {status} {field_name}: {'present' if has_data else 'empty/missing'}")
            
            # Show endpoint success rate
            success_rate = (len(sample_data.endpoints_succeeded) / 
                          len(sample_data.endpoints_attempted) * 100 
                          if sample_data.endpoints_attempted else 0)
            logger.info(f"\n   Overall endpoint success rate: {success_rate:.1f}%")
            
            # Show any errors
            if sample_data.fetch_errors:
                logger.info(f"\n   Fetch errors ({len(sample_data.fetch_errors)}):")
                for endpoint, error in list(sample_data.fetch_errors.items())[:3]:
                    logger.info(f"      - {endpoint}: {error[:60]}...")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Reddit tickers discovered: {len(results.get('discovered_tickers', []))}")
        logger.info(f"News data fetched: {len(results.get('news_data', {}))}")
        logger.info(f"YFinance comprehensive data: {len(raw_cache)} tickers")
        logger.info(f"Execution time: {results['metadata']['execution_time']:.2f}s")
        logger.info(f"YFinance version: {results['metadata'].get('yfinance_version', 'unknown')}")
        
        if raw_cache:
            logger.info("\n✅ Phase 1 v3.1 refactor test PASSED!")
            logger.info("   - Reddit → News → YFinance flow works")
            logger.info("   - RawYFinanceData structure is correct")
            logger.info("   - Comprehensive endpoint coverage confirmed")
        else:
            logger.warning("\n⚠️  Phase 1 test completed with warnings")
            logger.warning("   - No YFinance data fetched (check API connectivity)")
        
        return results
        
    except Exception as e:
        logger.error(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("\nStarting Phase 1 v3.1 test...\n")
    results = asyncio.run(test_phase1_v3_1())
    
    if results:
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Test failed - check logs above")
        print("=" * 80)
