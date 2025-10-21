"""
Quick test: Check if the 3 risk factors now work with 2y data
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath('.'))

from backend.phases.phase1_fetch import Phase1Fetcher
from backend.phases.phase2_calculate import Phase2Calculator

async def test_risk_factors():
    # Test with just AAPL
    tickers = ['AAPL']
    
    # Phase 1: Fetch data
    print("="*80)
    print("PHASE 1: Fetching data with 2y history...")
    print("="*80)
    phase1 = Phase1Fetcher()
    result = await phase1.fetch_all_data(tickers)
    
    raw_cache = result.get('raw_cache_by_ticker', {})
    reddit_data = result.get('reddit_data', {})
    news_data = result.get('news_data', {})
    market_data = result.get('market_data')
    
    # Check what we got
    if 'AAPL' in raw_cache:
        aapl_data = raw_cache['AAPL']
        if aapl_data.history is not None:
            print(f"\nAAPL history length: {len(aapl_data.history)} days")
            print(f"AAPL history date range: {aapl_data.history.index[0]} to {aapl_data.history.index[-1]}")
    
    if market_data and market_data.spy_history is not None:
        print(f"\nSPY history length: {len(market_data.spy_history)} days")
        print(f"SPY history date range: {market_data.spy_history.index[0]} to {market_data.spy_history.index[-1]}")
    
    # Phase 2: Calculate factors
    print("\n" + "="*80)
    print("PHASE 2: Calculating factors...")
    print("="*80)
    phase2 = Phase2Calculator()
    results = phase2.calculate_batch(raw_cache, reddit_data, news_data, market_data)
    
    # Check the three problem factors
    if 'AAPL' in results:
        risk_factors = results['AAPL'].risk_stability
        print(f"\nRISK FACTORS FOR AAPL:")
        print(f"  volatility_percentile: {risk_factors.get('volatility_percentile', 'MISSING')}")
        print(f"  calmar_ratio: {risk_factors.get('calmar_ratio', 'MISSING')}")
        print(f"  downside_capture_1y: {risk_factors.get('downside_capture_1y', 'MISSING')}")
        
        # Check if they're NaN
        import math
        for factor_name in ['volatility_percentile', 'calmar_ratio', 'downside_capture_1y']:
            value = risk_factors.get(factor_name)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                print(f"  ✓ {factor_name} is VALID!")
            else:
                print(f"  ✗ {factor_name} is NaN or missing")

if __name__ == "__main__":
    asyncio.run(test_risk_factors())
