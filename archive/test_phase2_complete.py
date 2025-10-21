"""
Test Phase 2 calculation engine with Phase 1 data.

Tests:
1. Load RawYFinanceData from Phase 1 results
2. Calculate all 141 factors using Phase2Calculator
3. Verify coverage statistics
4. Display sample factors by group
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict

# Add backend to path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

from backend.phases.phase1_fetch import Phase1Fetcher
from backend.phases.phase2_calculate import Phase2Calculator, GroupFactors
from backend.integrations.yfinance import RawYFinanceData


async def test_phase2_with_real_data(tickers: list[str] = ['AAPL', 'MSFT', 'GOOGL']):
    """Test Phase 2 calculations with real Phase 1 data"""
    
    print("=" * 80)
    print("PHASE 2 CALCULATION TEST")
    print("=" * 80)
    
    # Step 1: Fetch Phase 1 data
    print(f"\n📥 Step 1: Fetching Phase 1 data for {tickers}...")
    fetcher = Phase1Fetcher()
    phase1_results = await fetcher.fetch_all_data(tickers=tickers)
    
    raw_cache = phase1_results['raw_cache_by_ticker']
    reddit_cache = phase1_results.get('reddit_cache_by_ticker', {})
    news_cache = phase1_results.get('news_cache_by_ticker', {})
    
    print(f"   ✅ Fetched data for {len(raw_cache)} tickers")
    
    # Step 2: Initialize Phase 2 calculator
    print(f"\n🧮 Step 2: Initializing Phase2Calculator...")
    calculator = Phase2Calculator()
    print(f"   ✅ Calculator ready")
    
    # Step 3: Calculate factors for each ticker
    print(f"\n📊 Step 3: Calculating factors...")
    results: Dict[str, GroupFactors] = {}
    
    for ticker in tickers:
        raw_data = raw_cache.get(ticker)
        if not raw_data:
            print(f"   ⚠️  {ticker}: No raw data available")
            continue
        
        reddit_data = reddit_cache.get(ticker)
        news_data = news_cache.get(ticker)
        
        # Calculate all factors
        result = calculator.calculate_all_factors(
            ticker=ticker,
            raw_data=raw_data,
            reddit_data=reddit_data,
            news_data=news_data
        )
        
        results[ticker] = result
        
        # Display coverage
        cov_stats = result.get_coverage_stats()
        print(f"\n   {ticker} Coverage:")
        print(f"      Technical:       {cov_stats['technical_coverage']:>6.1f}% ({cov_stats['technical_populated']}/{cov_stats['technical_total']})")
        print(f"      Fundamental:     {cov_stats['fundamental_coverage']:>6.1f}% ({cov_stats['fundamental_populated']}/{cov_stats['fundamental_total']})")
        print(f"      News/Macro:      {cov_stats['news_macro_coverage']:>6.1f}% ({cov_stats['news_macro_populated']}/{cov_stats['news_macro_total']})")
        print(f"      Social:          {cov_stats['social_alternative_coverage']:>6.1f}% ({cov_stats['social_alternative_populated']}/{cov_stats['social_alternative_total']})")
        print(f"      Risk/Stability:  {cov_stats['risk_stability_coverage']:>6.1f}% ({cov_stats['risk_stability_populated']}/{cov_stats['risk_stability_total']})")
        print(f"      Institutional:   {cov_stats['institutional_smart_money_coverage']:>6.1f}% ({cov_stats['institutional_smart_money_populated']}/{cov_stats['institutional_smart_money_total']})")
        print(f"      ───────────────")
        print(f"      OVERALL:         {cov_stats['overall_coverage']:>6.1f}% ({cov_stats['overall_populated']}/{cov_stats['overall_total']})")
    
    # Step 4: Display sample factors
    print(f"\n" + "=" * 80)
    print("SAMPLE FACTORS (AAPL)")
    print("=" * 80)
    
    if 'AAPL' in results:
        aapl = results['AAPL']
        
        print("\n🔹 Technical (5 samples):")
        tech_samples = ['price_1d_pct', 'rsi_14', 'macd_hist', 'sma_50_vs_price', 'volume_spike_ratio']
        for factor in tech_samples:
            val = aapl.technical.get(factor, 'N/A')
            print(f"   {factor:25s}: {val}")
        
        print("\n🔹 Fundamental (5 samples):")
        fund_samples = ['pe_ratio', 'roe', 'revenue_growth_yoy', 'debt_to_equity', 'fcf_per_share']
        for factor in fund_samples:
            val = aapl.fundamental.get(factor, 'N/A')
            print(f"   {factor:25s}: {val}")
        
        print("\n🔹 News/Macro (3 samples):")
        news_samples = ['news_sentiment_7d', 'days_to_earnings', 'pre_earnings_flag']
        for factor in news_samples:
            val = aapl.news_macro.get(factor, 'N/A')
            print(f"   {factor:25s}: {val}")
        
        print("\n🔹 Social (3 samples):")
        social_samples = ['reddit_mentions_7d', 'reddit_sentiment_7d', 'buzz_vs_baseline']
        for factor in social_samples:
            val = aapl.social_alternative.get(factor, 'N/A')
            print(f"   {factor:25s}: {val}")
        
        print("\n🔹 Risk/Stability (5 samples):")
        risk_samples = ['volatility_60d', 'beta_60d', 'max_drawdown_1y', 'sharpe_ratio_60d', 'drawdown_current']
        for factor in risk_samples:
            val = aapl.risk_stability.get(factor, 'N/A')
            print(f"   {factor:25s}: {val}")
        
        print("\n🔹 Institutional (5 samples):")
        inst_samples = ['inst_ownership_pct', 'insider_buy_score', 'analyst_rating_avg', 'price_target_upside_pct', 'smart_money_composite']
        for factor in inst_samples:
            val = aapl.institutional_smart_money.get(factor, 'N/A')
            print(f"   {factor:25s}: {val}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully calculated factors for {len(results)} tickers")
    print(f"📊 Average coverage: {sum(r.get_coverage_stats()['overall_coverage'] for r in results.values()) / len(results):.1f}%")
    print(f"\n🎉 Phase 2 calculation engine test COMPLETE!")
    
    return results


if __name__ == '__main__':
    results = asyncio.run(test_phase2_with_real_data())
