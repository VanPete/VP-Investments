"""
Test Phase 3 normalization with Phase 2 calculated factors.

Tests:
1. Run Phase 1 + Phase 2 to get calculated factors
2. Normalize with Phase 3 using robust z-scores
3. Verify normalization statistics
4. Display sample normalized factors
"""

import sys
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict

# Add backend to path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

from backend.phases.phase1_fetch import Phase1Fetcher
from backend.phases.phase2_calculate import Phase2Calculator, GroupFactors
from backend.phases.phase3_normalize import Phase3Normalizer, NormalizedGroupFactors


async def test_phase3_normalization(tickers: list[str] = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']):
    """Test Phase 3 normalization with real data"""
    
    print("=" * 80)
    print("PHASE 3 NORMALIZATION TEST")
    print("=" * 80)
    
    # Step 1: Fetch Phase 1 data
    print(f"\n📥 Step 1: Fetching Phase 1 data for {len(tickers)} tickers...")
    fetcher = Phase1Fetcher()
    phase1_results = await fetcher.fetch_all_data(tickers=tickers)
    
    raw_cache = phase1_results['raw_cache_by_ticker']
    reddit_cache = phase1_results.get('reddit_cache_by_ticker', {})
    news_cache = phase1_results.get('news_cache_by_ticker', {})
    
    print(f"   ✅ Fetched data for {len(raw_cache)} tickers")
    
    # Step 2: Calculate factors with Phase 2
    print(f"\n🧮 Step 2: Calculating factors with Phase2Calculator...")
    calculator = Phase2Calculator()
    
    calculated_by_ticker: Dict[str, GroupFactors] = {}
    for ticker in tickers:
        raw_data = raw_cache.get(ticker)
        if not raw_data:
            print(f"   ⚠️  {ticker}: No raw data available")
            continue
        
        reddit_data = reddit_cache.get(ticker)
        news_data = news_cache.get(ticker)
        
        result = calculator.calculate_all_factors(
            ticker=ticker,
            raw_data=raw_data,
            reddit_data=reddit_data,
            news_data=news_data
        )
        
        calculated_by_ticker[ticker] = result
    
    print(f"   ✅ Calculated factors for {len(calculated_by_ticker)} tickers")
    
    # Step 3: Normalize with Phase 3
    print(f"\n📊 Step 3: Normalizing factors with Phase3Normalizer...")
    normalizer = Phase3Normalizer()
    normalized_by_ticker = normalizer.normalize_all_factors(calculated_by_ticker)
    
    print(f"   ✅ Normalized {len(normalized_by_ticker)} tickers")
    
    # Step 4: Analyze normalization statistics
    print(f"\n" + "=" * 80)
    print("NORMALIZATION STATISTICS")
    print("=" * 80)
    
    # Collect all normalized values across all tickers
    all_normalized_values = []
    for ticker, normalized in normalized_by_ticker.items():
        all_factors = normalized.get_all_factors()
        for factor_name, value in all_factors.items():
            if not np.isnan(value):
                all_normalized_values.append(value)
    
    if all_normalized_values:
        mean_z = np.mean(all_normalized_values)
        std_z = np.std(all_normalized_values)
        median_z = np.median(all_normalized_values)
        min_z = np.min(all_normalized_values)
        max_z = np.max(all_normalized_values)
        
        print(f"\n📈 Cross-Sectional Z-Score Statistics:")
        print(f"   Mean:   {mean_z:>8.4f} (should be close to 0)")
        print(f"   Std:    {std_z:>8.4f} (should be close to 1)")
        print(f"   Median: {median_z:>8.4f}")
        print(f"   Min:    {min_z:>8.4f}")
        print(f"   Max:    {max_z:>8.4f}")
        print(f"   Total non-NaN values: {len(all_normalized_values)}")
    
    # Step 5: Display sample factors (before vs after)
    print(f"\n" + "=" * 80)
    print("SAMPLE FACTORS (AAPL) - BEFORE vs AFTER NORMALIZATION")
    print("=" * 80)
    
    if 'AAPL' in calculated_by_ticker and 'AAPL' in normalized_by_ticker:
        calc = calculated_by_ticker['AAPL']
        norm = normalized_by_ticker['AAPL']
        
        # Technical samples
        print("\n🔹 Technical Factors:")
        tech_samples = ['price_1d_pct', 'rsi_14', 'macd_hist', 'volume_spike_ratio']
        print(f"   {'Factor':<25s} {'Raw Value':>15s} {'Z-Score':>12s}")
        print(f"   {'-'*25} {'-'*15} {'-'*12}")
        for factor in tech_samples:
            raw_val = calc.technical.get(factor, np.nan)
            z_val = norm.technical.get(factor, np.nan)
            print(f"   {factor:<25s} {raw_val:>15.4f} {z_val:>12.4f}")
        
        # Fundamental samples
        print("\n🔹 Fundamental Factors:")
        fund_samples = ['pe_ratio', 'roe', 'revenue_growth_yoy', 'debt_to_equity']
        print(f"   {'Factor':<25s} {'Raw Value':>15s} {'Z-Score':>12s}")
        print(f"   {'-'*25} {'-'*15} {'-'*12}")
        for factor in fund_samples:
            raw_val = calc.fundamental.get(factor, np.nan)
            z_val = norm.fundamental.get(factor, np.nan)
            print(f"   {factor:<25s} {raw_val:>15.4f} {z_val:>12.4f}")
        
        # Risk samples
        print("\n🔹 Risk/Stability Factors:")
        risk_samples = ['volatility_60d', 'beta_60d', 'sharpe_ratio_60d']
        print(f"   {'Factor':<25s} {'Raw Value':>15s} {'Z-Score':>12s}")
        print(f"   {'-'*25} {'-'*15} {'-'*12}")
        for factor in risk_samples:
            raw_val = calc.risk_stability.get(factor, np.nan)
            z_val = norm.risk_stability.get(factor, np.nan)
            print(f"   {factor:<25s} {raw_val:>15.4f} {z_val:>12.4f}")
    
    # Step 6: Display per-ticker normalized coverage
    print(f"\n" + "=" * 80)
    print("NORMALIZED COVERAGE BY TICKER")
    print("=" * 80)
    
    for ticker in sorted(normalized_by_ticker.keys()):
        norm = normalized_by_ticker[ticker]
        all_factors = norm.get_all_factors()
        
        total = len(all_factors)
        non_nan = sum(1 for v in all_factors.values() if not np.isnan(v))
        coverage_pct = (non_nan / total * 100) if total > 0 else 0
        
        print(f"{ticker:>6s}: {non_nan:>3d}/{total:<3d} factors ({coverage_pct:>5.1f}%)")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully normalized {len(normalized_by_ticker)} tickers")
    print(f"📊 Normalization method: {normalizer.method}")
    print(f"📊 Winsorization: {normalizer.winsorize_pct*100}%")
    print(f"📊 Z-score mean: {mean_z:.4f} (target: 0.00)")
    print(f"📊 Z-score std: {std_z:.4f} (target: 1.00)")
    print(f"\n🎉 Phase 3 normalization test COMPLETE!")
    
    return normalized_by_ticker


if __name__ == '__main__':
    results = asyncio.run(test_phase3_normalization())
