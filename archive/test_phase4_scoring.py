"""
Test Phase 4: Score & Assemble
================================

Tests the Phase 4 weighted scoring system with normalized data from Phase 3.

Test Plan:
1. Run Phase 1 (fetch) for 5 tickers: AAPL, MSFT, GOOGL, TSLA, NVDA
2. Run Phase 2 (calculate) to compute 145 factors
3. Run Phase 3 (normalize) to get z-scores
4. Run Phase 4 (score & assemble) to compute weighted scores
5. Display group scores and overall scores
6. Verify scoring logic and weights

Expected Results:
- Each ticker has 6 group scores (technical, fundamental, etc.)
- Overall score = weighted sum of group scores
- Group weights sum to 1.0
- Factor weights sum to 1.0 per group
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.phases.phase1_fetch import Phase1Fetcher
from backend.phases.phase2_calculate import Phase2Calculator
from backend.phases.phase3_normalize import Phase3Normalizer
from backend.phases.phase4_score_assemble import Phase4ScoreAssembler


async def test_phase4_scoring():
    """Test Phase 4 scoring with 5 tickers."""
    
    print("=" * 80)
    print("PHASE 4 SCORING TEST")
    print("=" * 80)
    
    # Test tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    # Step 1: Fetch Phase 1 data
    print("\n📥 Step 1: Fetching Phase 1 data for 5 tickers...")
    fetcher = Phase1Fetcher()
    phase1_results = await fetcher.fetch_all_data(tickers=tickers)
    
    if not phase1_results:
        print("❌ No Phase 1 data fetched")
        return
    
    print(f"   ✅ Fetched data for {len(phase1_results)} tickers")
    
    # Step 2: Calculate Phase 2 factors
    print("\n🧮 Step 2: Calculating factors with Phase2Calculator...")
    calculator = Phase2Calculator()
    
    calculated_by_ticker = {}
    for ticker, yfinance_data in phase1_results.items():
        if yfinance_data:
            try:
                group_factors = calculator.calculate_all_factors(ticker, yfinance_data)
                calculated_by_ticker[ticker] = group_factors
            except Exception as e:
                print(f"   ⚠️ Failed to calculate {ticker}: {e}")
    
    print(f"   ✅ Calculated factors for {len(calculated_by_ticker)} tickers")
    
    # Step 3: Normalize Phase 3 factors
    print("\n📊 Step 3: Normalizing factors with Phase3Normalizer...")
    normalizer = Phase3Normalizer()
    normalized_by_ticker = normalizer.normalize_all_factors(calculated_by_ticker)
    print(f"   ✅ Normalized {len(normalized_by_ticker)} tickers")
    
    # Step 4: Score with Phase 4
    print("\n🎯 Step 4: Scoring tickers with Phase4ScoreAssembler...")
    scorer = Phase4ScoreAssembler()
    scored_by_ticker = scorer.score_all_tickers(normalized_by_ticker)
    print(f"   ✅ Scored {len(scored_by_ticker)} tickers")
    
    # Display results
    print("\n" + "=" * 80)
    print("SCORING RESULTS")
    print("=" * 80)
    
    # Sort by overall score
    sorted_tickers = sorted(scored_by_ticker.items(), 
                           key=lambda x: x[1].overall_score, 
                           reverse=True)
    
    print("\n📊 Overall Scores (sorted):")
    print(f"{'Rank':<6} {'Ticker':<8} {'Overall Score':<15} {'Coverage':<10}")
    print("-" * 80)
    
    for rank, (ticker, result) in enumerate(sorted_tickers, 1):
        print(f"{rank:<6} {ticker:<8} {result.overall_score:>13.4f} {result.total_coverage:>9.1%}")
    
    # Detailed breakdown for top ticker
    if sorted_tickers:
        print("\n" + "=" * 80)
        print(f"DETAILED BREAKDOWN: {sorted_tickers[0][0]} (Top Score)")
        print("=" * 80)
        
        top_ticker, top_result = sorted_tickers[0]
        
        print(f"\n📈 Group Scores:")
        print(f"{'Group':<30} {'Score':<12} {'Coverage':<10} {'Weight':<10}")
        print("-" * 80)
        
        groups = [
            ('Technical', top_result.technical, scorer.group_weights['technical']),
            ('Fundamental', top_result.fundamental, scorer.group_weights['fundamental']),
            ('News/Macro', top_result.news_macro, scorer.group_weights['news_macro']),
            ('Social/Alternative', top_result.social_alternative, scorer.group_weights['social_alternative']),
            ('Risk/Stability', top_result.risk_stability, scorer.group_weights['risk_stability']),
            ('Institutional/Smart Money', top_result.institutional_smart_money, scorer.group_weights['institutional_smart_money'])
        ]
        
        for group_name, group_score, group_weight in groups:
            print(f"{group_name:<30} {group_score.score:>10.4f} {group_score.coverage:>9.1%} {group_weight:>9.1%}")
        
        # Verify weighted sum
        print("\n🔍 Verification:")
        weighted_sum = sum(gs.score * gw for _, gs, gw in groups)
        print(f"   Weighted sum of group scores: {weighted_sum:.4f}")
        print(f"   Overall score:                {top_result.overall_score:.4f}")
        print(f"   Match: {'✅ YES' if abs(weighted_sum - top_result.overall_score) < 0.0001 else '❌ NO'}")
        
        # Display group weight contribution
        print(f"\n📊 Group Score Contributions to Overall:")
        print(f"{'Group':<30} {'Contribution':<15}")
        print("-" * 80)
        for group_name, group_score, group_weight in groups:
            contribution = group_score.score * group_weight
            print(f"{group_name:<30} {contribution:>13.4f}")
        print("-" * 80)
        print(f"{'TOTAL (Overall Score)':<30} {top_result.overall_score:>13.4f}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    if scored_by_ticker:
        overall_scores = [r.overall_score for r in scored_by_ticker.values()]
        coverages = [r.total_coverage for r in scored_by_ticker.values()]
        
        print(f"\n📊 Overall Scores:")
        print(f"   Mean:     {sum(overall_scores) / len(overall_scores):>10.4f}")
        print(f"   Min:      {min(overall_scores):>10.4f}")
        print(f"   Max:      {max(overall_scores):>10.4f}")
        print(f"   Range:    {max(overall_scores) - min(overall_scores):>10.4f}")
        
        print(f"\n📊 Coverage:")
        print(f"   Mean:     {sum(coverages) / len(coverages):>10.1%}")
        print(f"   Min:      {min(coverages):>10.1%}")
        print(f"   Max:      {max(coverages):>10.1%}")
    
    # Config verification
    print("\n" + "=" * 80)
    print("CONFIG VERIFICATION")
    print("=" * 80)
    
    print(f"\n📋 Group Weights (from weights.yaml):")
    group_weight_sum = 0.0
    for group_name, weight in scorer.group_weights.items():
        print(f"   {group_name:<30} {weight:.4f}")
        group_weight_sum += weight
    print(f"   {'TOTAL':<30} {group_weight_sum:.4f} {'✅ Valid' if abs(group_weight_sum - 1.0) < 0.001 else '❌ Invalid'}")
    
    print(f"\n📋 Factor Weights (per group):")
    for group_name, factor_weights in scorer.factor_weights.items():
        factor_weight_sum = sum(factor_weights.values())
        status = '✅ Valid' if abs(factor_weight_sum - 1.0) < 0.001 else '❌ Invalid'
        print(f"   {group_name:<30} {len(factor_weights):>3} factors, sum={factor_weight_sum:.4f} {status}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully scored {len(scored_by_ticker)} tickers")
    print(f"📊 Overall score range: {min(overall_scores):.4f} to {max(overall_scores):.4f}")
    print(f"📊 Avg coverage: {sum(coverages) / len(coverages):.1%}")
    print(f"🎉 Phase 4 scoring test COMPLETE!")


if __name__ == "__main__":
    results = asyncio.run(test_phase4_scoring())
