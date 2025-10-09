"""
Test Phase 3 Scoring System with Enhanced Fundamental Data

Tests the Phase 3 scoring enhancements including:
- Analyst price targets and ratings
- Earnings surprise history
- Institutional ownership changes
- Insider trading activity
"""

import sys
import asyncio
from backend.integrations.yfinance import FinancialMetricsCalculator
from backend.pipeline import UnifiedPipeline

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

async def test_phase3_scoring():
    """Test Phase 3 scoring with real stocks."""
    
    print("=" * 70)
    print("PHASE 3 SCORING SYSTEM TEST")
    print("Testing enhanced fundamental data collection and scoring")
    print("=" * 70)
    print()
    
    # Initialize
    calculator = FinancialMetricsCalculator()
    pipeline = UnifiedPipeline()
    
    # Test stocks with different characteristics
    test_stocks = {
        'AAPL': 'High analyst coverage, strong fundamentals',
        'TSLA': 'Mixed analyst views, high institutional ownership',
        'NVDA': 'Strong earnings surprises, growth momentum',
        'AMD': 'Moderate analyst coverage, competitive sector',
        'F': 'Value stock, strong fundamentals, dividend',
    }
    
    print(f"\nTesting {len(test_stocks)} stocks with Phase 3 enhancements:\n")
    
    results = []
    
    for ticker, description in test_stocks.items():
        print(f"\n{'='*70}")
        print(f"Testing: {ticker} - {description}")
        print(f"{'='*70}")
        
        try:
            # Get comprehensive financial data (includes Phase 3 data)
            financial_data = calculator.get_comprehensive_financial_data(ticker)
            
            # Display Phase 3 specific data
            print(f"\n[PHASE 3 DATA COLLECTED]")
            
            # Analyst data
            print(f"\n  Analyst Data:")
            print(f"    Target Mean Price: ${financial_data.get('target_price_mean', 'N/A')}")
            print(f"    Target Upside: {financial_data.get('target_upside_pct', 'N/A')}%")
            print(f"    Recommendation: {financial_data.get('recommendation_mean', 'N/A')} (1=Strong Buy, 5=Sell)")
            print(f"    Number of Analysts: {financial_data.get('num_analysts', 'N/A')}")
            
            # Earnings surprise
            print(f"\n  Earnings Surprise:")
            print(f"    Last Surprise: {financial_data.get('last_earnings_surprise_pct', 'N/A')}%")
            print(f"    Avg Surprise (4Q): {financial_data.get('avg_earnings_surprise_pct', 'N/A')}%")
            print(f"    Trend: {financial_data.get('earnings_surprise_trend', 'N/A')}")
            
            # Institutional ownership
            print(f"\n  Institutional Ownership:")
            print(f"    Current %: {financial_data.get('institutional_ownership_pct', 'N/A')}%")
            print(f"    QoQ Change: {financial_data.get('institutional_change_qoq', 'N/A')}%")
            print(f"    Number of Institutions: {financial_data.get('num_institutions', 'N/A')}")
            print(f"    Top 10 Holders %: {financial_data.get('top_10_holders_pct', 'N/A')}%")
            
            # Insider trading
            print(f"\n  Insider Activity (3 months):")
            print(f"    Buy Transactions: {financial_data.get('insider_buy_transactions_3m', 'N/A')}")
            print(f"    Sell Transactions: {financial_data.get('insider_sell_transactions_3m', 'N/A')}")
            print(f"    Net Shares: {financial_data.get('insider_net_shares_3m', 'N/A'):,}")
            print(f"    Activity Score: {financial_data.get('insider_activity_score', 'N/A')} (0-100)")
            
            # Calculate financial score
            financial_score = pipeline._calculate_financial_score(financial_data)
            fundamentals_score = pipeline._calculate_fundamentals_score(financial_data)
            technical_score = pipeline._calculate_technical_score(financial_data)
            
            print(f"\n[SCORING RESULTS]")
            print(f"  Financial Score: {financial_score:.4f}")
            print(f"  => Fundamentals (30%): {fundamentals_score:.4f}")
            print(f"  => Technical (40%): {technical_score:.4f}")
            
            # Validation
            is_valid = 0 <= financial_score <= 1
            status = "✓ PASS" if is_valid else "✗ FAIL"
            
            print(f"\n[VALIDATION]")
            print(f"  Score in [0, 1] range: {status}")
            print(f"  Score value: {financial_score:.4f}")
            
            results.append({
                'ticker': ticker,
                'score': financial_score,
                'fundamentals': fundamentals_score,
                'technical': technical_score,
                'valid': is_valid,
                'has_analyst_data': financial_data.get('target_upside_pct') is not None,
                'has_earnings_data': financial_data.get('avg_earnings_surprise_pct') is not None,
                'has_institutional_data': financial_data.get('institutional_change_qoq') is not None,
                'has_insider_data': financial_data.get('insider_activity_score') is not None,
            })
            
        except Exception as e:
            print(f"\n✗ ERROR testing {ticker}: {e}")
            results.append({
                'ticker': ticker,
                'score': 0.0,
                'valid': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n\n{'='*70}")
    print("PHASE 3 TEST SUMMARY")
    print(f"{'='*70}\n")
    
    # Sort by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    print(f"{'Ticker':<8} {'Score':<8} {'Fundamentals':<14} {'Phase 3 Data':<20} {'Status'}")
    print(f"{'-'*70}")
    
    for result in results:
        if result.get('valid'):
            phase3_coverage = []
            if result.get('has_analyst_data'):
                phase3_coverage.append('Analyst')
            if result.get('has_earnings_data'):
                phase3_coverage.append('Earnings')
            if result.get('has_institutional_data'):
                phase3_coverage.append('Inst')
            if result.get('has_insider_data'):
                phase3_coverage.append('Insider')
            
            phase3_str = ', '.join(phase3_coverage) if phase3_coverage else 'None'
            
            print(
                f"{result['ticker']:<8} "
                f"{result['score']:<8.4f} "
                f"{result['fundamentals']:<14.4f} "
                f"{phase3_str:<20} "
                f"{'✓ PASS' if result['valid'] else '✗ FAIL'}"
            )
        else:
            print(f"{result['ticker']:<8} {'ERROR':<8} {'N/A':<14} {'N/A':<20} ✗ FAIL")
    
    # Statistics
    passed = sum(1 for r in results if r.get('valid', False))
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"Tests Passed: {passed}/{total}")
    print(f"Pass Rate: {(passed/total*100):.1f}%")
    
    # Phase 3 data coverage
    analyst_coverage = sum(1 for r in results if r.get('has_analyst_data', False))
    earnings_coverage = sum(1 for r in results if r.get('has_earnings_data', False))
    institutional_coverage = sum(1 for r in results if r.get('has_institutional_data', False))
    insider_coverage = sum(1 for r in results if r.get('has_insider_data', False))
    
    print(f"\n[PHASE 3 DATA COVERAGE]")
    print(f"  Analyst Data: {analyst_coverage}/{total} stocks ({analyst_coverage/total*100:.0f}%)")
    print(f"  Earnings Data: {earnings_coverage}/{total} stocks ({earnings_coverage/total*100:.0f}%)")
    print(f"  Institutional Data: {institutional_coverage}/{total} stocks ({institutional_coverage/total*100:.0f}%)")
    print(f"  Insider Data: {insider_coverage}/{total} stocks ({insider_coverage/total*100:.0f}%)")
    
    if passed == total:
        print(f"\n✓ ALL TESTS PASSED! Phase 3 scoring system is ready.")
    else:
        print(f"\n✗ Some tests failed. Review errors above.")
    
    print(f"{'='*70}\n")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(test_phase3_scoring())
    sys.exit(0 if success else 1)
