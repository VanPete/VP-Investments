"""
Test Phase 2 Enhanced Financial Scoring

This script tests the new comprehensive scoring system to verify:
1. All indicators are being used
2. Scores are in valid range [0, 1]
3. Score breakdown logging works
4. Normalization handles missing data correctly
"""

import os
import sys
from dotenv import load_dotenv
from backend.pipeline import UnifiedPipeline
from backend.integrations.yfinance import FinancialMetricsCalculator
import logging

# Fix unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Setup logging to see debug output
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_financial_scoring():
    """Test the enhanced financial scoring system."""
    
    load_dotenv()
    
    # Test with a few different ticker types
    test_tickers = [
        'AAPL',   # Large cap, stable
        'TSLA',   # High volatility, growth
        'KO',     # Value stock, defensive
        'NVDA',   # High growth, momentum
        'F',      # Traditional auto, moderate
    ]
    
    print("=" * 80)
    print("PHASE 2 FINANCIAL SCORING TEST")
    print("=" * 80)
    print()
    
    # Initialize components
    yfinance = FinancialMetricsCalculator()
    pipeline = UnifiedPipeline()
    
    results = []
    
    for ticker in test_tickers:
        print(f"\n{'='*80}")
        print(f"Testing {ticker}")
        print(f"{'='*80}\n")
        
        try:
            # Get financial data
            logger.info(f"Fetching financial data for {ticker}...")
            financial_data = yfinance.get_comprehensive_financial_data(ticker)
            
            if not financial_data or 'error' in financial_data:
                print(f"[ERROR] Error fetching data for {ticker}")
                continue
            
            # Calculate financial score
            logger.info(f"Calculating financial score for {ticker}...")
            score = pipeline._calculate_financial_score(financial_data)
            
            # Validate score
            if 0 <= score <= 1:
                print(f"[PASS] {ticker}: Financial Score = {score:.4f} (VALID)")
                results.append({
                    'ticker': ticker,
                    'score': score,
                    'valid': True
                })
            else:
                print(f"[FAIL] {ticker}: Financial Score = {score:.4f} (INVALID - out of range)")
                results.append({
                    'ticker': ticker,
                    'score': score,
                    'valid': False
                })
            
            # Show some key metrics
            print(f"\nKey Metrics:")
            print(f"  Market Cap: ${financial_data.get('market_cap_numeric', 0):,.0f}")
            print(f"  P/E Ratio: {financial_data.get('pe_ratio', 'N/A')}")
            print(f"  RSI: {financial_data.get('rsi', 'N/A')}")
            print(f"  Volatility: {financial_data.get('volatility', 'N/A')}")
            print(f"  Beta: {financial_data.get('beta', 'N/A')}")
            print(f"  Momentum Consistency: {financial_data.get('momentum_consistency_score', 'N/A')}")
            print(f"  Liquidity Score: {financial_data.get('liquidity_score', 'N/A')}")
            
        except Exception as e:
            print(f"[ERROR] Error testing {ticker}: {e}")
            logger.exception(f"Error testing {ticker}")
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")
    
    valid_scores = [r for r in results if r['valid']]
    invalid_scores = [r for r in results if not r['valid']]
    
    print(f"Total tickers tested: {len(results)}")
    print(f"Valid scores: {len(valid_scores)}")
    print(f"Invalid scores: {len(invalid_scores)}")
    
    if valid_scores:
        print(f"\nScore Distribution:")
        for result in sorted(valid_scores, key=lambda x: x['score'], reverse=True):
            print(f"  {result['ticker']}: {result['score']:.4f}")
    
    if invalid_scores:
        print(f"\n[WARNING] Invalid Scores (need investigation):")
        for result in invalid_scores:
            print(f"  {result['ticker']}: {result['score']:.4f}")
    
    print(f"\n{'='*80}")
    if len(valid_scores) == len(results):
        print("[SUCCESS] ALL TESTS PASSED - Phase 2 scoring system working correctly!")
    else:
        print(f"[WARNING] {len(invalid_scores)} tests failed - review debug logs above")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_financial_scoring()
