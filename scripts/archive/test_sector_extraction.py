"""
Test Phase 5 Sector Extraction Fix
===================================

Verifies that the getattr → .get() fix correctly extracts sector data.

Run: python scripts/test_sector_extraction.py
"""

import yfinance as yf
from backend.utils.sector_etfs import get_sector_etf

def test_sector_extraction():
    """Test that we can extract sector from yfinance info dict."""
    
    print("\n" + "="*80)
    print("TESTING SECTOR EXTRACTION FIX")
    print("="*80)
    print("Testing the fix: getattr(dict, 'sector') → dict.get('sector')\n")
    
    test_tickers = ['AAPL', 'GOOGL', 'JPM', 'XOM', 'PFE']
    
    for ticker in test_tickers:
        print(f"Testing {ticker}...")
        
        try:
            # Fetch ticker data
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # OLD BROKEN WAY (returns None for dicts)
            sector_old = getattr(info, 'sector', None)
            
            # NEW FIXED WAY (correct dict access)
            sector_new = info.get('sector')
            
            # Get sector ETF
            sector_etf = get_sector_etf(sector_new) if sector_new else None
            
            print(f"  Type of info: {type(info)}")
            print(f"  OLD (getattr): {sector_old}")
            print(f"  NEW (.get):    {sector_new}")
            print(f"  Sector ETF:    {sector_etf}")
            
            # Verify fix works
            assert sector_old is None, "getattr should return None for dicts"
            assert sector_new is not None, "dict.get() should return sector"
            assert sector_etf is not None, "Should map sector to ETF"
            
            print(f"  ✅ PASS\n")
            
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}\n")
        except Exception as e:
            print(f"  ⚠️  ERROR: {e}\n")
    
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nThe fix ensures Phase 5 will correctly populate sector data going forward.")
    print("Old data needs backfill script to populate missing sector fields.")
    print()

if __name__ == '__main__':
    test_sector_extraction()
