"""
Test script to verify company_name and current_price extraction from Phase 5
"""
import yfinance as yf

def test_extraction():
    """Test company name and price extraction for a few tickers"""
    
    # Test tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    print("\n🧪 Testing company_name and current_price extraction...")
    print("=" * 70)
    
    for ticker_symbol in test_tickers:
        print(f"\n📊 Testing {ticker_symbol}:")
        
        # Fetch raw data
        ticker = yf.Ticker(ticker_symbol)
        
        # Simulate ticker_data structure from Phase 1
        ticker_data = {
            'ticker': ticker_symbol,
            'raw_data': {
                'info': ticker.info,
                'fast_info': ticker.fast_info
            }
        }
        
        # Add history for price fallback
        try:
            history = ticker.history(period='5d')
            if not history.empty:
                ticker_data['raw_data']['history'] = history
        except Exception as e:
            print(f"  ⚠️  Could not fetch history: {e}")
        
        # Extract company name
        company_name = None
        info = ticker_data['raw_data'].get('info', {})
        if info:
            company_name = info.get('longName') or info.get('shortName')
        
        print(f"  Company Name: {company_name}")
        
        # Extract current price
        current_price = None
        fast_info = ticker_data['raw_data'].get('fast_info')
        if fast_info:
            try:
                current_price = float(fast_info.get('lastPrice', 0))
            except (ValueError, TypeError):
                current_price = None
        
        # Fallback to history
        if not current_price:
            history = ticker_data['raw_data'].get('history')
            if history is not None and not history.empty and 'Close' in history.columns:
                try:
                    current_price = float(history['Close'].iloc[-1])
                except (ValueError, TypeError, IndexError):
                    pass
        
        print(f"  Current Price: ${current_price:.2f}" if current_price else "  Current Price: None")
        
        # Check what would be stored
        if company_name and current_price:
            print(f"  ✅ Both fields extracted successfully")
        elif company_name:
            print(f"  ⚠️  Only company name extracted")
        elif current_price:
            print(f"  ⚠️  Only price extracted")
        else:
            print(f"  ❌ Neither field extracted")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")

if __name__ == "__main__":
    test_extraction()
