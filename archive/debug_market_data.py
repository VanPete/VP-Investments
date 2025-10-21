"""
Debug script to check market_data in the pipeline
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from backend.integrations.yfinance import ComprehensiveYFinanceFetcher
import pandas as pd

# Create fetcher
fetcher = ComprehensiveYFinanceFetcher()

# Fetch market data
print("Fetching market data...")
market_data = fetcher.fetch_market_data()

# Check what we got
print(f"\nMarket data object: {market_data}")
print(f"Market data type: {type(market_data)}")
print(f"Has spy_history: {hasattr(market_data, 'spy_history')}")

if market_data:
    print(f"spy_history is None: {market_data.spy_history is None}")
    if market_data.spy_history is not None:
        print(f"SPY history shape: {market_data.spy_history.shape}")
        print(f"SPY history columns: {market_data.spy_history.columns.tolist()}")
        print(f"SPY history head:\n{market_data.spy_history.head()}")
        print(f"SPY history tail:\n{market_data.spy_history.tail()}")
        print(f"SPY history empty: {market_data.spy_history.empty}")
    
    print(f"\nVIX current: {market_data.vix_current}")
    print(f"Treasury 10y: {market_data.treasury_yield_10y}")
    print(f"Treasury 2y: {market_data.treasury_yield_2y}")
    print(f"is_valid(): {market_data.is_valid()}")
else:
    print("market_data is None!")

# Now test the downside capture calculation manually
if market_data and market_data.spy_history is not None:
    print("\n" + "="*80)
    print("Testing downside capture calculation with AAPL")
    print("="*80)
    
    # Fetch AAPL data
    aapl_raw = fetcher.fetch_ticker("AAPL")
    
    if aapl_raw and aapl_raw.history is not None:
        aapl_returns = aapl_raw.history['Close'].pct_change().dropna()
        market_returns = market_data.spy_history['Close'].pct_change().dropna()
        
        print(f"\nAAPL returns length: {len(aapl_returns)}")
        print(f"Market returns length: {len(market_returns)}")
        
        # Align
        aligned = pd.DataFrame({'stock': aapl_returns, 'market': market_returns}).dropna()
        print(f"Aligned length: {len(aligned)}")
        
        if len(aligned) >= 252:
            down_days = aligned[aligned['market'] < 0].tail(252)
            print(f"Down days in last 252: {len(down_days)}")
            
            if len(down_days) >= 20:
                cov = down_days['stock'].cov(down_days['market'])
                var = down_days['market'].var()
                downside_capture = (cov / var) if var > 0 else None
                print(f"\nCovariance: {cov}")
                print(f"Variance: {var}")
                print(f"Downside capture: {downside_capture}")
            else:
                print("Not enough down days!")
        else:
            print("Not enough aligned data!")
