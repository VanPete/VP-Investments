"""
Debug script to test volatility_percentile and calmar_ratio with actual pipeline data
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from backend.integrations.yfinance import ComprehensiveYFinanceFetcher
import pandas as pd
import numpy as np

# Create fetcher
fetcher = ComprehensiveYFinanceFetcher()

# Fetch AAPL data
print("Fetching AAPL data...")
aapl_raw = fetcher.fetch_ticker("AAPL")

if aapl_raw and aapl_raw.history is not None:
    returns = aapl_raw.history['Close'].pct_change().dropna()
    print(f"Returns length: {len(returns)}")
    print(f"Returns head:\n{returns.head()}")
    print(f"Returns tail:\n{returns.tail()}")
    
    # Test volatility_percentile
    print("\n" + "="*80)
    print("Testing volatility_percentile")
    print("="*80)
    
    if len(returns) >= 252:
        vol_current = returns.tail(60).std() if len(returns) >= 60 else returns.std()
        rolling_vol = returns.rolling(60).std()
        
        print(f"vol_current: {vol_current}")
        print(f"rolling_vol length: {len(rolling_vol)}")
        print(f"rolling_vol (first 10):\n{rolling_vol.head(10)}")
        print(f"rolling_vol (last 10):\n{rolling_vol.tail(10)}")
        
        # Check for NaN values
        nan_count = rolling_vol.isna().sum()
        print(f"\nNaN values in rolling_vol: {nan_count}")
        
        # Original calculation
        percentile = (rolling_vol < vol_current).sum() / len(rolling_vol) * 100
        print(f"\nOriginal calculation: {percentile}%")
        
        # Try without NaN
        rolling_vol_clean = rolling_vol.dropna()
        percentile_clean = (rolling_vol_clean < vol_current).sum() / len(rolling_vol_clean) * 100
        print(f"Without NaN: {percentile_clean}%")
    else:
        print(f"Not enough data! Only {len(returns)} days, need 252")
    
    # Test calmar_ratio
    print("\n" + "="*80)
    print("Testing calmar_ratio")
    print("="*80)
    
    if len(returns) >= 252:
        # Calculate max_drawdown_1y first (this is what the code does)
        cumulative = (1 + returns.tail(252)).cumprod()
        running_max = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown_1y = drawdown.min()
        
        print(f"max_drawdown_1y: {max_drawdown_1y}")
        
        if max_drawdown_1y < 0:
            annual_return = returns.tail(252).mean() * 252
            max_dd = abs(max_drawdown_1y / 100)
            calmar_ratio = annual_return / max_dd if max_dd > 0 else np.nan
            
            print(f"annual_return: {annual_return}")
            print(f"max_dd (abs): {max_dd}")
            print(f"calmar_ratio: {calmar_ratio}")
        else:
            print("max_drawdown_1y is not negative!")
    else:
        print(f"Not enough data! Only {len(returns)} days, need 252")
else:
    print("Failed to fetch AAPL data")
