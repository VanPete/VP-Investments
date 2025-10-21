import yfinance as yf
import pandas as pd
import numpy as np

ticker = 'AAPL'
print(f"Testing {ticker}...")

# Fetch data
tk = yf.Ticker(ticker)
earnings_history = tk.earnings_history
history = tk.history(period='1y')

print(f"\nEarnings history: {earnings_history is not None and not earnings_history.empty}")
if earnings_history is not None and not earnings_history.empty:
    print(f"Rows: {len(earnings_history)}")
    print(f"\nMost recent earnings date: {earnings_history.index[-1]}")
    
    last_earnings_date = earnings_history.index[-1]
    
    # FIXED: Remove timezone from BOTH
    if hasattr(last_earnings_date, 'tz') and last_earnings_date.tz is not None:
        last_earnings_date = last_earnings_date.tz_localize(None)
    
    print(f"Stock history: {history is not None and not history.empty}")
    if history is not None and not history.empty:
        # Remove timezone from history index if present
        history_index = history.index
        if hasattr(history_index, 'tz') and history_index.tz is not None:
            history_index = history_index.tz_localize(None)
            history_naive = history.copy()
            history_naive.index = history_index
        else:
            history_naive = history
        
        # Try to filter
        print(f"\nFiltering for dates after {last_earnings_date}...")
        post_earnings = history_naive[history_naive.index > last_earnings_date]
        print(f"Post-earnings rows: {len(post_earnings)}")
        
        if len(post_earnings) >= 21:
            drift = ((post_earnings['Close'].iloc[20] - post_earnings['Close'].iloc[0]) 
                    / post_earnings['Close'].iloc[0]) * 100
            print(f"✓ Drift (21d): {drift:.2f}%")
        else:
            print(f"Not enough data (need 21, got {len(post_earnings)})")

