import yfinance as yf
import pandas as pd
import numpy as np

ticker = 'AAPL'
tk = yf.Ticker(ticker)
history = tk.history(period='2y')
returns = history['Close'].pct_change().dropna()

print(f"Returns: {len(returns)} days")

# Current calculation (WRONG)
vol_current = returns.tail(60).std()
rolling_vol = returns.rolling(60).std()
print(f"\nCurrent calculation:")
print(f"  rolling_vol length: {len(rolling_vol)}")
print(f"  rolling_vol NaN count: {rolling_vol.isna().sum()}")
print(f"  vol_current: {vol_current:.6f}")
print(f"  (rolling_vol < vol_current).sum(): {(rolling_vol < vol_current).sum()}")
percentile_wrong = (rolling_vol < vol_current).sum() / len(rolling_vol) * 100
print(f"  Percentile (WRONG): {percentile_wrong:.2f}%")

# Fixed calculation
rolling_vol_valid = rolling_vol.dropna()
print(f"\nFixed calculation:")
print(f"  rolling_vol_valid length: {len(rolling_vol_valid)}")
print(f"  (rolling_vol_valid < vol_current).sum(): {(rolling_vol_valid < vol_current).sum()}")
percentile_fixed = (rolling_vol_valid < vol_current).sum() / len(rolling_vol_valid) * 100
print(f"  Percentile (FIXED): {percentile_fixed:.2f}%")
