"""Quick script to inspect yfinance data structure for debugging"""
import yfinance as yf
import pandas as pd

# Pick a ticker
ticker = yf.Ticker("ORCL")

# Get the data
info = ticker.info
earnings_history = ticker.earnings_history

print("="*80)
print("SECTOR & INDUSTRY INFO:")
print("="*80)
for key in ['sector', 'industry', 'sectorDisp', 'industryDisp', 
            'sectorKey', 'industryKey']:
    print(f"  {key}: {info.get(key, 'NOT FOUND')}")

print("\n" + "="*80)
print("EARNINGS HISTORY:")
print("="*80)
if earnings_history is not None:
    print(f"Type: {type(earnings_history)}")
    if isinstance(earnings_history, pd.DataFrame):
        print(f"Shape: {earnings_history.shape}")
        print(f"Columns: {list(earnings_history.columns)}")
        print(f"\nFirst row:")
        print(earnings_history.iloc[0] if not earnings_history.empty else "EMPTY")
    else:
        print(f"Not a DataFrame: {earnings_history}")
else:
    print("None")
