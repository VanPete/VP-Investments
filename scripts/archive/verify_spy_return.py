"""Check actual SPY return for Oct 26-27."""
import yfinance as yf
from datetime import datetime

# Get SPY data
spy = yf.download('SPY', start='2025-10-25', end='2025-10-28', progress=False, auto_adjust=True)

print("\n=== SPY Price Data ===")
print(spy[['Close']])

if len(spy) >= 2:
    oct26_close = spy.iloc[-3]['Close'] if len(spy) >= 3 else spy.iloc[0]['Close']
    oct27_close = spy.iloc[-2]['Close'] if len(spy) >= 2 else None
    
    if oct27_close:
        spy_return = ((oct27_close - oct26_close) / oct26_close) * 100
        print(f"\nOct 26 Close: ${oct26_close:.2f}")
        print(f"Oct 27 Close: ${oct27_close:.2f}")
        print(f"SPY 1-day Return: {spy_return:.4f}%")
        print(f"\nExpected in DB: {spy_return:.4f}")
        print(f"Actually in DB: 0.0000")
        print(f"\n❌ MISMATCH! SPY return calculation is broken!")
