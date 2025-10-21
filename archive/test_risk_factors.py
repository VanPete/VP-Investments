import yfinance as yf
import pandas as pd
import numpy as np

ticker = 'AAPL'
print(f"Testing risk factors for {ticker}...")

# Fetch data
tk = yf.Ticker(ticker)
history = tk.history(period='2y')
returns = history['Close'].pct_change().dropna()

print(f"\nHistory: {len(history)} days")
print(f"Returns: {len(returns)} days")

# Test 1: volatility_percentile
print("\n" + "="*60)
print("TEST 1: volatility_percentile")
print("="*60)
if len(returns) >= 252:
    vol_current = returns.tail(60).std() if len(returns) >= 60 else returns.std()
    rolling_vol = returns.rolling(60).std()
    percentile = (rolling_vol < vol_current).sum() / len(rolling_vol) * 100
    print(f"✓ Current vol (60d): {vol_current:.6f}")
    print(f"✓ Rolling vol shape: {rolling_vol.shape}")
    print(f"✓ Percentile: {percentile:.2f}%")
else:
    print(f"✗ Not enough data (need 252, got {len(returns)})")

# Test 2: calmar_ratio
print("\n" + "="*60)
print("TEST 2: calmar_ratio")
print("="*60)
if len(returns) >= 252:
    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown_1y = drawdown.tail(252).min()
    
    print(f"Max drawdown (1y): {max_drawdown_1y:.2f}%")
    
    if max_drawdown_1y < 0:
        annual_return = returns.tail(252).mean() * 252
        max_dd = abs(max_drawdown_1y / 100)  # Convert from percentage
        calmar = annual_return / max_dd if max_dd > 0 else np.nan
        print(f"✓ Annual return: {annual_return:.4f}")
        print(f"✓ Max DD (abs): {max_dd:.4f}")
        print(f"✓ Calmar ratio: {calmar:.4f}")
    else:
        print(f"✗ Max drawdown not negative: {max_drawdown_1y}")
else:
    print(f"✗ Not enough data (need 252, got {len(returns)})")

# Test 3: downside_capture_1y
print("\n" + "="*60)
print("TEST 3: downside_capture_1y")
print("="*60)

# Fetch SPY data
spy = yf.Ticker('SPY')
spy_history = spy.history(period='2y')
print(f"SPY history: {len(spy_history)} days")

if len(returns) >= 252 and len(spy_history) >= 252:
    market_returns = spy_history['Close'].pct_change().dropna()
    aligned = pd.DataFrame({'stock': returns, 'market': market_returns}).dropna()
    
    print(f"Aligned data: {len(aligned)} days")
    
    if len(aligned) >= 252:
        # Only use days when market was down
        down_days = aligned[aligned['market'] < 0].tail(252)
        
        print(f"Down days (1y): {len(down_days)}")
        
        if len(down_days) >= 20:
            # Covariance / Variance of down days
            cov = down_days['stock'].cov(down_days['market'])
            var = down_days['market'].var()
            downside_capture = (cov / var) if var > 0 else np.nan
            print(f"✓ Covariance: {cov:.6f}")
            print(f"✓ Variance: {var:.6f}")
            print(f"✓ Downside capture: {downside_capture:.4f}")
        else:
            print(f"✗ Not enough down days (need 20, got {len(down_days)})")
    else:
        print(f"✗ Not enough aligned data (need 252, got {len(aligned)})")
else:
    print(f"✗ Not enough data")

print("\n" + "="*60)
