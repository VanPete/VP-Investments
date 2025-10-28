"""Manually test the SPY calculation logic from Phase 6."""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

def _get_price_at_date(df, target_date, price_col='Close'):
    """Replicate the Phase 6 price lookup logic."""
    try:
        # Normalize to date
        target_ts = pd.Timestamp(target_date.date())
        
        print(f"  Looking for price on {target_ts}")
        print(f"  Available dates: {list(df.index)}")
        
        # Try exact date first
        if target_ts in df.index:
            price = float(df.loc[target_ts, price_col].iloc[0] if hasattr(df.loc[target_ts, price_col], 'iloc') else df.loc[target_ts, price_col])
            print(f"  ✅ Found exact date: ${price:.2f}")
            return price
        
        # Forward fill - find next available date
        available_dates = [d for d in df.index if d >= target_ts]
        print(f"  Forward fill candidates: {available_dates}")
        if available_dates:
            price = float(df.loc[available_dates[0], price_col].iloc[0] if hasattr(df.loc[available_dates[0], price_col], 'iloc') else df.loc[available_dates[0], price_col])
            print(f"  ✅ Forward fill to {available_dates[0]}: ${price:.2f}")
            return price
        
        print(f"  ❌ No price found")
        return None
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

# Test with Oct 26 signal
baseline_date = datetime.fromisoformat('2025-10-26T15:15:46.353579+00:00')
end_date = datetime.now()
start_date = baseline_date - timedelta(days=2)

print(f"\n=== Testing SPY Return Calculation ===")
print(f"Baseline: {baseline_date}")
print(f"Fetch range: {start_date.date()} to {end_date.date()}\n")

# Download SPY data
spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)
print(f"\nSPY Data ({len(spy_df)} rows):")
print(spy_df[['Close']])

# Get baseline price
print(f"\n--- Getting SPY Baseline Price ---")
spy_baseline = _get_price_at_date(spy_df, baseline_date, 'Close')

# Get target price (1 day later)
target_date = baseline_date + timedelta(days=1)
print(f"\n--- Getting SPY Target Price (1d) ---")
spy_target = _get_price_at_date(spy_df, target_date, 'Close')

# Calculate return
if spy_baseline and spy_target:
    spy_return = ((spy_target - spy_baseline) / spy_baseline) * 100
    print(f"\n✅ SPY 1d Return: {spy_return:.4f}%")
    print(f"   Baseline: ${spy_baseline:.2f}")
    print(f"   Target: ${spy_target:.2f}")
else:
    print(f"\n❌ Cannot calculate - baseline={spy_baseline}, target={spy_target}")
