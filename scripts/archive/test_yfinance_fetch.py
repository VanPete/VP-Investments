"""Test interval return calculation manually."""
import asyncio
import yfinance as yf
from datetime import datetime, timedelta, timezone

async def main():
    # Test with a record from Oct 27
    ticker = "LPG"
    baseline_date_str = "2025-10-27T13:13:57.355506+00:00"
    baseline_date = datetime.fromisoformat(baseline_date_str.replace('Z', '+00:00'))
    
    print(f"\n=== Testing Return Calculation for {ticker} ===")
    print(f"Baseline Date: {baseline_date}")
    print(f"Age: {(datetime.now(timezone.utc) - baseline_date).total_seconds() / 3600:.1f} hours")
    print(f"Age: {(datetime.now(timezone.utc) - baseline_date).days} days\n")
    
    # Try to fetch price data like Phase 6 does
    end_date = datetime.now()
    start_date = baseline_date - timedelta(days=2)
    
    print(f"Fetching data from {start_date.date()} to {end_date.date()}...\n")
    
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        print(f"Downloaded {len(df)} rows of data")
        if not df.empty:
            print(f"\nData range: {df.index[0]} to {df.index[-1]}")
            print(f"\nLast 5 rows:")
            print(df.tail())
        else:
            print("ERROR: No data returned!")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == '__main__':
    asyncio.run(main())
