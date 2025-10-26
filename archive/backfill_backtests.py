"""
Backtest Backfill Script - Following phase6_backtest.py Logic

BASELINE LOGIC:
- backtest_baseline_date = signal_date + 1 day (next trading day after signal)
- backtest_baseline_price = next day's OPEN price (entry price for trading)

RETURNS LOGIC:
- return_Xd = % change from baseline_price to Close price X days after baseline_date
- spy_return_Xd = same calculation for SPY benchmark

SKIP CRITERIA:
- Skip signals < 2 days old (need baseline_date + 1d for first return)
- Skip signals that already have baseline_price (already backtested)
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from typing import Dict, Optional
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from backend.storage.database import SupabaseInterface
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def get_price_at_date(df: pd.DataFrame, target_date: datetime, price_type: str = 'Open') -> Optional[float]:
    """Get price at date with forward fill for weekends (matches phase6_backtest.py)"""
    if df.empty:
        return None
    future = df[df.index >= target_date]
    if not future.empty:
        return float(future.iloc[0][price_type])
    return None

def backtest_signal(ticker: str, signal_date: datetime) -> Optional[Dict]:
    """
    Backtest a signal following phase6_backtest.py logic.
    
    Args:
        ticker: Stock ticker symbol
        signal_date: When signal was created
        
    Returns:
        Dict with all backtest columns, or None if failed
    """
    try:
        # Get stock and SPY data
        stock = yf.Ticker(ticker)
        spy = yf.Ticker('SPY')
        
        # Baseline date is NEXT DAY after signal (when you'd actually trade)
        baseline_date = signal_date + timedelta(days=1)
        
        # Fetch data from signal_date to 100 days after baseline
        start_str = signal_date.strftime('%Y-%m-%d')
        end_str = (baseline_date + timedelta(days=100)).strftime('%Y-%m-%d')
        
        hist = stock.history(start=start_str, end=end_str)
        spy_hist = spy.history(start=start_str, end=end_str)
        
        if hist.empty:
            logger.warning(f"No price data for {ticker}")
            return None
            
        if spy_hist.empty:
            logger.warning(f"No SPY data for comparison")
            return None
        
        # Get BASELINE PRICE = next day's OPEN (this is your entry price)
        baseline_price = get_price_at_date(hist, baseline_date, 'Open')
        spy_baseline = get_price_at_date(spy_hist, baseline_date, 'Open')
        
        if not baseline_price or not spy_baseline:
            logger.warning(f"No baseline price for {ticker} at {baseline_date.date()}")
            return None
        
        # Start building result
        result = {
            'backtest_baseline_price': round(baseline_price, 2),
            'backtest_baseline_date': baseline_date,
            'backtest_last_update': datetime.now(),
        }
        
        # Calculate returns for all periods
        # Measured from baseline_date (not signal_date!)
        periods = [1, 3, 7, 10, 14, 30, 90]
        completed_count = 0
        
        for days in periods:
            # Target date is X days AFTER baseline_date
            target_date = baseline_date + timedelta(days=days)
            
            # Stock return (Close price at target vs Open at baseline)
            target_price = get_price_at_date(hist, target_date, 'Close')
            if target_price and baseline_price > 0:
                pct = ((target_price - baseline_price) / baseline_price) * 100
                result[f'return_{days}d'] = round(pct, 2)
                completed_count += 1
            
            # SPY return (benchmark comparison)
            spy_target = get_price_at_date(spy_hist, target_date, 'Close')
            if spy_target and spy_baseline > 0:
                spy_pct = ((spy_target - spy_baseline) / spy_baseline) * 100
                result[f'spy_return_{days}d'] = round(spy_pct, 2)
        
        # Set status based on what returns we could calculate
        if completed_count == len(periods):
            result['backtest_status'] = 'completed'
        elif completed_count > 0:
            result['backtest_status'] = 'partial'
        else:
            result['backtest_status'] = 'baseline_set'
            
        return result
        
    except Exception as e:
        logger.error(f"Error backtesting {ticker}: {e}")
        return None

async def backfill_backtests():
    """Main function to backfill old signals"""
    db = SupabaseInterface()
    await db.connect()
    
    print("=" * 80)
    print("BACKTEST BACKFILL - Following phase6_backtest.py")
    print("=" * 80)
    print("Logic:")
    print("  - baseline_date = signal_date + 1 day")
    print("  - baseline_price = next day's OPEN")
    print("  - returns measured from baseline_date")
    print("Filters:")
    print("  - Skip signals < 2 days old")
    print("  - Skip signals with baseline already set")
    print("=" * 80 + "\n")
    
    # Get signals WITHOUT baseline (never backtested)
    query = """
    SELECT id, ticker, created_at
    FROM signals
    WHERE backtest_baseline_price IS NULL
    ORDER BY created_at ASC
    """
    
    signals = await db.execute_query(query)
    print(f"Found {len(signals)} signals to backfill\n")
    
    success = error = skipped = 0
    
    for i, sig in enumerate(signals, 1):
        ticker = sig['ticker']
        created = sig['created_at']
        # Remove timezone for age calculation
        if hasattr(created, 'tzinfo') and created.tzinfo is not None:
            created_naive = created.replace(tzinfo=None)
        else:
            created_naive = created
        days_old = (datetime.now() - created_naive).days
        
        # Skip if < 2 days old (need baseline + 1d)
        if days_old < 2:
            print(f"[{i}/{len(signals)}] Skip {ticker} - only {days_old}d old")
            skipped += 1
            continue
        
        print(f"[{i}/{len(signals)}] {ticker} (signal: {created_naive.date()}, {days_old}d old)...", end=" ", flush=True)
        
        # Calculate backtest data (sync yfinance in executor)
        result = await asyncio.get_event_loop().run_in_executor(
            None, backtest_signal, ticker, created_naive
        )
        
        if not result:
            print("❌ No data")
            error += 1
            continue
        
        # Build update query with positional params
        update_parts = []
        params = []
        param_num = 1
        
        for key, val in result.items():
            if val is not None:
                update_parts.append(f"{key} = ${param_num}")
                params.append(val)
                param_num += 1
        
        if update_parts:
            params.append(sig['id'])  # Add ID as last param
            
            update_query = f"""
            UPDATE signals 
            SET {', '.join(update_parts)} 
            WHERE id = ${param_num}
            """
            
            await db.execute_non_query(update_query, params)
            
            status = result.get('backtest_status', '?')
            baseline = result.get('backtest_baseline_price', 0)
            print(f"✓ [{status}] baseline=${baseline:.2f}")
            success += 1
        else:
            print("⚠️ No updates")
            error += 1
        
        # Rate limit
        if i % 5 == 0:
            await asyncio.sleep(1)
    
    print("\n" + "=" * 80)
    print(f"✓ Success: {success} | ❌ Errors: {error} | ⏭️ Skipped: {skipped}")
    print("=" * 80)
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(backfill_backtests())
