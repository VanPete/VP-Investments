"""
Backtest ALL signals - populate ALL backtest columns properly.

Columns to populate:
- backtest_baseline_price: Price when signal was created  
- backtest_baseline_date: Date when signal was created
- return_1d, return_3d, return_7d, return_10d, return_14d, return_30d, return_90d
- spy_return_1d, spy_return_3d, spy_return_7d, spy_return_10d, spy_return_14d, spy_return_30d, spy_return_90d  
- backtest_status: 'completed', 'partial', 'pending', or 'error'
- backtest_last_update: Timestamp of last update
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent))

from backend.storage.database import SupabaseInterface
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_all_returns(ticker: str, signal_date: datetime, days_old: int) -> Optional[Dict]:
    """
    Calculate baseline data (ALWAYS) and returns (based on age).
    
    Strategy:
    1. ALWAYS get baseline price/date (needed for all signals)
    2. Calculate returns only for periods where signal is old enough
    3. Mark status based on what SHOULD be available vs what IS available
    """
    try:
        stock = yf.Ticker(ticker)
        spy = yf.Ticker('SPY')
        
        signal_str = signal_date.strftime('%Y-%m-%d')
        end_str = (signal_date + timedelta(days=100)).strftime('%Y-%m-%d')
        
        hist = stock.history(start=signal_str, end=end_str)
        spy_hist = spy.history(start=signal_str, end=end_str)
        
        if hist.empty:
            logger.warning(f"No price data for {ticker}")
            return {
                'backtest_status': 'error',
                'backtest_last_update': datetime.now(),
            }
        
        if spy_hist.empty:
            logger.warning(f"No SPY data available")
            return {
                'backtest_status': 'error', 
                'backtest_last_update': datetime.now(),
            }
            
        # ALWAYS get baseline data
        baseline_price = float(hist.iloc[0]['Close'])
        baseline_date = hist.index[0]
        spy_baseline = float(spy_hist.iloc[0]['Close'])
        
        result = {
            'backtest_baseline_price': baseline_price,
            'backtest_baseline_date': baseline_date,
            'backtest_last_update': datetime.now(),
        }
        
        # Calculate returns for periods where signal is old enough
        periods = [1, 3, 7, 10, 14, 30, 90]
        expected_periods = [p for p in periods if days_old >= p]  # What SHOULD be available
        completed = 0
        
        for days in periods:
            target = signal_date + timedelta(days=days)
            
            # Skip if signal not old enough yet
            if days_old < days:
                continue
            
            # Ticker returns
            future = hist[hist.index >= target]
            if not future.empty:
                price = float(future.iloc[0]['Close'])
                result[f'return_{days}d'] = ((price - baseline_price) / baseline_price) * 100
                completed += 1
            
            # SPY returns  
            spy_future = spy_hist[spy_hist.index >= target]
            if not spy_future.empty:
                spy_price = float(spy_future.iloc[0]['Close'])
                result[f'spy_return_{days}d'] = ((spy_price - spy_baseline) / spy_baseline) * 100
        
        # Set status based on what SHOULD be available
        if len(expected_periods) == 0:
            result['backtest_status'] = 'pending'  # Too recent for any returns
        elif completed == len(expected_periods):
            result['backtest_status'] = 'completed'  # Got all expected returns
        elif completed > 0:
            result['backtest_status'] = 'partial'  # Got some but not all expected
        else:
            result['backtest_status'] = 'pending'  # Old enough but no data available
            
        return result
        
    except Exception as e:
        logger.error(f"Error for {ticker}: {e}")
        return {
            'backtest_status': 'error',
            'backtest_last_update': datetime.now(),
        }

async def backtest_all():
    db = SupabaseInterface()
    await db.connect()
    
    print("=" * 80)
    print("BACKTEST BACKFILL - Following phase6_backtest.py Logic")
    print("=" * 80)
    print("Baseline: next_day_open (signal_date + 1 day)")
    print("Returns: measured from baseline_date")
    print("Skip: signals < 2 days old, signals with baseline already set")
    print("=" * 80)
    
    # Get signals missing backtest data (only those without baseline set)
    query = """
    SELECT id, ticker, created_at
    FROM signals
    WHERE backtest_baseline_price IS NULL
    ORDER BY created_at ASC
    """
    
    signals = await db.execute_query(query)
    print(f"\n{len(signals)} signals to backtest")
    print("-" * 80)
    
    success = error = 0
    
    for i, sig in enumerate(signals, 1):
        ticker = sig['ticker']
        created = sig['created_at']
        days_old = (datetime.now(created.tzinfo) - created).days
        
        print(f"[{i}/{len(signals)}] {ticker} ({created.date()}, {days_old}d old)...", end=" ", flush=True)
        
        # Calculate in executor to not block (ALWAYS process, even if < 1 day)
        returns = await asyncio.get_event_loop().run_in_executor(
            None, calculate_all_returns, ticker, created, days_old
        )
        
        if not returns:
            print("❌")
            error += 1
            continue
        
        # Build update query with positional parameters ($1, $2, etc.)
        update_parts = []
        param_values = []
        param_num = 1
        
        for k, v in returns.items():
            if v is not None:
                update_parts.append(f"{k} = ${param_num}")
                param_values.append(v)
                param_num += 1
        
        if update_parts:
            # Add signal ID as last parameter
            param_values.append(sig['id'])
            
            await db.execute_non_query(
                f"UPDATE signals SET {', '.join(update_parts)} WHERE id = ${param_num}",
                param_values
            )
            
            status = returns.get('backtest_status', '?')
            print(f"✓ [{status}]")
            success += 1
        else:
            print("⚠️")
            error += 1
        
        if i % 5 == 0:
            await asyncio.sleep(1)  # Rate limit
    
    print("\n" + "=" * 80)
    print(f"Success: {success} | Errors: {error}")
    print(f"Note: Signals < 1 day old get baseline data only (status='pending')")
    print(f"      Returns calculated only when signal is old enough for each period")
    print("=" * 80)
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(backtest_all())
