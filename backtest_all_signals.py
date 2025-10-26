"""
Backtest all signals in the database by calculating historical returns.
This will:
1. Get all signals from database that don't have return data
2. For each signal, fetch historical price data
3. Calculate 1D, 7D, 30D, and 90D returns
4. Update the signals table with the results
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.storage.database import DatabaseService
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

async def calculate_returns(ticker: str, signal_date: datetime):
    """Calculate returns for a signal"""
    try:
        # Use yfinance directly for simplicity
        stock = yf.Ticker(ticker)
        
        # Calculate date ranges
        signal_date_str = signal_date.strftime('%Y-%m-%d')
        future_1d = (signal_date + timedelta(days=1)).strftime('%Y-%m-%d')
        future_7d = (signal_date + timedelta(days=7)).strftime('%Y-%m-%d')
        future_30d = (signal_date + timedelta(days=30)).strftime('%Y-%m-%d')
        future_90d = (signal_date + timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = (signal_date + timedelta(days=100)).strftime('%Y-%m-%d')  # Buffer
        
        # Get historical data with buffer
        hist = stock.history(start=signal_date_str, end=end_date)
        
        if hist.empty or len(hist) < 2:
            logger.warning(f"No price data found for {ticker} from {signal_date_str}")
            return None
            
        # Get signal price (first available price on or after signal date)
        signal_price = hist.iloc[0]['Close']
        
        returns = {}
        
        # Find closest prices for each period
        for days, key in [(1, 'return_1d'), (7, 'return_7d'), (30, 'return_30d'), (90, 'return_90d')]:
            target_date = signal_date + timedelta(days=days)
            
            # Find closest date in historical data
            future_prices = hist[hist.index >= target_date]
            
            if not future_prices.empty:
                future_price = future_prices.iloc[0]['Close']
                returns[key] = ((future_price - signal_price) / signal_price) * 100
            
        return returns
        
    except Exception as e:
        logger.error(f"Error calculating returns for {ticker}: {e}")
        return None

async def backtest_signals():
    """Main function to backtest all signals"""
    db = DatabaseService()
    
    print("=" * 80)
    print("BACKTESTING ALL SIGNALS")
    print("=" * 80)
    
    # Get all signals that don't have complete backtest data
    query = """
    SELECT id, ticker, created_at, return_1d, return_7d, return_30d, return_90d
    FROM signals
    WHERE return_1d IS NULL OR return_7d IS NULL OR return_30d IS NULL OR return_90d IS NULL
    ORDER BY created_at DESC
    """
    
    signals = await db.execute_query(query)
    
    print(f"\nFound {len(signals)} signals to backtest")
    print("-" * 80)
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, signal in enumerate(signals, 1):
        signal_id = signal['id']
        ticker = signal['ticker']
        created_at = signal['created_at']
        
        # Skip if signal is too recent for long-term returns
        days_since_signal = (datetime.now(created_at.tzinfo) - created_at).days
        
        if days_since_signal < 1:
            print(f"[{i}/{len(signals)}] Skipping {ticker} - signal too recent (< 1 day old)")
            skipped_count += 1
            continue
            
        print(f"[{i}/{len(signals)}] Backtesting {ticker} (signal from {created_at.date()}, {days_since_signal} days ago)...", end=" ")
        
        # Calculate returns
        returns = await calculate_returns(ticker, created_at)
        
        if returns is None:
            print("❌ ERROR")
            error_count += 1
            continue
            
        # Update database
        update_parts = []
        params = {'signal_id': signal_id}
        
        if 'return_1d' in returns:
            update_parts.append("return_1d = %(return_1d)s")
            params['return_1d'] = returns['return_1d']
            
        if 'return_7d' in returns:
            update_parts.append("return_7d = %(return_7d)s")
            params['return_7d'] = returns['return_7d']
            
        if 'return_30d' in returns:
            update_parts.append("return_30d = %(return_30d)s")
            params['return_30d'] = returns['return_30d']
            
        if 'return_90d' in returns:
            update_parts.append("return_90d = %(return_90d)s")
            params['return_90d'] = returns['return_90d']
            
        if update_parts:
            update_query = f"""
            UPDATE signals
            SET {', '.join(update_parts)}, updated_at = NOW()
            WHERE id = %(signal_id)s
            """
            
            await db.execute_query(update_query, params)
            
            # Show which returns were calculated
            returns_str = ", ".join([f"{k}={v:.2f}%" for k, v in returns.items()])
            print(f"✓ {returns_str}")
            success_count += 1
        else:
            print("⚠️ No returns calculated")
            error_count += 1
            
        # Rate limiting - don't hammer yfinance
        if i % 10 == 0:
            await asyncio.sleep(2)
            
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print(f"Total signals: {len(signals)}")
    print(f"Successfully backtested: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped (too recent): {skipped_count}")
    print("=" * 80)
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(backtest_signals())
