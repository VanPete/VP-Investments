"""Quick check: What time are signals created and what should baseline be?"""
import asyncio
from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    await db.connect()
    
    if db.pool:
        async with db.pool.acquire() as conn:
            # Get sample signals with timestamps
            signals = await conn.fetch("""
                SELECT 
                    ticker,
                    created_at,
                    backtest_baseline_price,
                    backtest_baseline_date
                FROM signals
                WHERE backtest_baseline_price IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            print("\nSignal Creation Times vs Baseline Dates:")
            print("="*80)
            for sig in signals:
                created = sig['created_at']
                baseline_date = sig['backtest_baseline_date']
                baseline_price = sig['backtest_baseline_price']
                
                print(f"{sig['ticker']:6} | Created: {created} | Baseline: {baseline_date} | ${baseline_price:.2f}")
                print(f"        Time: {created.time()} | Days diff: {(baseline_date - created).days}")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
