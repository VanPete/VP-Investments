"""Quick check for Phase 1 data in most recent pipeline run"""
from backend.storage.database import SupabaseInterface
import asyncio

async def check():
    db = SupabaseInterface()
    await db.connect()
    
    # Get any recent signal to check structure
    result = db.supabase.from_('signals') \
        .select('ticker, run_id, market_cap_category, expected_hold_duration, float_turnover_ratio, momentum_consistency_score, volume_price_correlation, earnings_date') \
        .order('created_at', desc=True) \
        .limit(10) \
        .execute()
    
    print(f"\nFound {len(result.data)} recent signals:\n")
    for s in result.data:
        print(f"{s['ticker']} (run: {s['run_id']})")
        print(f"  Market Cap Category: {s.get('market_cap_category')}")
        print(f"  Expected Hold Duration: {s.get('expected_hold_duration')}")
        print(f"  Float Turnover: {s.get('float_turnover_ratio')}")
        print(f"  Momentum Consistency: {s.get('momentum_consistency_score')}")
        print(f"  Volume Price Corr: {s.get('volume_price_correlation')}")
        print(f"  Earnings Date: {s.get('earnings_date')}")
        print()

if __name__ == '__main__':
    asyncio.run(check())
