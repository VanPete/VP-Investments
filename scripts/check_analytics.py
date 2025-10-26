"""Quick check of analytics table"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
from backend.storage.database import get_database

async def main():
    db = get_database()
    await db.connect()
    
    # Get latest analytics
    result = db.client.table('analytics').select('*').order('created_at', desc=True).limit(1).execute()
    
    if result.data:
        analytics = result.data[0]
        print(f"✅ Analytics table has data!")
        print(f"   Total signals: {analytics['total_signals']}")
        print(f"   Period: {analytics['period_type']}")
        print(f"   Win rate 1d: {analytics.get('win_rate_1d')}%")
        print(f"   Win rate 30d: {analytics.get('win_rate_30d')}%")
        print(f"   Sharpe 30d: {analytics.get('sharpe_ratio_30d')}")
        print(f"   Top sector: {analytics.get('top_sector')}")
    else:
        print("❌ No analytics data found")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
