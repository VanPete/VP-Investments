"""Quick database clear without confirmation"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def clear():
    db = SupabaseInterface()
    await db.connect()
    
    print("🗑️  Clearing all database data...")
    
    # Delete in correct order (foreign key constraints)
    tables = [
        'signals_technical',
        'signals_fundamental',
        'signals_news_macro',
        'signals_social_alternative',
        'signals_risk_stability',
        'signals_institutional_smart_money',
        'performance',
        'analytics',
        'signals',
        'signal_runs'
    ]
    
    for table in tables:
        await db.execute_non_query(f'DELETE FROM {table}')
        print(f'  ✓ Cleared {table}')
    
    print("\n✅ Database cleared successfully")
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(clear())
