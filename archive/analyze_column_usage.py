"""Analyze which columns in signals table are actually being used"""
from backend.storage.database import SupabaseInterface
import asyncio

async def get_all_columns():
    db = SupabaseInterface()
    await db.connect()
    
    # Get one signal to see all columns
    result = db.supabase.from_('signals') \
        .select('*') \
        .limit(1) \
        .execute()
    
    if result.data:
        columns = list(result.data[0].keys())
        columns.sort()
        
        print(f"\n{'='*80}")
        print(f"SIGNALS TABLE COLUMNS ({len(columns)} total)")
        print(f"{'='*80}\n")
        
        for i, col in enumerate(columns, 1):
            print(f"{i:3d}. {col}")
        
        return columns
    
    return []

if __name__ == '__main__':
    asyncio.run(get_all_columns())
