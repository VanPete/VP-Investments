"""Check structure of signals_technical table"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.storage.database import SupabaseInterface
import json

async def check():
    db = SupabaseInterface()
    
    # Check columns
    cols = await db.execute_query("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'signals_technical'
        ORDER BY ordinal_position
    """)
    
    print("signals_technical columns:")
    for col in cols:
        print(f"  {col['column_name']}: {col['data_type']}")
    
    # Get sample data
    sample = await db.execute_query("""
        SELECT *
        FROM signals_technical
        LIMIT 1
    """)
    
    if sample:
        print("\nSample data:")
        for key, value in sample[0].items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                print(json.dumps(value, indent=2)[:500])
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(check())
