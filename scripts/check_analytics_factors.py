"""Check analytics table for factor data"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.storage.database import SupabaseInterface

async def check():
    db = SupabaseInterface()
    
    # Check all columns first
    all_cols = await db.execute_query("""
        SELECT column_name, data_type
        FROM information_schema.columns 
        WHERE table_name = 'analytics'
        ORDER BY ordinal_position
        LIMIT 10
    """)
    
    print("First 10 columns in analytics table:")
    for r in all_cols:
        print(f"  {r['column_name']}: {r['data_type']}")
    
    # Check for factor columns
    result = await db.execute_query("""
        SELECT column_name, data_type
        FROM information_schema.columns 
        WHERE table_name = 'analytics' 
        AND (column_name LIKE '%factor%' OR column_name = 'top_factors')
        ORDER BY ordinal_position
    """)
    
    print("\nFactor-related columns in analytics table:")
    for r in result:
        print(f"  {r['column_name']}: {r['data_type']}")
    
    # Check sample data
    sample = await db.execute_query("""
        SELECT *
        FROM analytics
        LIMIT 1
    """)
    
    if sample:
        print(f"\nSample data columns:")
        print(", ".join(sample[0].keys()))
        if 'top_factors' in sample[0]:
            import json
            print(f"\nSample top_factors:")
            print(json.dumps(sample[0]['top_factors'], indent=2))

if __name__ == "__main__":
    asyncio.run(check())
