"""Check detailed signals table structure including JSONB columns"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.storage.database import SupabaseInterface

async def check():
    db = SupabaseInterface()
    
    # Get sample signal to see JSONB structure
    sample = await db.execute_query("""
        SELECT *
        FROM signals
        LIMIT 1
    """)
    
    if sample:
        print("Sample signal columns and types:")
        for key, value in sample[0].items():
            value_type = type(value).__name__
            if isinstance(value, dict):
                print(f"\n{key} ({value_type}):")
                for k, v in value.items():
                    print(f"  {k}: {type(v).__name__} = {v if not isinstance(v, dict) else '...'}")
            else:
                print(f"{key}: {value_type} = {value}")
    
    # Check performance table structure
    print("\n" + "="*80)
    print("PERFORMANCE TABLE STRUCTURE")
    print("="*80)
    
    perf_cols = await db.execute_query("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'performance'
        ORDER BY ordinal_position
        LIMIT 20
    """)
    
    print("\nFirst 20 columns:")
    for col in perf_cols:
        print(f"  {col['column_name']}: {col['data_type']}")
    
    # Check if performance has signal_id
    signal_id_check = await db.execute_query("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'performance' AND column_name = 'signal_id'
    """)
    
    print(f"\nPerformance has signal_id: {len(signal_id_check) > 0}")

if __name__ == "__main__":
    asyncio.run(check())
