"""Check for detail and factor tables"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.storage.database import SupabaseInterface

async def check():
    db = SupabaseInterface()
    
    # Check for detail tables
    tables = await db.execute_query("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND (table_name LIKE '%detail%' OR table_name LIKE '%factor%')
        ORDER BY table_name
    """)
    
    print("Detail/Factor tables:")
    for t in tables:
        print(f"  {t['table_name']}")
    
    # Also check all public tables
    all_tables = await db.execute_query("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    
    print("\nAll public tables:")
    for t in all_tables:
        print(f"  {t['table_name']}")

if __name__ == "__main__":
    asyncio.run(check())
