"""Check analytics table constraints."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def check_constraints():
    db = SupabaseInterface()
    await db.connect()
    
    try:
        # Check constraints on analytics table
        result = await db.execute_query("""
            SELECT 
                conname as constraint_name,
                contype as constraint_type,
                pg_get_constraintdef(oid) as definition
            FROM pg_constraint
            WHERE conrelid = 'analytics'::regclass
            ORDER BY conname
        """)
        
        print(f"\n{'='*80}")
        print("ANALYTICS TABLE CONSTRAINTS")
        print(f"{'='*80}\n")
        
        for row in result:
            ctype = {
                'p': 'PRIMARY KEY',
                'u': 'UNIQUE',
                'f': 'FOREIGN KEY',
                'c': 'CHECK'
            }.get(row['constraint_type'], row['constraint_type'])
            
            print(f"{row['constraint_name']}: {ctype}")
            print(f"  {row['definition']}\n")
            
        # Check indexes
        print(f"{'='*80}")
        print("ANALYTICS TABLE INDEXES")
        print(f"{'='*80}\n")
        
        result = await db.execute_query("""
            SELECT 
                indexname,
                indexdef
            FROM pg_indexes
            WHERE tablename = 'analytics'
            ORDER BY indexname
        """)
        
        for row in result:
            print(f"{row['indexname']}:")
            print(f"  {row['indexdef']}\n")
            
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check_constraints())
