"""Check if alpha columns are generated columns"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.storage.database import get_supabase_database

async def check_generated_columns():
    db = await get_supabase_database()
    
    result = await db.execute_query("""
        SELECT column_name, is_generated, generation_expression 
        FROM information_schema.columns 
        WHERE table_name = 'performance' 
        AND (column_name LIKE 'alpha%' OR column_name LIKE 'sector_alpha%')
        ORDER BY column_name
    """)
    
    print("Alpha Column Generation Status:")
    print("=" * 100)
    for r in result:
        gen = r['is_generated']
        expr = r['generation_expression'] or 'N/A'
        print(f"  {r['column_name']:<30} Generated: {gen:<10} Expression: {expr}")

if __name__ == "__main__":
    asyncio.run(check_generated_columns())
