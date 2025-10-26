"""Check all backtest-related columns in signals table"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backend.storage.database import SupabaseInterface

async def check_schema():
    db = SupabaseInterface()
    await db.connect()
    
    # Get all columns
    query = """
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'signals'
    ORDER BY ordinal_position
    """
    
    result = await db.execute_query(query)
    
    print("=" * 80)
    print("ALL SIGNALS TABLE COLUMNS")
    print("=" * 80)
    
    backtest_cols = []
    return_cols = []
    other_cols = []
    
    for r in result:
        col = r['column_name']
        dtype = r['data_type']
        
        if 'backtest' in col:
            backtest_cols.append(f"{col}: {dtype}")
        elif 'return' in col:
            return_cols.append(f"{col}: {dtype}")
        else:
            other_cols.append(f"{col}: {dtype}")
    
    print("\nBACKTEST COLUMNS:")
    for col in backtest_cols:
        print(f"  {col}")
    
    print("\nRETURN COLUMNS:")
    for col in return_cols:
        print(f"  {col}")
    
    print(f"\nOther columns: {len(other_cols)}")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check_schema())
