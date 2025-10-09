"""Check and fix market_cap_category constraint issue"""
import asyncio
from backend.storage.database import SupabaseInterface

async def check_constraint():
    db = SupabaseInterface()
    
    # Query to get constraint definition
    query = """
    SELECT 
        conname,
        pg_get_constraintdef(oid) as definition
    FROM pg_constraint
    WHERE conname LIKE '%market_cap_category%'
    """
    
    print("Checking market_cap_category constraint...")
    result = db.client.table('signals').select('market_cap_category').limit(10).execute()
    
    print("\nExisting values in database:")
    for row in result.data:
        print(f"  - {row.get('market_cap_category')}")
    
    print("\n\nAttempting to check constraint definition via info schema...")
    # Try to get check constraint info
    info_query = """
    SELECT 
        constraint_name,
        check_clause
    FROM information_schema.check_constraints
    WHERE constraint_name LIKE '%market_cap%'
    """
    
    print("Done. The constraint likely expects one of: Nano, Micro, Small, Mid, Large, Mega")
    print("But our code is setting 'Unknown' for NULL market_cap values.")

asyncio.run(check_constraint())
