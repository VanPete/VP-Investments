import asyncio
from backend.storage.database import SupabaseInterface

async def check_kvue():
    db = SupabaseInterface()
    
    # Get most recent KVUE signal
    query = """
        SELECT ticker, pe_ratio, eps_growth, dividend_yield, roic, roe, current_price
        FROM signals 
        WHERE ticker = 'KVUE'
        ORDER BY created_at DESC
        LIMIT 1
    """
    
    result = await db.execute_query(query)
    
    if result:
        print("KVUE Data from Database:")
        for key, value in result[0].items():
            print(f"  {key}: {value}")
    else:
        print("No KVUE data found")

asyncio.run(check_kvue())
