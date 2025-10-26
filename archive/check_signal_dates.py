import asyncio
from datetime import datetime
from backend.storage.database import SupabaseInterface

async def check():
    db = SupabaseInterface()
    await db.connect()
    
    sigs = await db.execute_query("""
        SELECT created_at, ticker 
        FROM signals 
        WHERE backtest_baseline_price IS NULL
        ORDER BY created_at 
        LIMIT 10
    """)
    
    print(f"Current time: {datetime.now()}")
    print(f"Current UTC: {datetime.utcnow()}\n")
    
    for s in sigs:
        created = s['created_at']
        print(f"{s['ticker']}: {created} (tzinfo: {created.tzinfo})")
        
        # Calculate age different ways
        if hasattr(created, 'tzinfo') and created.tzinfo is not None:
            created_naive = created.replace(tzinfo=None)
        else:
            created_naive = created
            
        age1 = (datetime.now() - created_naive).days
        age2 = (datetime.utcnow() - created_naive).days
        print(f"  Age (local): {age1}d, Age (UTC): {age2}d\n")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check())
