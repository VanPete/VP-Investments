"""Quick script to check SPY returns in performance table."""
import asyncio
from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    
    # Get sample performance data with signal ticker
    result = db.client.table('performance').select(
        'id, return_1d, spy_return_1d, return_7d, spy_return_7d, baseline_date, signals!inner(ticker)'
    ).limit(10).execute()
    
    print("\n=== Sample Performance Data ===")
    print(f"Found {len(result.data)} records\n")
    
    for record in result.data:
        ticker = record['signals']['ticker']
        print(f"Ticker: {ticker}")
        print(f"  Baseline Date: {record['baseline_date']}")
        print(f"  Return 1d: {record.get('return_1d', 'NULL')}")
        print(f"  SPY Return 1d: {record.get('spy_return_1d', 'NULL')}")
        print(f"  Return 7d: {record.get('return_7d', 'NULL')}")
        print(f"  SPY Return 7d: {record.get('spy_return_7d', 'NULL')}")
        print()

if __name__ == '__main__':
    asyncio.run(main())
