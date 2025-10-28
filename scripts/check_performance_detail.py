"""Check performance table status in detail."""
import asyncio
from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    
    # Get detailed performance data
    result = db.client.table('performance').select(
        'id, return_1d, spy_return_1d, intervals_completed, status, baseline_date, signals!inner(ticker)'
    ).order('baseline_date', desc=True).limit(15).execute()
    
    print("\n=== Performance Table Detail ===")
    print(f"Found {len(result.data)} records\n")
    
    for record in result.data:
        ticker = record['signals']['ticker']
        print(f"Ticker: {ticker}")
        print(f"  Status: {record.get('status')}")
        print(f"  Baseline Date: {record['baseline_date']}")
        print(f"  Return 1d: {record.get('return_1d', 'NULL')}")
        print(f"  SPY Return 1d: {record.get('spy_return_1d', 'NULL')}")
        print(f"  Intervals Completed: {record.get('intervals_completed', [])}")
        print()

if __name__ == '__main__':
    asyncio.run(main())
