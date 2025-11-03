"""Check IC series data from analytics table."""

import json
from backend.storage.database import get_database

def main():
    db = get_database()
    
    # Get analytics records
    result = db.client.table('analytics').select(
        'id, period_start, period_end, ic_mean, ic_series'
    ).order('created_at', desc=True).limit(5).execute()
    
    print("=" * 80)
    print(f"ANALYTICS TABLE - IC DATA")
    print("=" * 80)
    
    if result.data:
        print(f"\nFound {len(result.data)} records\n")
        
        for i, row in enumerate(result.data, 1):
            print(f"\n[{i}] Period: {row.get('period_start')} to {row.get('period_end')}")
            print(f"    IC Mean: {row.get('ic_mean')}")
            
            ic_series = row.get('ic_series')
            if ic_series:
                print(f"    IC Series length: {len(ic_series)}")
                if len(ic_series) > 0:
                    print(f"    First entry: {ic_series[0]}")
                    print(f"    Last entry: {ic_series[-1]}")
            else:
                print(f"    IC Series: None")
    else:
        print("\n❌ No analytics data found")
    
    print("\n" + "=" * 80)
    
    db.close()

if __name__ == '__main__':
    main()
