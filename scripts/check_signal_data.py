"""Check if signals have composite_score and confidence data."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage import get_database


async def main():
    db = get_database()
    
    # Check latest signals
    print("=" * 80)
    print("Latest signals from database:")
    print("=" * 80)
    
    result = await db.execute_query(
        """
        SELECT 
            ticker,
            overall_score,
            created_at
        FROM signals 
        ORDER BY created_at DESC 
        LIMIT 5
        """
    )
    
    if result:
        for row in result:
            print(f"\nTicker: {row['ticker']}")
            print(f"  overall_score: {row['overall_score']}")
            print(f"  created_at: {row['created_at']}")
    else:
        print("No signals found!")
    
    # Check the performance table structure
    print("\n" + "=" * 80)
    print("Performance table structure (via Supabase query):")
    print("=" * 80)
    
    from supabase import create_client
    import os
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        
        # Try to fetch with nested signals data
        result = supabase.table('performance').select('''
            *,
            signals!inner(
                ticker,
                overall_score
            )
        ''').limit(2).execute()
        
        if result.data:
            print(f"\nFound {len(result.data)} records")
            for row in result.data:
                print(f"\n{row}")
        else:
            print("No data found!")
    else:
        print("Supabase credentials not found!")


if __name__ == "__main__":
    asyncio.run(main())
