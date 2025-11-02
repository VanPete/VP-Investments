"""Quick check of analytics table."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def check_analytics():
    db = SupabaseInterface()
    await db.connect()
    
    try:
        result = await db.execute_query("""
            SELECT 
                id,
                period_type,
                period_start,
                period_end,
                total_signals,
                performance_records_used,
                avg_overall_score,
                cagr,
                top_sector
            FROM analytics 
            ORDER BY period_type
        """)
        
        print(f"\n{'='*80}")
        print(f"ANALYTICS TABLE - {len(result)} rows")
        print(f"{'='*80}\n")
        
        if not result:
            print("❌ No analytics rows found!\n")
            return
        
        for row in result:
            print(f"Period Type: {row['period_type']}")
            print(f"  ID: {row['id']}")
            print(f"  Period: {row['period_start']} to {row['period_end']}")
            print(f"  Signals: {row['total_signals']}")
            print(f"  Performance Records: {row['performance_records_used']}")
            print(f"  Avg Score: {row['avg_overall_score']}")
            print(f"  CAGR: {row['cagr']}")
            print(f"  Top Sector: {row['top_sector']}")
            print()
        
        # Check if we should have more rows
        if len(result) == 1:
            print("⚠️  WARNING: Only 1 analytics row found!")
            print("   Expected: 3 rows (all_time, 90d, 30d)")
            print("   This suggests Phase 7 only completed 1 window calculation.")
            print()
            
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check_analytics())
