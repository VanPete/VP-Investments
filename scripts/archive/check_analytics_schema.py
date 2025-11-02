"""Check analytics table schema and data"""
import asyncio
import os
from dotenv import load_dotenv
import asyncpg

load_dotenv()

async def main():
    # Connect to database
    conn = await asyncpg.connect(
        "postgresql://postgres.rdkxwoqevjicupmefbem:1qaz1QAZ2wsx2WSX@aws-1-us-east-2.pooler.supabase.com:6543/postgres"
    )
    
    try:
        # Get column info
        print("=" * 80)
        print("ANALYTICS TABLE SCHEMA")
        print("=" * 80)
        
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'analytics' 
            ORDER BY ordinal_position;
        """)
        
        print(f"\nTotal columns: {len(columns)}\n")
        
        for i, col in enumerate(columns, 1):
            print(f"{i:2d}. {col['column_name']:<30} {col['data_type']:<20} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
        
        # Check for data
        print("\n" + "=" * 80)
        print("ANALYTICS TABLE DATA")
        print("=" * 80)
        
        count = await conn.fetchval("SELECT COUNT(*) FROM analytics;")
        print(f"\nTotal rows: {count}")
        
        if count > 0:
            # Get latest row
            latest = await conn.fetchrow("""
                SELECT run_id, total_signals, avg_overall_score, created_at
                FROM analytics 
                ORDER BY created_at DESC 
                LIMIT 1;
            """)
            print(f"\nLatest analytics:")
            print(f"  Run ID: {latest['run_id']}")
            print(f"  Total Signals: {latest['total_signals']}")
            print(f"  Avg Score: {latest['avg_overall_score']:.3f}")
            print(f"  Created: {latest['created_at']}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
