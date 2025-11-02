"""Clear old analytics data and prepare for clean backfill"""
import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect(
        "postgresql://postgres.rdkxwoqevjicupmefbem:1qaz1QAZ2wsx2WSX@aws-1-us-east-2.pooler.supabase.com:6543/postgres"
    )
    
    try:
        print("=" * 80)
        print("ANALYTICS TABLE CLEANUP")
        print("=" * 80)
        
        # Check current row count
        count_before = await conn.fetchval("SELECT COUNT(*) FROM analytics;")
        print(f"\nCurrent rows in analytics table: {count_before}")
        
        # Check column count
        col_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = 'analytics';
        """)
        print(f"Current column count: {col_count}")
        
        if count_before > 0:
            print(f"\n⚠️  WARNING: This will DELETE {count_before} existing analytics rows!")
            print("These rows contain old schema data with NULL columns that were removed.")
            print("\nAfter deletion, run the pipeline to backfill with clean data.")
            print("\nProceed? (This script will delete in 3 seconds...)")
            
            # Give user time to cancel
            await asyncio.sleep(3)
            
            # Delete all analytics data
            await conn.execute("DELETE FROM analytics;")
            
            count_after = await conn.fetchval("SELECT COUNT(*) FROM analytics;")
            print(f"\n✅ Analytics table cleared!")
            print(f"Rows before: {count_before}")
            print(f"Rows after: {count_after}")
            print("\n" + "=" * 80)
            print("NEXT STEPS:")
            print("=" * 80)
            print("1. Run the pipeline: python run_pipeline_and_push.py")
            print("2. Phase 7 will now successfully insert analytics data")
            print("3. Frontend will display clean metrics with no NULL columns")
            print("=" * 80)
        else:
            print("\n✅ Analytics table is already empty - ready for fresh data!")
            print("\nNext: Run pipeline to generate analytics")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
