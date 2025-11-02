"""Apply migration 019 to remove unused analytics columns"""
import asyncio
import asyncpg

async def main():
    # Read migration file
    with open('migrations/019_cleanup_analytics_unused_columns.sql', 'r') as f:
        sql = f.read()
    
    # Connect to database
    conn = await asyncpg.connect(
        "postgresql://postgres.rdkxwoqevjicupmefbem:1qaz1QAZ2wsx2WSX@aws-1-us-east-2.pooler.supabase.com:6543/postgres"
    )
    
    try:
        print("Applying migration 019: Remove unused analytics columns...")
        print("=" * 80)
        
        # Execute migration
        await conn.execute(sql)
        
        print("✅ Migration applied successfully!")
        print("\nVerifying schema...")
        
        # Count columns after migration
        count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = 'analytics';
        """)
        
        print(f"Analytics table now has {count} columns (reduced from 82)")
        
        # List remaining columns
        columns = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns 
            WHERE table_name = 'analytics' 
            ORDER BY ordinal_position;
        """)
        
        print("\nRemaining columns:")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col['column_name']:<35} {col['data_type']}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
