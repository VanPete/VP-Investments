"""Apply migration 020"""
import asyncio
import asyncpg

async def main():
    with open('migrations/020_add_period_columns_for_windows.sql', 'r') as f:
        sql = f.read()
    
    conn = await asyncpg.connect(
        "postgresql://postgres.rdkxwoqevjicupmefbem:1qaz1QAZ2wsx2WSX@aws-1-us-east-2.pooler.supabase.com:6543/postgres",
        statement_cache_size=0
    )
    
    try:
        print("Applying migration 020: Add period columns for time windows...")
        await conn.execute(sql)
        print("✅ Migration applied successfully!")
        
        # Verify
        cols = await conn.fetch("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'analytics' 
            AND column_name IN ('period_type', 'period_start', 'period_end')
            ORDER BY column_name;
        """)
        
        print(f"\nVerified columns added: {[c['column_name'] for c in cols]}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
