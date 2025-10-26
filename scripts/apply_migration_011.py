"""Apply migration 011: Add company_name and current_price to signals table"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.storage.database import SupabaseInterface

async def apply_migration():
    db = SupabaseInterface()
    
    print("=" * 80)
    print("MIGRATION 011: Add company_name and current_price to signals")
    print("=" * 80)
    
    # Read migration SQL
    migration_file = os.path.join(
        os.path.dirname(__file__),
        '../migrations/011_add_company_price_to_signals.sql'
    )
    
    with open(migration_file, 'r') as f:
        sql = f.read()
    
    print("\nExecuting migration...")
    try:
        # Execute migration commands separately
        commands = [
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS company_name TEXT",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS current_price NUMERIC(12, 4)",
            "CREATE INDEX IF NOT EXISTS idx_signals_company_name ON signals(company_name)",
            "COMMENT ON COLUMN signals.company_name IS 'Company full name fetched from yfinance'",
            "COMMENT ON COLUMN signals.current_price IS 'Stock price at signal creation time (matches performance baseline_price)'"
        ]
        
        for cmd in commands:
            await db.execute_query(cmd)
        
        print("✅ Migration applied successfully")
        
        # Verify columns exist
        cols = await db.execute_query("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'signals'
            AND column_name IN ('company_name', 'current_price')
            ORDER BY column_name
        """)
        
        print("\n✅ Verified new columns:")
        for col in cols:
            print(f"   {col['column_name']}: {col['data_type']}")
        
        # Check current data
        count = await db.execute_query("""
            SELECT COUNT(*) as total,
                   COUNT(company_name) as with_name,
                   COUNT(current_price) as with_price
            FROM signals
        """)
        
        if count:
            c = count[0]
            print(f"\n📊 Current data:")
            print(f"   Total signals: {c['total']}")
            print(f"   With company_name: {c['with_name']}")
            print(f"   With current_price: {c['with_price']}")
            print(f"\n   Next pipeline run will populate these fields")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(apply_migration())
