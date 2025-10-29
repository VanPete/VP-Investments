"""Apply migration 013: Add QQQ benchmark columns."""
import asyncio
from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    
    print("\n=== Applying Migration 013: Add QQQ Benchmark Columns ===\n")
    
    # Read migration SQL
    with open('migrations/013_add_qqq_benchmark.sql', 'r') as f:
        sql = f.read()
    
    # Execute migration
    try:
        # Note: Supabase client doesn't support raw SQL execution directly
        # Need to use the REST API or apply via Supabase dashboard
        print("❌ Cannot apply SQL migrations via Supabase Python client")
        print("📋 Please apply this migration manually via Supabase dashboard:")
        print("\n" + "="*80)
        print(sql)
        print("="*80)
        print("\n✅ After applying the migration, run:")
        print("   python scripts/fix_benchmark_returns.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())
