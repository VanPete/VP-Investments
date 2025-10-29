"""Apply comprehensive database fixes."""
import asyncio
from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    
    # Read SQL file
    with open('scripts/comprehensive_db_fixes.sql', 'r') as f:
        sql = f.read()
    
    print("\n" + "="*80)
    print("APPLYING DATABASE FIXES")
    print("="*80)
    print("\nExecuting SQL migration...")
    
    try:
        # Execute the SQL
        result = await db.execute_raw_sql(sql)
        print("\n✅ Migration complete!")
        print("\nResults:", result)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
