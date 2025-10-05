#!/usr/bin/env python3
"""
Check existing schema of signal_metrics and signal_performance tables
"""

import asyncio
from backend.storage.database import SupabaseInterface


async def check_schema():
    db = SupabaseInterface()
    await db.connect()
    
    print("=" * 80)
    print("CHECKING EXISTING TABLE SCHEMAS")
    print("=" * 80)
    
    # Check signal_metrics
    print("\n📊 signal_metrics columns:")
    print("-" * 80)
    query1 = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public' 
    AND table_name = 'signal_metrics'
    ORDER BY ordinal_position;
    """
    result = await db.pool.fetch(query1)
    for i, row in enumerate(result, 1):
        print(f"{i:3}. {row['column_name']:30} | {row['data_type']:20} | NULL: {row['is_nullable']}")
    print(f"\nTotal: {len(result)} columns")
    
    # Check signal_performance
    print("\n\n📈 signal_performance columns:")
    print("-" * 80)
    query2 = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public' 
    AND table_name = 'signal_performance'
    ORDER BY ordinal_position;
    """
    result = await db.pool.fetch(query2)
    for i, row in enumerate(result, 1):
        print(f"{i:3}. {row['column_name']:30} | {row['data_type']:20} | NULL: {row['is_nullable']}")
    print(f"\nTotal: {len(result)} columns")
    
    # Check for data
    print("\n\n📊 Row Counts:")
    print("-" * 80)
    count1 = await db.pool.fetchval("SELECT COUNT(*) FROM signals")
    count2 = await db.pool.fetchval("SELECT COUNT(*) FROM signal_metrics")
    count3 = await db.pool.fetchval("SELECT COUNT(*) FROM signal_performance")
    print(f"signals:             {count1:6} rows")
    print(f"signal_metrics:      {count2:6} rows (should match signals)")
    print(f"signal_performance:  {count3:6} rows")
    
    if count2 < count1:
        print(f"\n⚠️  WARNING: signal_metrics has {count1 - count2} fewer rows than signals!")
    
    if count3 == 0:
        print(f"\n⚠️  WARNING: signal_performance is EMPTY - need to populate!")
    
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(check_schema())
