"""Check complete signals table schema including indexes and constraints"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.storage.database import SupabaseInterface

async def check_full_schema():
    db = SupabaseInterface()
    
    print("=" * 100)
    print("SIGNALS TABLE - COMPLETE SCHEMA ANALYSIS")
    print("=" * 100)
    
    # 1. Check all columns
    columns = await db.execute_query("""
        SELECT 
            column_name, 
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = 'signals'
        ORDER BY ordinal_position;
    """)
    
    print("\n📋 COLUMNS:")
    print("-" * 100)
    print(f"{'Column Name':<40} {'Type':<20} {'Precision':<12} {'Nullable':<10} {'Default':<20}")
    print("-" * 100)
    
    coverage_cols = []
    backtest_cols = []
    
    for col in columns:
        type_str = col['data_type']
        if col['character_maximum_length']:
            type_str += f"({col['character_maximum_length']})"
        elif col['numeric_precision'] and col['numeric_scale']:
            type_str += f"({col['numeric_precision']},{col['numeric_scale']})"
        
        default = col['column_default'] or ''
        if len(default) > 18:
            default = default[:15] + '...'
        
        print(f"{col['column_name']:<40} {type_str:<20} {str(col['numeric_precision'] or ''):<12} {col['is_nullable']:<10} {default:<20}")
        
        if 'coverage' in col['column_name']:
            coverage_cols.append(col['column_name'])
        if any(x in col['column_name'] for x in ['backtest', 'return_', 'spy_']):
            backtest_cols.append(col['column_name'])
    
    # 2. Check indexes
    indexes = await db.execute_query("""
        SELECT
            i.relname as index_name,
            a.attname as column_name,
            ix.indisunique as is_unique,
            ix.indisprimary as is_primary,
            pg_get_indexdef(ix.indexrelid) as index_definition
        FROM pg_class t
        JOIN pg_index ix ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
        WHERE t.relname = 'signals'
        ORDER BY i.relname, a.attnum;
    """)
    
    print("\n\n📊 INDEXES:")
    print("-" * 100)
    
    # Group by index name
    index_dict = {}
    for idx in indexes:
        name = idx['index_name']
        if name not in index_dict:
            index_dict[name] = {
                'columns': [],
                'is_unique': idx['is_unique'],
                'is_primary': idx['is_primary'],
                'definition': idx['index_definition']
            }
        index_dict[name]['columns'].append(idx['column_name'])
    
    backtest_indexes = []
    for name, info in index_dict.items():
        unique_str = " [UNIQUE]" if info['is_unique'] else ""
        primary_str = " [PRIMARY KEY]" if info['is_primary'] else ""
        columns_str = ", ".join(info['columns'])
        
        print(f"\n{name}{unique_str}{primary_str}")
        print(f"  Columns: {columns_str}")
        if 'WHERE' in info['definition']:
            where_clause = info['definition'].split('WHERE')[1].strip()
            print(f"  Where: {where_clause}")
        
        if any(x in name for x in ['backtest', 'return', 'spy', 'performance', 'baseline']):
            backtest_indexes.append(name)
    
    # 3. Check constraints
    constraints = await db.execute_query("""
        SELECT
            con.conname as constraint_name,
            con.contype as constraint_type,
            pg_get_constraintdef(con.oid) as constraint_definition
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        WHERE rel.relname = 'signals'
        ORDER BY con.conname;
    """)
    
    print("\n\n🔒 CONSTRAINTS:")
    print("-" * 100)
    
    constraint_types = {
        'p': 'PRIMARY KEY',
        'f': 'FOREIGN KEY',
        'u': 'UNIQUE',
        'c': 'CHECK'
    }
    
    for con in constraints:
        con_type = constraint_types.get(con['constraint_type'], con['constraint_type'])
        print(f"\n{con['constraint_name']} [{con_type}]")
        print(f"  {con['constraint_definition']}")
    
    # 4. Summary
    print("\n\n" + "=" * 100)
    print("MIGRATION IMPACT ANALYSIS")
    print("=" * 100)
    
    print(f"\n✅ COVERAGE COLUMNS (TO BE ADDED):")
    if coverage_cols:
        print(f"   ⚠️  Already exist: {len(coverage_cols)} columns")
        for col in coverage_cols:
            print(f"      • {col}")
    else:
        print(f"   ✓ None exist - will add 7 new columns")
    
    print(f"\n❌ BACKTEST COLUMNS (TO BE REMOVED):")
    if backtest_cols:
        print(f"   Found {len(backtest_cols)} columns to remove:")
        for col in backtest_cols:
            print(f"      • {col}")
    else:
        print(f"   ✓ None exist - already clean")
    
    print(f"\n🗑️  BACKTEST INDEXES (TO BE DROPPED):")
    if backtest_indexes:
        print(f"   Found {len(backtest_indexes)} indexes to drop:")
        for idx in backtest_indexes:
            print(f"      • {idx}")
    else:
        print(f"   ✓ None exist - already clean")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    
    if coverage_cols and backtest_cols:
        print("⚠️  WARNING: Coverage columns already exist AND backtest columns exist")
        print("   Suggested action: Modify migration to handle existing coverage columns")
    elif coverage_cols:
        print("⚠️  Coverage columns already exist (migration may need adjustment)")
    elif backtest_cols:
        print("✅ Safe to run migration - will add coverage and remove backtest columns")
    else:
        print("ℹ️  Both coverage and backtest columns missing - verify table state")

if __name__ == "__main__":
    asyncio.run(check_full_schema())
