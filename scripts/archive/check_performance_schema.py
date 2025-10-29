"""Check performance table schema for v3.2 sector performance columns"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.storage.database import get_supabase_database

async def check_performance_schema():
    db = await get_supabase_database()
    
    print("=" * 100)
    print("PERFORMANCE TABLE - SCHEMA ANALYSIS (v3.2)")
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
        WHERE table_name = 'performance'
        ORDER BY ordinal_position;
    """)
    
    print("\n📋 COLUMNS:")
    print("-" * 100)
    print(f"{'Column Name':<40} {'Type':<20} {'Nullable':<10} {'Default':<20}")
    print("-" * 100)
    
    baseline_cols = []
    interval_cols = []
    spy_cols = []
    sector_cols = []
    
    for col in columns:
        type_str = col['data_type']
        if col['character_maximum_length']:
            type_str += f"({col['character_maximum_length']})"
        elif col['numeric_precision'] and col['numeric_scale']:
            type_str += f"({col['numeric_precision']},{col['numeric_scale']})"
        
        default = col['column_default'] or ''
        if len(default) > 18:
            default = default[:15] + '...'
        
        col_name = col['column_name']
        print(f"{col_name:<40} {type_str:<20} {col['is_nullable']:<10} {default:<20}")
        
        # Categorize columns
        if 'baseline' in col_name:
            baseline_cols.append(col_name)
        elif col_name.startswith('return_'):
            interval_cols.append(col_name)
        elif col_name.startswith('spy_'):
            spy_cols.append(col_name)
        elif col_name.startswith('sector_') or col_name == 'sector':
            sector_cols.append(col_name)
    
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
        WHERE t.relname = 'performance'
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
    
    sector_indexes = []
    for name, info in index_dict.items():
        unique_str = " [UNIQUE]" if info['is_unique'] else ""
        primary_str = " [PRIMARY KEY]" if info['is_primary'] else ""
        columns_str = ", ".join(info['columns'])
        
        print(f"\n{name}{unique_str}{primary_str}")
        print(f"  Columns: {columns_str}")
        if 'WHERE' in info['definition']:
            where_clause = info['definition'].split('WHERE')[1].strip()
            print(f"  Where: {where_clause}")
        
        if 'sector' in name:
            sector_indexes.append(name)
    
    # 3. Check constraints
    constraints = await db.execute_query("""
        SELECT
            con.conname as constraint_name,
            con.contype as constraint_type,
            pg_get_constraintdef(con.oid) as constraint_definition
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        WHERE rel.relname = 'performance'
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
    print("v3.2 SECTOR PERFORMANCE ANALYSIS")
    print("=" * 100)
    
    print(f"\n📍 BASELINE COLUMNS ({len(baseline_cols)}):")
    for col in baseline_cols:
        print(f"   • {col}")
    
    print(f"\n📈 INTERVAL RETURN COLUMNS ({len(interval_cols)}):")
    for col in sorted(interval_cols):
        print(f"   • {col}")
    
    print(f"\n📊 SPY BENCHMARK COLUMNS ({len(spy_cols)}):")
    for col in sorted(spy_cols):
        print(f"   • {col}")
    
    print(f"\n🎯 SECTOR PERFORMANCE COLUMNS ({len(sector_cols)}) - v3.2:")
    if sector_cols:
        print(f"   ✅ Sector columns exist:")
        for col in sorted(sector_cols):
            print(f"      • {col}")
    else:
        print(f"   ❌ No sector columns found - need to run migration 006")
    
    print(f"\n🗂️  SECTOR INDEXES ({len(sector_indexes)}):")
    if sector_indexes:
        print(f"   ✅ Sector indexes exist:")
        for idx in sector_indexes:
            print(f"      • {idx}")
    else:
        print(f"   ❌ No sector indexes found")
    
    # Expected v3.2 columns
    expected_sector_cols = [
        'sector',
        'sector_etf',
        'sector_return_1d', 'sector_return_3d', 'sector_return_7d', 
        'sector_return_10d', 'sector_return_14d', 'sector_return_30d', 'sector_return_90d',
        'sector_alpha_1d', 'sector_alpha_3d', 'sector_alpha_7d',
        'sector_alpha_10d', 'sector_alpha_14d', 'sector_alpha_30d', 'sector_alpha_90d'
    ]
    
    print("\n\n" + "=" * 100)
    print("MIGRATION 006 STATUS")
    print("=" * 100)
    
    missing_cols = [col for col in expected_sector_cols if col not in sector_cols]
    extra_cols = [col for col in sector_cols if col not in expected_sector_cols]
    
    if not missing_cols and not extra_cols:
        print("✅ ALL v3.2 SECTOR COLUMNS PRESENT (16/16)")
        print("   Migration 006 successfully applied!")
    elif missing_cols:
        print(f"⚠️  MISSING COLUMNS ({len(missing_cols)}/16):")
        for col in missing_cols:
            print(f"   • {col}")
        print("\n   Action: Run migration 006")
    
    if extra_cols:
        print(f"\nℹ️  Extra columns found: {extra_cols}")
    
    print("\n" + "=" * 100)
    print("SCHEMA RECOMMENDATIONS")
    print("=" * 100)
    
    if len(sector_cols) == 16 and len(sector_indexes) >= 2:
        print("✅ Schema is ready for v3.2 sector performance tracking")
        print("   - All 16 sector columns present")
        print("   - Sector indexes created")
        print("   - Ready to run pipeline with sector comparison")
    elif len(sector_cols) == 16:
        print("⚠️  Sector columns exist but indexes may be missing")
        print("   - Check if idx_performance_sector and idx_performance_sector_etf exist")
    else:
        print("❌ Migration 006 not applied or incomplete")
        print("   - Run: python scripts/apply_migration_006.py")
    
    # Count total columns
    total_cols = len(columns)
    print(f"\n📊 TOTAL COLUMNS: {total_cols}")
    print(f"   Expected v3.2: ~43 columns (27 original + 16 sector)")

if __name__ == "__main__":
    asyncio.run(check_performance_schema())
