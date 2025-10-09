"""
Comprehensive Supabase Schema Inspection Utility

This tool provides complete visibility into the database schema, data quality,
and actionable recommendations for optimization and cleanup.

Usage:
    Interactive mode: python tables.py
    List all tables: python tables.py --list
    Table schema: python tables.py --schema signals
    NULL coverage: python tables.py --nulls signals
    Recommendations: python tables.py --recommend
    Full report: python tables.py --report
    Export: python tables.py --export output.md
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment from project root
project_root = Path(__file__).parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Supabase setup - try multiple key names
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_ANON_KEY') or os.getenv('supabase.anon_key')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Missing SUPABASE_URL or SUPABASE_KEY in environment")
    print(f"   Checked: {env_path}")
    print(f"   SUPABASE_URL present: {bool(SUPABASE_URL)}")
    print(f"   SUPABASE_KEY present: {bool(SUPABASE_KEY)}")
    print(f"   Tried keys: SUPABASE_KEY, SUPABASE_ANON_KEY, supabase.anon_key")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================================
# CORE DATA FETCHING
# ============================================================================

def get_all_tables() -> List[Dict[str, Any]]:
    """Get all tables with row counts."""
    try:
        # Query information_schema for all tables
        result = supabase.rpc('get_table_list').execute()
        
        if not result.data:
            # Fallback: manually query known tables
            known_tables = [
                'signals', 'signal_metrics', 'ai_strategies', 'runs',
                'company_tickers', 'guardrails_config', 'backtest_results',
                'signal_performance', 'signal_scoring_factors'
            ]
            
            tables = []
            for table_name in known_tables:
                try:
                    count_result = supabase.table(table_name).select('*', count='exact').limit(0).execute()
                    tables.append({
                        'table_name': table_name,
                        'row_count': count_result.count if count_result.count else 0
                    })
                except Exception:
                    continue
            
            return tables
        
        return result.data
    
    except Exception:
        # Manual fallback for known tables
        known_tables = [
            'signals', 'signal_metrics', 'ai_strategies', 'runs',
            'company_tickers', 'guardrails_config', 'backtest_results'
        ]
        
        tables = []
        for table_name in known_tables:
            try:
                count_result = supabase.table(table_name).select('*', count='exact').limit(0).execute()
                tables.append({
                    'table_name': table_name,
                    'row_count': count_result.count if count_result.count else 0
                })
            except Exception:
                continue
        
        return tables


def get_table_schema(table_name: str) -> List[Dict[str, Any]]:
    """Get detailed schema for a specific table."""
    try:
        # Get column information from information_schema
        query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        
        result = supabase.rpc('run_sql', {'query': query}).execute()
        
        if result.data:
            return result.data
        
        # Fallback: sample a single row to infer schema
        sample = supabase.table(table_name).select('*').limit(1).execute()
        
        if sample.data and len(sample.data) > 0:
            row = sample.data[0]
            schema = []
            for col_name, value in row.items():
                col_type = type(value).__name__ if value is not None else 'unknown'
                schema.append({
                    'column_name': col_name,
                    'data_type': col_type,
                    'is_nullable': 'YES',
                    'column_default': None
                })
            return schema
        
        return []
    
    except Exception as e:
        # Last resort: just get column names from a sample row
        try:
            sample = supabase.table(table_name).select('*').limit(1).execute()
            if sample.data and len(sample.data) > 0:
                return [{'column_name': col, 'data_type': 'unknown', 'is_nullable': 'YES', 'column_default': None} 
                        for col in sample.data[0].keys()]
        except Exception:
            pass
        
        return []


def analyze_column_nulls(table_name: str, sample_size: int = 1000) -> Dict[str, Dict[str, Any]]:
    """Analyze NULL coverage and data quality for each column."""
    try:
        # Get sample of data
        result = supabase.table(table_name).select('*').limit(sample_size).execute()
        
        if not result.data:
            return {}
        
        total_rows = len(result.data)
        column_stats = defaultdict(lambda: {
            'null_count': 0,
            'null_pct': 0.0,
            'unique_values': set(),
            'sample_values': []
        })
        
        # Analyze each row
        for row in result.data:
            for col_name, value in row.items():
                if value is None:
                    column_stats[col_name]['null_count'] += 1
                else:
                    column_stats[col_name]['unique_values'].add(str(value))
                    if len(column_stats[col_name]['sample_values']) < 5:
                        column_stats[col_name]['sample_values'].append(value)
        
        # Calculate percentages and finalize
        for col_name, stats in column_stats.items():
            stats['null_pct'] = (stats['null_count'] / total_rows) * 100
            stats['unique_count'] = len(stats['unique_values'])
            stats['unique_pct'] = (stats['unique_count'] / total_rows) * 100 if total_rows > 0 else 0
            # Convert set to list for JSON serialization
            stats['unique_values'] = list(stats['unique_values'])[:10]  # Keep max 10 samples
        
        return dict(column_stats)
    
    except Exception as e:
        print(f"⚠️ Warning: Could not analyze {table_name}: {e}")
        return {}


def get_foreign_keys(table_name: str) -> List[Dict[str, str]]:
    """Get foreign key relationships for a table."""
    try:
        query = f"""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = '{table_name}'
        """
        
        result = supabase.rpc('run_sql', {'query': query}).execute()
        return result.data if result.data else []
    
    except Exception:
        return []


# ============================================================================
# ANALYSIS & RECOMMENDATIONS
# ============================================================================

def generate_recommendations(tables_data: List[Dict[str, Any]]) -> List[str]:
    """Generate actionable recommendations for schema optimization."""
    recommendations = []
    
    for table_info in tables_data:
        table_name = table_info['table_name']
        row_count = table_info['row_count']
        
        # Empty tables
        if row_count == 0:
            recommendations.append(f"[X] DROP EMPTY TABLE: {table_name} (0 rows)")
        
        # Analyze columns if data exists
        if row_count > 0:
            column_stats = analyze_column_nulls(table_name, sample_size=min(row_count, 1000))
            
            for col_name, stats in column_stats.items():
                null_pct = stats['null_pct']
                unique_count = stats['unique_count']
                unique_pct = stats['unique_pct']
                
                # 100% NULL columns
                if null_pct == 100.0:
                    recommendations.append(f"[X] DROP NULL COLUMN: {table_name}.{col_name} (100% NULL - no data)")
                
                # 100% same value (low variance)
                elif unique_count == 1 and row_count > 10:
                    sample_val = stats['sample_values'][0] if stats['sample_values'] else 'unknown'
                    recommendations.append(f"[!] CONSTANT COLUMN: {table_name}.{col_name} (100% = {sample_val} - verify calculation)")
                
                # Very low variance (95%+ same value)
                elif unique_pct < 5.0 and row_count > 100:
                    recommendations.append(f"[!] LOW VARIANCE: {table_name}.{col_name} ({unique_pct:.1f}% unique - verify data quality)")
                
                # High NULL rate (>80%)
                elif null_pct > 80.0 and null_pct < 100.0:
                    recommendations.append(f"[!] HIGH NULL RATE: {table_name}.{col_name} ({null_pct:.1f}% NULL - improve data collection)")
            
            # Check for redundant columns
            if table_name == 'signals':
                schema = get_table_schema(table_name)
                col_names = [col['column_name'] for col in schema]
                
                # Redundant score columns
                if 'reddit_score' in col_names and 'weighted_reddit_score' in col_names:
                    recommendations.append(f"[~] REDUNDANT COLUMNS: signals.reddit_score + weighted_reddit_score (combine or remove one)")
                
                # Calculated columns that can be derived
                if 'backtest_eligible' in col_names and 'created_at' in col_names:
                    recommendations.append(f"[~] CALCULATED COLUMN: signals.backtest_eligible (can derive from created_at + interval)")
                
                if 'backtest_intervals' in col_names and 'created_at' in col_names:
                    recommendations.append(f"[~] CALCULATED COLUMN: signals.backtest_intervals (can derive from created_at + time logic)")
            
            # Check for redundant tables
            if table_name == 'backtest_results' and 'signals' in [t['table_name'] for t in tables_data]:
                recommendations.append(f"[X] REDUNDANT TABLE: backtest_results (signals table has all backtest columns)")
            
            if table_name == 'signal_performance' and 'signals' in [t['table_name'] for t in tables_data]:
                recommendations.append(f"[X] REDUNDANT TABLE: signal_performance (signals table has backtest columns)")
    
    return recommendations


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_table_list(tables_data: List[Dict[str, Any]]) -> None:
    """Display formatted list of all tables."""
    print("\n" + "="*80)
    print("DATABASE TABLES")
    print("="*80 + "\n")
    
    if not tables_data:
        print("[!] No tables found or unable to query schema")
        return
    
    # Sort by row count descending
    sorted_tables = sorted(tables_data, key=lambda x: x['row_count'], reverse=True)
    
    print(f"{'TABLE NAME':<30} {'ROW COUNT':>15} {'STATUS':<20}")
    print("-" * 80)
    
    for table in sorted_tables:
        name = table['table_name']
        count = table['row_count']
        
        if count == 0:
            status = "[!] EMPTY"
        elif count < 100:
            status = "[OK] Active (small)"
        elif count < 1000:
            status = "[OK] Active (medium)"
        else:
            status = "[OK] Active (large)"
        
        print(f"{name:<30} {count:>15,} {status:<20}")
    
    print("\n" + "="*80)
    print(f"Total tables: {len(tables_data)}")
    print(f"Active tables: {len([t for t in tables_data if t['row_count'] > 0])}")
    print(f"Empty tables: {len([t for t in tables_data if t['row_count'] == 0])}")
    print("="*80 + "\n")


def display_table_schema(table_name: str) -> None:
    """Display detailed schema for a table."""
    print("\n" + "="*80)
    print(f"SCHEMA: {table_name}")
    print("="*80 + "\n")
    
    schema = get_table_schema(table_name)
    
    if not schema:
        print(f"[!] Could not retrieve schema for {table_name}")
        return
    
    print(f"{'COLUMN':<35} {'TYPE':<20} {'NULLABLE':<10} {'DEFAULT':<15}")
    print("-" * 80)
    
    for col in schema:
        col_name = col['column_name']
        data_type = col.get('data_type', 'unknown')
        nullable = col.get('is_nullable', 'unknown')
        default = col.get('column_default', '-')
        
        if default and len(str(default)) > 12:
            default = str(default)[:12] + "..."
        
        print(f"{col_name:<35} {data_type:<20} {nullable:<10} {str(default):<15}")
    
    print("\n" + "="*80)
    print(f"Total columns: {len(schema)}")
    print("="*80 + "\n")


def display_null_analysis(table_name: str) -> None:
    """Display NULL coverage analysis for a table."""
    print("\n" + "="*80)
    print(f"NULL COVERAGE ANALYSIS: {table_name}")
    print("="*80 + "\n")
    
    # Get row count
    count_result = supabase.table(table_name).select('*', count='exact').limit(0).execute()
    total_rows = count_result.count if count_result.count else 0
    
    print(f"Total rows: {total_rows:,}\n")
    
    if total_rows == 0:
        print("[!] Table is empty - no data to analyze")
        return
    
    column_stats = analyze_column_nulls(table_name, sample_size=min(total_rows, 1000))
    
    if not column_stats:
        print("[!] Could not analyze column data")
        return
    
    # Sort by NULL percentage descending
    sorted_stats = sorted(column_stats.items(), key=lambda x: x[1]['null_pct'], reverse=True)
    
    print(f"{'COLUMN':<35} {'NULL %':<10} {'UNIQUE':<10} {'STATUS':<25}")
    print("-" * 80)
    
    for col_name, stats in sorted_stats:
        null_pct = stats['null_pct']
        unique_count = stats['unique_count']
        
        # Status indicator (ASCII for Windows compatibility)
        if null_pct == 100.0:
            status = "[X] 100% NULL"
        elif null_pct > 80.0:
            status = "[!] High NULL rate"
        elif null_pct > 50.0:
            status = "[!] Moderate NULL rate"
        elif unique_count == 1:
            status = "[!] Constant value"
        elif stats['unique_pct'] < 5.0 and total_rows > 100:
            status = "[!] Low variance"
        else:
            status = "[OK] Good"
        
        print(f"{col_name:<35} {null_pct:>6.1f}%   {unique_count:>8,}   {status:<25}")
    
    print("\n" + "="*80)
    null_columns = len([s for s in column_stats.values() if s['null_pct'] == 100.0])
    high_null = len([s for s in column_stats.values() if s['null_pct'] > 80.0 and s['null_pct'] < 100.0])
    print(f"100% NULL columns: {null_columns}")
    print(f"High NULL (>80%): {high_null}")
    print("="*80 + "\n")


def display_recommendations(tables_data: List[Dict[str, Any]]) -> None:
    """Display all recommendations."""
    print("\n" + "="*80)
    print("SCHEMA OPTIMIZATION RECOMMENDATIONS")
    print("="*80 + "\n")
    
    recommendations = generate_recommendations(tables_data)
    
    if not recommendations:
        print("[OK] No issues found - schema looks good!")
        return
    
    # Group by type
    drop_tables = [r for r in recommendations if 'DROP EMPTY TABLE' in r or 'REDUNDANT TABLE' in r]
    drop_columns = [r for r in recommendations if 'DROP NULL COLUMN' in r]
    data_quality = [r for r in recommendations if 'HIGH NULL RATE' in r or 'CONSTANT COLUMN' in r or 'LOW VARIANCE' in r]
    redundant = [r for r in recommendations if 'REDUNDANT COLUMNS' in r or 'CALCULATED COLUMN' in r]
    
    if drop_tables:
        print("[DROP] TABLES TO DROP:")
        for rec in drop_tables:
            print(f"   {rec}")
        print()
    
    if drop_columns:
        print("[DROP] COLUMNS TO DROP:")
        for rec in drop_columns:
            print(f"   {rec}")
        print()
    
    if data_quality:
        print("[WARN] DATA QUALITY ISSUES:")
        for rec in data_quality:
            print(f"   {rec}")
        print()
    
    if redundant:
        print("[INFO] REDUNDANCY ISSUES:")
        for rec in redundant:
            print(f"   {rec}")
        print()
    
    print("="*80)
    print(f"Total recommendations: {len(recommendations)}")
    print("="*80 + "\n")


def generate_full_report() -> str:
    """Generate comprehensive report of entire database."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("SUPABASE SCHEMA REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Get all tables
    tables_data = get_all_tables()
    
    # Summary
    report_lines.append("## SUMMARY")
    report_lines.append("")
    report_lines.append(f"Total tables: {len(tables_data)}")
    report_lines.append(f"Active tables: {len([t for t in tables_data if t['row_count'] > 0])}")
    report_lines.append(f"Empty tables: {len([t for t in tables_data if t['row_count'] == 0])}")
    report_lines.append("")
    
    # Table list
    report_lines.append("## TABLES")
    report_lines.append("")
    sorted_tables = sorted(tables_data, key=lambda x: x['row_count'], reverse=True)
    for table in sorted_tables:
        status = "EMPTY" if table['row_count'] == 0 else f"{table['row_count']:,} rows"
        report_lines.append(f"- {table['table_name']}: {status}")
    report_lines.append("")
    
    # Recommendations
    recommendations = generate_recommendations(tables_data)
    if recommendations:
        report_lines.append("## RECOMMENDATIONS")
        report_lines.append("")
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        report_lines.append("")
    
    report_lines.append("="*80)
    
    return "\n".join(report_lines)


# ============================================================================
# IMPORTABLE FUNCTIONS (for use in other scripts)
# ============================================================================

def check_table_exists(table_name: str) -> bool:
    """Check if a table exists."""
    try:
        supabase.table(table_name).select('*').limit(0).execute()
        return True
    except Exception:
        return False


def get_row_count(table_name: str) -> int:
    """Get row count for a table."""
    try:
        result = supabase.table(table_name).select('*', count='exact').limit(0).execute()
        return result.count if result.count else 0
    except Exception:
        return 0


def get_column_names(table_name: str) -> List[str]:
    """Get list of column names for a table."""
    schema = get_table_schema(table_name)
    return [col['column_name'] for col in schema]


def check_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    columns = get_column_names(table_name)
    return column_name in columns


# ============================================================================
# INTERACTIVE MENU
# ============================================================================

def show_menu():
    """Display interactive menu."""
    print("\n" + "="*80)
    print("SUPABASE SCHEMA INSPECTOR")
    print("="*80)
    print("\n1. List all tables")
    print("2. Show table schema")
    print("3. Analyze NULL coverage")
    print("4. Get recommendations")
    print("5. Generate full report")
    print("6. Export report to file")
    print("0. Exit")
    print("\n" + "="*80)


def interactive_mode():
    """Run in interactive mode."""
    while True:
        show_menu()
        choice = input("\nSelect option: ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        
        elif choice == '1':
            tables_data = get_all_tables()
            display_table_list(tables_data)
        
        elif choice == '2':
            table_name = input("\nEnter table name: ").strip()
            if table_name:
                display_table_schema(table_name)
        
        elif choice == '3':
            table_name = input("\nEnter table name: ").strip()
            if table_name:
                display_null_analysis(table_name)
        
        elif choice == '4':
            tables_data = get_all_tables()
            display_recommendations(tables_data)
        
        elif choice == '5':
            report = generate_full_report()
            print("\n" + report)
        
        elif choice == '6':
            filename = input("\nEnter output filename (default: schema_report.md): ").strip()
            if not filename:
                filename = "schema_report.md"
            
            report = generate_full_report()
            with open(filename, 'w') as f:
                f.write(report)
            
            print(f"\n[OK] Report exported to {filename}")
        
        else:
            print("\n[X] Invalid option")
        
        input("\nPress Enter to continue...")


# ============================================================================
# CLI MODE
# ============================================================================

def cli_mode(args):
    """Run in CLI mode with arguments."""
    if '--list' in args:
        tables_data = get_all_tables()
        display_table_list(tables_data)
    
    elif '--schema' in args:
        idx = args.index('--schema')
        if idx + 1 < len(args):
            table_name = args[idx + 1]
            display_table_schema(table_name)
        else:
            print("❌ ERROR: --schema requires a table name")
    
    elif '--nulls' in args:
        idx = args.index('--nulls')
        if idx + 1 < len(args):
            table_name = args[idx + 1]
            display_null_analysis(table_name)
        else:
            print("❌ ERROR: --nulls requires a table name")
    
    elif '--recommend' in args:
        tables_data = get_all_tables()
        display_recommendations(tables_data)
    
    elif '--report' in args:
        report = generate_full_report()
        print("\n" + report)
    
    elif '--export' in args:
        idx = args.index('--export')
        filename = args[idx + 1] if idx + 1 < len(args) else 'schema_report.md'
        
        report = generate_full_report()
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"\n[OK] Report exported to {filename}")
    
    elif '--help' in args or '-h' in args:
        print(__doc__)
    
    else:
        print("[X] ERROR: Unknown argument")
        print("\nUse --help for usage information")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # CLI mode
        cli_mode(sys.argv[1:])
    else:
        # Interactive mode
        interactive_mode()
