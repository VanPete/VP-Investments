#!/usr/bin/env python3
"""
VP Investments - Comprehensive Database Schema Analysis Tool

Analyzes all database tables and columns to identify:
- NULL values and their percentages
- Completely empty columns (100% NULL)
- Empty tables (0 rows)
- Data quality issues requiring backend fixes
- Schema optimization opportunities

Usage:
    python tables.py                          # Analyze all tables
    python tables.py --table signals          # Analyze specific table
    python tables.py --gaps-only              # Show only critical issues
    python tables.py --empty-only             # Show only empty tables/columns
    python tables.py --detailed               # Show detailed column statistics
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import argparse
from collections import defaultdict

from backend.storage.database import SupabaseInterface


class ComprehensiveDatabaseAnalyzer:
    """Comprehensive database analysis tool for schema and data quality."""
    
    def __init__(self):
        self.db = SupabaseInterface()
        
    async def connect(self):
        """Initialize database connection."""
        await self.db.connect()
        
    async def disconnect(self):
        """Close database connection."""
        await self.db.disconnect()
        
    async def get_all_tables(self) -> List[str]:
        """Get list of all tables in the public schema."""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        
        try:
            if self.db.pool:
                async with self.db.pool.acquire() as conn:
                    rows = await conn.fetch(query)
                    return [row['table_name'] for row in rows]
        except Exception as e:
            print(f"⚠️  Could not fetch tables from database: {e}")
            # Fallback to known tables
            return [
                'ai_strategies', 'ai_strategy_performance', 'backtest_trades', 'backtests',
                'company_tickers', 'guardrails_config', 'market_conditions', 'runs',
                'signal_calibration_log', 'signal_performance', 'signal_performance_history',
                'signal_scoring_factors', 'signals', 'signals_norm'
            ]
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a table's structure and data."""
        print(f"🔍 Analyzing: {table_name}")
        
        try:
            if not self.db.pool:
                return {'table_name': table_name, 'error': 'No database connection'}
                
            async with self.db.pool.acquire() as conn:
                # Get row count
                count_result = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                total_rows = count_result or 0
                
                # Get table size
                size_query = """
                SELECT pg_size_pretty(pg_total_relation_size($1)) as size,
                       pg_total_relation_size($1) as size_bytes
                """
                size_result = await conn.fetchrow(size_query, table_name)
                
                # Get all columns with detailed metadata
                columns_query = """
                SELECT 
                    column_name, 
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = $1
                ORDER BY ordinal_position;
                """
                columns = await conn.fetch(columns_query, table_name)
                
                if total_rows == 0:
                    return {
                        'table_name': table_name,
                        'total_rows': 0,
                        'status': 'EMPTY_TABLE',
                        'columns': [{'name': col['column_name'], 'type': col['data_type']} for col in columns],
                        'column_count': len(columns),
                        'size': size_result['size'] if size_result else 'N/A',
                        'implementation_priority': self._determine_table_priority(table_name)
                    }
                
                # Build query to analyze all columns in one shot
                column_analysis = []
                for col in columns:
                    col_name = col['column_name']
                    col_type = col['data_type']
                    # Properly quote column names
                    quoted_col = f'"{col_name}"'
                    
                    # Build analysis based on column type
                    base_analysis = f"""
                        'column_name', '{col_name}',
                        'null_count', COUNT(*) - COUNT({quoted_col}),
                        'null_percentage', ROUND((COUNT(*) - COUNT({quoted_col})) * 100.0 / COUNT(*), 2),
                        'populated_count', COUNT({quoted_col}),
                        'distinct_count', COUNT(DISTINCT {quoted_col})
                    """
                    
                    # Add zero/invalid checks for numeric columns
                    if col_type in ['integer', 'bigint', 'smallint', 'numeric', 'real', 'double precision', 'decimal']:
                        extra_checks = f""",
                        'zero_count', COUNT(CASE WHEN {quoted_col} = 0 THEN 1 END),
                        'zero_percentage', ROUND(COUNT(CASE WHEN {quoted_col} = 0 THEN 1 END) * 100.0 / COUNT(*), 2),
                        'negative_count', COUNT(CASE WHEN {quoted_col} < 0 THEN 1 END),
                        'min_value', MIN({quoted_col}),
                        'max_value', MAX({quoted_col}),
                        'avg_value', AVG({quoted_col})
                        """
                    # Add empty string checks for text columns
                    elif col_type in ['text', 'character varying', 'varchar', 'char', 'character']:
                        extra_checks = f""",
                        'empty_string_count', COUNT(CASE WHEN {quoted_col} = '' THEN 1 END),
                        'empty_string_percentage', ROUND(COUNT(CASE WHEN {quoted_col} = '' THEN 1 END) * 100.0 / COUNT(*), 2)
                        """
                    else:
                        extra_checks = ""
                    
                    column_analysis.append(f"""
                        json_build_object(
                            {base_analysis}{extra_checks}
                        )
                    """)
                
                if column_analysis:
                    analysis_query = f"""
                    SELECT json_agg(stats) as column_stats
                    FROM (
                        SELECT {', '.join(column_analysis)} as stats
                        FROM {table_name}
                    ) subquery
                    """
                    
                    analysis_result = await conn.fetchrow(analysis_query)
                    column_stats_raw = analysis_result['column_stats'] if analysis_result else None
                    
                    # Parse JSON if it's a string
                    if column_stats_raw:
                        if isinstance(column_stats_raw, str):
                            import json
                            column_stats = json.loads(column_stats_raw)
                        else:
                            column_stats = column_stats_raw
                    else:
                        column_stats = []
                    
                    # Process column statistics
                    columns_detail = []
                    empty_columns = []
                    high_null_columns = []
                    
                    for i, col_meta in enumerate(columns):
                        if column_stats and i < len(column_stats):
                            stats = column_stats[i]
                            null_pct = float(stats.get('null_percentage', 0)) if isinstance(stats, dict) else 0
                            col_type = col_meta['data_type']
                            
                            col_detail = {
                                'name': col_meta['column_name'],
                                'type': col_type,
                                'nullable': col_meta['is_nullable'] == 'YES',
                                'default': col_meta['column_default'],
                                'null_count': stats.get('null_count', 0),
                                'null_percentage': null_pct,
                                'populated_count': stats.get('populated_count', 0),
                                'distinct_count': stats.get('distinct_count', 0),
                                'max_length': col_meta['character_maximum_length']
                            }
                            
                            # Add data quality metrics for numeric columns
                            if col_type in ['integer', 'bigint', 'smallint', 'numeric', 'real', 'double precision', 'decimal']:
                                col_detail['zero_count'] = stats.get('zero_count', 0)
                                col_detail['zero_percentage'] = float(stats.get('zero_percentage', 0) or 0)
                                col_detail['negative_count'] = stats.get('negative_count', 0)
                                col_detail['min_value'] = stats.get('min_value')
                                col_detail['max_value'] = stats.get('max_value')
                                col_detail['avg_value'] = stats.get('avg_value')
                            
                            # Add empty string metrics for text columns
                            if col_type in ['text', 'character varying', 'varchar', 'char', 'character']:
                                col_detail['empty_string_count'] = stats.get('empty_string_count', 0)
                                col_detail['empty_string_percentage'] = float(stats.get('empty_string_percentage', 0) or 0)
                            
                            # Assess data quality
                            issues = []
                            if null_pct >= 100:
                                col_detail['status'] = 'EMPTY'
                                empty_columns.append(col_meta['column_name'])
                                issues.append('100% NULL')
                            elif null_pct >= 95:
                                col_detail['status'] = 'CRITICAL'
                                high_null_columns.append(col_meta['column_name'])
                                issues.append(f'{null_pct:.0f}% NULL')
                            elif null_pct >= 80:
                                col_detail['status'] = 'HIGH'
                                high_null_columns.append(col_meta['column_name'])
                                issues.append(f'{null_pct:.0f}% NULL')
                            elif null_pct >= 50:
                                col_detail['status'] = 'MEDIUM'
                                issues.append(f'{null_pct:.0f}% NULL')
                            elif null_pct > 0:
                                col_detail['status'] = 'LOW'
                            else:
                                col_detail['status'] = 'OK'
                            
                            # Check for problematic zeros in numeric columns (where 0 might be invalid)
                            zero_pct = col_detail.get('zero_percentage', 0)
                            if zero_pct > 50 and col_type in ['integer', 'bigint', 'smallint', 'numeric', 'real', 'double precision', 'decimal']:
                                # Check if column name suggests zeros are problematic
                                suspicious_zero_columns = [
                                    'price', 'market_cap', 'volume', 'mentions', 'score', 'pe_ratio',
                                    'eps', 'revenue', 'profit', 'return', 'yield'
                                ]
                                if any(keyword in col_meta['column_name'].lower() for keyword in suspicious_zero_columns):
                                    if col_detail['status'] == 'OK':
                                        col_detail['status'] = 'WARNING'
                                    issues.append(f'{zero_pct:.0f}% zeros')
                            
                            # Check for suspicious negatives in columns that should be positive
                            neg_count = col_detail.get('negative_count', 0)
                            if neg_count > 0:
                                positive_only_columns = [
                                    'market_cap', 'volume', 'mentions', 'upvotes', 'shares', 'count'
                                ]
                                if any(keyword in col_meta['column_name'].lower() for keyword in positive_only_columns):
                                    if col_detail['status'] == 'OK':
                                        col_detail['status'] = 'WARNING'
                                    issues.append(f'{neg_count} negative values')
                            
                            # Check for empty strings
                            empty_str_pct = col_detail.get('empty_string_percentage', 0)
                            if empty_str_pct > 50:
                                if col_detail['status'] == 'OK':
                                    col_detail['status'] = 'WARNING'
                                issues.append(f'{empty_str_pct:.0f}% empty strings')
                            
                            col_detail['issues'] = ', '.join(issues) if issues else None
                            
                            # Add implementation priority and action
                            priority_info = self._get_column_priority_and_action(
                                table_name, col_meta['column_name'], null_pct
                            )
                            col_detail.update(priority_info)
                            
                            columns_detail.append(col_detail)
                
                return {
                    'table_name': table_name,
                    'total_rows': total_rows,
                    'column_count': len(columns),
                    'columns': columns_detail,
                    'empty_columns': empty_columns,
                    'high_null_columns': high_null_columns,
                    'size': size_result['size'] if size_result else 'N/A',
                    'size_bytes': size_result['size_bytes'] if size_result else 0,
                    'status': 'OK'
                }
                
        except Exception as e:
            print(f"❌ Error analyzing {table_name}: {e}")
            return {'table_name': table_name, 'error': str(e)}
    
    def _determine_table_priority(self, table_name: str) -> str:
        """Determine priority for populating an empty table."""
        critical_tables = ['signals', 'runs']
        high_priority_tables = [
            'ai_strategies', 'signal_performance', 'market_conditions',
            'company_tickers'
        ]
        
        if table_name in critical_tables:
            return 'CRITICAL'
        elif table_name in high_priority_tables:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _get_column_priority_and_action(
        self, table_name: str, column_name: str, null_pct: float
    ) -> Dict[str, str]:
        """Determine implementation priority and suggested action for a column."""
        
        # Define critical columns that should always be populated
        critical_fields = {
            'signals': [
                'ticker', 'weighted_score', 'trade_type', 'current_price',
                'macd_line', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
                'beta', 'ai_commentary', 'score_explanation'
            ],
            'ai_strategies': ['signal_id', 'ticker', 'strategy_type', 'confidence_score'],
            'runs': ['run_id', 'status', 'started_at']
        }
        
        # Define high priority columns
        high_priority_fields = {
            'signals': [
                'ai_trends_commentary', 'options_flow_score', 'sector_relative_strength',
                'ml_confidence_score', 'expected_hold_duration', 'reddit_summary',
                'ai_news_summary'
            ],
            'ai_strategies': ['performance_metrics', 'backtest_results', 'risk_metrics'],
            'signal_performance': ['actual_return', 'hold_duration', 'exit_reason'],
            'signal_performance_history': ['price_history', 'returns_1d', 'returns_7d']
        }
        
        # Backend action suggestions
        action_map = {
            ('signals', 'macd_line'): 'Implement MACD calculation in signal_processing.py',
            ('signals', 'macd_signal'): 'Implement MACD signal calculation in signal_processing.py',
            ('signals', 'bollinger_upper'): 'Implement Bollinger Bands in signal_processing.py',
            ('signals', 'bollinger_lower'): 'Implement Bollinger Bands in signal_processing.py',
            ('signals', 'beta'): 'Fix Beta vs SPY calculation in signal_processing.py',
            ('signals', 'ai_commentary'): 'Enhance AI commentary generation in ai.py',
            ('signals', 'score_explanation'): 'Add score explanation in signal scoring logic',
            ('signals', 'ai_trends_commentary'): 'Implement trends analysis in ai.py',
            ('signals', 'reddit_summary'): 'Implement Reddit summary in reddit.py integration',
            ('signals', 'ai_news_summary'): 'Implement news summarization in news.py integration',
            ('ai_strategies', 'performance_metrics'): 'Implement strategy performance tracking',
            ('signal_performance', 'actual_return'): 'Implement performance tracking system',
            ('signal_performance_history', 'price_history'): 'Implement historical price tracking'
        }
        
        # Determine priority
        priority = 'INFO'
        if table_name in critical_fields and column_name in critical_fields[table_name]:
            priority = 'CRITICAL' if null_pct > 50 else 'HIGH'
        elif table_name in high_priority_fields and column_name in high_priority_fields[table_name]:
            priority = 'HIGH' if null_pct > 70 else 'MEDIUM'
        elif null_pct >= 100:
            priority = 'HIGH'  # Completely empty column
        elif null_pct >= 95:
            priority = 'MEDIUM'
        elif null_pct >= 80:
            priority = 'LOW'
        
        # Get suggested action
        key = (table_name, column_name)
        action = action_map.get(
            key, 
            f'Review {table_name}.{column_name} population in backend pipeline'
        )
        
        # Add schema recommendation
        schema_action = ''
        if null_pct >= 100:
            schema_action = 'Consider removing column if not used, or implement population logic'
        elif null_pct >= 95 and priority in ['CRITICAL', 'HIGH']:
            schema_action = 'Add NOT NULL constraint after population, or make explicitly optional'
        
        return {
            'priority': priority,
            'backend_action': action,
            'schema_recommendation': schema_action
        }
    
    async def analyze_all_tables(
        self, target_table: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze all tables or a specific table."""
        tables = [target_table] if target_table else await self.get_all_tables()
        
        results = {}
        total_tables = len(tables)
        
        print(f"📊 Analyzing {total_tables} table(s)...\n")
        
        for table in tables:
            results[table] = await self.get_table_info(table)
            
        return results
    
    def print_summary_report(
        self, 
        results: Dict[str, Any],
        gaps_only: bool = False,
        empty_only: bool = False,
        detailed: bool = False
    ):
        """Print comprehensive analysis report."""
        print("\n" + "=" * 100)
        print("🔍 VP INVESTMENTS - COMPREHENSIVE DATABASE SCHEMA ANALYSIS")
        print("=" * 100)
        
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Tables Analyzed: {len(results)}")
        
        # Collect statistics
        empty_tables = []
        tables_with_issues = []
        total_empty_columns = 0
        total_high_null_columns = 0
        critical_issues = []
        high_issues = []
        medium_issues = []
        total_rows = 0
        total_size_bytes = 0
        
        for table_name, data in results.items():
            if 'error' in data:
                print(f"❌ {table_name}: {data['error']}")
                continue
            
            if data.get('status') == 'EMPTY_TABLE':
                empty_tables.append({
                    'name': table_name,
                    'columns': data.get('column_count', 0),
                    'priority': data.get('implementation_priority', 'MEDIUM')
                })
                continue
            
            total_rows += data.get('total_rows', 0)
            total_size_bytes += data.get('size_bytes', 0)
            
            empty_cols = data.get('empty_columns', [])
            high_null_cols = data.get('high_null_columns', [])
            
            if empty_cols or high_null_cols:
                tables_with_issues.append(table_name)
                total_empty_columns += len(empty_cols)
                total_high_null_columns += len(high_null_cols)
            
            # Categorize issues
            for col in data.get('columns', []):
                if col.get('priority') == 'CRITICAL':
                    critical_issues.append({
                        'table': table_name,
                        'column': col['name'],
                        'null_pct': col.get('null_percentage', 0),
                        'action': col.get('backend_action', ''),
                        'schema': col.get('schema_recommendation', '')
                    })
                elif col.get('priority') == 'HIGH':
                    high_issues.append({
                        'table': table_name,
                        'column': col['name'],
                        'null_pct': col.get('null_percentage', 0),
                        'action': col.get('backend_action', ''),
                        'schema': col.get('schema_recommendation', '')
                    })
                elif col.get('priority') == 'MEDIUM' and col.get('null_percentage', 0) >= 50:
                    medium_issues.append({
                        'table': table_name,
                        'column': col['name'],
                        'null_pct': col.get('null_percentage', 0),
                        'action': col.get('backend_action', '')
                    })
        
        # Summary Statistics
        if not gaps_only and not empty_only:
            print(f"\n📈 DATABASE OVERVIEW:")
            print("-" * 80)
            print(f"Total Rows Across All Tables: {total_rows:,}")
            print(f"Total Database Size: {self._format_bytes(total_size_bytes)}")
            print(f"Tables with Data Issues: {len(tables_with_issues)}")
            print(f"Empty Tables: {len(empty_tables)}")
            print(f"Completely Empty Columns: {total_empty_columns}")
            print(f"High NULL% Columns: {total_high_null_columns}")
        
        # Table-by-Table NULL Summary
        if not empty_only and not detailed:
            print(f"\n📊 TABLE NULL SUMMARY:")
            print("-" * 80)
            print(f"{'Table':<30} {'Rows':<10} {'Columns':<10} {'Cols w/ NULLs':<15} {'Avg NULL%':<12}")
            print("-" * 80)
            
            for table_name, data in results.items():
                if 'error' in data or data.get('status') == 'EMPTY_TABLE':
                    continue
                
                total_cols = data.get('column_count', 0)
                cols_with_nulls = sum(1 for col in data.get('columns', []) if col.get('null_percentage', 0) > 0)
                avg_null = sum(col.get('null_percentage', 0) for col in data.get('columns', [])) / total_cols if total_cols > 0 else 0
                
                status_icon = "✅" if cols_with_nulls == 0 else "⚠️" if cols_with_nulls > total_cols * 0.1 else "🟢"
                
                print(f"{status_icon} {table_name:<27} {data['total_rows']:<10,} {total_cols:<10} {cols_with_nulls:<15} {avg_null:>10.1f}%")
        
        # Issue Summary
        print(f"\n🚨 ISSUES SUMMARY:")
        print("-" * 80)
        print(f"🔴 Critical Issues: {len(critical_issues)}")
        print(f"🟡 High Priority Issues: {len(high_issues)}")
        print(f"🟠 Medium Priority Issues: {len(medium_issues)}")
        
        # Empty Tables
        if empty_tables and not gaps_only:
            print(f"\n📭 EMPTY TABLES ({len(empty_tables)}):")
            print("-" * 80)
            for table in empty_tables:
                priority_icon = "🔴" if table['priority'] == 'CRITICAL' else "🟡" if table['priority'] == 'HIGH' else "🟢"
                print(f"{priority_icon} {table['name']}")
                print(f"   Columns: {table['columns']} | Priority: {table['priority']}")
                print(f"   → Implement data population pipeline for this table")
                print()
        
        # Critical Issues
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES - IMMEDIATE ACTION REQUIRED:")
            print("-" * 80)
            for issue in critical_issues:
                print(f"🔴 {issue['table']}.{issue['column']} ({issue['null_pct']:.1f}% NULL)")
                print(f"   Backend: {issue['action']}")
                if issue['schema']:
                    print(f"   Schema: {issue['schema']}")
                print()
        
        # High Priority Issues
        if high_issues:
            max_show = 10 if not detailed else len(high_issues)
            print(f"\n⚠️  HIGH PRIORITY ISSUES:")
            print("-" * 80)
            for issue in high_issues[:max_show]:
                print(f"🟡 {issue['table']}.{issue['column']} ({issue['null_pct']:.1f}% NULL)")
                print(f"   Backend: {issue['action']}")
                if issue['schema']:
                    print(f"   Schema: {issue['schema']}")
                print()
            
            if len(high_issues) > max_show:
                print(f"   ... and {len(high_issues) - max_show} more high priority issues\n")
        
        # Medium Priority Issues (summary only unless detailed)
        if medium_issues and not empty_only:
            if detailed:
                print(f"\n🟠 MEDIUM PRIORITY ISSUES:")
                print("-" * 80)
                for issue in medium_issues:
                    print(f"🟠 {issue['table']}.{issue['column']} ({issue['null_pct']:.1f}% NULL)")
                    print(f"   → {issue['action']}")
                    print()
            else:
                print(f"\n🟠 MEDIUM PRIORITY: {len(medium_issues)} columns with 50-80% NULL values")
                print("   Run with --detailed flag for full list")
        
        # Detailed table-by-table breakdown
        if detailed and not empty_only:
            print(f"\n📋 DETAILED TABLE-BY-TABLE ANALYSIS:")
            print("=" * 100)
            for table_name, data in results.items():
                if 'error' in data or data.get('status') == 'EMPTY_TABLE':
                    continue
                
                has_issues = any(
                    col.get('null_percentage', 0) > 0 
                    for col in data.get('columns', [])
                )
                
                if gaps_only and not has_issues:
                    continue
                
                print(f"\n📊 {table_name.upper()}")
                print("-" * 100)
                print(f"Rows: {data['total_rows']:,} | Size: {data['size']} | Columns: {data['column_count']}")
                
                # Always show column details in detailed mode with quality metrics
                print(f"\n{'Column':<35} {'Type':<15} {'Rows':<10} {'NULL%':<8} {'Quality Issues':<30} {'Status':<8}")
                print("-" * 120)
                
                for col in data.get('columns', []):
                    null_pct = col.get('null_percentage', 0)
                    populated = col.get('populated_count', 0)
                    col_type = col.get('type', 'unknown')
                    
                    # Skip columns with no issues if gaps_only is set
                    if gaps_only and null_pct == 0 and not col.get('issues'):
                        continue
                    
                    status_icon = {
                        'OK': '✅',
                        'LOW': '🟢',
                        'MEDIUM': '🟠',
                        'HIGH': '🟡',
                        'CRITICAL': '🔴',
                        'EMPTY': '❌',
                        'WARNING': '⚠️'
                    }.get(col.get('status', 'OK'), '❓')
                    
                    # Format populated count
                    pop_display = f"{populated:,}"
                    
                    # Build quality issues summary
                    issues_list = []
                    if null_pct > 0:
                        issues_list.append(f"NULL:{null_pct:.0f}%")
                    
                    # Add numeric quality issues
                    if 'zero_percentage' in col and col['zero_percentage'] > 50:
                        issues_list.append(f"ZERO:{col['zero_percentage']:.0f}%")
                    if 'negative_count' in col and col['negative_count'] > 0:
                        issues_list.append(f"NEG:{col['negative_count']}")
                    
                    # Add text quality issues
                    if 'empty_string_percentage' in col and col['empty_string_percentage'] > 50:
                        issues_list.append(f"EMPTY:{col['empty_string_percentage']:.0f}%")
                    
                    # Add value range if numeric
                    if 'min_value' in col and 'max_value' in col:
                        min_val = col['min_value']
                        max_val = col['max_value']
                        if min_val is not None and max_val is not None:
                            issues_list.append(f"Range:[{min_val:.2f}-{max_val:.2f}]" if isinstance(min_val, float) else f"Range:[{min_val}-{max_val}]")
                    
                    issues_display = ', '.join(issues_list[:3]) if issues_list else '✓'
                    
                    print(f"{col['name']:<35} {col_type:<15} {pop_display:<10} {null_pct:>6.1f}% {issues_display:<30} {status_icon}")
                    
                    # Show detailed issues and actions for problematic columns
                    if col.get('status') in ['EMPTY', 'CRITICAL', 'HIGH', 'WARNING']:
                        if col.get('issues'):
                            print(f"   ⚠️  Issues: {col['issues']}")
                        if col.get('backend_action'):
                            print(f"   → {col.get('backend_action')}")
                        if col.get('schema_recommendation'):
                            print(f"   ⚙️  {col.get('schema_recommendation')}")
                
                # Summary line
                null_count = sum(1 for col in data.get('columns', []) if col.get('null_percentage', 0) > 0)
                if null_count > 0:
                    print(f"\n⚠️  {null_count} column(s) with NULL values")
                else:
                    print(f"\n✅ All columns fully populated")
        
        # Implementation Roadmap
        if critical_issues or high_issues:
            print(f"\n🗺️  IMPLEMENTATION ROADMAP:")
            print("-" * 80)
            print("1. 🔴 Fix Critical Issues (signals table core fields)")
            print("2. 🟡 Implement High Priority Features (AI commentary, technical indicators)")
            print("3. 📭 Populate Empty Tables (if required for functionality)")
            print("4. 🟠 Address Medium Priority Enhancements")
            print("5. ✅ Validate and add schema constraints")
            
            print(f"\n🚀 QUICK ACTIONS:")
            print("-" * 80)
            print("# Run the main pipeline to populate core data:")
            print("python backend/pipeline.py")
            print()
            print("# Add AI commentary and analysis:")
            print("python backend/pipeline.py --enhanced")
            print()
            print("# Check specific table:")
            print(f"python tables.py --table signals --detailed")
        
        # Schema Optimization Suggestions
        if not empty_only and (critical_issues or high_issues):
            print(f"\n💡 SCHEMA OPTIMIZATION SUGGESTIONS:")
            print("-" * 80)
            print("1. Add NOT NULL constraints to critical columns after population")
            print("2. Consider default values for optional fields")
            print("3. Remove unused columns that are 100% empty")
            print("4. Add indexes on frequently queried NULL columns")
            print("5. Review data types for optimal storage efficiency")
        
        print("\n" + "=" * 100)
    
    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes into human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="VP Investments - Comprehensive Database Schema Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--table', type=str, help='Analyze specific table only')
    parser.add_argument('--gaps-only', action='store_true', 
                       help='Show only tables/columns with data gaps')
    parser.add_argument('--empty-only', action='store_true',
                       help='Show only empty tables and columns')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed column-by-column analysis')
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveDatabaseAnalyzer()
    
    try:
        await analyzer.connect()
        results = await analyzer.analyze_all_tables(args.table)
        analyzer.print_summary_report(
            results, 
            gaps_only=args.gaps_only,
            empty_only=args.empty_only,
            detailed=args.detailed
        )
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await analyzer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
