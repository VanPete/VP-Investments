#!/usr/bin/env python3
"""
VP Investments Signal Analysis & Quality Check Tool

This tool performs comprehensive analysis of all 6 signal groups:
1. Technical (price/momentum)
2. Fundamental (financial metrics)
3. News/Macro (sentiment)
4. Social/Alternative (Reddit, social media)
5. Risk/Stability (volatility, risk factors)
6. Institutional/Smart Money (ownership, analyst data)

Functions:
- Validates Supabase schema columns are being used in calculations
- Checks data quality (NULLs, outliers, suspicious values)
- Generates column usage matrix
- Provides summary statistics per signal group
- Recommends improvements

Author: VP Investments
Date: October 10, 2025
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()


class SignalAnalyzer:
    """Comprehensive signal analysis tool"""
    
    # Define the 6 signal groups and their associated columns
    SIGNAL_GROUPS = {
        'Technical': {
            'description': 'Price action, momentum, and technical indicators',
            'columns': [
                # Price and momentum (23% of technical score)
                'current_price', 'price_1d_pct', 'price_7d_pct', 'momentum_30d_pct',
                # Volume analysis (12%)
                'volume', 'avg_volume_30d', 'volume_spike_ratio', 'volume_price_correlation',
                # RSI (12%)
                'rsi',
                # Moving averages (12%)
                'above_50d_ma_pct', 'above_200d_ma_pct',
                # MACD (10%)
                'macd', 'macd_signal', 'macd_line', 'macd_histogram',
                # Bollinger Bands (part of volatility)
                'bollinger_upper', 'bollinger_lower', 'bollinger_position', 'bollinger_width',
                # Volatility (10%)
                'volatility', 'volatility_rank', 'historical_volatility',
                # Relative strength (10%)
                'relative_strength', 'sector_relative_strength',
                # Beta (8%)
                'beta',
                # ATR indicators
                'atr', 'atr_percent',
                # Phase 1.4 ML metrics (13% combined)
                'momentum_consistency_score', 'liquidity_score',
                # Phase 2 z-scores
                'z_score_momentum', 'z_score_volume', 'z_score_volatility',
                # Phase 8 backtest
                'backtest_stop_loss_price', 'backtest_take_profit_price'
            ],
            'score_fields': ['technical_score', 'signal_score']
        },
        'Fundamental': {
            'description': 'Financial metrics, valuation, and profitability',
            'columns': [
                'market_cap', 'pe_ratio', 'pb_ratio', 'earnings_gap_pct', 'revenue_growth',
                'eps_growth', 'roe', 'debt_equity', 'fcf_margin', 'profit_margin',
                'revenue_per_share', 'book_value_per_share', 'operating_cash_flow',
                'earnings_per_share', 'dividend_yield',
                # Phase 5 enhanced
                'operating_margin', 'debt_to_equity', 'current_ratio',
                # Phase 2
                'z_score_valuation'
            ],
            'score_fields': ['fundamental_score', 'signal_score']
        },
        'News_Macro': {
            'description': 'News sentiment and macro events',
            'columns': [
                'news_sentiment', 'news_volume', 'news_impact_score', 'sec_filing_sentiment',
                'earnings_surprise', 'earnings_call_sentiment', 'analyst_rating_change',
                'news_buzz_7d', 'news_score'
            ],
            'score_fields': ['news_score', 'signal_score']
        },
        'Social_Alternative': {
            'description': 'Social media sentiment and alternative data',
            'columns': [
                'reddit_mentions', 'reddit_sentiment', 'reddit_score', 'reddit_posts_analyzed',
                'reddit_bullish_ratio', 'retail_holding_pct', 'social_sentiment',
                'social_score', 'social_volume'
            ],
            'score_fields': ['social_score', 'reddit_score', 'signal_score']
        },
        'Risk_Stability': {
            'description': 'Risk assessment and stability metrics',
            'columns': [
                'risk_score', 'risk_level', 'liquidity_score', 'beta', 'volatility_risk',
                'liquidity_risk', 'leverage_risk', 'concentration_risk', 'technical_risk',
                'fundamental_risk', 'sentiment_risk',
                # Phase 6 adjustments
                'adjusted_signal_score', 'position_size_recommendation', 'entry_threshold',
                # Phase 8 backtest risk
                'backtest_risk_reward_ratio', 'backtest_position_size_pct', 'backtest_hold_period_days'
            ],
            'score_fields': ['risk_score', 'adjusted_signal_score']
        },
        'Institutional_Smart_Money': {
            'description': 'Institutional activity and smart money indicators',
            'columns': [
                'institutional_ownership_pct', 'insider_activity_score', 'insider_buy_volume',
                'insider_sell_volume', 'shares_short', 'short_pct_float', 'days_to_cover',
                'analyst_count', 'analyst_target_price', 'analyst_target_upside_pct',
                'analyst_rating', 'top_10_institutional_holders_pct', 'num_institutional_holders',
                'institutional_buying_pressure', 'fund_ownership_change',
                # Phase 5 enhanced
                'institutional_ownership', 'insider_ownership', 'short_interest',
                'put_call_ratio', 'open_interest'
            ],
            'score_fields': ['institutional_score', 'signal_score']
        }
    }
    
    def __init__(self):
        """Initialize analyzer with database connection"""
        self.conn = self._connect_db()
        self.schema_columns = self._fetch_schema_columns()
        self.analysis_results = {}
        
    def _connect_db(self):
        """Connect to Supabase PostgreSQL"""
        db_url = os.getenv('DATABASE_URL') or os.getenv('SUPABASE_DATABASE_URL')
        if not db_url:
            raise ValueError("Missing DATABASE_URL or SUPABASE_DATABASE_URL in .env")
        
        return psycopg2.connect(db_url)
    
    def _fetch_schema_columns(self) -> List[str]:
        """Get all column names from signals table schema"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'signals'
            ORDER BY ordinal_position;
        """)
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return columns
    
    def _fetch_recent_signals(self, limit: int = 100) -> List[Dict]:
        """Fetch recent signals for analysis"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(f"""
            SELECT * FROM signals 
            ORDER BY created_at DESC 
            LIMIT {limit}
        """)
        signals = cursor.fetchall()
        cursor.close()
        return [dict(signal) for signal in signals]
    
    def analyze_column_usage(self) -> Dict[str, Any]:
        """
        Analyze which columns are defined in schema vs used in signal groups
        Returns column usage matrix
        """
        print("\n" + "="*80)
        print("COLUMN USAGE ANALYSIS")
        print("="*80 + "\n")
        
        # Collect all columns referenced in signal groups
        used_columns = set()
        for group_name, group_info in self.SIGNAL_GROUPS.items():
            used_columns.update(group_info['columns'])
        
        # Find columns in schema but not in any signal group
        unused_columns = [col for col in self.schema_columns if col not in used_columns]
        
        # Filter out metadata columns (expected to not be in signal groups)
        metadata_columns = {
            'id', 'ticker', 'created_at', 'updated_at', 'run_id', 'signal_id',
            'trade_type', 'trade_type_confidence', 'ai_commentary', 'risk_narrative',
            'backtest_entry_threshold', 'confidence_score', 'signal_strength'
        }
        
        truly_unused = [col for col in unused_columns if col not in metadata_columns]
        
        # Create usage matrix
        usage_matrix = {}
        for col in self.schema_columns:
            usage_matrix[col] = {
                'in_schema': True,
                'used_in_groups': [],
                'is_metadata': col in metadata_columns,
                'is_unused': col in truly_unused
            }
            
            for group_name, group_info in self.SIGNAL_GROUPS.items():
                if col in group_info['columns']:
                    usage_matrix[col]['used_in_groups'].append(group_name)
        
        # Print summary
        print(f"📊 Total Schema Columns: {len(self.schema_columns)}")
        print(f"✅ Columns Used in Signal Groups: {len(used_columns)}")
        print(f"📋 Metadata Columns (expected): {len(metadata_columns)}")
        print(f"⚠️  Potentially Unused Columns: {len(truly_unused)}\n")
        
        if truly_unused:
            print("🔍 Columns in schema but NOT used in any signal group:")
            for col in sorted(truly_unused):
                print(f"   - {col}")
            print()
        
        # Show column distribution across groups
        print("📈 Column Distribution by Signal Group:")
        for group_name, group_info in self.SIGNAL_GROUPS.items():
            col_count = len(group_info['columns'])
            print(f"   {group_name}: {col_count} columns")
        
        return {
            'usage_matrix': usage_matrix,
            'total_columns': len(self.schema_columns),
            'used_columns': len(used_columns),
            'unused_columns': truly_unused,
            'metadata_columns': list(metadata_columns)
        }
    
    def analyze_data_quality(self, signals: List[Dict]) -> Dict[str, Any]:
        """
        Analyze data quality for each signal group
        - NULL rates
        - Zero/invalid values
        - Outliers
        """
        print("\n" + "="*80)
        print("DATA QUALITY ANALYSIS")
        print("="*80 + "\n")
        
        quality_report = {}
        
        for group_name, group_info in self.SIGNAL_GROUPS.items():
            print(f"\n📊 {group_name} Signal Group")
            print(f"   {group_info['description']}")
            print(f"   Columns: {len(group_info['columns'])}\n")
            
            group_quality = {
                'column_stats': {},
                'null_rates': {},
                'zero_rates': {},
                'issues': []
            }
            
            for col in group_info['columns']:
                if col not in self.schema_columns:
                    group_quality['issues'].append(f"Column '{col}' not in schema")
                    continue
                
                # Calculate NULL rate
                null_count = sum(1 for s in signals if s.get(col) is None)
                null_rate = (null_count / len(signals)) * 100 if signals else 0
                
                # Calculate zero rate for numeric columns
                zero_count = sum(1 for s in signals if s.get(col) == 0 or s.get(col) == 0.0)
                zero_rate = (zero_count / len(signals)) * 100 if signals else 0
                
                # Collect non-null values for stats
                values = [s.get(col) for s in signals if s.get(col) is not None]
                
                group_quality['null_rates'][col] = null_rate
                group_quality['zero_rates'][col] = zero_rate
                group_quality['column_stats'][col] = {
                    'null_rate': null_rate,
                    'zero_rate': zero_rate,
                    'populated_count': len(values),
                    'sample_values': values[:3] if values else []
                }
                
                # Flag quality issues
                if null_rate > 50:
                    status = "🔴"
                    group_quality['issues'].append(f"{col}: {null_rate:.1f}% NULL (CRITICAL)")
                elif null_rate > 30:
                    status = "🟡"
                    group_quality['issues'].append(f"{col}: {null_rate:.1f}% NULL (HIGH)")
                elif null_rate > 10:
                    status = "🟢"
                else:
                    status = "✅"
                
                if null_rate > 10 or (col in group_info.get('score_fields', []) and null_rate > 0):
                    print(f"   {status} {col}: {null_rate:.1f}% NULL, {zero_rate:.1f}% zeros")
            
            quality_report[group_name] = group_quality
        
        return quality_report
    
    def analyze_signal_scores(self, signals: List[Dict]) -> Dict[str, Any]:
        """
        Analyze signal score distributions and relationships
        """
        print("\n" + "="*80)
        print("SIGNAL SCORE ANALYSIS")
        print("="*80 + "\n")
        
        score_analysis = {}
        
        for group_name, group_info in self.SIGNAL_GROUPS.items():
            score_fields = group_info.get('score_fields', [])
            if not score_fields:
                continue
            
            print(f"\n📈 {group_name} Scores:")
            
            group_scores = {}
            for score_field in score_fields:
                scores = [s.get(score_field) for s in signals if s.get(score_field) is not None]
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    min_score = min(scores)
                    max_score = max(scores)
                    
                    group_scores[score_field] = {
                        'count': len(scores),
                        'avg': avg_score,
                        'min': min_score,
                        'max': max_score,
                        'null_rate': ((len(signals) - len(scores)) / len(signals)) * 100
                    }
                    
                    print(f"   {score_field}:")
                    print(f"      Avg: {avg_score:.3f}, Min: {min_score:.3f}, Max: {max_score:.3f}")
                    print(f"      Populated: {len(scores)}/{len(signals)} ({(len(scores)/len(signals)*100):.1f}%)")
            
            score_analysis[group_name] = group_scores
        
        return score_analysis
    
    def generate_recommendations(self, quality_report: Dict, usage_analysis: Dict) -> List[str]:
        """
        Generate actionable recommendations based on analysis
        """
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80 + "\n")
        
        recommendations = []
        
        # Check for unused columns
        if usage_analysis['unused_columns']:
            rec = f"⚠️  Found {len(usage_analysis['unused_columns'])} unused columns in schema"
            print(f"\n{rec}")
            print("   Consider either:")
            print("   - Adding these columns to appropriate signal groups")
            print("   - Removing them from schema if truly not needed")
            print("   Columns:", ", ".join(usage_analysis['unused_columns'][:5]))
            recommendations.append(rec)
        
        # Check for high NULL rates
        critical_nulls = []
        for group_name, group_data in quality_report.items():
            for issue in group_data.get('issues', []):
                if 'CRITICAL' in issue:
                    critical_nulls.append(f"{group_name}: {issue}")
        
        if critical_nulls:
            rec = f"🔴 Found {len(critical_nulls)} columns with >50% NULL rate"
            print(f"\n{rec}")
            for issue in critical_nulls[:5]:
                print(f"   - {issue}")
            recommendations.append(rec)
        
        # Check for columns in signal groups but not in schema
        missing_from_schema = []
        for group_name, group_info in self.SIGNAL_GROUPS.items():
            for col in group_info['columns']:
                if col not in self.schema_columns:
                    missing_from_schema.append(f"{group_name}.{col}")
        
        if missing_from_schema:
            rec = f"❌ Found {len(missing_from_schema)} columns referenced in code but NOT in schema"
            print(f"\n{rec}")
            for col in missing_from_schema[:5]:
                print(f"   - {col}")
            recommendations.append(rec)
        
        # Positive findings
        if not recommendations:
            rec = "✅ All signal groups are properly configured with good data quality"
            print(f"\n{rec}")
            recommendations.append(rec)
        
        return recommendations
    
    def generate_usage_matrix_report(self, usage_matrix: Dict) -> str:
        """
        Generate detailed column usage matrix report
        """
        report_lines = []
        report_lines.append("\n" + "="*100)
        report_lines.append("DETAILED COLUMN USAGE MATRIX")
        report_lines.append("="*100 + "\n")
        
        # Group columns by usage pattern
        by_group_count = defaultdict(list)
        for col, info in usage_matrix.items():
            if info['is_metadata']:
                continue
            group_count = len(info['used_in_groups'])
            by_group_count[group_count].append((col, info['used_in_groups']))
        
        # Show multi-group columns (used in multiple signal groups)
        if by_group_count.get(2) or by_group_count.get(3) or by_group_count.get(4):
            report_lines.append("🔗 Columns Used in Multiple Signal Groups:")
            for count in sorted([k for k in by_group_count.keys() if k > 1], reverse=True):
                for col, groups in sorted(by_group_count[count]):
                    report_lines.append(f"   {col}: {', '.join(groups)}")
            report_lines.append("")
        
        # Show single-group columns
        if by_group_count.get(1):
            report_lines.append(f"📍 Columns Used in Single Signal Group: {len(by_group_count[1])}")
            report_lines.append("")
        
        # Show unused columns
        if by_group_count.get(0):
            report_lines.append(f"⚠️  Columns Not Used in Any Signal Group: {len(by_group_count[0])}")
            for col, _ in sorted(by_group_count[0])[:10]:
                report_lines.append(f"   - {col}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def run_full_analysis(self, save_report: bool = True) -> Dict[str, Any]:
        """
        Run complete signal analysis
        """
        print("\n" + "="*100)
        print(" "*30 + "VP INVESTMENTS SIGNAL ANALYSIS")
        print(" "*35 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*100)
        
        # Fetch recent signals
        print("\n📥 Fetching recent signals from database...")
        signals = self._fetch_recent_signals(limit=100)
        print(f"✅ Retrieved {len(signals)} signals for analysis\n")
        
        # Run analyses
        usage_analysis = self.analyze_column_usage()
        quality_report = self.analyze_data_quality(signals)
        score_analysis = self.analyze_signal_scores(signals)
        recommendations = self.generate_recommendations(quality_report, usage_analysis)
        
        # Generate detailed reports
        usage_matrix_report = self.generate_usage_matrix_report(usage_analysis['usage_matrix'])
        print(usage_matrix_report)
        
        # Compile full report
        full_report = {
            'timestamp': datetime.now().isoformat(),
            'signals_analyzed': len(signals),
            'usage_analysis': usage_analysis,
            'quality_report': quality_report,
            'score_analysis': score_analysis,
            'recommendations': recommendations
        }
        
        # Save to file
        if save_report:
            report_file = f"signal_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            print(f"\n💾 Full report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*100)
        print("ANALYSIS COMPLETE")
        print("="*100)
        print(f"\n📊 Summary:")
        print(f"   - Total Schema Columns: {usage_analysis['total_columns']}")
        print(f"   - Columns Used in Signal Groups: {usage_analysis['used_columns']}")
        print(f"   - Unused Columns: {len(usage_analysis['unused_columns'])}")
        print(f"   - Signals Analyzed: {len(signals)}")
        print(f"   - Recommendations: {len(recommendations)}")
        print()
        
        return full_report
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main entry point"""
    try:
        analyzer = SignalAnalyzer()
        report = analyzer.run_full_analysis(save_report=True)
        analyzer.close()
        
        # Exit with appropriate code
        if len(report['recommendations']) > 2:
            print("⚠️  Action items found - review recommendations above")
            sys.exit(1)
        else:
            print("✅ Signal analysis complete - no critical issues found")
            sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
