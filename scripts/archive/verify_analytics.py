"""Verify analytics data after pipeline run"""
import asyncio
import asyncpg
import json

async def main():
    conn = await asyncpg.connect(
        "postgresql://postgres.rdkxwoqevjicupmefbem:1qaz1QAZ2wsx2WSX@aws-1-us-east-2.pooler.supabase.com:6543/postgres",
        statement_cache_size=0  # Required for pgbouncer in transaction mode
    )
    
    try:
        print("=" * 80)
        print("ANALYTICS VERIFICATION")
        print("=" * 80)
        
        # Check row count
        count = await conn.fetchval("SELECT COUNT(*) FROM analytics;")
        print(f"\n✓ Total analytics rows: {count}")
        
        if count == 0:
            print("\n⚠️  No analytics data found!")
            print("Wait for pipeline Phase 7 to complete.")
            return
        
        # Get latest analytics
        latest = await conn.fetchrow("""
            SELECT * FROM analytics 
            ORDER BY created_at DESC 
            LIMIT 1;
        """)
        
        print(f"\n✓ Latest analytics created: {latest['created_at']}")
        print(f"✓ Run ID: {latest['run_id']}")
        print(f"✓ Total signals: {latest['total_signals']}")
        
        # Check each column group
        print("\n" + "=" * 80)
        print("COLUMN DATA VERIFICATION")
        print("=" * 80)
        
        checks = {
            "Basic Metrics": [
                ('avg_overall_score', latest['avg_overall_score']),
                ('signals_analyzed', latest['signals_analyzed']),
                ('performance_records_used', latest['performance_records_used']),
            ],
            "Factor Group Scores": [
                ('avg_technical_score', latest['avg_technical_score']),
                ('avg_fundamental_score', latest['avg_fundamental_score']),
                ('avg_news_macro_score', latest['avg_news_macro_score']),
                ('avg_social_alternative_score', latest['avg_social_alternative_score']),
                ('avg_risk_stability_score', latest['avg_risk_stability_score']),
                ('avg_institutional_score', latest['avg_institutional_score']),
            ],
            "Sector Analysis": [
                ('top_sector', latest['top_sector']),
                ('top_sector_avg_return', latest['top_sector_avg_return']),
                ('top_sector_count', latest['top_sector_count']),
                ('worst_sector', latest['worst_sector']),
                ('worst_sector_avg_return', latest['worst_sector_avg_return']),
                ('worst_sector_count', latest['worst_sector_count']),
            ],
            "Performance Metrics": [
                ('cagr', latest['cagr']),
                ('volatility', latest['volatility']),
                ('sortino_ratio', latest['sortino_ratio']),
                ('calmar_ratio', latest['calmar_ratio']),
            ],
            "Benchmark Metrics": [
                ('alpha_vs_spy', latest['alpha_vs_spy']),
                ('beta_vs_spy', latest['beta_vs_spy']),
                ('alpha_vs_qqq', latest['alpha_vs_qqq']),
                ('beta_vs_qqq', latest['beta_vs_qqq']),
            ],
            "Predictive Strength": [
                ('ic_mean', latest['ic_mean']),
                ('ic_std', latest['ic_std']),
                ('hit_rate_top_decile', latest['hit_rate_top_decile']),
                ('profit_factor', latest['profit_factor']),
                ('win_loss_ratio', latest['win_loss_ratio']),
            ],
            "JSONB Columns": [
                ('score_bucket_performance', 'JSONB' if latest['score_bucket_performance'] else None),
                ('factor_correlations', 'JSONB' if latest['factor_correlations'] else None),
                ('factor_contributions', 'JSONB' if latest['factor_contributions'] else None),
                ('backtest_cumulative_returns', 'JSONB' if latest['backtest_cumulative_returns'] else None),
                ('ic_series', 'JSONB' if latest['ic_series'] else None),
                ('rolling_sharpe_30d', 'JSONB' if latest['rolling_sharpe_30d'] else None),
                ('benchmark_correlations', 'JSONB' if latest['benchmark_correlations'] else None),
                ('signal_correlations', 'JSONB' if latest['signal_correlations'] else None),
                ('top_positive_pairs', 'JSONB' if latest['top_positive_pairs'] else None),
                ('top_negative_pairs', 'JSONB' if latest['top_negative_pairs'] else None),
            ],
        }
        
        issues = []
        
        for group_name, columns in checks.items():
            print(f"\n{group_name}:")
            for col_name, value in columns:
                if value is None:
                    status = "❌ NULL"
                    issues.append(f"{col_name} is NULL")
                elif isinstance(value, str) and value == 'JSONB':
                    status = "✅ Has data"
                else:
                    status = f"✅ {value}"
                print(f"  {col_name:<30} {status}")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if issues:
            print(f"\n⚠️  Found {len(issues)} columns with NULL values:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ All columns have data! Analytics table is fully populated.")
        
        # Check if score_bucket_performance has interval data
        if latest['score_bucket_performance']:
            sbp = json.loads(latest['score_bucket_performance']) if isinstance(latest['score_bucket_performance'], str) else latest['score_bucket_performance']
            print(f"\n✓ Score bucket performance structure:")
            for bucket in ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']:
                if bucket in sbp:
                    intervals = [k for k in sbp[bucket].keys() if k not in ['threshold', 'count']]
                    print(f"  {bucket}: {len(intervals)} intervals ({', '.join(intervals[:3])}...)")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
