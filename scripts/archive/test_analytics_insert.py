"""Test analytics INSERT with minimal data"""
import asyncio
import asyncpg
import json
from uuid import uuid4

async def main():
    # Connect to database
    conn = await asyncpg.connect(
        "postgresql://postgres.rdkxwoqevjicupmefbem:1qaz1QAZ2wsx2WSX@aws-1-us-east-2.pooler.supabase.com:6543/postgres"
    )
    
    try:
        # Test minimal INSERT
        test_run_id = uuid4()
        
        query = """
            INSERT INTO analytics (
                run_id,
                total_signals, avg_overall_score,
                avg_technical_score, avg_fundamental_score, avg_news_macro_score,
                avg_social_alternative_score, avg_risk_stability_score, avg_institutional_score,
                top_sector, top_sector_avg_return, top_sector_count,
                worst_sector, worst_sector_avg_return, worst_sector_count,
                signals_analyzed, performance_records_used,
                score_bucket_performance, factor_correlations, factor_contributions,
                backtest_cumulative_returns,
                ic_series, ic_mean, ic_std, hit_rate_top_decile, profit_factor, win_loss_ratio,
                cagr, volatility, sortino_ratio, calmar_ratio,
                alpha_vs_spy, beta_vs_spy, alpha_vs_qqq, beta_vs_qqq,
                rolling_sharpe_30d, benchmark_correlations,
                signal_correlations, top_positive_pairs, top_negative_pairs
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9,
                $10, $11, $12, $13, $14, $15,
                $16, $17, $18, $19, $20, $21,
                $22, $23, $24, $25, $26, $27, $28,
                $29, $30, $31, $32,
                $33, $34, $35, $36,
                $37, $38,
                $39, $40
            )
            RETURNING id
        """
        
        # Count columns and placeholders
        columns = query.split("INSERT INTO analytics (")[1].split(") VALUES")[0]
        column_list = [c.strip() for c in columns.split(",")]
        print(f"Columns in INSERT: {len(column_list)}")
        for i, col in enumerate(column_list, 1):
            print(f"  {i:2d}. {col}")
        
        values = query.split("VALUES (")[1].split(")")[0]
        placeholder_list = [p.strip() for p in values.split(",")]
        print(f"\nPlaceholders in VALUES: {len(placeholder_list)}")
        
        # Create minimal test parameters
        params = [
            test_run_id,                # 1: run_id
            10,                         # 2: total_signals
            0.5,                        # 3: avg_overall_score
            0.6, 0.5, 0.4,            # 4-6: avg scores
            0.3, 0.7, 0.6,            # 7-9: avg scores
            "Technology", 0.15, 5,     # 10-12: top sector
            "Energy", -0.05, 2,        # 13-15: worst sector
            10, 100,                   # 16-17: signals_analyzed, performance_records_used
            json.dumps({}), json.dumps({}), json.dumps({}),  # 18-20: JSONB
            json.dumps({}),            # 21: backtest
            json.dumps([]), 0.5, 0.2, 0.6, 1.5, 2.0,  # 22-27: IC metrics
            0.25, 0.18, 1.2, 3.5,     # 28-31: performance metrics
            0.08, 0.95, 0.06, 1.02,   # 32-35: alpha/beta
            json.dumps({}), json.dumps({}),  # 36-37: JSONB
            json.dumps({}), json.dumps([]), json.dumps([])  # 38-40: JSONB
        ]
        
        print(f"\nParameters provided: {len(params)}")
        
        # Try INSERT
        print("\nAttempting INSERT...")
        result = await conn.fetchrow(query, *params)
        print(f"✅ INSERT successful! ID: {result['id']}")
        
    except Exception as e:
        print(f"❌ INSERT failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
