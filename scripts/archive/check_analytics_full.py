"""Check all analytics columns for NULL values and data quality."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    await db.connect()
    
    try:
        result = await db.execute_query("""
            SELECT 
                period_type,
                total_signals,
                avg_overall_score,
                avg_technical_score,
                avg_fundamental_score,
                avg_news_macro_score,
                avg_social_alternative_score,
                avg_risk_stability_score,
                avg_institutional_score,
                top_sector,
                win_rate,
                sharpe_ratio,
                max_drawdown,
                avg_return,
                avg_alpha,
                ic_mean,
                ic_std,
                hit_rate_top_decile,
                profit_factor,
                win_loss_ratio,
                alpha_vs_spy,
                alpha_vs_qqq,
                beta_vs_spy,
                beta_vs_qqq
            FROM analytics 
            ORDER BY 
                CASE period_type
                    WHEN '1d' THEN 1
                    WHEN '3d' THEN 2
                    WHEN '7d' THEN 3
                    WHEN '10d' THEN 4
                    WHEN '14d' THEN 5
                    WHEN '30d' THEN 6
                    WHEN '90d' THEN 7
                    WHEN 'all_time' THEN 8
                END
        """)
        
        print(f"\n{'='*120}")
        print(f"Analytics Table - NULL Value Check")
        print(f"{'='*120}\n")
        
        # Check for NULL values in each column
        for row in result:
            period = row['period_type']
            print(f"\n{period.upper()} Interval:")
            print("-" * 80)
            
            # Group scores
            print(f"  Group Scores:")
            print(f"    avg_technical_score:           {row['avg_technical_score']}")
            print(f"    avg_fundamental_score:         {row['avg_fundamental_score']}")
            print(f"    avg_news_macro_score:          {row['avg_news_macro_score']}")
            print(f"    avg_social_alternative_score:  {row['avg_social_alternative_score']}")
            print(f"    avg_risk_stability_score:      {row['avg_risk_stability_score']} {'❌ NULL' if row['avg_risk_stability_score'] is None else '✓'}")
            print(f"    avg_institutional_score:       {row['avg_institutional_score']}")
            
            # Interval metrics
            print(f"\n  Interval Metrics:")
            print(f"    win_rate:        {row['win_rate']} {'❌ NULL' if row['win_rate'] is None else '✓'}")
            print(f"    sharpe_ratio:    {row['sharpe_ratio']} {'❌ NULL' if row['sharpe_ratio'] is None else '✓'}")
            print(f"    max_drawdown:    {row['max_drawdown']} {'❌ NULL' if row['max_drawdown'] is None else '✓'}")
            print(f"    avg_return:      {row['avg_return']} {'❌ NULL' if row['avg_return'] is None else '✓'}")
            print(f"    avg_alpha:       {row['avg_alpha']} {'❌ NULL' if row['avg_alpha'] is None else '✓'}")
            
            # Benchmark metrics
            print(f"\n  Benchmark Metrics:")
            print(f"    alpha_vs_spy:    {row['alpha_vs_spy']}")
            print(f"    beta_vs_spy:     {row['beta_vs_spy']}")
            print(f"    alpha_vs_qqq:    {row['alpha_vs_qqq']}")
            print(f"    beta_vs_qqq:     {row['beta_vs_qqq']}")
            
            # Other metrics
            print(f"\n  Other Metrics:")
            print(f"    ic_mean:              {row['ic_mean']} {'❌ NULL' if row['ic_mean'] is None else '✓'}")
            print(f"    ic_std:               {row['ic_std']} {'❌ NULL' if row['ic_std'] is None else '✓'}")
            print(f"    hit_rate_top_decile:  {row['hit_rate_top_decile']} {'❌ NULL' if row['hit_rate_top_decile'] is None else '✓'}")
            print(f"    profit_factor:        {row['profit_factor']} {'❌ NULL' if row['profit_factor'] is None else '✓'}")
            print(f"    win_loss_ratio:       {row['win_loss_ratio']} {'❌ NULL' if row['win_loss_ratio'] is None else '✓'}")
        
        print(f"\n{'='*120}\n")
        
    finally:
        await db.disconnect()

asyncio.run(main())
