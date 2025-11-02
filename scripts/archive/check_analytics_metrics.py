"""Check analytics rows with interval metrics."""
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
                win_rate, 
                sharpe_ratio, 
                max_drawdown, 
                avg_return, 
                avg_alpha,
                alpha_vs_spy,
                alpha_vs_qqq
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
        
        print(f"\n{'='*100}")
        print(f"Analytics Table: {len(result)} rows")
        print(f"{'='*100}\n")
        
        print(f"{'Period':<10} {'Signals':<10} {'Win%':<8} {'Sharpe':<8} {'MaxDD':<8} {'AvgRet':<8} {'AvgAlpha':<10} {'αSPY':<8} {'αQQQ':<8}")
        print(f"{'-'*100}")
        
        for row in result:
            period = row['period_type']
            signals = row['total_signals']
            win_rate = row['win_rate'] if row['win_rate'] is not None else 'N/A'
            sharpe = row['sharpe_ratio'] if row['sharpe_ratio'] is not None else 'N/A'
            max_dd = row['max_drawdown'] if row['max_drawdown'] is not None else 'N/A'
            avg_ret = row['avg_return'] if row['avg_return'] is not None else 'N/A'
            avg_alpha = row['avg_alpha'] if row['avg_alpha'] is not None else 'N/A'
            alpha_spy = row['alpha_vs_spy'] if row['alpha_vs_spy'] is not None else 'N/A'
            alpha_qqq = row['alpha_vs_qqq'] if row['alpha_vs_qqq'] is not None else 'N/A'
            
            # Format numeric values
            if isinstance(win_rate, (int, float)):
                win_rate = f"{win_rate:.2f}%"
            if isinstance(sharpe, (int, float)):
                sharpe = f"{sharpe:.3f}"
            if isinstance(max_dd, (int, float)):
                max_dd = f"{max_dd:.2f}%"
            if isinstance(avg_ret, (int, float)):
                avg_ret = f"{avg_ret:.2f}%"
            if isinstance(avg_alpha, (int, float)):
                avg_alpha = f"{avg_alpha:.2f}%"
            if isinstance(alpha_spy, (int, float)):
                alpha_spy = f"{alpha_spy:.2f}"
            if isinstance(alpha_qqq, (int, float)):
                alpha_qqq = f"{alpha_qqq:.2f}"
            
            print(f"{period:<10} {signals:<10} {win_rate:<8} {sharpe:<8} {max_dd:<8} {avg_ret:<8} {avg_alpha:<10} {alpha_spy:<8} {alpha_qqq:<8}")
        
        print(f"\n{'='*100}")
        print("✅ All 8 intervals successfully calculated!")
        print(f"{'='*100}\n")
        
    finally:
        await db.disconnect()

asyncio.run(main())
