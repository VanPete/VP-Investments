"""Quick test to verify the new interval-specific methods work"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.phases.phase7_analytics import AnalyticsEngine
from backend.storage.database import get_supabase_database

load_dotenv()

async def test_interval_methods():
    print("\n" + "="*80)
    print("TESTING INTERVAL-SPECIFIC METRIC CALCULATIONS")
    print("="*80)
    
    # Initialize
    db = await get_supabase_database()
    engine = AnalyticsEngine(db=db)
    
    # Fetch sample data
    from datetime import datetime, timezone
    period_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    period_end = datetime.now(timezone.utc)
    
    result = db.client.table('performance').select('''
        *,
        signals!inner(
            ticker,
            overall_score,
            technical_score,
            fundamental_score,
            news_macro_score,
            social_alternative_score,
            risk_stability_score,
            institutional_smart_money_score
        )
    ''').gte('baseline_date', period_start.isoformat()).lte('baseline_date', period_end.isoformat()).limit(100).execute()
    
    performance_data = result.data
    print(f"\nFetched {len(performance_data)} performance records")
    
    # Test each interval
    intervals = ['1d', '3d', '7d']
    
    for interval in intervals:
        print(f"\n{'='*80}")
        print(f"TESTING INTERVAL: {interval}")
        print(f"{'='*80}")
        
        try:
            cagr = engine._calculate_cagr_for_interval(performance_data, interval)
            print(f"  CAGR: {cagr}")
            
            vol = engine._calculate_volatility_for_interval(performance_data, interval)
            print(f"  Volatility: {vol}")
            
            sortino = engine._calculate_sortino_ratio_for_interval(performance_data, interval)
            print(f"  Sortino: {sortino}")
            
            calmar = engine._calculate_calmar_ratio_for_interval(performance_data, interval)
            print(f"  Calmar: {calmar}")
            
            bench_corr = engine._calculate_benchmark_correlation_for_interval(performance_data, interval)
            print(f"  Benchmark Correlations: {bench_corr}")
            
            rolling = engine._calculate_rolling_sharpe_for_interval(performance_data, interval)
            print(f"  Rolling Sharpe: {len(rolling)} datapoints")
            
            print(f"  ✅ All methods executed successfully")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test_interval_methods())
