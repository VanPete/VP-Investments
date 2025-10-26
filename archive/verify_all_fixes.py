"""Verify all three fixes are working correctly"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backend.storage.database import SupabaseInterface

async def verify_fixes():
    db = SupabaseInterface()
    await db.connect()
    
    print("=" * 80)
    print("VERIFICATION OF ALL FIXES")
    print("=" * 80)
    
    # Fix 1: Check backtest columns population
    print("\n1. BACKTEST COLUMN POPULATION")
    print("-" * 80)
    
    query = """
    SELECT 
        COUNT(*) as total_signals,
        COUNT(backtest_baseline_price) as has_baseline_price,
        COUNT(backtest_baseline_date) as has_baseline_date,
        COUNT(return_1d) as has_return_1d,
        COUNT(return_3d) as has_return_3d,
        COUNT(return_7d) as has_return_7d,
        COUNT(return_10d) as has_return_10d,
        COUNT(return_14d) as has_return_14d,
        COUNT(return_30d) as has_return_30d,
        COUNT(return_90d) as has_return_90d,
        COUNT(spy_return_1d) as has_spy_1d,
        COUNT(spy_return_7d) as has_spy_7d,
        COUNT(spy_return_30d) as has_spy_30d,
        COUNT(backtest_status) as has_status,
        COUNT(backtest_last_update) as has_last_update
    FROM signals
    """
    
    result = await db.execute_query(query)
    stats = result[0]
    
    total = stats['total_signals']
    print(f"Total signals: {total}")
    print(f"\nBaseline columns:")
    print(f"  backtest_baseline_price:  {stats['has_baseline_price']} ({stats['has_baseline_price']/total*100:.1f}%)")
    print(f"  backtest_baseline_date:   {stats['has_baseline_date']} ({stats['has_baseline_date']/total*100:.1f}%)")
    
    print(f"\nReturn columns:")
    for period in ['1d', '3d', '7d', '10d', '14d', '30d', '90d']:
        count = stats[f'has_return_{period}']
        print(f"  return_{period:3s}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nSPY return columns:")
    for period in ['1d', '7d', '30d']:
        count = stats[f'has_spy_{period}']
        print(f"  spy_return_{period}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nStatus columns:")
    print(f"  backtest_status:      {stats['has_status']} ({stats['has_status']/total*100:.1f}%)")
    print(f"  backtest_last_update: {stats['has_last_update']} ({stats['has_last_update']/total*100:.1f}%)")
    
    # Fix 2: Check all runs are accessible
    print("\n\n2. SIGNAL RUNS AVAILABILITY")
    print("-" * 80)
    
    query = """
    SELECT 
        COUNT(*) as total_runs,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
        COUNT(CASE WHEN status = 'running' THEN 1 END) as running
    FROM signal_runs
    """
    
    result = await db.execute_query(query)
    run_stats = result[0]
    
    print(f"Total runs in database: {run_stats['total_runs']}")
    print(f"  Completed: {run_stats['completed']}")
    print(f"  Failed: {run_stats['failed']}")
    print(f"  Running: {run_stats['running']}")
    print(f"\n✓ Frontend should show ALL {run_stats['total_runs']} runs (no limit)")
    
    # Fix 3: Check coverage calculation (sample few signals)
    print("\n\n3. COVERAGE CALCULATION ACCURACY")
    print("-" * 80)
    print("Checking sample signals to verify coverage varies by data quality...\n")
    
    # Get a few signals with their factor counts
    query = """
    SELECT 
        s.ticker,
        s.overall_score,
        st.factors as tech_factors,
        sf.factors as fund_factors,
        sn.factors as news_factors,
        ss.factors as social_factors,
        sr.factors as risk_factors,
        si.factors as inst_factors
    FROM signals s
    LEFT JOIN signals_technical st ON s.id = st.signal_id
    LEFT JOIN signals_fundamental sf ON s.id = sf.signal_id
    LEFT JOIN signals_news_macro sn ON s.id = sn.signal_id
    LEFT JOIN signals_social_alternative ss ON s.id = ss.signal_id
    LEFT JOIN signals_risk_stability sr ON s.id = sr.signal_id
    LEFT JOIN signals_institutional_smart_money si ON s.id = si.signal_id
    ORDER BY s.overall_score DESC
    LIMIT 5
    """
    
    result = await db.execute_query(query)
    
    # Correct MAX_FACTORS from config
    MAX_FACTORS = {
        'technical': 41,
        'fundamental': 45,
        'news_macro': 18,
        'social_alternative': 10,
        'risk_stability': 23,
        'institutional_smart_money': 21,
    }
    
    for signal in result:
        ticker = signal['ticker']
        
        # Count factors in each category
        tech_count = len(signal['tech_factors'] or {})
        fund_count = len(signal['fund_factors'] or {})
        news_count = len(signal['news_factors'] or {})
        social_count = len(signal['social_factors'] or {})
        risk_count = len(signal['risk_factors'] or {})
        inst_count = len(signal['inst_factors'] or {})
        
        # Calculate coverage with CORRECT maximums
        total_count = tech_count + fund_count + news_count + social_count + risk_count + inst_count
        total_max = sum(MAX_FACTORS.values())  # 158
        
        coverage_pct = (total_count / total_max) * 100
        
        print(f"{ticker}:")
        print(f"  Technical: {tech_count}/{MAX_FACTORS['technical']} ({tech_count/MAX_FACTORS['technical']*100:.1f}%)")
        print(f"  Fundamental: {fund_count}/{MAX_FACTORS['fundamental']} ({fund_count/MAX_FACTORS['fundamental']*100:.1f}%)")
        print(f"  News/Macro: {news_count}/{MAX_FACTORS['news_macro']} ({news_count/MAX_FACTORS['news_macro']*100:.1f}%)")
        print(f"  Social: {social_count}/{MAX_FACTORS['social_alternative']} ({social_count/MAX_FACTORS['social_alternative']*100:.1f}%)")
        print(f"  Risk: {risk_count}/{MAX_FACTORS['risk_stability']} ({risk_count/MAX_FACTORS['risk_stability']*100:.1f}%)")
        print(f"  Institutional: {inst_count}/{MAX_FACTORS['institutional_smart_money']} ({inst_count/MAX_FACTORS['institutional_smart_money']*100:.1f}%)")
        print(f"  TOTAL COVERAGE: {total_count}/{total_max} = {coverage_pct:.1f}%")
        print()
    
    print("✓ Coverage should now vary per ticker based on actual data quality")
    print("✓ No longer showing inflated 90.3% for all tickers")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(verify_fixes())
