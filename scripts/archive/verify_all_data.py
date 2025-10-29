"""Verify that our fixes are working correctly"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backend.storage.database import SupabaseInterface

async def verify_fixes():
    db = SupabaseInterface()
    await db.connect()
    
    print("=" * 80)
    print("VERIFICATION OF FIXES")
    print("=" * 80)
    
    # Fix 1: Check news_macro_score is NOT NULL
    print("\n1. CHECKING news_macro_score POPULATION")
    print("-" * 80)
    
    query = """
    SELECT 
        ticker, 
        news_macro_score,
        (SELECT COUNT(*) FROM signals_news_macro WHERE signal_id = signals.id) as factor_count
    FROM signals
    ORDER BY created_at DESC
    LIMIT 5
    """
    
    results = await db.execute_query(query)
    
    print(f"\nRecent signals:")
    for row in results:
        ticker = row['ticker']
        score = row['news_macro_score']
        factors = row['factor_count']
        status = "✅ OK" if score is not None else "❌ NULL"
        score_str = f"{score:.4f}" if score is not None else "NULL"
        print(f"  {ticker:6} | news_macro_score: {score_str:>8} | factors: {factors:>2} | {status}")
    
    # Count NULL news_macro_scores
    null_query = """
    SELECT COUNT(*) as null_count
    FROM signals
    WHERE news_macro_score IS NULL
      AND EXISTS (SELECT 1 FROM signals_news_macro WHERE signal_id = signals.id)
    """
    null_result = await db.execute_query(null_query)
    null_count = null_result[0]['null_count']
    
    if null_count == 0:
        print(f"\n✅ FIX #1 VERIFIED: All signals with news_macro factors have scores populated")
    else:
        print(f"\n❌ FIX #1 FAILED: {null_count} signals have NULL news_macro_score despite having factors")
    
    # Fix 2: Check total_coverage is populated
    print("\n\n2. CHECKING total_coverage POPULATION")
    print("-" * 80)
    
    coverage_query = """
    SELECT ticker, total_coverage, created_at
    FROM signals
    ORDER BY created_at DESC
    LIMIT 5
    """
    
    cov_results = await db.execute_query(coverage_query)
    
    print(f"\nRecent signals:")
    for row in cov_results:
        ticker = row['ticker']
        coverage = row['total_coverage']
        status = "✅ OK" if coverage is not None else "❌ NULL"
        print(f"  {ticker:6} | total_coverage: {coverage*100 if coverage else 0:>6.2f}% | {status}")
    
    # Count NULL total_coverage
    null_cov_query = "SELECT COUNT(*) as count FROM signals WHERE total_coverage IS NULL"
    null_cov_result = await db.execute_query(null_cov_query)
    null_cov_count = null_cov_result[0]['count']
    
    if null_cov_count == 0:
        print(f"\n✅ FIX #2 VERIFIED: All signals have total_coverage populated")
    else:
        print(f"\n❌ FIX #2 FAILED: {null_cov_count} signals have NULL total_coverage")
    
    # Summary statistics
    print("\n\n3. SUMMARY STATISTICS")
    print("-" * 80)
    
    stats_query = """
    SELECT 
        COUNT(*) as total_signals,
        COUNT(news_macro_score) as has_news_score,
        COUNT(total_coverage) as has_coverage,
        AVG(total_coverage) as avg_coverage,
        MIN(total_coverage) as min_coverage,
        MAX(total_coverage) as max_coverage
    FROM signals
    """
    
    stats = await db.execute_query(stats_query)
    s = stats[0]
    
    print(f"\nTotal signals: {s['total_signals']}")
    print(f"Signals with news_macro_score: {s['has_news_score']} ({s['has_news_score']/s['total_signals']*100:.1f}%)")
    print(f"Signals with total_coverage: {s['has_coverage']} ({s['has_coverage']/s['total_signals']*100:.1f}%)")
    print(f"Average coverage: {s['avg_coverage']*100:.2f}%")
    print(f"Coverage range: {s['min_coverage']*100:.2f}% - {s['max_coverage']*100:.2f}%")
    
    # Check a specific ticker's detail
    print("\n\n4. DETAILED TICKER CHECK (NVDA)")
    print("-" * 80)
    
    detail_query = """
    SELECT 
        s.ticker,
        s.overall_score,
        s.technical_score,
        s.fundamental_score,
        s.news_macro_score,
        s.social_alternative_score,
        s.risk_stability_score,
        s.institutional_smart_money_score,
        s.total_coverage
    FROM signals s
    WHERE s.ticker = 'NVDA'
    ORDER BY s.created_at DESC
    LIMIT 1
    """
    
    nvda = await db.execute_query(detail_query)
    
    if nvda:
        n = nvda[0]
        print(f"\nNVDA Signal Details:")
        print(f"  Overall Score: {n['overall_score']:.4f}")
        print(f"  Technical: {n['technical_score']:.4f}")
        print(f"  Fundamental: {n['fundamental_score']:.4f}")
        print(f"  News/Macro: {n['news_macro_score']:.4f}" if n['news_macro_score'] else "  News/Macro: NULL ❌")
        print(f"  Social: {n['social_alternative_score']:.4f}")
        print(f"  Risk: {n['risk_stability_score']:.4f}")
        print(f"  Institutional: {n['institutional_smart_money_score']:.4f}")
        print(f"  Total Coverage: {n['total_coverage']*100:.2f}%" if n['total_coverage'] else "  Total Coverage: NULL ❌")
    
    print("\n" + "=" * 80)
    if null_count == 0 and null_cov_count == 0:
        print("✅ ALL FIXES VERIFIED SUCCESSFULLY!")
    else:
        print("❌ SOME FIXES FAILED - SEE DETAILS ABOVE")
    print("=" * 80)
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(verify_fixes())
