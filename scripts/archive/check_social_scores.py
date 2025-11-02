"""Check signal scores to diagnose the social_alternative_score issue."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    await db.connect()
    
    try:
        # Get sample signals
        result = await db.execute_query("""
            SELECT 
                ticker,
                overall_score,
                technical_score,
                fundamental_score,
                news_macro_score,
                social_alternative_score,
                risk_stability_score,
                institutional_smart_money_score
            FROM signals 
            LIMIT 10
        """)
        
        print(f"\n{'='*100}")
        print(f"Sample Signal Scores (first 10)")
        print(f"{'='*100}\n")
        
        print(f"{'Ticker':<8} {'Overall':<10} {'Tech':<8} {'Fund':<8} {'News':<8} {'Social':<8} {'Risk':<8} {'Inst':<8}")
        print(f"{'-'*100}")
        
        for row in result:
            print(f"{row['ticker']:<8} "
                  f"{row['overall_score']:<10.3f} "
                  f"{row['technical_score']:<8.3f} "
                  f"{row['fundamental_score']:<8.3f} "
                  f"{row['news_macro_score']:<8.3f} "
                  f"{row['social_alternative_score']:<8} "
                  f"{row['risk_stability_score']:<8.3f} "
                  f"{row['institutional_smart_money_score']:<8.3f}")
        
        # Check for NULL vs 0
        null_check = await db.execute_query("""
            SELECT 
                COUNT(*) as total,
                COUNT(social_alternative_score) as has_social,
                SUM(CASE WHEN social_alternative_score = 0 THEN 1 ELSE 0 END) as zero_social,
                SUM(CASE WHEN social_alternative_score IS NULL THEN 1 ELSE 0 END) as null_social,
                AVG(social_alternative_score) as avg_social
            FROM signals
        """)
        
        print(f"\n{'='*100}")
        print(f"Social Alternative Score Statistics")
        print(f"{'='*100}\n")
        print(f"Total signals: {null_check[0]['total']}")
        print(f"Has social score: {null_check[0]['has_social']}")
        print(f"Zero values: {null_check[0]['zero_social']}")
        print(f"NULL values: {null_check[0]['null_social']}")
        print(f"Average: {null_check[0]['avg_social']}")
        
    finally:
        await db.disconnect()

asyncio.run(main())
