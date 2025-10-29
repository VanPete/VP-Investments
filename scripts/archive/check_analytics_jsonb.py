"""
Quick script to verify new JSONB analytics columns are populated.
"""
import asyncio
import json
from backend.storage.database import get_supabase_database

async def check_analytics():
    db = await get_supabase_database()
    
    # Get latest analytics record
    result = await db.execute_query("""
        SELECT 
            id,
            period_type,
            period_start,
            period_end,
            total_signals,
            score_bucket_performance,
            factor_correlations,
            factor_contributions,
            group_performance,
            backtest_cumulative_returns
        FROM analytics
        ORDER BY created_at DESC
        LIMIT 1
    """)
    
    if not result:
        print("❌ No analytics records found")
        return
    
    record = result[0]
    
    print("=" * 80)
    print("LATEST ANALYTICS RECORD")
    print("=" * 80)
    print(f"ID: {record['id']}")
    print(f"Period: {record['period_type']}")
    print(f"Date range: {record['period_start']} to {record['period_end']}")
    print(f"Total signals: {record['total_signals']}")
    print()
    
    # Check each JSONB field
    fields = [
        'score_bucket_performance',
        'factor_correlations', 
        'factor_contributions',
        'group_performance',
        'backtest_cumulative_returns'
    ]
    
    for field in fields:
        data = record.get(field)
        if data:
            print(f"✅ {field}: POPULATED")
            
            # Show sample structure
            if isinstance(data, dict):
                print(f"   Keys: {list(data.keys())[:10]}")  # First 10 keys
                
                # Show specific details
                if field == 'score_bucket_performance':
                    if 'strong_buy' in data:
                        print(f"   Strong Buy signals: {data['strong_buy'].get('count', 0)}")
                        print(f"   Buy signals: {data.get('buy', {}).get('count', 0)}")
                        print(f"   Hold signals: {data.get('hold', {}).get('count', 0)}")
                
                elif field == 'factor_correlations':
                    if 'group_correlations' in data:
                        labels = data['group_correlations'].get('labels', [])
                        print(f"   Groups: {labels}")
                
                elif field == 'backtest_cumulative_returns':
                    if 'summary' in data:
                        summary = data['summary']
                        print(f"   VP Total Return: {summary.get('vp_total_return', 0):.2%}")
                        print(f"   SPY Total Return: {summary.get('spy_total_return', 0):.2%}")
                        print(f"   VP Sharpe: {summary.get('vp_sharpe', 0):.2f}")
        else:
            print(f"❌ {field}: NULL or empty")
        print()
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check_analytics())
