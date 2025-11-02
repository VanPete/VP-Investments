"""
Verify that group_performance has all 8 metrics per quintile.
"""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from backend.storage.supabase_client import SupabaseClient

# Load environment variables
load_dotenv()

def verify_group_performance():
    """Check if group_performance has all metrics for each quintile."""
    
    # Initialize database connection
    db = SupabaseClient()
    
    # Query analytics table
    print("Fetching analytics data...")
    response = db.client.table('analytics').select('*').order('period_type').execute()
    
    if not response.data:
        print("❌ No analytics data found!")
        return
    
    print(f"\n✅ Found {len(response.data)} analytics rows\n")
    print("=" * 120)
    
    # Check each interval
    for row in response.data:
        interval = row['period_type']
        group_performance = row.get('group_performance')
        
        print(f"\n📊 Interval: {interval}")
        print("-" * 120)
        
        if not group_performance:
            print("  ⚠ No group_performance data")
            continue
        
        # Parse if string
        if isinstance(group_performance, str):
            group_performance = json.loads(group_performance)
        
        # Check each scoring group
        groups = ['technical', 'fundamental', 'news_macro', 'social_alternative', 'risk_stability', 'institutional']
        
        for group in groups:
            if group not in group_performance:
                print(f"  ❌ {group}: MISSING")
                continue
            
            group_data = group_performance[group]
            quintiles = ['top_20pct', 'q2', 'q3', 'q4', 'bottom_20pct']
            
            print(f"\n  📈 {group.replace('_', ' ').title()}:")
            
            for quintile in quintiles:
                if quintile not in group_data:
                    print(f"    ❌ {quintile}: MISSING")
                    continue
                
                metrics = group_data[quintile]
                
                # Expected 8 metrics
                expected_metrics = ['count', 'avg_return', 'win_rate', 'sharpe', 'max_drawdown', 'volatility', 'sortino', 'calmar']
                missing = [m for m in expected_metrics if m not in metrics]
                
                if missing:
                    print(f"    ⚠ {quintile}: Missing {missing}")
                    print(f"       Has: {list(metrics.keys())}")
                else:
                    # Show sample metrics
                    volatility = metrics.get('volatility')
                    sortino = metrics.get('sortino')
                    calmar = metrics.get('calmar')
                    print(f"    ✅ {quintile}: count={metrics['count']}, vol={volatility:.2f if volatility else 'N/A'}, sortino={sortino:.2f if sortino else 'N/A'}, calmar={calmar:.2f if calmar else 'N/A'}")
    
    print("\n" + "=" * 120)
    
    # Also check the individual interval metrics
    print("\n📋 Individual Interval Metrics:")
    print("-" * 120)
    
    for row in response.data:
        interval = row['period_type']
        cagr = row.get('cagr')
        volatility = row.get('volatility')
        sortino = row.get('sortino_ratio')
        calmar = row.get('calmar_ratio')
        rolling_sharpe = row.get('rolling_sharpe_30d')
        
        null_count = sum([
            cagr is None,
            volatility is None,
            sortino is None,
            calmar is None,
            rolling_sharpe is None
        ])
        
        status = "✅" if null_count == 0 else f"⚠ {null_count} NULL"
        
        print(f"  {interval:12s}: {status:15s} | CAGR={cagr:.2f if cagr else 'NULL':>8s} | Vol={volatility:.2f if volatility else 'NULL':>8s} | Sortino={sortino:.2f if sortino else 'NULL':>8s} | Calmar={calmar:.2f if calmar else 'NULL':>8s}")
    
    print("\n" + "=" * 120)

if __name__ == "__main__":
    verify_group_performance()
