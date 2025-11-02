"""
Quick script to check if system is ready for factor-return correlations.
Checks:
1. How many performance records exist per interval
2. Sample factor data structure
3. Estimated computation time
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from backend.storage.database import get_supabase_database
import yaml

async def check_readiness():
    print("=" * 80)
    print("🔍 FACTOR-RETURN CORRELATION READINESS CHECK")
    print("=" * 80)
    
    # Connect to database
    db = await get_supabase_database()
    
    # 1. Check performance data availability
    print("\n📊 PERFORMANCE DATA AVAILABILITY:")
    print("-" * 80)
    
    intervals = ['1d', '3d', '7d', '10d', '14d', '30d', '90d']
    
    for interval in intervals:
        result = db.client.table('performance').select('signal_id', count='exact').not_.is_(f'return_{interval}', 'null').execute()
        count = result.count if hasattr(result, 'count') else len(result.data)
        status = "✅" if count >= 30 else "⚠️" if count >= 10 else "❌"
        print(f"  {status} {interval:>4}: {count:>4} records with non-null returns")
    
    # 2. Load factor definitions
    print("\n📋 FACTOR DEFINITIONS:")
    print("-" * 80)
    
    factor_file = project_root / 'config' / 'factor_to_group.yaml'
    with open(factor_file, 'r') as f:
        factor_config = yaml.safe_load(f)
    
    total_factors = 0
    for group, factors in factor_config.items():
        if group == 'validation':
            continue
        factor_count = len(factors)
        total_factors += factor_count
        print(f"  • {group:30} {factor_count:>3} factors")
    
    print(f"\n  📌 TOTAL FACTORS: {total_factors}")
    
    # 3. Sample factor data structure
    print("\n🔬 SAMPLE FACTOR DATA STRUCTURE:")
    print("-" * 80)
    
    result = db.client.table('performance').select(
        'signal_id, signals(technical_data, fundamental_data)'
    ).limit(1).execute()
    
    if result.data:
        sample = result.data[0]
        print(f"  Signal ID: {sample.get('signal_id')}")
        
        signals = sample.get('signals', {})
        if signals:
            tech_data = signals.get('technical_data')
            fund_data = signals.get('fundamental_data')
            
            print(f"\n  Technical data type: {type(tech_data)}")
            if tech_data:
                if isinstance(tech_data, dict):
                    print(f"  Technical factors available: {len(tech_data)} keys")
                    print(f"  Sample keys: {list(tech_data.keys())[:5]}")
                else:
                    print(f"  Technical data: {tech_data}")
            
            print(f"\n  Fundamental data type: {type(fund_data)}")
            if fund_data:
                if isinstance(fund_data, dict):
                    print(f"  Fundamental factors available: {len(fund_data)} keys")
                    print(f"  Sample keys: {list(fund_data.keys())[:5]}")
                else:
                    print(f"  Fundamental data: {fund_data}")
        else:
            print("  ⚠️ No signals data found in sample")
    
    # 4. Estimate computation time
    print("\n⏱️  ESTIMATED COMPUTATION TIME:")
    print("-" * 80)
    
    correlations_per_interval = total_factors
    total_correlations = correlations_per_interval * len(intervals)
    
    # Assume 1ms per correlation (scipy.pearsonr is fast)
    estimated_time = total_correlations * 0.001
    
    print(f"  • Correlations per interval: {correlations_per_interval}")
    print(f"  • Number of intervals: {len(intervals)}")
    print(f"  • Total correlations: {total_correlations}")
    print(f"  • Estimated time: ~{estimated_time:.2f} seconds")
    
    if estimated_time < 2:
        print(f"  ✅ FAST - Well within acceptable range")
    elif estimated_time < 5:
        print(f"  ⚠️ MODERATE - May want to optimize")
    else:
        print(f"  ❌ SLOW - Need optimization (parallelization)")
    
    # 5. Readiness summary
    print("\n" + "=" * 80)
    print("✅ READINESS SUMMARY")
    print("=" * 80)
    
    result = db.client.table('performance').select('signal_id', count='exact').not_.is_('return_1d', 'null').execute()
    ready_intervals = sum(1 for interval in intervals if (db.client.table('performance').select('signal_id', count='exact').not_.is_(f'return_{interval}', 'null').execute().count if hasattr(db.client.table('performance').select('signal_id', count='exact').not_.is_(f'return_{interval}', 'null').execute(), 'count') else 0) >= 30)
    
    print(f"  • Total factors defined: {total_factors}")
    print(f"  • Intervals with data (n≥30): {ready_intervals}/{len(intervals)}")
    print(f"  • Estimated computation: ~{estimated_time:.2f}s")
    
    if ready_intervals >= 2 and estimated_time < 5:
        print(f"\n  ✅ SYSTEM READY for factor-return correlation implementation!")
    else:
        print(f"\n  ⚠️ System may benefit from waiting for more data or optimization")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(check_readiness())
