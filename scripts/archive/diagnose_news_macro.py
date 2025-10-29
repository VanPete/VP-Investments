"""Diagnose why news/macro scores are still 0.000"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

# 1. Check latest pipeline results
print("=" * 80)
print("PIPELINE RESULTS CHECK")
print("=" * 80)
result_files = sorted(Path('frontend/public/results').glob('pipeline_results_*.json'), reverse=True)
if result_files:
    with open(result_files[0]) as f:
        data = json.load(f)
    
    print(f"Latest file: {result_files[0].name}")
    signals = data.get('signals', [])
    print(f"Total signals: {len(signals)}")
    
    if signals:
        nm_scores = [s.get('news_macro_score', 0) for s in signals]
        print(f"\nNews/Macro Score Stats:")
        print(f"  Min: {min(nm_scores)}")
        print(f"  Max: {max(nm_scores)}")
        print(f"  Unique values: {len(set(nm_scores))}")
        print(f"  All zeros?: {all(s == 0.0 for s in nm_scores)}")
        
        print(f"\nSample tickers with scores:")
        for s in signals[:10]:
            print(f"  {s['ticker']}: news_macro={s.get('news_macro_score', 0):.4f}, coverage={s.get('news_macro_coverage', 0):.2f}")

# 2. Check latest factor monitoring log
print("\n" + "=" * 80)
print("FACTOR MONITORING LOG")
print("=" * 80)
log_files = sorted(Path('logs').glob('factor_monitoring_*.json'), reverse=True)
if log_files:
    with open(log_files[0]) as f:
        data = json.load(f)
    
    print(f"Latest log: {log_files[0].name}")
    nm = data['factors']['news_macro']
    print(f"\nNews/Macro Group:")
    print(f"  Success Rate: {nm['success_rate']*100:.1f}%")
    print(f"  Total calculations: {nm['total_calculations']}")
    print(f"  Successful: {nm['successful']}")
    print(f"  Failed: {nm['failed']}")
    
    print("\nFactors with 0% success rate:")
    for factor, stats in nm['factor_stats'].items():
        if stats['success_rate'] == 0:
            failure_types = stats.get('failures', {})
            print(f"  - {factor}: {sum(failure_types.values())} failures")
            if failure_types:
                for fail_type, count in failure_types.items():
                    print(f"      {fail_type}: {count}")
