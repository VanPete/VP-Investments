"""
Phase 5.5: Volume Testing
Tests pipeline performance with increasing ticker volumes (50, 100, 500)
Runs pipeline directly via Python imports instead of subprocess
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import psutil
import os
import csv
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.pipeline import run_pipeline
from loguru import logger


# Test configurations
VOLUME_TESTS = {
    "small": {
        "name": "50 Tickers",
        "ticker_count": 50,
        "expected_records": 351,  # 1 run + 50 signals + 300 factors
        "description": "Small volume - baseline performance"
    },
    "medium": {
        "name": "100 Tickers",
        "ticker_count": 100,
        "expected_records": 701,  # 1 run + 100 signals + 600 factors
        "description": "Medium volume - 2x scaling test"
    },
    "large": {
        "name": "500 Tickers",
        "ticker_count": 500,
        "expected_records": 3501,  # 1 run + 500 signals + 3000 factors
        "description": "Large volume - high load test"
    }
}


def get_memory_usage() -> Dict[str, float]:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024)
    }


def read_ticker_pool() -> List[str]:
    """Read available tickers from NYSE CSV"""
    csv_path = Path(__file__).parent / "backend" / "core" / "nyse.csv"
    
    if not csv_path.exists():
        print(f"❌ ERROR: nyse.csv not found at {csv_path}")
        return []
    
    tickers = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try both 'Symbol' and 'ticker' column names
            ticker = row.get('Symbol', row.get('ticker', '')).strip().upper()
            if ticker and not any(char in ticker for char in ['$', '^', '.', '-']):
                # Skip tickers with special characters (preferred shares, warrants, etc.)
                tickers.append(ticker)
    
    print(f"✅ Loaded {len(tickers)} tickers from NYSE CSV")
    return tickers


async def run_pipeline_test(config_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run pipeline with specified ticker count and collect metrics"""
    
    print(f"\n{'='*80}")
    print(f"VOLUME TEST: {config['name']}")
    print(f"{'='*80}")
    print(f"Description: {config['description']}")
    print(f"Ticker count: {config['ticker_count']}")
    print(f"Expected DB records: {config['expected_records']}")
    print()
    
    # Load all tickers
    all_tickers = read_ticker_pool()
    if not all_tickers:
        return {"error": "No tickers available", "success": False}
    
    if len(all_tickers) < config['ticker_count']:
        print(f"⚠️  WARNING: Only {len(all_tickers)} tickers available, requested {config['ticker_count']}")
        print(f"⚠️  Using all {len(all_tickers)} tickers instead")
        config['ticker_count'] = len(all_tickers)
    
    # Select ticker subset
    test_tickers = all_tickers[:config['ticker_count']]
    print(f"✅ Selected {len(test_tickers)} tickers for testing")
    print(f"   First 10: {', '.join(test_tickers[:10])}")
    print()
    
    try:
        # Record start metrics
        start_time = time.time()
        start_mem = get_memory_usage()
        
        print(f"🚀 Starting pipeline with {len(test_tickers)} tickers...")
        print(f"📊 Initial memory: RSS={start_mem['rss_mb']:.1f}MB, VMS={start_mem['vms_mb']:.1f}MB")
        print()
        
        # Run pipeline directly
        phase_timings = await run_pipeline(tickers=test_tickers)
        
        # Record end metrics
        end_time = time.time()
        end_mem = get_memory_usage()
        duration = end_time - start_time
        
        # Calculate metrics
        mem_delta = {
            "rss_mb": end_mem['rss_mb'] - start_mem['rss_mb'],
            "vms_mb": end_mem['vms_mb'] - start_mem['vms_mb']
        }
        
        per_ticker_time = duration / len(test_tickers) if test_tickers else 0
        
        # Extract phase timings from result
        timings = {}
        if isinstance(phase_timings, dict):
            timings = {
                'phase1': phase_timings.get('phase1', 0),
                'phase2': phase_timings.get('phase2', 0),
                'phase3': phase_timings.get('phase3', 0),
                'phase4': phase_timings.get('phase4', 0),
                'phase5': phase_timings.get('phase5', 0)
            }
        
        metrics = {
            "config_name": config_name,
            "ticker_count": len(test_tickers),
            "duration_seconds": duration,
            "per_ticker_seconds": per_ticker_time,
            "expected_records": config['expected_records'],
            "phase_timings": timings,
            "memory_start_mb": start_mem['rss_mb'],
            "memory_end_mb": end_mem['rss_mb'],
            "memory_delta_mb": mem_delta['rss_mb'],
            "success": True
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS: {config['name']}")
        print(f"{'='*80}")
        print(f"✅ Status: SUCCESS")
        print(f"⏱️  Total duration: {duration:.1f}s")
        print(f"📊 Per-ticker time: {per_ticker_time:.2f}s")
        print(f"💾 Memory delta: {mem_delta['rss_mb']:+.1f}MB")
        print()
        
        if timings:
            print("Phase breakdown:")
            for phase, timing_val in timings.items():
                if isinstance(timing_val, (int, float)):
                    pct = (timing_val / duration * 100) if duration > 0 else 0
                    print(f"  {phase}: {timing_val:6.1f}s ({pct:5.1f}%)")
        
        print(f"{'='*80}\n")
        
        return metrics
        
    except Exception as e:
        print(f"❌ ERROR running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {
            "config_name": config_name,
            "ticker_count": config['ticker_count'],
            "error": str(e),
            "success": False
        }


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and compare results across volume tests"""
    
    print(f"\n{'='*80}")
    print("VOLUME TEST ANALYSIS")
    print(f"{'='*80}\n")
    
    # Summary table
    print("Performance Summary:")
    print(f"{'Volume':<15} {'Tickers':>10} {'Duration':>12} {'Per-Ticker':>12} {'Phase 5':>12} {'Memory Δ':>12}")
    print(f"{'-'*80}")
    
    for result in results:
        if result.get('success'):
            duration = result.get('duration_seconds', 0)
            per_ticker = result.get('per_ticker_seconds', 0)
            phase5 = result.get('phase_timings', {}).get('phase5', 0)
            mem_delta = result.get('memory_delta_mb', 0)
            
            print(f"{result['config_name']:<15} "
                  f"{result['ticker_count']:>10} "
                  f"{duration:>10.1f}s "
                  f"{per_ticker:>10.2f}s "
                  f"{phase5:>10.1f}s "
                  f"{mem_delta:>+10.1f}MB")
    
    print()
    
    # Scaling analysis
    if len(results) >= 2:
        print("Scaling Analysis:")
        base = results[0]
        for result in results[1:]:
            if base.get('success') and result.get('success'):
                ticker_ratio = result['ticker_count'] / base['ticker_count']
                time_ratio = result['duration_seconds'] / base['duration_seconds']
                
                scaling_efficiency = (ticker_ratio / time_ratio) * 100
                
                print(f"\n{result['config_name']} vs {base['config_name']}:")
                print(f"  Ticker increase: {ticker_ratio:.1f}x")
                print(f"  Time increase: {time_ratio:.1f}x")
                print(f"  Scaling efficiency: {scaling_efficiency:.1f}% (100% = linear scaling)")
    
    print(f"\n{'='*80}\n")


def save_results(results: List[Dict[str, Any]]):
    """Save results to JSON file"""
    output_path = Path(__file__).parent / "docs" / "PHASE5_5_VOLUME_TEST_RESULTS.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": results
        }, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")


async def main():
    """Run volume tests"""
    
    print(f"\n{'='*80}")
    print("PHASE 5.5: VOLUME TESTING")
    print(f"{'='*80}")
    print("Testing pipeline scalability with increasing ticker volumes")
    print()
    
    # Ask which tests to run
    print("Available tests:")
    for key, config in VOLUME_TESTS.items():
        print(f"  {key}: {config['name']} ({config['ticker_count']} tickers)")
    
    print("\nEnter test names to run (comma-separated), or 'all' for all tests:")
    user_input = input("> ").strip().lower()
    
    if user_input == 'all':
        tests_to_run = list(VOLUME_TESTS.keys())
    else:
        tests_to_run = [t.strip() for t in user_input.split(',') if t.strip() in VOLUME_TESTS]
    
    if not tests_to_run:
        print("❌ No valid tests selected. Exiting.")
        return
    
    print(f"\n✅ Running {len(tests_to_run)} test(s): {', '.join(tests_to_run)}")
    
    # Run tests
    results = []
    for test_name in tests_to_run:
        config = VOLUME_TESTS[test_name]
        result = await run_pipeline_test(test_name, config)
        results.append(result)
        
        # Brief pause between tests
        if test_name != tests_to_run[-1]:
            print("\n⏸️  Pausing 5 seconds before next test...")
            await asyncio.sleep(5)
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    save_results(results)
    
    print("\n✅ Volume testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
