from backend.storage.database import SupabaseInterface
import json

db = SupabaseInterface()

# Get the 1d analytics row
result = db.client.table('analytics').select('*').eq('period_type', '1d').execute()

if result.data:
    analytics = result.data[0]
    
    print("\n" + "="*80)
    print("1D ANALYTICS DATA INSPECTION")
    print("="*80)
    
    # Basic metrics
    print(f"\n📊 Basic Metrics:")
    print(f"  Total Signals: {analytics.get('total_signals')}")
    print(f"  Period: {analytics.get('period_start')} to {analytics.get('period_end')}")
    
    # Performance metrics
    print(f"\n💰 Performance Metrics:")
    print(f"  CAGR: {analytics.get('cagr')}")
    print(f"  Volatility: {analytics.get('volatility')}")
    print(f"  Sharpe Ratio: {analytics.get('sharpe_ratio')}")
    print(f"  Sortino Ratio: {analytics.get('sortino_ratio')}")
    print(f"  Calmar Ratio: {analytics.get('calmar_ratio')}")
    
    print(f"\n📈 Benchmark Metrics:")
    print(f"  Alpha vs SPY: {analytics.get('alpha_vs_spy')}")
    print(f"  Beta vs SPY: {analytics.get('beta_vs_spy')}")
    print(f"  Alpha vs QQQ: {analytics.get('alpha_vs_qqq')}")
    print(f"  Beta vs QQQ: {analytics.get('beta_vs_qqq')}")
    
    print(f"\n🎯 Predictive Metrics:")
    print(f"  Win Rate: {analytics.get('win_rate')}")
    print(f"  IC Mean: {analytics.get('ic_mean')}")
    print(f"  IC Std: {analytics.get('ic_std')}")
    print(f"  Hit Rate (Top 10%): {analytics.get('hit_rate_top_decile')}")
    print(f"  Profit Factor: {analytics.get('profit_factor')}")
    print(f"  Win/Loss Ratio: {analytics.get('win_loss_ratio')}")
    
    # Score bucket performance
    print(f"\n📊 Score Bucket Performance:")
    if analytics.get('score_bucket_performance'):
        sbp = analytics['score_bucket_performance']
        print(f"  Type: {type(sbp)}")
        if isinstance(sbp, dict):
            print(f"  Keys: {list(sbp.keys())}")
            # Show structure of first bucket
            if sbp:
                first_bucket = list(sbp.keys())[0]
                print(f"\n  Sample bucket ({first_bucket}):")
                print(f"    Structure: {json.dumps(sbp[first_bucket], indent=6)[:500]}...")
    else:
        print("  ❌ No score bucket performance data")
    
    # Backtest data
    print(f"\n📉 Backtest Cumulative Returns:")
    if analytics.get('backtest_cumulative_returns'):
        bt = analytics['backtest_cumulative_returns']
        print(f"  Type: {type(bt)}")
        if isinstance(bt, dict):
            print(f"  Keys: {list(bt.keys())}")
            if 'summary' in bt:
                print(f"  Summary: {bt['summary']}")
            if 'daily_returns' in bt:
                print(f"  Daily returns count: {len(bt['daily_returns'])}")
    else:
        print("  ❌ No backtest data")
    
    print("\n" + "="*80)
else:
    print("❌ No 1d analytics found")
