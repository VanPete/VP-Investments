"""Check current database state for backfill planning."""
import asyncio
from backend.storage.database import SupabaseInterface
from datetime import datetime, timezone

async def main():
    db = SupabaseInterface()
    
    print("\n" + "="*80)
    print("DATABASE BACKFILL STATUS CHECK")
    print("="*80)
    
    # 1. Check signals table
    print("\n[1] SIGNALS TABLE")
    print("-" * 80)
    signals_result = db.client.table('signals').select(
        'id, ticker, created_at, run_id, sector, current_price'
    ).order('created_at', desc=True).limit(100).execute()
    
    if signals_result.data:
        signals = signals_result.data
        total = len(signals)
        with_sector = sum(1 for s in signals if s.get('sector'))
        with_price = sum(1 for s in signals if s.get('current_price'))
        
        print(f"Total signals: {total}")
        print(f"With sector: {with_sector}/{total} ({with_sector/total*100:.1f}%)")
        print(f"With current_price: {with_price}/{total} ({with_price/total*100:.1f}%)")
        
        # Group by run_id
        run_ids = {}
        for s in signals:
            run_id = s.get('run_id')
            if run_id:
                if run_id not in run_ids:
                    run_ids[run_id] = {
                        'count': 0,
                        'created_at': s.get('created_at'),
                        'with_sector': 0,
                        'with_price': 0
                    }
                run_ids[run_id]['count'] += 1
                if s.get('sector'):
                    run_ids[run_id]['with_sector'] += 1
                if s.get('current_price'):
                    run_ids[run_id]['with_price'] += 1
        
        print(f"\nSignals by run_id (last 5 runs):")
        for run_id, info in sorted(run_ids.items(), key=lambda x: x[1]['created_at'], reverse=True)[:5]:
            created = info['created_at'][:10] if info['created_at'] else 'N/A'
            print(f"  {run_id[:8]}... | {created} | {info['count']:2d} signals | "
                  f"sector:{info['with_sector']:2d}/{info['count']:2d} | "
                  f"price:{info['with_price']:2d}/{info['count']:2d}")
    
    # 2. Check performance table
    print("\n[2] PERFORMANCE TABLE")
    print("-" * 80)
    perf_result = db.client.table('performance').select(
        'id, baseline_date, return_1d, spy_return_1d, qqq_return_1d, '
        'sector, intervals_completed, signals!inner(ticker)'
    ).order('baseline_date', desc=True).limit(100).execute()
    
    if perf_result.data:
        perf_records = perf_result.data
        total_perf = len(perf_records)
        
        # Check data completeness
        with_return_1d = sum(1 for p in perf_records if p.get('return_1d') is not None)
        with_spy_1d = sum(1 for p in perf_records if p.get('spy_return_1d') is not None)
        with_qqq_1d = sum(1 for p in perf_records if p.get('qqq_return_1d') is not None)
        with_sector = sum(1 for p in perf_records if p.get('sector'))
        
        # Check intervals
        completed_intervals = {}
        for p in perf_records:
            intervals = p.get('intervals_completed', [])
            for interval in intervals:
                completed_intervals[interval] = completed_intervals.get(interval, 0) + 1
        
        print(f"Total performance records: {total_perf}")
        print(f"With return_1d: {with_return_1d}/{total_perf} ({with_return_1d/total_perf*100:.1f}%)")
        print(f"With spy_return_1d: {with_spy_1d}/{total_perf} ({with_spy_1d/total_perf*100:.1f}%)")
        print(f"With qqq_return_1d: {with_qqq_1d}/{total_perf} ({with_qqq_1d/total_perf*100:.1f}%)")
        print(f"With sector: {with_sector}/{total_perf} ({with_sector/total_perf*100:.1f}%)")
        
        print(f"\nCompleted intervals distribution:")
        for interval in sorted(completed_intervals.keys()):
            count = completed_intervals[interval]
            print(f"  {interval:2d}d: {count:3d} records ({count/total_perf*100:.1f}%)")
        
        # Check age of signals
        now = datetime.now(timezone.utc)
        age_buckets = {'<1d': 0, '1-7d': 0, '7-30d': 0, '30-90d': 0, '>90d': 0}
        
        for p in perf_records:
            baseline_date = p.get('baseline_date')
            if baseline_date:
                baseline_dt = datetime.fromisoformat(baseline_date.replace('Z', '+00:00'))
                days_old = (now - baseline_dt).days
                
                if days_old < 1:
                    age_buckets['<1d'] += 1
                elif days_old < 7:
                    age_buckets['1-7d'] += 1
                elif days_old < 30:
                    age_buckets['7-30d'] += 1
                elif days_old < 90:
                    age_buckets['30-90d'] += 1
                else:
                    age_buckets['>90d'] += 1
        
        print(f"\nAge distribution:")
        for bucket, count in age_buckets.items():
            print(f"  {bucket:6s}: {count:3d} records ({count/total_perf*100:.1f}%)")
    
    # 3. Check analytics table
    print("\n[3] ANALYTICS TABLE")
    print("-" * 80)
    analytics_result = db.client.table('analytics').select(
        'id, created_at, total_signals, win_rate_1d, win_rate_7d, win_rate_30d, '
        'sharpe_ratio_1d, sharpe_ratio_7d'
    ).order('created_at', desc=True).limit(10).execute()
    
    if analytics_result.data:
        print(f"Total analytics records: {len(analytics_result.data)}")
        print(f"\nRecent analytics:")
        for a in analytics_result.data[:5]:
            created = a.get('created_at', '')[:19]
            total_sig = a.get('total_signals', 0)
            wr_1d = a.get('win_rate_1d')
            wr_7d = a.get('win_rate_7d')
            wr_30d = a.get('win_rate_30d')
            
            wr_1d_str = f"{wr_1d:.2f}%" if wr_1d is not None else "NULL"
            wr_7d_str = f"{wr_7d:.2f}%" if wr_7d is not None else "NULL"
            wr_30d_str = f"{wr_30d:.2f}%" if wr_30d is not None else "NULL"
            
            print(f"  {created} | {total_sig:3d} signals | "
                  f"WR: 1d={wr_1d_str:7s} 7d={wr_7d_str:7s} 30d={wr_30d_str:7s}")
    else:
        print("No analytics records found")
    
    # 4. Summary & Recommendations
    print("\n" + "="*80)
    print("BACKFILL RECOMMENDATIONS")
    print("="*80)
    
    if signals_result.data:
        signals_needing_sector = [s for s in signals if not s.get('sector')]
        signals_needing_price = [s for s in signals if not s.get('current_price')]
        
        print(f"\n[SIGNALS TABLE]")
        if signals_needing_sector:
            print(f"  ⚠️  {len(signals_needing_sector)} signals missing sector data")
            print(f"     → Can backfill from yfinance")
        else:
            print(f"  ✅ All signals have sector data")
        
        if signals_needing_price:
            print(f"  ⚠️  {len(signals_needing_price)} signals missing current_price")
            print(f"     → Can backfill from yfinance")
        else:
            print(f"  ✅ All signals have current_price")
    
    if perf_result.data:
        perf_needing_qqq = [p for p in perf_records if p.get('qqq_return_1d') is None and p.get('intervals_completed')]
        perf_needing_sector = [p for p in perf_records if not p.get('sector')]
        
        # Count how many could have more intervals
        now = datetime.now(timezone.utc)
        can_add_intervals = []
        for p in perf_records:
            baseline_date = p.get('baseline_date')
            intervals = p.get('intervals_completed', [])
            if baseline_date:
                baseline_dt = datetime.fromisoformat(baseline_date.replace('Z', '+00:00'))
                days_old = (now - baseline_dt).days
                
                # Check which intervals could be calculated
                possible_intervals = [i for i in [1, 3, 7, 10, 14, 30, 90] if days_old >= i]
                missing_intervals = [i for i in possible_intervals if i not in intervals]
                
                if missing_intervals:
                    can_add_intervals.append({
                        'ticker': p.get('signals', {}).get('ticker'),
                        'days_old': days_old,
                        'missing': missing_intervals
                    })
        
        print(f"\n[PERFORMANCE TABLE]")
        if perf_needing_qqq:
            print(f"  ⚠️  {len(perf_needing_qqq)} records missing QQQ returns")
            print(f"     → Can backfill QQQ benchmark data")
        
        if perf_needing_sector:
            print(f"  ⚠️  {len(perf_needing_sector)} records missing sector")
            print(f"     → Can backfill from signals table")
        
        if can_add_intervals:
            print(f"  ⚠️  {len(can_add_intervals)} records could have additional intervals calculated")
            print(f"     → Run Phase 6 performance update to backfill")
            print(f"\n     Examples:")
            for rec in can_add_intervals[:3]:
                print(f"       {rec['ticker']:6s} ({rec['days_old']:2d}d old) → missing intervals: {rec['missing']}")
    
    print(f"\n[ANALYTICS TABLE]")
    if not analytics_result.data:
        print(f"  ⚠️  No analytics records - need to run Phase 7")
    else:
        print(f"  ℹ️  {len(analytics_result.data)} analytics records exist")
        print(f"     → May want to recalculate with updated performance data")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    asyncio.run(main())
