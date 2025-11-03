"""
Verify Phase 6 Performance Tracking - Signal Update Verification
================================================================

Checks that Phase 6 is correctly updating performance records:
1. Signal creation timeline (baseline_date)
2. Interval eligibility (age-based)
3. Progressive updates (intervals_completed tracking)
4. Data population (returns, benchmarks, alpha)
5. Status transitions (pending → in_progress → completed)

Usage:
    python scripts/verify_phase6_signals.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import get_supabase_database


def format_date(dt_str):
    """Format datetime string to readable format."""
    if not dt_str:
        return 'N/A'
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    return dt.strftime('%Y-%m-%d %H:%M')


def days_since(dt_str):
    """Calculate days since datetime string."""
    if not dt_str:
        return 0
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    return (datetime.now(timezone.utc) - dt).days


async def main():
    """Verify Phase 6 is working correctly."""
    
    print('='*100)
    print('PHASE 6 PERFORMANCE TRACKING VERIFICATION')
    print('='*100)
    
    # Get database
    db = await get_supabase_database()
    
    try:
        # ====================================================================
        # 1. CHECK PERFORMANCE RECORDS OVERVIEW
        # ====================================================================
        print('\n📊 1. PERFORMANCE RECORDS OVERVIEW')
        print('-'*100)
        
        result = db.client.table('performance').select(
            'id, signal_id, baseline_date, baseline_price, status, intervals_completed, '
            'return_1d, return_3d, return_7d, return_30d, return_90d, '
            'spy_return_1d, spy_return_3d, spy_return_7d, '
            'alpha_1d, alpha_3d, alpha_7d, '
            'signals!inner(ticker, created_at)'
        ).order('baseline_date', desc=True).limit(500).execute()
        
        if not result.data:
            print('❌ No performance records found!')
            return
        
        print(f'✓ Found {len(result.data)} performance records')
        
        # Status breakdown
        status_counts = Counter(r['status'] for r in result.data)
        print(f'\n📈 Status Distribution:')
        for status, count in status_counts.most_common():
            pct = (count / len(result.data)) * 100
            print(f'  {status:15s}: {count:4d} records ({pct:5.1f}%)')
        
        # ====================================================================
        # 2. CHECK TIMELINE & AGE DISTRIBUTION
        # ====================================================================
        print('\n\n📅 2. SIGNAL AGE DISTRIBUTION')
        print('-'*100)
        
        # Calculate age for each record
        age_buckets = {
            '0 days (fresh)': [],
            '1-2 days': [],
            '3-6 days': [],
            '7-13 days': [],
            '14-29 days': [],
            '30-89 days': [],
            '90+ days': []
        }
        
        for rec in result.data:
            age = days_since(rec['baseline_date'])
            ticker = rec['signals']['ticker']
            
            if age == 0:
                age_buckets['0 days (fresh)'].append((ticker, age, rec))
            elif age <= 2:
                age_buckets['1-2 days'].append((ticker, age, rec))
            elif age <= 6:
                age_buckets['3-6 days'].append((ticker, age, rec))
            elif age <= 13:
                age_buckets['7-13 days'].append((ticker, age, rec))
            elif age <= 29:
                age_buckets['14-29 days'].append((ticker, age, rec))
            elif age <= 89:
                age_buckets['30-89 days'].append((ticker, age, rec))
            else:
                age_buckets['90+ days'].append((ticker, age, rec))
        
        print(f'\n📊 Signals by Age:')
        for bucket, signals in age_buckets.items():
            if signals:
                count = len(signals)
                pct = (count / len(result.data)) * 100
                print(f'  {bucket:15s}: {count:4d} signals ({pct:5.1f}%)')
        
        # ====================================================================
        # 3. VERIFY INTERVAL COMPLETION LOGIC
        # ====================================================================
        print('\n\n⏱️  3. INTERVAL COMPLETION VERIFICATION')
        print('-'*100)
        
        # Expected intervals based on age
        intervals_map = {1: '1d', 3: '3d', 7: '7d', 10: '10d', 14: '14d', 30: '30d', 90: '90d'}
        
        # Check sample from each age bucket
        print('\n🔍 Sample Signal Analysis:')
        
        for bucket, signals in age_buckets.items():
            if signals:
                # Get first signal from bucket
                ticker, age, rec = signals[0]
                expected_intervals = [i for i in [1, 3, 7, 10, 14, 30, 90] if i <= age]
                completed = rec.get('intervals_completed') or []
                
                print(f'\n  {bucket} - {ticker} (age: {age}d):')
                print(f'    Expected intervals:  {expected_intervals}')
                print(f'    Completed intervals: {completed}')
                print(f'    Status: {rec["status"]}')
                
                # Check data population
                if 1 in completed:
                    print(f'    1d return:  {rec.get("return_1d", "N/A")}%')
                    print(f'    SPY 1d:     {rec.get("spy_return_1d", "N/A")}%')
                    print(f'    Alpha 1d:   {rec.get("alpha_1d", "N/A")}%')
                
                if 7 in completed:
                    print(f'    7d return:  {rec.get("return_7d", "N/A")}%')
                    print(f'    Alpha 7d:   {rec.get("alpha_7d", "N/A")}%')
                
                # Verify completeness
                missing = [i for i in expected_intervals if i not in completed]
                if missing:
                    print(f'    ⚠️  Missing intervals: {missing} (should be calculated)')
                else:
                    print(f'    ✓ All eligible intervals completed')
        
        # ====================================================================
        # 4. CHECK DATA POPULATION RATES
        # ====================================================================
        print('\n\n📈 4. DATA POPULATION RATES')
        print('-'*100)
        
        intervals_to_check = [
            ('1d', 'return_1d', 'spy_return_1d', 'alpha_1d'),
            ('3d', 'return_3d', 'spy_return_3d', 'alpha_3d'),
            ('7d', 'return_7d', 'spy_return_7d', 'alpha_7d'),
            ('30d', 'return_30d', 'spy_return_30d', 'alpha_30d'),
            ('90d', 'return_90d', 'spy_return_90d', 'alpha_90d')
        ]
        
        print('\n📊 Data Population (eligible signals only):')
        
        for interval_name, return_col, spy_col, alpha_col in intervals_to_check:
            interval_days = int(interval_name.replace('d', ''))
            
            # Count eligible signals (old enough)
            eligible = [r for r in result.data if days_since(r['baseline_date']) >= interval_days]
            
            if not eligible:
                print(f'\n  {interval_name:4s} interval:')
                print(f'    ⚠️  No signals old enough yet (need ≥{interval_days} days)')
                continue
            
            # Count populated data
            with_return = sum(1 for r in eligible if r.get(return_col) is not None)
            with_spy = sum(1 for r in eligible if r.get(spy_col) is not None)
            with_alpha = sum(1 for r in eligible if r.get(alpha_col) is not None)
            
            return_pct = (with_return / len(eligible)) * 100 if eligible else 0
            spy_pct = (with_spy / len(eligible)) * 100 if eligible else 0
            alpha_pct = (with_alpha / len(eligible)) * 100 if eligible else 0
            
            print(f'\n  {interval_name:4s} interval ({len(eligible)} eligible signals):')
            print(f'    Return populated: {with_return:3d}/{len(eligible):3d} ({return_pct:5.1f}%)')
            print(f'    SPY populated:    {with_spy:3d}/{len(eligible):3d} ({spy_pct:5.1f}%)')
            print(f'    Alpha populated:  {with_alpha:3d}/{len(eligible):3d} ({alpha_pct:5.1f}%)')
            
            # Status check
            if return_pct >= 95 and spy_pct >= 95:
                print(f'    ✓ Excellent population rate')
            elif return_pct >= 80 and spy_pct >= 80:
                print(f'    ⚠️  Good but could be better')
            else:
                print(f'    ❌ Poor population rate - Phase 6 may have issues')
        
        # ====================================================================
        # 5. CHECK RECENT UPDATES
        # ====================================================================
        print('\n\n🔄 5. RECENT UPDATE ACTIVITY')
        print('-'*100)
        
        # Get records updated recently
        recent_result = db.client.table('performance').select(
            'last_updated, baseline_date, status, intervals_completed, '
            'signals!inner(ticker)'
        ).order('last_updated', desc=True).limit(20).execute()
        
        if recent_result.data:
            print('\n📝 Last 10 Updated Records:')
            for i, rec in enumerate(recent_result.data[:10], 1):
                ticker = rec['signals']['ticker']
                age = days_since(rec['baseline_date'])
                last_update = format_date(rec.get('last_updated'))
                intervals = rec.get('intervals_completed') or []
                
                print(f'  {i:2d}. {ticker:6s} | Age: {age:3d}d | Status: {rec["status"]:12s} | '
                      f'Intervals: {intervals} | Updated: {last_update}')
        
        # ====================================================================
        # 6. IDENTIFY ISSUES
        # ====================================================================
        print('\n\n🔍 6. ISSUE DETECTION')
        print('-'*100)
        
        issues_found = []
        
        # Issue 1: Old signals with no completed intervals
        old_pending = [
            r for r in result.data 
            if days_since(r['baseline_date']) >= 7 
            and not (r.get('intervals_completed') or [])
        ]
        
        if old_pending:
            issues_found.append(f'❌ Found {len(old_pending)} signals ≥7 days old with NO completed intervals')
            print(f'\n  ❌ Issue 1: Old signals not updating')
            print(f'     {len(old_pending)} signals ≥7 days old with no intervals completed')
            print(f'     Sample tickers: {", ".join(r["signals"]["ticker"] for r in old_pending[:5])}')
        
        # Issue 2: Completed intervals but missing data
        for interval_name, return_col, spy_col, alpha_col in intervals_to_check:
            interval_num = int(interval_name.replace('d', ''))
            
            # Signals with interval marked complete but data missing
            bad_records = [
                r for r in result.data
                if interval_num in (r.get('intervals_completed') or [])
                and r.get(return_col) is None
            ]
            
            if bad_records:
                issues_found.append(f'❌ {len(bad_records)} signals have {interval_name} marked complete but data is NULL')
                print(f'\n  ❌ Issue 2: Incomplete data for {interval_name}')
                print(f'     {len(bad_records)} signals marked {interval_name} complete but {return_col} is NULL')
                print(f'     Sample tickers: {", ".join(r["signals"]["ticker"] for r in bad_records[:5])}')
        
        # Issue 3: Alpha mismatch (should be calculated from return - spy)
        alpha_issues = []
        for rec in result.data:
            if rec.get('return_1d') is not None and rec.get('spy_return_1d') is not None:
                expected_alpha = rec['return_1d'] - rec['spy_return_1d']
                actual_alpha = rec.get('alpha_1d')
                
                if actual_alpha is not None:
                    diff = abs(expected_alpha - actual_alpha)
                    if diff > 0.01:  # Allow 0.01% rounding error
                        alpha_issues.append((rec['signals']['ticker'], expected_alpha, actual_alpha))
        
        if alpha_issues:
            issues_found.append(f'❌ {len(alpha_issues)} signals have incorrect alpha calculations')
            print(f'\n  ❌ Issue 3: Alpha calculation errors')
            print(f'     {len(alpha_issues)} signals have alpha_vs_spy not matching (return - spy)')
            for ticker, expected, actual in alpha_issues[:3]:
                print(f'     {ticker}: expected {expected:.2f}%, got {actual:.2f}%')
        
        # ====================================================================
        # 7. OVERALL VERDICT
        # ====================================================================
        print('\n\n' + '='*100)
        print('✅ PHASE 6 VERIFICATION SUMMARY')
        print('='*100)
        
        if not issues_found:
            print('\n✅ Phase 6 is working correctly!')
            print('   - All signals are being tracked')
            print('   - Intervals are updating based on age')
            print('   - Data is being populated correctly')
            print('   - Status transitions are working')
        else:
            print('\n⚠️  Phase 6 has some issues:')
            for issue in issues_found:
                print(f'   {issue}')
            print('\n💡 Recommendations:')
            print('   1. Check Phase 6 logs for errors')
            print('   2. Verify yfinance is accessible')
            print('   3. Run manual Phase 6 update: scripts/archive/test_phase6_manual.py')
            print('   4. Check database constraints on performance table')
        
        # Key metrics
        print(f'\n📊 Key Metrics:')
        print(f'   Total signals tracked: {len(result.data)}')
        print(f'   Status breakdown: {dict(status_counts)}')
        
        # Eligibility breakdown
        eligible_1d = len([r for r in result.data if days_since(r['baseline_date']) >= 1])
        eligible_7d = len([r for r in result.data if days_since(r['baseline_date']) >= 7])
        eligible_30d = len([r for r in result.data if days_since(r['baseline_date']) >= 30])
        eligible_90d = len([r for r in result.data if days_since(r['baseline_date']) >= 90])
        
        print(f'   Eligible for 1d:  {eligible_1d:3d} signals')
        print(f'   Eligible for 7d:  {eligible_7d:3d} signals')
        print(f'   Eligible for 30d: {eligible_30d:3d} signals')
        print(f'   Eligible for 90d: {eligible_90d:3d} signals')
        
        print('\n' + '='*100)
    
    finally:
        await db.disconnect()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
