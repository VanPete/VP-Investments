"""
Performance Data Verification Script
====================================

Verifies that Phase 6 is correctly populating:
1. SPY benchmark data (all 7 horizons)
2. QQQ benchmark data (all 7 horizons)
3. Sector benchmark data (all 7 horizons)
4. Alpha calculations (auto-generated columns)
5. Market cap and beta (from Phase 5)

This confirms the backend is ready for the Performance Tab frontend.

Usage:
    python scripts/verify_performance_data.py
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

# Load environment variables
load_dotenv()

def main():
    """Verify performance data for frontend readiness."""
    
    print("\n" + "=" * 80)
    print("Performance Data Verification for Frontend Performance Tab")
    print("=" * 80 + "\n")
    
    # Step 1: Connect to Supabase
    print("STEP 1: Connecting to Supabase")
    print("-" * 80)
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL or SUPABASE_ANON_KEY not found in environment")
        return
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase\n")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return
    
    # Step 2: Get latest signal run
    print("STEP 2: Fetching Latest Signal Run")
    print("-" * 80)
    
    try:
        response = supabase.table('signal_runs').select(
            'id, run_timestamp, successful_tickers, status'
        ).order('run_timestamp', desc=True).limit(1).execute()
        
        if not response.data:
            print("❌ No signal runs found")
            return
        
        latest_run = response.data[0]
        run_id = latest_run['id']
        
        print(f"✅ Latest Run: {run_id}")
        print(f"   Timestamp: {latest_run['run_timestamp']}")
        print(f"   Tickers: {latest_run['successful_tickers']}")
        print(f"   Status: {latest_run['status']}\n")
        
    except Exception as e:
        print(f"❌ Failed to fetch signal runs: {e}")
        return
    
    # Step 3: Get sample performance records with all columns
    print("STEP 3: Fetching Performance Records (Sample)")
    print("-" * 80)
    
    try:
        # Fetch performance records with signal data
        perf_response = supabase.table('performance').select('''
            id,
            signal_id,
            baseline_price,
            baseline_date,
            return_1d, return_3d, return_7d, return_10d, return_14d, return_30d, return_90d,
            spy_return_1d, spy_return_3d, spy_return_7d, spy_return_10d, spy_return_14d, spy_return_30d, spy_return_90d,
            qqq_return_1d, qqq_return_3d, qqq_return_7d, qqq_return_10d, qqq_return_14d, qqq_return_30d, qqq_return_90d,
            sector_return_1d, sector_return_3d, sector_return_7d, sector_return_10d, sector_return_14d, sector_return_30d, sector_return_90d,
            alpha_1d, alpha_3d, alpha_7d, alpha_10d, alpha_14d, alpha_30d, alpha_90d,
            qqq_alpha_1d, qqq_alpha_3d, qqq_alpha_7d, qqq_alpha_10d, qqq_alpha_14d, qqq_alpha_30d, qqq_alpha_90d,
            sector_alpha_1d, sector_alpha_3d, sector_alpha_7d, sector_alpha_10d, sector_alpha_14d, sector_alpha_30d, sector_alpha_90d,
            signals!inner(
                ticker,
                company_name,
                sector,
                overall_score,
                market_cap,
                beta,
                run_id
            )
        ''').eq('signals.run_id', run_id).limit(5).execute()
        
        if not perf_response.data:
            print(f"❌ No performance records found for run {run_id}")
            print("   This could mean Phase 6 didn't run or failed")
            return
        
        records = perf_response.data
        print(f"✅ Found {len(records)} performance records\n")
        
    except Exception as e:
        print(f"❌ Failed to fetch performance data: {e}")
        return
    
    # Step 4: Verify SPY data
    print("STEP 4: Verifying SPY Benchmark Data")
    print("-" * 80)
    
    spy_horizons = ['1d', '3d', '7d', '10d', '14d', '30d', '90d']
    spy_columns = [f'spy_return_{h}' for h in spy_horizons]
    spy_alpha_columns = [f'alpha_{h}' for h in spy_horizons]
    
    spy_populated = 0
    spy_alpha_populated = 0
    
    for record in records:
        for col in spy_columns:
            if record.get(col) is not None:
                spy_populated += 1
        for col in spy_alpha_columns:
            if record.get(col) is not None:
                spy_alpha_populated += 1
    
    total_expected = len(records) * len(spy_horizons)
    spy_pct = (spy_populated / total_expected) * 100 if total_expected > 0 else 0
    spy_alpha_pct = (spy_alpha_populated / total_expected) * 100 if total_expected > 0 else 0
    
    print(f"SPY Returns:      {spy_populated}/{total_expected} ({spy_pct:.1f}%)")
    print(f"SPY Alpha:        {spy_alpha_populated}/{total_expected} ({spy_alpha_pct:.1f}%)")
    
    if spy_pct >= 50:
        print("✅ SPY benchmark data looks good\n")
    else:
        print("⚠️  SPY benchmark data incomplete\n")
    
    # Step 5: Verify QQQ data
    print("STEP 5: Verifying QQQ Benchmark Data")
    print("-" * 80)
    
    qqq_columns = [f'qqq_return_{h}' for h in spy_horizons]
    qqq_alpha_columns = [f'qqq_alpha_{h}' for h in spy_horizons]
    
    qqq_populated = 0
    qqq_alpha_populated = 0
    
    for record in records:
        for col in qqq_columns:
            if record.get(col) is not None:
                qqq_populated += 1
        for col in qqq_alpha_columns:
            if record.get(col) is not None:
                qqq_alpha_populated += 1
    
    qqq_pct = (qqq_populated / total_expected) * 100 if total_expected > 0 else 0
    qqq_alpha_pct = (qqq_alpha_populated / total_expected) * 100 if total_expected > 0 else 0
    
    print(f"QQQ Returns:      {qqq_populated}/{total_expected} ({qqq_pct:.1f}%)")
    print(f"QQQ Alpha:        {qqq_alpha_populated}/{total_expected} ({qqq_alpha_pct:.1f}%)")
    
    if qqq_pct >= 50:
        print("✅ QQQ benchmark data looks good\n")
    else:
        print("⚠️  QQQ benchmark data incomplete\n")
    
    # Step 6: Verify Sector data
    print("STEP 6: Verifying Sector Benchmark Data")
    print("-" * 80)
    
    sector_columns = [f'sector_return_{h}' for h in spy_horizons]
    sector_alpha_columns = [f'sector_alpha_{h}' for h in spy_horizons]
    
    sector_populated = 0
    sector_alpha_populated = 0
    
    for record in records:
        for col in sector_columns:
            if record.get(col) is not None:
                sector_populated += 1
        for col in sector_alpha_columns:
            if record.get(col) is not None:
                sector_alpha_populated += 1
    
    sector_pct = (sector_populated / total_expected) * 100 if total_expected > 0 else 0
    sector_alpha_pct = (sector_alpha_populated / total_expected) * 100 if total_expected > 0 else 0
    
    print(f"Sector Returns:   {sector_populated}/{total_expected} ({sector_pct:.1f}%)")
    print(f"Sector Alpha:     {sector_alpha_populated}/{total_expected} ({sector_alpha_pct:.1f}%)")
    
    if sector_pct >= 50:
        print("✅ Sector benchmark data looks good\n")
    else:
        print("⚠️  Sector benchmark data incomplete\n")
    
    # Step 7: Verify Market Cap and Beta
    print("STEP 7: Verifying Market Cap and Beta (Header Data)")
    print("-" * 80)
    
    mktcap_populated = 0
    beta_populated = 0
    
    for record in records:
        signal = record.get('signals', {})
        if signal.get('market_cap') is not None:
            mktcap_populated += 1
        if signal.get('beta') is not None:
            beta_populated += 1
    
    mktcap_pct = (mktcap_populated / len(records)) * 100 if len(records) > 0 else 0
    beta_pct = (beta_populated / len(records)) * 100 if len(records) > 0 else 0
    
    print(f"Market Cap:       {mktcap_populated}/{len(records)} ({mktcap_pct:.1f}%)")
    print(f"Beta:             {beta_populated}/{len(records)} ({beta_pct:.1f}%)")
    
    if mktcap_pct >= 50 and beta_pct >= 50:
        print("✅ Header data (MktCap/Beta) looks good\n")
    else:
        print("⚠️  Header data incomplete\n")
    
    # Step 8: Display sample data
    print("STEP 8: Sample Performance Data (Top 5 Signals)")
    print("-" * 80)
    
    for i, record in enumerate(records, 1):
        signal = record.get('signals', {})
        ticker = signal.get('ticker', 'N/A')
        company = signal.get('company_name', 'N/A')[:30]
        score = signal.get('overall_score', 0)
        
        # Market cap formatting
        mktcap = signal.get('market_cap')
        if mktcap and mktcap >= 1_000_000_000_000:
            mktcap_str = f"${mktcap / 1_000_000_000_000:.2f}T"
        elif mktcap and mktcap >= 1_000_000_000:
            mktcap_str = f"${mktcap / 1_000_000_000:.2f}B"
        else:
            mktcap_str = "N/A"
        
        beta = signal.get('beta')
        beta_str = f"β {beta:.2f}" if beta else "N/A"
        
        print(f"\n{i}. {ticker:6s} | {company:30s} | Score: {score:.2f}")
        print(f"   MktCap: {mktcap_str:10s} | Beta: {beta_str:8s}")
        
        # 7-horizon returns
        returns_7h = [
            record.get('return_1d'),
            record.get('return_3d'),
            record.get('return_7d'),
            record.get('return_10d'),
            record.get('return_14d'),
            record.get('return_30d'),
            record.get('return_90d')
        ]
        
        # SPY alpha
        spy_alpha_7h = [
            record.get('alpha_1d'),
            record.get('alpha_3d'),
            record.get('alpha_7d'),
            record.get('alpha_10d'),
            record.get('alpha_14d'),
            record.get('alpha_30d'),
            record.get('alpha_90d')
        ]
        
        # QQQ alpha
        qqq_alpha_7h = [
            record.get('qqq_alpha_1d'),
            record.get('qqq_alpha_3d'),
            record.get('qqq_alpha_7d'),
            record.get('qqq_alpha_10d'),
            record.get('qqq_alpha_14d'),
            record.get('qqq_alpha_30d'),
            record.get('qqq_alpha_90d')
        ]
        
        print(f"   Returns:    ", end="")
        for j, ret in enumerate(returns_7h):
            if ret is not None:
                print(f"{spy_horizons[j]:>4s}:{ret:>6.1%} ", end="")
        print()
        
        print(f"   SPY Alpha:  ", end="")
        for j, alpha in enumerate(spy_alpha_7h):
            if alpha is not None:
                print(f"{spy_horizons[j]:>4s}:{alpha:>6.1%} ", end="")
        print()
        
        print(f"   QQQ Alpha:  ", end="")
        for j, alpha in enumerate(qqq_alpha_7h):
            if alpha is not None:
                print(f"{spy_horizons[j]:>4s}:{alpha:>6.1%} ", end="")
        print()
    
    print("\n" + "-" * 80)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_checks_passed = (
        spy_pct >= 50 and 
        qqq_pct >= 50 and 
        sector_pct >= 50 and
        spy_alpha_pct >= 50 and
        qqq_alpha_pct >= 50 and
        sector_alpha_pct >= 50 and
        mktcap_pct >= 50 and
        beta_pct >= 50
    )
    
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Backend ready for Performance Tab frontend!")
        print("\nData Available:")
        print("  ✅ SPY benchmark (all 7 horizons)")
        print("  ✅ QQQ benchmark (all 7 horizons)")
        print("  ✅ Sector benchmark (all 7 horizons)")
        print("  ✅ Auto-calculated alpha columns")
        print("  ✅ Market cap and beta (header data)")
        print("\nNext Step: Build Performance Tab frontend")
        print("  → See: docs/deployment/PHASE_6_ASSESSMENT.md Section 6")
    else:
        print("⚠️  SOME CHECKS FAILED - Review warnings above")
        print("\nMissing/Incomplete:")
        if spy_pct < 50:
            print(f"  ❌ SPY returns ({spy_pct:.1f}%)")
        if spy_alpha_pct < 50:
            print(f"  ❌ SPY alpha ({spy_alpha_pct:.1f}%)")
        if qqq_pct < 50:
            print(f"  ❌ QQQ returns ({qqq_pct:.1f}%)")
        if qqq_alpha_pct < 50:
            print(f"  ❌ QQQ alpha ({qqq_alpha_pct:.1f}%)")
        if sector_pct < 50:
            print(f"  ❌ Sector returns ({sector_pct:.1f}%)")
        if sector_alpha_pct < 50:
            print(f"  ❌ Sector alpha ({sector_alpha_pct:.1f}%)")
        if mktcap_pct < 50:
            print(f"  ❌ Market cap ({mktcap_pct:.1f}%)")
        if beta_pct < 50:
            print(f"  ❌ Beta ({beta_pct:.1f}%)")
    
    print()

if __name__ == "__main__":
    main()
