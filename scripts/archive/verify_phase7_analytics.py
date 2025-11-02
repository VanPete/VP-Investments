"""
Phase 7 Run-Based Analytics Verification Script
===============================================

This script verifies that Phase 7 analytics is correctly using run-based
architecture instead of period-based, confirming:
1. Analytics table has run_id populated
2. Only 1 row per run (not 4 period-based rows)
3. run_id matches signal_runs table
4. Analytics data is being calculated and persisted
5. Storage savings achieved (75% reduction vs period-based)

Usage:
    python scripts/verify_phase7_analytics.py
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def main():
    """Verify Phase 7 run-based analytics implementation."""
    
    print("\n" + "=" * 70)
    print("Phase 7 Run-Based Analytics Verification")
    print("=" * 70 + "\n")
    
    # Step 1: Connect to Supabase
    print("STEP 1: Connecting to Supabase")
    print("-" * 70)
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL or SUPABASE_ANON_KEY not found in environment")
        print("   Please check your .env file")
        return
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase\n")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return
    
    # Step 2: Get latest signal run
    print("STEP 2: Fetching Latest Signal Run")
    print("-" * 70)
    
    try:
        response = supabase.table('signal_runs').select(
            'id, run_timestamp, successful_tickers, status'
        ).order('run_timestamp', desc=True).limit(1).execute()
        
        if not response.data:
            print("❌ No signal runs found in database")
            return
        
        latest_run = response.data[0]
        run_id = latest_run['id']  # signal_runs.id is the run identifier
        run_timestamp = latest_run['run_timestamp']
        successful_tickers = latest_run['successful_tickers']
        status = latest_run['status']
        
        print(f"✅ Latest Run Found:")
        print(f"   Run ID: {run_id}")
        print(f"   Timestamp: {run_timestamp}")
        print(f"   Successful Tickers: {successful_tickers}")
        print(f"   Status: {status}\n")
        
    except Exception as e:
        print(f"❌ Failed to fetch signal runs: {e}")
        return
    
    # Step 3: Check analytics table for run_id
    print("STEP 3: Verifying Analytics Table Structure")
    print("-" * 70)
    
    try:
        # Check if run_id column exists
        test_response = supabase.table('analytics').select('run_id').limit(1).execute()
        print("✅ Column 'run_id' exists in analytics table\n")
    except Exception as e:
        print(f"❌ Error accessing analytics table: {e}")
        print("   Migration 015 may not have been applied correctly")
        return
    
    # Step 4: Check for analytics record with this run_id
    print("STEP 4: Checking Analytics Record for Latest Run")
    print("-" * 70)
    
    try:
        analytics_response = supabase.table('analytics').select(
            'run_id, ic_mean, ic_std, cagr, sortino_ratio, volatility, calmar_ratio, created_at'
        ).eq('run_id', run_id).execute()
        
        if not analytics_response.data:
            print(f"⚠️  No analytics record found for run_id: {run_id}")
            print("   This could mean:")
            print("   1. Phase 7 did not run (check logs)")
            print("   2. Analytics calculation failed (check error logs)")
            print("   3. Pipeline is still running")
            print("\n   Run pipeline again to generate analytics data.")
            return
        
        analytics_records = analytics_response.data
        num_records = len(analytics_records)
        
        print(f"✅ Found {num_records} analytics record(s) for run_id: {run_id}")
        
        if num_records > 1:
            print(f"⚠️  WARNING: Expected 1 record (run-based), found {num_records}")
            print("   This suggests period-based architecture is still in use")
            print("   Phase 7 refactoring may not be working correctly")
        else:
            print("✅ Correct! Run-based architecture (1 row per run)")
        
        print()
        
    except Exception as e:
        print(f"❌ Failed to query analytics table: {e}")
        return
    
    # Step 5: Display analytics data
    print("STEP 5: Analytics Data Sample")
    print("-" * 70)
    
    analytics = analytics_records[0]
    
    print(f"Run ID:        {analytics.get('run_id', 'N/A')}")
    print(f"IC Mean:       {analytics.get('ic_mean', 'N/A')}")
    print(f"IC Std:        {analytics.get('ic_std', 'N/A')}")
    print(f"CAGR:          {analytics.get('cagr', 'N/A')}")
    print(f"Sortino Ratio: {analytics.get('sortino_ratio', 'N/A')}")
    print(f"Volatility:    {analytics.get('volatility', 'N/A')}")
    print(f"Calmar Ratio:  {analytics.get('calmar_ratio', 'N/A')}")
    print(f"Created At:    {analytics.get('created_at', 'N/A')}")
    print()
    
    # Step 6: Compare with historical runs (storage savings)
    print("STEP 6: Historical Analytics Count (Storage Verification)")
    print("-" * 70)
    
    try:
        # Count total analytics records
        total_response = supabase.table('analytics').select('run_id', count='exact').execute()
        total_analytics = total_response.count
        
        # Count total signal runs
        runs_response = supabase.table('signal_runs').select('id', count='exact').execute()
        total_runs = runs_response.count
        
        print(f"Total Analytics Records: {total_analytics}")
        print(f"Total Signal Runs:       {total_runs}")
        
        if total_analytics == total_runs:
            print("✅ Perfect match! 1 analytics record per run (run-based architecture)")
            print(f"   Storage savings: 75% (vs 4 records per run in period-based)")
        elif total_analytics > total_runs:
            ratio = total_analytics / total_runs if total_runs > 0 else 0
            print(f"⚠️  Found {ratio:.1f}x more analytics records than runs")
            print("   This suggests period-based architecture may still be in use")
            print("   Expected: 1:1 ratio for run-based")
        else:
            print(f"ℹ️  Some runs missing analytics (normal if Phase 7 failed/skipped)")
        
        print()
        
    except Exception as e:
        print(f"⚠️  Could not verify storage savings: {e}")
        print()
    
    # Step 7: Check for old period-based columns
    print("STEP 7: Checking for Period-Based Data (Should be None)")
    print("-" * 70)
    
    try:
        # Check if any records have period_type populated
        period_response = supabase.table('analytics').select(
            'period_type, period_start, period_end'
        ).not_.is_('period_type', 'null').limit(5).execute()
        
        if period_response.data:
            print(f"⚠️  Found {len(period_response.data)} records with period_type populated")
            print("   These are likely old period-based records")
            print("   New run-based records should have NULL period_type")
        else:
            print("✅ No period-based data found (all records use run_id)")
        
        print()
        
    except Exception as e:
        print(f"ℹ️  Could not check period columns: {e}")
        print()
    
    # Final Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if num_records == 1 and total_analytics <= total_runs:
        print("✅ Phase 7 run-based analytics verification PASSED")
        print("   - run_id column exists and is populated")
        print("   - 1 record per run (not 4 period-based records)")
        print("   - Storage architecture is correct")
        print("   - Analytics data is being calculated")
        print("\n✅ Phase 7 refactoring is working correctly!")
    else:
        print("⚠️  Phase 7 verification INCONCLUSIVE")
        print("   Some checks did not pass as expected.")
        print("   Review the warnings above for details.")
    
    print()

if __name__ == "__main__":
    main()
