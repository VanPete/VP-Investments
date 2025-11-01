"""
Verify Migration 017 - MktCap and Beta Column Population

This script checks if market_cap and beta columns were successfully
populated in the signals table after migration 017 execution.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_migration_017():
    """Verify migration 017 market_cap and beta population."""
    
    try:
        from supabase import create_client, Client
    except ImportError:
        print("❌ Error: supabase-py not installed")
        print("   Install with: pip install supabase")
        return False
    
    # Initialize Supabase client
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL or SUPABASE_ANON_KEY not found in environment")
        print("   Check your .env file")
        return False
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase\n")
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        return False
    
    # Step 1: Get latest signal run
    print("=" * 80)
    print("STEP 1: Fetching Latest Signal Run")
    print("=" * 80)
    
    try:
        response = supabase.table('signal_runs') \
            .select('id, run_timestamp, successful_tickers, status') \
            .order('run_timestamp', desc=True) \
            .limit(1) \
            .execute()
        
        if not response.data:
            print("❌ No signal runs found in database")
            return False
        
        run = response.data[0]
        run_id = run['id']
        run_timestamp = run['run_timestamp']
        successful_tickers = run['successful_tickers']
        status = run['status']
        
        print(f"✅ Latest Run Found:")
        print(f"   Run ID: {run_id}")
        print(f"   Timestamp: {run_timestamp}")
        print(f"   Successful Tickers: {successful_tickers}")
        print(f"   Status: {status}\n")
        
    except Exception as e:
        print(f"❌ Error fetching signal runs: {e}")
        return False
    
    # Step 2: Check column existence
    print("=" * 80)
    print("STEP 2: Verifying Columns Exist")
    print("=" * 80)
    
    try:
        # Try to query with market_cap and beta columns
        test_response = supabase.table('signals') \
            .select('market_cap, beta') \
            .limit(1) \
            .execute()
        
        print("✅ Columns 'market_cap' and 'beta' exist in signals table\n")
        
    except Exception as e:
        print(f"❌ Error: Columns may not exist - {e}")
        print("   Run migration 017 in Supabase SQL Editor first")
        return False
    
    # Step 3: Get population statistics
    print("=" * 80)
    print("STEP 3: Analyzing Data Population")
    print("=" * 80)
    
    try:
        response = supabase.table('signals') \
            .select('ticker, market_cap, beta') \
            .eq('run_id', run_id) \
            .execute()
        
        signals = response.data
        total_signals = len(signals)
        
        if total_signals == 0:
            print(f"❌ No signals found for run_id: {run_id}")
            return False
        
        mktcap_populated = sum(1 for s in signals if s.get('market_cap') is not None)
        beta_populated = sum(1 for s in signals if s.get('beta') is not None)
        
        mktcap_pct = (mktcap_populated / total_signals) * 100
        beta_pct = (beta_populated / total_signals) * 100
        
        print(f"📊 Population Statistics:")
        print(f"   Total Signals: {total_signals}")
        print(f"   Market Cap Populated: {mktcap_populated}/{total_signals} ({mktcap_pct:.1f}%)")
        print(f"   Beta Populated: {beta_populated}/{total_signals} ({beta_pct:.1f}%)\n")
        
        # Check success criteria
        success = True
        if mktcap_pct >= 50:
            print(f"   ✅ Market Cap: {mktcap_pct:.1f}% >= 50% (PASS)")
        else:
            print(f"   ⚠️  Market Cap: {mktcap_pct:.1f}% < 50% (LOW)")
            success = False
        
        if beta_pct >= 50:
            print(f"   ✅ Beta: {beta_pct:.1f}% >= 50% (PASS)")
        else:
            print(f"   ⚠️  Beta: {beta_pct:.1f}% < 50% (LOW)")
            success = False
        
        print()
        
    except Exception as e:
        print(f"❌ Error analyzing population: {e}")
        return False
    
    # Step 4: Show sample data
    print("=" * 80)
    print("STEP 4: Sample Data (Top 10 by Score)")
    print("=" * 80)
    
    try:
        response = supabase.table('signals') \
            .select('ticker, company_name, sector, market_cap, beta, overall_score') \
            .eq('run_id', run_id) \
            .order('overall_score', desc=True) \
            .limit(10) \
            .execute()
        
        signals = response.data
        
        # Print header
        print(f"{'Ticker':<8} {'Company':<25} {'Sector':<15} {'MktCap':<15} {'Beta':<8} {'Score':<8}")
        print("-" * 95)
        
        for signal in signals:
            ticker = signal.get('ticker', 'N/A')[:8]
            company = (signal.get('company_name') or 'N/A')[:25]
            sector = (signal.get('sector') or 'N/A')[:15]
            
            # Format market cap
            mktcap = signal.get('market_cap')
            if mktcap is not None:
                if mktcap >= 1_000_000_000_000:  # Trillions
                    mktcap_str = f"${mktcap / 1_000_000_000_000:.2f}T"
                elif mktcap >= 1_000_000_000:  # Billions
                    mktcap_str = f"${mktcap / 1_000_000_000:.2f}B"
                elif mktcap >= 1_000_000:  # Millions
                    mktcap_str = f"${mktcap / 1_000_000:.2f}M"
                else:
                    mktcap_str = f"${mktcap:,.0f}"
            else:
                mktcap_str = "N/A"
            
            beta = signal.get('beta')
            beta_str = f"{beta:.2f}" if beta is not None else "N/A"
            
            score = signal.get('overall_score')
            score_str = f"{score:.2f}" if score is not None else "N/A"
            
            print(f"{ticker:<8} {company:<25} {sector:<15} {mktcap_str:<15} {beta_str:<8} {score_str:<8}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error fetching sample data: {e}")
        return False
    
    # Final summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if success:
        print("✅ Migration 017 verification PASSED")
        print("   - Columns exist")
        print("   - Data population meets requirements (≥50%)")
        print("   - Sample data looks correct")
        print("\n🎯 Next Steps:")
        print("   1. Review Phase 6 Assessment: docs/deployment/PHASE_6_ASSESSMENT.md")
        print("   2. Test Phase 7 run-based analytics")
        print("   3. Start frontend Performance Tab refactor")
    else:
        print("⚠️  Migration 017 verification completed with WARNINGS")
        print("   - Columns exist but population is low (<50%)")
        print("   - Some tickers may not have market_cap/beta in YFinance")
        print("   - This is acceptable if most major stocks have data")
        print("\n🔍 Troubleshooting:")
        print("   - Check if low-data tickers are delisted/obscure")
        print("   - Review Phase 5 logs for extraction issues")
        print("   - Verify YFinance data availability for your ticker universe")
    
    print()
    return success


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Migration 017 Verification Script")
    print("Checking market_cap and beta column population")
    print("=" * 80 + "\n")
    
    try:
        success = verify_migration_017()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
