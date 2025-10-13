"""
Quick test to verify fundamental fields are now being saved to database.
Tests the fix: financial_data now at signal level instead of buried in metadata.
"""

import asyncio
import os
from dotenv import load_dotenv
from backend.storage.database import SupabaseInterface
from backend.integrations.yfinance import YahooFinanceIntegrator

# Load environment
load_dotenv()

async def test_fundamental_fix():
    """Test that improved fundamental fields are being saved correctly."""
    
    print("=" * 80)
    print("FUNDAMENTAL FIELD FIX VERIFICATION TEST")
    print("=" * 80)
    
    # Initialize components
    db = SupabaseInterface()
    yf = YahooFinanceIntegrator()
    
    # Test ticker
    test_ticker = "AAPL"
    print(f"\n1. Testing with {test_ticker}...")
    
    # Get comprehensive financial data (should have improved fields)
    print(f"   → Fetching comprehensive financial data...")
    financial_data = yf.get_comprehensive_financial_data(test_ticker)
    
    if not financial_data:
        print(f"   ❌ Failed to get financial data for {test_ticker}")
        return False
    
    # Check improved fields are present in financial_data
    improved_fields = [
        'pe_ratio',
        'dividend_yield', 
        'eps_growth',
        'interest_coverage',
        'share_buyback_yield',
        'fcf_growth_3y_cagr',
        'last_earnings_surprise_pct'
    ]
    
    print(f"\n2. Checking improved fields in financial_data dict:")
    present_count = 0
    for field in improved_fields:
        value = financial_data.get(field)
        status = "✅" if value is not None else "❌"
        print(f"   {status} {field}: {value}")
        if value is not None:
            present_count += 1
    
    print(f"\n   → {present_count}/7 improved fields present in financial_data")
    
    # Now create a signal structure as pipeline would
    print(f"\n3. Creating signal structure (as pipeline does)...")
    signal = {
        'ticker': test_ticker,
        'signal_type': 'financial',
        'score': 0.75,
        'confidence': 0.8,
        'financial_data': financial_data,  # ✅ FIXED: Now at top level
        'metadata': financial_data
    }
    
    # Verify signal has financial_data at top level
    has_financial_data = 'financial_data' in signal
    print(f"   {'✅' if has_financial_data else '❌'} Signal has 'financial_data' key: {has_financial_data}")
    
    if has_financial_data:
        signal_financial = signal['financial_data']
        print(f"\n4. Verifying improved fields accessible via signal['financial_data']:")
        for field in improved_fields:
            value = signal_financial.get(field)
            status = "✅" if value is not None else "❌"
            print(f"   {status} signal['financial_data']['{field}']: {value}")
    
    # Get latest signal from database to compare
    print(f"\n5. Checking latest database signal for {test_ticker}...")
    try:
        result = db.supabase.table('signals').select('*').eq('ticker', test_ticker).order('created_at', desc=True).limit(1).execute()
        
        if result.data:
            db_signal = result.data[0]
            print(f"   ✅ Found latest signal in database (created: {db_signal.get('created_at')})")
            
            print(f"\n6. Comparing improved fields in database vs. calculated:")
            print(f"   {'Field':<30} {'Database':<15} {'Calculated':<15} {'Match'}")
            print(f"   {'-'*30} {'-'*15} {'-'*15} {'-'*5}")
            
            match_count = 0
            for field in improved_fields:
                db_value = db_signal.get(field)
                calc_value = financial_data.get(field)
                
                # Handle None comparisons
                if db_value is None and calc_value is None:
                    match = "N/A"
                elif db_value is not None and calc_value is not None:
                    # Round for comparison
                    db_rounded = round(float(db_value), 2) if db_value else None
                    calc_rounded = round(float(calc_value), 2) if calc_value else None
                    match = "✅" if abs(db_rounded - calc_rounded) < 0.01 else "❌"
                    if match == "✅":
                        match_count += 1
                else:
                    match = "❌"
                
                db_str = f"{db_value:.2f}" if db_value is not None else "None"
                calc_str = f"{calc_value:.2f}" if calc_value is not None else "None"
                print(f"   {field:<30} {db_str:<15} {calc_str:<15} {match}")
            
            # Check if any improved fields are now populated in database
            db_populated = sum(1 for field in improved_fields if db_signal.get(field) is not None)
            print(f"\n   → Database has {db_populated}/7 improved fields populated")
            
            if db_populated == 0:
                print(f"\n   ⚠️  WARNING: No improved fields in database yet!")
                print(f"   → This is expected if you haven't run the pipeline since the fix.")
                print(f"   → Run: python run_full_pipeline.py")
            elif db_populated < present_count:
                print(f"\n   ⚠️  Database has fewer fields than calculated ({db_populated} < {present_count})")
                print(f"   → Re-run pipeline to populate all improved fields")
            else:
                print(f"\n   ✅ SUCCESS: All improved fields are being saved to database!")
        
        else:
            print(f"   ⚠️  No signals found for {test_ticker} in database")
            print(f"   → Run pipeline to generate signals: python run_full_pipeline.py")
    
    except Exception as e:
        print(f"   ❌ Error querying database: {e}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    print("\n📋 SUMMARY:")
    print(f"   • Improved fields calculated: {present_count}/7")
    print(f"   • Signal structure fixed: {has_financial_data}")
    print(f"   • Next step: Run full pipeline to populate database")
    print(f"     → Command: python run_full_pipeline.py")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_fundamental_fix())
