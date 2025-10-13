"""
Troubleshoot: Check if improved fundamental fields are being calculated but not saved.
Tests the complete data flow: yfinance → signal generation → database
"""
import asyncio
from backend.integrations.yfinance import YahooFinanceIntegrator
from backend.pipeline import UnifiedPipeline
from backend.storage.database import SupabaseInterface

async def test_data_flow():
    """Test if improved fields are calculated, in signals, and saved to DB."""
    
    print("=" * 80)
    print("TROUBLESHOOTING: FUNDAMENTAL FIELDS DATA FLOW")
    print("=" * 80)
    print()
    
    # Test with a known good ticker
    test_ticker = "AAPL"
    
    print(f"🔍 Testing with {test_ticker}...\n")
    
    # STEP 1: Check what yfinance returns
    print("STEP 1: YahooFinanceIntegrator.get_comprehensive_financial_data()")
    print("-" * 80)
    
    yf = YahooFinanceIntegrator()
    financial_data = yf.get_comprehensive_financial_data(test_ticker)
    
    if not financial_data:
        print("❌ No financial data returned!")
        return
    
    # Check which improved fields are present
    improved_fields = [
        'pe_ratio', 'dividend_yield', 'eps_growth', 'interest_coverage',
        'share_buyback_yield', 'fcf_growth_3y_cagr', 'last_earnings_surprise_pct'
    ]
    
    print(f"\n✅ Financial data retrieved for {test_ticker}")
    print(f"\nImproved fields in financial_data:")
    for field in improved_fields:
        value = financial_data.get(field)
        status = "✅" if value is not None else "❌"
        print(f"  {status} {field}: {value}")
    
    # STEP 2: Check what the pipeline produces
    print("\n\nSTEP 2: UnifiedPipeline - Generate financial signal")
    print("-" * 80)
    
    pipeline = UnifiedPipeline()
    signal = pipeline._create_financial_signal(test_ticker, financial_data)
    
    if not signal:
        print("❌ No signal generated!")
        return
    
    print(f"\n✅ Signal generated for {test_ticker}")
    print(f"\nImproved fields in signal dict:")
    for field in improved_fields:
        value = signal.get(field)
        status = "✅" if value is not None else "❌"
        print(f"  {status} {field}: {value}")
    
    # STEP 3: Check what gets saved to database
    print("\n\nSTEP 3: SupabaseInterface.save_signal()")
    print("-" * 80)
    
    db = SupabaseInterface()
    
    # Save the signal
    await db.save_signal(signal)
    print(f"\n✅ Signal saved to database")
    
    # Retrieve it back
    print("\nRetrieving signal from database...")
    result = await db.execute_query(f"""
        SELECT ticker, {', '.join(improved_fields)}
        FROM signals
        WHERE ticker = '{test_ticker}'
        ORDER BY id DESC
        LIMIT 1
    """)
    
    if not result:
        print("❌ Could not retrieve signal from database!")
        return
    
    saved_signal = result[0]
    print(f"\n✅ Signal retrieved from database")
    print(f"\nImproved fields in database:")
    for field in improved_fields:
        value = saved_signal.get(field)
        status = "✅" if value is not None else "❌"
        print(f"  {status} {field}: {value}")
    
    # COMPARISON: Check if anything got lost
    print("\n\n" + "=" * 80)
    print("COMPARISON: WHAT GOT LOST?")
    print("=" * 80)
    print()
    
    lost_fields = []
    for field in improved_fields:
        in_financial = financial_data.get(field) is not None
        in_signal = signal.get(field) is not None
        in_db = saved_signal.get(field) is not None
        
        if in_financial and not in_signal:
            lost_fields.append((field, "signal generation"))
        elif in_signal and not in_db:
            lost_fields.append((field, "database save"))
    
    if lost_fields:
        print("❌ FOUND ISSUES:")
        for field, stage in lost_fields:
            print(f"  • {field} lost at: {stage}")
    else:
        print("✅ No data loss detected! All fields flow through correctly.")
    
    # DETAILED FIELD COMPARISON
    print("\n\n" + "=" * 80)
    print("DETAILED FIELD FLOW")
    print("=" * 80)
    print()
    print(f"{'Field':<30} {'Financial Data':<15} {'Signal':<15} {'Database'}")
    print("-" * 80)
    
    for field in improved_fields:
        fd_val = financial_data.get(field)
        sig_val = signal.get(field)
        db_val = saved_signal.get(field)
        
        fd_str = f"{fd_val:.2f}" if isinstance(fd_val, (int, float)) else str(fd_val)[:10]
        sig_str = f"{sig_val:.2f}" if isinstance(sig_val, (int, float)) else str(sig_val)[:10]
        db_str = f"{db_val:.2f}" if isinstance(db_val, (int, float)) else str(db_val)[:10]
        
        print(f"{field:<30} {fd_str:<15} {sig_str:<15} {db_str}")
    
    # CHECK: Are improved methods being called?
    print("\n\n" + "=" * 80)
    print("VERIFICATION: ARE IMPROVED METHODS BEING USED?")
    print("=" * 80)
    print()
    
    # Check if improved_calc exists
    if hasattr(yf, 'improved_calc'):
        print("✅ yf.improved_calc exists")
        
        # Test a specific improved method
        import yfinance as yf_lib
        stock = yf_lib.Ticker(test_ticker)
        
        pe_improved = yf.improved_calc.calculate_pe_ratio_improved(stock)
        div_improved = yf.improved_calc.calculate_dividend_yield_improved(stock)
        
        print(f"✅ calculate_pe_ratio_improved(): {pe_improved}")
        print(f"✅ calculate_dividend_yield_improved(): {div_improved}")
    else:
        print("❌ yf.improved_calc DOES NOT EXIST!")
        print("   This means improved methods are NOT being used!")
    
    print("\n\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if lost_fields:
        print("🔴 ISSUES DETECTED:")
        print()
        for field, stage in lost_fields:
            if stage == "signal generation":
                print(f"• {field} not being copied from financial_data to signal dict")
                print("  → Check SignalGenerator.generate_financial_signal()")
                print("  → Ensure all fields are included in signal dict")
            elif stage == "database save":
                print(f"• {field} not being saved to database")
                print("  → Check database schema - column may not exist")
                print("  → Check SupabaseInterface.save_signal() field mapping")
    else:
        print("✅ No obvious issues detected in data flow")
        print()
        print("The low coverage in production may be due to:")
        print("1. Tickers with missing/invalid data from yfinance")
        print("2. Error handling that returns None for failed calculations")
        print("3. API rate limits or temporary failures")
    
    print()

if __name__ == "__main__":
    asyncio.run(test_data_flow())
