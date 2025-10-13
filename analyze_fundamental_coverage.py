"""
Simple fundamental field coverage analysis.
"""
import asyncio
from backend.storage.database import SupabaseInterface
import pandas as pd

FUNDAMENTAL_FIELDS = [
    'revenue_growth', 'eps_growth', 'fcf_growth_3y_cagr',
    'roe', 'roic', 'fcf_margin',
    'pe_ratio', 'price_to_sales', 'price_to_book', 'sector_relative_percentile',
    'debt_to_equity', 'current_ratio', 'interest_coverage',
    'last_earnings_surprise_pct', 'earnings_surprise_streak',
    'dividend_yield', 'share_buyback_yield'
]

async def analyze():
    print("=" * 80)
    print("FUNDAMENTAL FIELD COVERAGE ANALYSIS")
    print("=" * 80)
    print()
    
    db = SupabaseInterface()
    
    # Get signals directly
    print("📊 Fetching signals...")
    query = "SELECT ticker, sector, " + ", ".join(FUNDAMENTAL_FIELDS) + " FROM signals ORDER BY id DESC LIMIT 50"
    
    result = await db.execute_query(query)
    
    if not result:
        print("❌ No signals found")
        return
    
    df = pd.DataFrame(result)
    total = len(df)
    
    print(f"✅ Found {total} signals\n")
    print("=" * 80)
    print("COVERAGE BY FIELD")
    print("=" * 80)
    print()
    
    # Calculate coverage
    coverage_data = []
    for field in FUNDAMENTAL_FIELDS:
        non_null = df[field].notna().sum()
        coverage_pct = (non_null / total) * 100
        
        coverage_data.append({
            'Field': field,
            'Populated': non_null,
            'Coverage %': coverage_pct
        })
    
    coverage_df = pd.DataFrame(coverage_data).sort_values('Coverage %', ascending=False)
    
    # Print table
    print(f"{'Field':<30} {'Coverage':<15} {'Status'}")
    print("-" * 70)
    
    for _, row in coverage_df.iterrows():
        field = row['Field']
        coverage = row['Coverage %']
        populated = row['Populated']
        
        if coverage >= 70:
            status = "✅ Good"
        elif coverage >= 40:
            status = "⚠️  Fair"
        else:
            status = "❌ Poor"
        
        print(f"{field:<30} {populated:>2}/{total} ({coverage:>5.1f}%)  {status}")
    
    # Overall stats
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print()
    
    avg_coverage = coverage_df['Coverage %'].mean()
    fields_good = (coverage_df['Coverage %'] >= 70).sum()
    fields_poor = (coverage_df['Coverage %'] < 40).sum()
    
    print(f"📊 Average Coverage: {avg_coverage:.1f}%")
    print(f"✅ Fields with >70% coverage: {fields_good}/{len(FUNDAMENTAL_FIELDS)}")
    print(f"❌ Fields with <40% coverage: {fields_poor}/{len(FUNDAMENTAL_FIELDS)}")
    
    # Low coverage fields detail
    if fields_poor > 0:
        print("\n" + "=" * 80)
        print("PROBLEMATIC FIELDS (Coverage < 40%)")
        print("=" * 80)
        print()
        
        low_fields = coverage_df[coverage_df['Coverage %'] < 40]
        for _, row in low_fields.iterrows():
            field = row['Field']
            coverage = row['Coverage %']
            populated = row['Populated']
            
            print(f"❌ {field}: {coverage:.1f}% ({populated}/{total})")
            
            # Show which tickers have this field
            tickers_with = df[df[field].notna()]['ticker'].tolist()
            if tickers_with:
                print(f"   Populated for: {', '.join(tickers_with)}")
            
            tickers_missing = df[df[field].isna()]['ticker'].tolist()
            print(f"   Missing for: {', '.join(tickers_missing[:5])}" + 
                  (f" ... and {len(tickers_missing)-5} more" if len(tickers_missing) > 5 else ""))
            print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if avg_coverage < 60:
        print("🔴 CRITICAL: Overall coverage is below 60%")
        print("\nImmediate actions:")
        print("1. Review yfinance_improvements.py - ensure all methods are being called")
        print("2. Check if improved_calc is initialized in YahooFinanceIntegrator")
        print("3. Verify fallback calculations are working")
    elif avg_coverage < 75:
        print("⚠️  WARNING: Overall coverage is below target (75%)")
        print("\nSuggested improvements:")
        print("1. Focus on fields with <40% coverage")
        print("2. Add additional fallback methods")
    else:
        print("✅ Coverage is good! Above 75% target")
    
    print()

if __name__ == "__main__":
    asyncio.run(analyze())
