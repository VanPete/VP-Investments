"""
Detailed NULL analysis for signals table - identifies all columns with missing data
"""
import asyncio
from backend.storage.database import SupabaseInterface

async def analyze_signals_nulls():
    """Analyze all columns in signals table for NULL values"""
    db = SupabaseInterface()
    await db.connect()
    
    print("\n" + "="*80)
    print("DETAILED SIGNALS TABLE NULL ANALYSIS")
    print("="*80)
    
    # Get all signals data
    query = """
    SELECT * FROM signals 
    ORDER BY created_at DESC 
    LIMIT 254
    """
    
    result = await db.execute_query(query)
    
    if not result or len(result) == 0:
        print("❌ No signals found in table")
        return
    
    total_rows = len(result)
    print(f"\n📊 Total signals analyzed: {total_rows}\n")
    
    # Get all column names from first row
    if result:
        columns = list(result[0].keys())
        
        # Analyze each column for NULLs
        null_analysis = []
        
        for col in columns:
            null_count = sum(1 for row in result if row.get(col) is None)
            null_pct = (null_count / total_rows) * 100
            
            if null_count > 0:
                null_analysis.append({
                    'column': col,
                    'null_count': null_count,
                    'null_pct': null_pct,
                    'populated_count': total_rows - null_count
                })
        
        # Sort by null percentage (highest first)
        null_analysis.sort(key=lambda x: x['null_pct'], reverse=True)
        
        if null_analysis:
            print(f"🔴 COLUMNS WITH NULL VALUES ({len(null_analysis)} columns affected):\n")
            print(f"{'Column':<40} {'NULL Count':<12} {'NULL %':<10} {'Populated':<12}")
            print("-" * 80)
            
            for item in null_analysis:
                status = "🔴" if item['null_pct'] > 50 else "🟡" if item['null_pct'] > 10 else "🟢"
                print(f"{status} {item['column']:<37} {item['null_count']:<12} {item['null_pct']:<9.1f}% {item['populated_count']:<12}")
            
            # Category breakdown
            print("\n" + "="*80)
            print("CATEGORIZED NULL ANALYSIS")
            print("="*80)
            
            critical = [x for x in null_analysis if x['null_pct'] > 50]
            moderate = [x for x in null_analysis if 10 < x['null_pct'] <= 50]
            minor = [x for x in null_analysis if x['null_pct'] <= 10]
            
            if critical:
                print(f"\n🔴 CRITICAL (>50% NULL): {len(critical)} columns")
                for item in critical:
                    print(f"   - {item['column']}: {item['null_pct']:.1f}% NULL")
            
            if moderate:
                print(f"\n🟡 MODERATE (10-50% NULL): {len(moderate)} columns")
                for item in moderate:
                    print(f"   - {item['column']}: {item['null_pct']:.1f}% NULL")
            
            if minor:
                print(f"\n🟢 MINOR (<10% NULL): {len(minor)} columns")
                for item in minor:
                    print(f"   - {item['column']}: {item['null_pct']:.1f}% NULL")
            
            # Sample some NULL records for critical fields
            if critical:
                print("\n" + "="*80)
                print("SAMPLE RECORDS WITH CRITICAL NULLS")
                print("="*80)
                
                for item in critical[:5]:  # Top 5 critical fields
                    col = item['column']
                    print(f"\n🔍 {col} (NULL in {item['null_count']} records):")
                    
                    # Find 3 records where this column is NULL
                    null_samples = [row for row in result if row.get(col) is None][:3]
                    for i, sample in enumerate(null_samples, 1):
                        ticker = sample.get('ticker', 'UNKNOWN')
                        signal_type = sample.get('signal_type', 'UNKNOWN')
                        print(f"   Sample {i}: {ticker} ({signal_type})")
        else:
            print("✅ NO NULL VALUES FOUND - All columns fully populated!")
        
        # Show fully populated columns
        populated_cols = [col for col in columns if all(row.get(col) is not None for row in result)]
        if populated_cols:
            print(f"\n✅ FULLY POPULATED COLUMNS ({len(populated_cols)} columns):")
            print(f"   {', '.join(populated_cols[:20])}")
            if len(populated_cols) > 20:
                print(f"   ... and {len(populated_cols) - 20} more")
    
    print("\n" + "="*80)
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(analyze_signals_nulls())
