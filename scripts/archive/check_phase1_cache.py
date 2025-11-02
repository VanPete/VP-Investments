"""Check Phase 1 benchmark cache to see what data is being passed to Phase 6."""
import json
import os

# Check the latest pipeline results to see what Phase 1 cached
results_file = 'frontend/public/results/pipeline_results_20251101_003647.json'

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("Checking Phase 1 benchmark cache from latest pipeline run...\n")
    
    # Check if sector_etf_data exists
    if 'sector_etf_data' in data:
        etf_data = data['sector_etf_data']
        print(f"✅ sector_etf_data found with {len(etf_data)} ETFs:\n")
        
        for etf, df_data in etf_data.items():
            if isinstance(df_data, dict):
                # It's a serialized DataFrame
                columns = df_data.get('columns', [])
                index_len = len(df_data.get('index', []))
                print(f"  {etf:10s} | Columns: {columns} | Rows: {index_len}")
                
                # Check if it has Close column
                if 'Close' not in columns:
                    print(f"             ⚠️  Missing 'Close' column!")
                
                # Check date range if index exists
                if 'index' in df_data and df_data['index']:
                    first_date = df_data['index'][0]
                    last_date = df_data['index'][-1]
                    print(f"             Date range: {first_date} to {last_date}")
            else:
                print(f"  {etf:10s} | ⚠️  Not a valid DataFrame structure")
        
        print("\n" + "="*80)
        print("Checking for required benchmarks:")
        print("="*80)
        
        required = ['SPY', 'QQQ', 'XLK', 'XLF', 'XLV']  # Sample sector ETFs
        for req in required:
            if req in etf_data:
                print(f"  ✅ {req} present")
            else:
                print(f"  ❌ {req} MISSING")
    else:
        print("❌ sector_etf_data NOT found in pipeline results")
        print("\nAvailable keys:")
        for key in data.keys():
            print(f"  - {key}")
else:
    print(f"❌ Results file not found: {results_file}")
