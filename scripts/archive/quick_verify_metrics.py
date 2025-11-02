"""
Quick verification of group_performance metrics.
"""
import os
import json
from dotenv import load_dotenv
from supabase import create_client

# Load env
load_dotenv()

# Initialize Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(url, key)

# Query analytics
response = supabase.table('analytics').select('*').order('period_type').execute()

print(f"Found {len(response.data)} analytics rows\n")
print("=" * 120)

for row in response.data:
    interval = row['period_type']
    
    # Check individual metrics
    cagr = row.get('cagr')
    vol = row.get('volatility')
    sortino = row.get('sortino_ratio')
    calmar = row.get('calmar_ratio')
    
    print(f"\n📊 {interval}:")
    print(f"  CAGR: {f'{cagr:.2f}' if cagr else 'NULL':>8s}")
    print(f"  Volatility: {f'{vol:.2f}' if vol else 'NULL':>8s}")
    print(f"  Sortino: {f'{sortino:.2f}' if sortino else 'NULL':>8s}")
    print(f"  Calmar: {f'{calmar:.2f}' if calmar else 'NULL':>8s}")
    
    # Check group performance
    gp = row.get('group_performance')
    if gp:
        if isinstance(gp, str):
            gp = json.loads(gp)
        
        # Check one group's one quintile to see structure
        if 'technical' in gp and 'top_20pct' in gp['technical']:
            metrics = gp['technical']['top_20pct']
            has_vol = 'volatility' in metrics
            has_sortino = 'sortino' in metrics
            has_calmar = 'calmar' in metrics
            
            print(f"  Group Performance (technical/top_20pct):")
            print(f"    Has volatility: {has_vol}")
            print(f"    Has sortino: {has_sortino}")
            print(f"    Has calmar: {has_calmar}")
            if has_vol:
                print(f"    Values: vol={metrics['volatility']}, sortino={metrics['sortino']}, calmar={metrics['calmar']}")

print("\n" + "=" * 120)
