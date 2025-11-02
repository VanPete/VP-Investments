"""Check performance data for older signals."""
from supabase import create_client
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))

print("Checking performance data for signals >= 1 day old...\n")

# Get performance records that are at least 1 day old
perf = client.table('performance').select(
    '''
    status, baseline_date, intervals_completed,
    return_1d, spy_return_1d, qqq_return_1d, sector_return_1d,
    alpha_1d, qqq_alpha_1d, sector_alpha_1d,
    signals!inner(ticker, created_at)
    '''
).limit(10).order('baseline_date', desc=False).execute()

print(f"Found {len(perf.data)} older performance records:\n")

for p in perf.data:
    ticker = p['signals']['ticker']
    baseline = p.get('baseline_date', '')
    
    # Calculate age
    if baseline:
        baseline_dt = datetime.fromisoformat(baseline.replace('Z', '+00:00'))
        age_days = (datetime.now(timezone.utc) - baseline_dt).days
    else:
        age_days = 0
    
    status = p.get('status', 'N/A')
    completed = p.get('intervals_completed') or []
    
    # Check if returns are populated
    ret_1d = p.get('return_1d')
    spy_ret_1d = p.get('spy_return_1d')
    qqq_ret_1d = p.get('qqq_return_1d')
    sector_ret_1d = p.get('sector_return_1d')
    alpha_1d = p.get('alpha_1d')
    qqq_alpha_1d = p.get('qqq_alpha_1d')
    
    print(f"{ticker:6s} | Age: {age_days:3d}d | Status: {status:10s} | Completed: {len(completed)}/7")
    
    if age_days >= 1:
        print(f"         Return: {f'{ret_1d:.2%}' if ret_1d is not None else 'NULL':>8s}")
        print(f"         SPY:    {f'{spy_ret_1d:.2%}' if spy_ret_1d is not None else 'NULL':>8s} | Alpha: {f'{alpha_1d:.2%}' if alpha_1d is not None else 'NULL':>8s}")
        print(f"         QQQ:    {f'{qqq_ret_1d:.2%}' if qqq_ret_1d is not None else 'NULL':>8s} | Alpha: {f'{qqq_alpha_1d:.2%}' if qqq_alpha_1d is not None else 'NULL':>8s}")
        print(f"         Sector: {f'{sector_ret_1d:.2%}' if sector_ret_1d is not None else 'NULL':>8s}")
    print()
