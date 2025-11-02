"""Check historical benchmark data across all performance records."""
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))

print("Checking ALL performance records for benchmark data...\n")

# Get all performance records ordered by baseline date
perf = client.table('performance').select(
    '''
    baseline_date,
    return_1d, spy_return_1d, qqq_return_1d, sector_return_1d,
    intervals_completed,
    signals!inner(ticker)
    '''
).not_.is_('return_1d', 'null').order('baseline_date', desc=False).limit(50).execute()

print(f"Analyzed {len(perf.data)} performance records with returns:\n")

spy_count = 0
qqq_count = 0
sector_count = 0
total_with_returns = 0

for p in perf.data:
    ticker = p['signals']['ticker']
    baseline = p.get('baseline_date', '')[:10]  # Just date
    
    ret_1d = p.get('return_1d')
    spy_ret = p.get('spy_return_1d')
    qqq_ret = p.get('qqq_return_1d')
    sector_ret = p.get('sector_return_1d')
    
    if ret_1d is not None:
        total_with_returns += 1
        
        if spy_ret is not None:
            spy_count += 1
        if qqq_ret is not None:
            qqq_count += 1
        if sector_ret is not None:
            sector_count += 1

print(f"Benchmark Population Statistics:")
print(f"  Total records with ticker returns: {total_with_returns}")
print(f"  SPY populated:    {spy_count}/{total_with_returns} ({spy_count/total_with_returns*100:.1f}%)")
print(f"  QQQ populated:    {qqq_count}/{total_with_returns} ({qqq_count/total_with_returns*100:.1f}%)")
print(f"  Sector populated: {sector_count}/{total_with_returns} ({sector_count/total_with_returns*100:.1f}%)")

print("\n" + "="*80)
print("Sample Records (first 10 with returns):")
print("="*80)

for i, p in enumerate(perf.data[:10], 1):
    ticker = p['signals']['ticker']
    baseline = p.get('baseline_date', '')[:10]
    
    ret = p.get('return_1d')
    spy = p.get('spy_return_1d')
    qqq = p.get('qqq_return_1d')
    sector = p.get('sector_return_1d')
    
    print(f"\n{i}. {ticker:6s} | Baseline: {baseline}")
    print(f"   Ticker:  {f'{ret:.2%}' if ret is not None else 'NULL'}")
    print(f"   SPY:     {f'{spy:.2%}' if spy is not None else 'NULL'}")
    print(f"   QQQ:     {f'{qqq:.2%}' if qqq is not None else 'NULL'}")
    print(f"   Sector:  {f'{sector:.2%}' if sector is not None else 'NULL'}")

# Check if there are ANY records with QQQ or sector data
print("\n" + "="*80)
print("Searching for ANY records with QQQ or Sector data...")
print("="*80)

qqq_any = client.table('performance').select('signals!inner(ticker), qqq_return_1d').not_.is_('qqq_return_1d', 'null').limit(5).execute()
sector_any = client.table('performance').select('signals!inner(ticker), sector_return_1d').not_.is_('sector_return_1d', 'null').limit(5).execute()

print(f"\nRecords with QQQ data: {len(qqq_any.data)}")
if qqq_any.data:
    for r in qqq_any.data[:3]:
        print(f"  {r['signals']['ticker']}: qqq_return_1d = {r['qqq_return_1d']}")

print(f"\nRecords with Sector data: {len(sector_any.data)}")
if sector_any.data:
    for r in sector_any.data[:3]:
        print(f"  {r['signals']['ticker']}: sector_return_1d = {r['sector_return_1d']}")
