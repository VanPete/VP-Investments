"""
Data Quality Analysis Script
Analyzes the most recent signals run to identify data quality issues
"""

import asyncio
import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime
from collections import Counter

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')

async def analyze_latest_signals():
    """Analyze the most recent signals for data quality issues"""
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("=" * 80)
    print("DATA QUALITY ANALYSIS - LATEST SIGNALS RUN")
    print("=" * 80)
    
    # Get the most recent run
    runs = supabase.table('runs').select('*').order('id', desc=True).limit(1).execute()
    
    if not runs.data:
        print("❌ No runs found!")
        return
    
    latest_run = runs.data[0]
    run_id = latest_run['run_id']
    
    print(f"\n📊 Latest Run: {run_id}")
    print(f"   Run ID: {latest_run.get('id')}")
    print(f"   Started: {latest_run.get('started_at')}")
    print(f"   Status: {latest_run.get('status')}")
    print(f"   Total Signals: {latest_run.get('total_signals')}")
    
    # Get signals from this run
    signals = supabase.table('signals').select('*').eq('run_id', run_id).execute()
    
    if not signals.data:
        print(f"\n❌ No signals found for run_id: {run_id}")
        return
    
    print(f"\n✅ Retrieved {len(signals.data)} signals")
    
    # Analyze data quality issues
    issues = {
        'missing_rank': [],
        'missing_sector': [],
        'missing_company': [],
        'missing_scores': [],
        'missing_technical': [],
        'missing_fundamental': [],
        'null_phase7_scores': []
    }
    
    print("\n" + "=" * 80)
    print("CHECKING CORE FIELDS")
    print("=" * 80)
    
    for signal in signals.data:
        ticker = signal['ticker']
        
        # Check rank
        if signal.get('rank') is None:
            issues['missing_rank'].append(ticker)
        
        # Check sector
        if signal.get('sector') is None or signal.get('sector') == '':
            issues['missing_sector'].append(ticker)
        
        # Check company
        if signal.get('company') is None or signal.get('company') == ticker:
            issues['missing_company'].append(ticker)
        
        # Check Phase 7 scores
        phase7_scores = {
            'technical_score': signal.get('technical_score'),
            'fundamental_score': signal.get('fundamental_score'),
            'news_macro_score': signal.get('news_macro_score'),
            'social_alternative_score': signal.get('social_alternative_score'),
            'risk_stability_score': signal.get('risk_stability_score'),
            'institutional_smart_money_score': signal.get('institutional_smart_money_score')
        }
        
        null_scores = [k for k, v in phase7_scores.items() if v is None or v == 0]
        if len(null_scores) >= 3:  # If 3+ scores are null/zero
            issues['null_phase7_scores'].append((ticker, null_scores))
        
        # Check key technical indicators
        technical_indicators = ['rsi', 'macd_histogram', 'above_50d_ma_pct', 'above_200d_ma_pct']
        missing_tech = [k for k in technical_indicators if signal.get(k) is None]
        if len(missing_tech) >= 2:
            issues['missing_technical'].append((ticker, missing_tech))
        
        # Check key fundamental indicators
        fundamental_indicators = ['pe_ratio', 'eps_growth', 'market_cap']
        missing_fund = [k for k in fundamental_indicators if signal.get(k) is None]
        if len(missing_fund) >= 2:
            issues['missing_fundamental'].append((ticker, missing_fund))
    
    # Print results
    print(f"\n🔢 Rank Issues: {len(issues['missing_rank'])} signals")
    if issues['missing_rank']:
        print(f"   Missing: {', '.join(issues['missing_rank'][:10])}")
        if len(issues['missing_rank']) > 10:
            print(f"   ... and {len(issues['missing_rank']) - 10} more")
    
    print(f"\n🏢 Sector Issues: {len(issues['missing_sector'])} signals")
    if issues['missing_sector']:
        print(f"   Missing: {', '.join(issues['missing_sector'][:10])}")
        if len(issues['missing_sector']) > 10:
            print(f"   ... and {len(issues['missing_sector']) - 10} more")
    
    print(f"\n🏭 Company Name Issues: {len(issues['missing_company'])} signals")
    if issues['missing_company']:
        print(f"   Missing: {', '.join(issues['missing_company'][:10])}")
    
    print(f"\n📈 Phase 7 Score Issues: {len(issues['null_phase7_scores'])} signals")
    if issues['null_phase7_scores']:
        for ticker, missing in issues['null_phase7_scores'][:5]:
            print(f"   {ticker}: Missing {len(missing)} scores - {', '.join(missing)}")
    
    print(f"\n📉 Technical Indicator Issues: {len(issues['missing_technical'])} signals")
    if issues['missing_technical']:
        for ticker, missing in issues['missing_technical'][:5]:
            print(f"   {ticker}: Missing {', '.join(missing)}")
    
    print(f"\n💰 Fundamental Indicator Issues: {len(issues['missing_fundamental'])} signals")
    if issues['missing_fundamental']:
        for ticker, missing in issues['missing_fundamental'][:5]:
            print(f"   {ticker}: Missing {', '.join(missing)}")
    
    # Sample a few signals to show actual data
    print("\n" + "=" * 80)
    print("SAMPLE SIGNALS (First 3)")
    print("=" * 80)
    
    for i, signal in enumerate(signals.data[:3], 1):
        print(f"\n{i}. {signal['ticker']} - {signal.get('company', 'NO COMPANY')}")
        print(f"   Rank: {signal.get('rank', 'NULL')}")
        print(f"   Sector: {signal.get('sector', 'NULL')}")
        print(f"   Signal Score: {signal.get('signal_score', 'NULL')}")
        print(f"   Phase 7 Scores:")
        print(f"      Technical (25%): {signal.get('technical_score', 'NULL')}")
        print(f"      Fundamental (25%): {signal.get('fundamental_score', 'NULL')}")
        print(f"      News/Macro (20%): {signal.get('news_macro_score', 'NULL')}")
        print(f"      Social/Alt (15%): {signal.get('social_alternative_score', 'NULL')}")
        print(f"      Risk/Stability (10%): {signal.get('risk_stability_score', 'NULL')}")
        print(f"      Institutional (5%): {signal.get('institutional_smart_money_score', 'NULL')}")
        print(f"   Key Metrics:")
        print(f"      RSI: {signal.get('rsi', 'NULL')}")
        print(f"      Above 50D MA: {signal.get('above_50d_ma_pct', 'NULL')}%")
        print(f"      Above 200D MA: {signal.get('above_200d_ma_pct', 'NULL')}%")
        print(f"      PE Ratio: {signal.get('pe_ratio', 'NULL')}")
        print(f"      Market Cap: {signal.get('market_cap', 'NULL')}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_issues = sum([
        len(issues['missing_rank']),
        len(issues['missing_sector']),
        len(issues['missing_company']),
        len(issues['null_phase7_scores']),
        len(issues['missing_technical']),
        len(issues['missing_fundamental'])
    ])
    
    if total_issues == 0:
        print("\n✅ No major data quality issues found!")
    else:
        print(f"\n⚠️  Found {total_issues} potential issues across {len(signals.data)} signals")
        print("\nRecommendations:")
        
        if issues['missing_sector']:
            print("   1. ✋ Sector field is NULL - Check Yahoo Finance API response")
            print("      - info.get('sector') may be returning None")
            print("      - Verify tickers are valid and have sector data")
        
        if issues['missing_rank']:
            print("   2. ✋ Rank field is NULL - Check signal ranking logic")
            print("      - Ranks should be assigned in save_signals_to_database()")
        
        if issues['missing_company']:
            print("   3. ✋ Company names missing - Check Yahoo Finance company field")
            print("      - info.get('longName') or info.get('shortName') may be None")
        
        if issues['null_phase7_scores']:
            print("   4. ✋ Phase 7 scores are NULL/0 - Check scoring calculation")
            print("      - Verify SignalScorer is calculating all 6 components")
        
        if issues['missing_technical'] or issues['missing_fundamental']:
            print("   5. ✋ Technical/Fundamental data sparse - Check data fetching")
            print("      - Verify yfinance data retrieval is comprehensive")

if __name__ == '__main__':
    asyncio.run(analyze_latest_signals())
