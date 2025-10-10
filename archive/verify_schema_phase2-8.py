"""
Comprehensive Schema Verification for Phase 2-8 Enhancements
Verifies Supabase schema has all required columns properly organized
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Dict, List, Set
import json

# Load environment variables
load_dotenv()

print("="*80)
print("PHASE 2-8 SCHEMA VERIFICATION")
print("="*80)
print("\nVerifying all enhancement columns are present and properly grouped...\n")

# Expected columns by phase
PHASE_2_COLUMNS = {
    'z_score_momentum': 'NUMERIC',
    'z_score_volume': 'NUMERIC',
    'z_score_volatility': 'NUMERIC',
    'z_score_valuation': 'NUMERIC'
}

PHASE_3_COLUMNS = {
    'trade_type': 'TEXT',
    'trade_type_confidence': 'NUMERIC'
}

PHASE_4_COLUMNS = {
    'risk_score': 'NUMERIC',
    'risk_level': 'TEXT',
    'volatility_risk': 'NUMERIC',
    'liquidity_risk': 'NUMERIC',
    'leverage_risk': 'NUMERIC',
    'concentration_risk': 'NUMERIC',
    'technical_risk': 'NUMERIC',
    'fundamental_risk': 'NUMERIC',
    'sentiment_risk': 'NUMERIC'
}

PHASE_5_COLUMNS = {
    'atr': 'NUMERIC',
    'atr_percent': 'NUMERIC',
    'historical_volatility': 'NUMERIC',
    'implied_volatility': 'NUMERIC',
    'put_call_ratio': 'NUMERIC',
    'open_interest': 'BIGINT',
    'profit_margin': 'NUMERIC',
    'operating_margin': 'NUMERIC',
    'roe': 'NUMERIC',
    'debt_to_equity': 'NUMERIC',
    'current_ratio': 'NUMERIC',
    'institutional_ownership': 'NUMERIC',
    'insider_ownership': 'NUMERIC',
    'short_interest': 'NUMERIC'
}

PHASE_6_COLUMNS = {
    'adjusted_signal_score': 'NUMERIC',
    'position_size_recommendation': 'NUMERIC',
    'entry_threshold': 'NUMERIC'
}

PHASE_7_COLUMNS = {
    'risk_narrative': 'TEXT'
}

PHASE_8_COLUMNS = {
    'backtest_entry_threshold': 'NUMERIC',
    'backtest_hold_period_days': 'INTEGER',
    'backtest_position_size_pct': 'NUMERIC',
    'backtest_stop_loss_price': 'NUMERIC',
    'backtest_take_profit_price': 'NUMERIC',
    'backtest_risk_reward_ratio': 'NUMERIC'
}

# Signal Performance table columns
SIGNAL_PERFORMANCE_COLUMNS = {
    'return_1d': 'NUMERIC',
    'return_3d': 'NUMERIC',
    'return_7d': 'NUMERIC',
    'return_10d': 'NUMERIC',
    'return_30d': 'NUMERIC',
    'spy_1d_return': 'NUMERIC',
    'spy_3d_return': 'NUMERIC',
    'spy_7d_return': 'NUMERIC',
    'spy_10d_return': 'NUMERIC',
    'spy_30d_return': 'NUMERIC',
    'beat_spy_1d': 'BOOLEAN',
    'beat_spy_3d': 'BOOLEAN',
    'beat_spy_7d': 'BOOLEAN',
    'beat_spy_10d': 'BOOLEAN',
    'beat_spy_30d': 'BOOLEAN'
}


def connect_to_supabase() -> Client:
    """Connect to Supabase"""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    
    return create_client(url, key)


def get_table_columns(supabase: Client, table_name: str) -> Dict[str, str]:
    """Get all columns for a table"""
    try:
        # Query the table to get column info
        # We'll use a LIMIT 0 query to get column names without data
        result = supabase.table(table_name).select("*").limit(1).execute()
        
        if result.data:
            # Get column names from the first row
            columns = list(result.data[0].keys()) if result.data else []
            return {col: 'UNKNOWN' for col in columns}
        else:
            # Try empty result to still get column names
            return {}
    except Exception as e:
        print(f"   Error querying {table_name}: {e}")
        return {}


def verify_phase_columns(current_columns: Set[str], expected_columns: Dict[str, str], phase_name: str) -> tuple[List[str], bool]:
    """Verify columns for a specific phase"""
    missing = []
    present = []
    
    for col_name in expected_columns.keys():
        if col_name in current_columns:
            present.append(col_name)
        else:
            missing.append(col_name)
    
    all_present = len(missing) == 0
    status = "✅ PASS" if all_present else "❌ MISSING"
    
    print(f"\n{phase_name}:")
    print(f"   Status: {status}")
    print(f"   Present: {len(present)}/{len(expected_columns)}")
    
    if present:
        print(f"   ✅ Found: {', '.join(present[:5])}")
        if len(present) > 5:
            print(f"            ... and {len(present)-5} more")
    
    if missing:
        print(f"   ❌ Missing: {', '.join(missing)}")
    
    return missing, all_present


def generate_alter_statements(table_name: str, missing_columns: Dict[str, str]) -> List[str]:
    """Generate ALTER TABLE statements for missing columns"""
    statements = []
    
    for col_name, col_type in missing_columns.items():
        # Determine constraints based on column name and type
        constraints = []
        
        if col_type == 'NUMERIC':
            if 'score' in col_name or 'confidence' in col_name or 'pct' in col_name or 'recommendation' in col_name or 'threshold' in col_name:
                constraints.append('CHECK ({} >= 0 AND {} <= 1)'.format(col_name, col_name))
            elif 'risk' in col_name and col_name.endswith('_risk'):
                constraints.append('CHECK ({} >= 0 AND {} <= 100)'.format(col_name, col_name))
        
        constraint_str = ' ' + ' '.join(constraints) if constraints else ''
        statement = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {col_type}{constraint_str};"
        statements.append(statement)
    
    return statements


def query_recent_signals(supabase: Client, limit: int = 5) -> List[Dict]:
    """Query recent signals to check data population"""
    try:
        result = supabase.table('signals').select('*').order('created_at', desc=True).limit(limit).execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"   Error querying signals: {e}")
        return []


def check_data_population(signals: List[Dict], phase_columns: Dict[str, Dict[str, str]]) -> None:
    """Check if Phase 2-8 columns have actual data"""
    if not signals:
        print("\n❌ No recent signals found to verify data population")
        return
    
    print(f"\n{'='*80}")
    print(f"DATA POPULATION CHECK (Most Recent {len(signals)} Signals)")
    print(f"{'='*80}")
    
    for phase_name, columns in phase_columns.items():
        print(f"\n{phase_name}:")
        populated = 0
        null_count = 0
        
        for col_name in columns.keys():
            has_data = False
            for signal in signals:
                if signal.get(col_name) is not None:
                    has_data = True
                    populated += 1
                    break
            
            if not has_data:
                null_count += 1
        
        status = "✅ POPULATED" if null_count == 0 else f"⚠️  {null_count}/{len(columns)} NULL"
        print(f"   Status: {status}")
        
        # Show sample data from first signal
        if signals:
            print(f"   Sample data from ticker '{signals[0].get('ticker', 'N/A')}':")
            for col_name in list(columns.keys())[:3]:  # Show first 3
                value = signals[0].get(col_name, 'NOT FOUND')
                if value == 'NOT FOUND':
                    print(f"      {col_name}: ❌ COLUMN NOT FOUND")
                elif value is None:
                    print(f"      {col_name}: ⚠️  NULL")
                else:
                    print(f"      {col_name}: ✅ {value}")


def main():
    """Main verification function"""
    
    # Connect to Supabase
    print("[1/6] Connecting to Supabase...")
    try:
        supabase = connect_to_supabase()
        print("   ✅ Connected successfully\n")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Get current signals table columns
    print("[2/6] Retrieving signals table schema...")
    current_columns = get_table_columns(supabase, 'signals')
    current_column_names = set(current_columns.keys())
    print(f"   ✅ Found {len(current_column_names)} columns in signals table\n")
    
    # Verify each phase
    print(f"[3/6] Verifying Phase 2-8 columns...")
    print("="*80)
    
    all_missing = {}
    phase_results = {}
    
    phases = {
        'Phase 2 (Z-Score Normalization)': PHASE_2_COLUMNS,
        'Phase 3 (Trade Type Classification)': PHASE_3_COLUMNS,
        'Phase 4 (Risk Scoring System)': PHASE_4_COLUMNS,
        'Phase 5 (Enhanced Data Collection)': PHASE_5_COLUMNS,
        'Phase 6 (Score Adjustments)': PHASE_6_COLUMNS,
        'Phase 7 (AI-Enhanced Narratives)': PHASE_7_COLUMNS,
        'Phase 8 (Backtesting Integration)': PHASE_8_COLUMNS
    }
    
    for phase_name, expected_cols in phases.items():
        missing, all_present = verify_phase_columns(current_column_names, expected_cols, phase_name)
        phase_results[phase_name] = all_present
        if missing:
            all_missing.update({col: expected_cols[col] for col in missing})
    
    # Verify signal_performance table
    print(f"\n{'='*80}")
    print("[4/6] Verifying signal_performance table...")
    print("="*80)
    
    perf_columns = get_table_columns(supabase, 'signal_performance')
    perf_column_names = set(perf_columns.keys())
    print(f"   Found {len(perf_column_names)} columns in signal_performance table")
    
    perf_missing, perf_ok = verify_phase_columns(
        perf_column_names, 
        SIGNAL_PERFORMANCE_COLUMNS, 
        'Signal Performance Columns'
    )
    
    # Query recent data
    print(f"\n{'='*80}")
    print("[5/6] Checking data population...")
    print("="*80)
    
    signals = query_recent_signals(supabase, limit=5)
    if signals:
        print(f"   ✅ Retrieved {len(signals)} recent signals")
        check_data_population(signals, phases)
    else:
        print("   ⚠️  No recent signals found")
    
    # Generate migration SQL
    print(f"\n{'='*80}")
    print("[6/6] Generating schema updates...")
    print("="*80)
    
    if all_missing or perf_missing:
        print("\n📝 MISSING COLUMNS DETECTED - Generating ALTER statements...\n")
        
        # Generate signals table updates
        if all_missing:
            print("-- Signals Table Updates")
            print("-- Add missing Phase 2-8 columns\n")
            statements = generate_alter_statements('public.signals', all_missing)
            for stmt in statements:
                print(stmt)
        
        # Generate signal_performance table updates
        if perf_missing:
            print("\n-- Signal Performance Table Updates")
            print("-- Add missing return columns\n")
            perf_missing_dict = {col: SIGNAL_PERFORMANCE_COLUMNS[col] for col in perf_missing}
            perf_statements = generate_alter_statements('public.signal_performance', perf_missing_dict)
            for stmt in perf_statements:
                print(stmt)
        
        # Save to file
        migration_file = 'migrations/phase2-8_schema_update.sql'
        try:
            os.makedirs('migrations', exist_ok=True)
            with open(migration_file, 'w') as f:
                f.write("-- Phase 2-8 Schema Update\n")
                f.write("-- Generated: " + str(__import__('datetime').datetime.now()) + "\n\n")
                
                if all_missing:
                    f.write("-- Signals Table Updates\n")
                    f.write("-- Add missing Phase 2-8 columns\n\n")
                    for stmt in statements:
                        f.write(stmt + "\n")
                
                if perf_missing:
                    f.write("\n-- Signal Performance Table Updates\n")
                    f.write("-- Add missing return columns\n\n")
                    for stmt in perf_statements:
                        f.write(stmt + "\n")
            
            print(f"\n✅ Migration SQL saved to: {migration_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save migration file: {e}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    total_phases = len(phases)
    passed_phases = sum(1 for result in phase_results.values() if result)
    
    print(f"\n📊 Phase Status: {passed_phases}/{total_phases} phases complete")
    
    for phase_name, passed in phase_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {phase_name}")
    
    print(f"\n📊 Signal Performance Table: {'✅ PASS' if perf_ok else '❌ FAIL'}")
    
    if signals:
        print(f"\n📊 Data Population:")
        print(f"   Recent signals found: {len(signals)}")
        print(f"   Latest ticker: {signals[0].get('ticker', 'N/A')}")
        print(f"   Latest timestamp: {signals[0].get('created_at', 'N/A')}")
    
    if all_missing or perf_missing:
        print(f"\n⚠️  SCHEMA UPDATES REQUIRED")
        print(f"   Missing signals columns: {len(all_missing)}")
        print(f"   Missing performance columns: {len(perf_missing)}")
        print(f"\n   Run the generated migration SQL to update schema:")
        print(f"   {migration_file}")
    else:
        print(f"\n✅ SCHEMA VERIFICATION COMPLETE")
        print(f"   All Phase 2-8 columns present and accounted for!")
    
    return passed_phases == total_phases and perf_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
