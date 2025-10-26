"""
Supabase Database Debug Utility
Comprehensive tool for inspecting and troubleshooting the Supabase database
"""
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

load_dotenv()


class SupabaseDebugger:
    """Debug utility for Supabase database inspection"""
    
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        
        if not url or not key:
            raise ValueError("Missing Supabase credentials in .env file")
        
        self.client: Client = create_client(url, key)
        self.separator = "=" * 80
    
    def print_section(self, title: str):
        """Print a formatted section header"""
        print(f"\n{self.separator}")
        print(f"  {title}")
        print(f"{self.separator}\n")
    
    def check_signal_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Check all signal runs in the database"""
        self.print_section("SIGNAL RUNS")
        
        response = self.client.table('signal_runs').select('*').order('run_timestamp', desc=True).limit(limit).execute()
        
        if response.data:
            print(f"Found {len(response.data)} runs (showing last {limit}):\n")
            
            for i, run in enumerate(response.data, 1):
                print(f"Run #{i}:")
                print(f"  ID: {run['id']}")
                print(f"  Timestamp: {run['run_timestamp']}")
                print(f"  Status: {run['status']}")
                print(f"  Total Tickers: {run.get('total_tickers', 'N/A')}")
                print(f"  Successful: {run.get('successful_tickers', 'N/A')}")
                print(f"  Failed: {run.get('failed_tickers', 'N/A')}")
                print(f"  Pipeline Version: {run.get('pipeline_version', 'N/A')}")
                print(f"  Duration: {run.get('duration_seconds', 'N/A')}s")
                print(f"  Error: {run.get('error_message', 'None')}")
                
                # Count associated signals
                signals_count = self.client.table('signals').select('id', count='exact').eq('run_id', run['id']).execute()
                count = getattr(signals_count, 'count', 0) if hasattr(signals_count, 'count') else len(signals_count.data or [])
                print(f"  Signals Count: {count}")
                print("-" * 60)
            
            return response.data
        else:
            print("❌ No runs found in database")
            return []
    
    def check_signals(self, run_id: Optional[str] = None, limit: int = 10):
        """Check signals for a specific run or latest run"""
        self.print_section(f"SIGNALS{f' (Run: {run_id[:8]}...)' if run_id else ' (Latest Run)'}")
        
        query = self.client.table('signals').select('*')
        
        if run_id:
            query = query.eq('run_id', run_id)
        else:
            # Get latest run
            runs = self.client.table('signal_runs').select('id').order('run_timestamp', desc=True).limit(1).execute()
            if runs.data:
                run_id = runs.data[0]['id']
                query = query.eq('run_id', run_id)
        
        response = query.limit(limit).execute()
        
        if response.data:
            print(f"Found {len(response.data)} signals (showing first {limit}):\n")
            
            for i, signal in enumerate(response.data, 1):
                print(f"Signal #{i}:")
                print(f"  Ticker: {signal['ticker']}")
                print(f"  Rank: {signal.get('rank', 'N/A')}")
                print(f"  Overall Score: {signal.get('overall_score', 'N/A')}")
                print(f"  Coverage: {signal.get('total_coverage', 'N/A')}")
                print(f"  Backtest Status: {signal.get('backtest_status', 'N/A')}")
                print(f"  Return 1D: {signal.get('return_1d', 'N/A')}")
                print(f"  Return 7D: {signal.get('return_7d', 'N/A')}")
                print("-" * 40)
        else:
            print(f"❌ No signals found for run_id: {run_id}")
    
    def check_table_schema(self, table_name: str):
        """Check the schema/columns of a table by fetching one row"""
        self.print_section(f"TABLE SCHEMA: {table_name}")
        
        try:
            response = self.client.table(table_name).select('*').limit(1).execute()
            
            if response.data and len(response.data) > 0:
                print(f"Columns in '{table_name}' table:\n")
                for key, value in response.data[0].items():
                    value_type = type(value).__name__
                    value_str = str(value)[:50] if value else "None"
                    print(f"  - {key:30} ({value_type:10}) = {value_str}")
            else:
                print(f"⚠️  Table '{table_name}' exists but is empty")
                
        except Exception as e:
            print(f"❌ Error accessing table '{table_name}': {e}")
    
    def check_run_signals_relationship(self, run_id: str):
        """Check the relationship between a run and its signals"""
        self.print_section(f"RUN-SIGNALS RELATIONSHIP: {run_id[:8]}...")
        
        # Get run details
        run = self.client.table('signal_runs').select('*').eq('id', run_id).execute()
        if not run.data:
            print(f"❌ Run not found: {run_id}")
            return
        
        print(f"Run Details:")
        print(f"  ID: {run.data[0]['id']}")
        print(f"  Timestamp: {run.data[0]['run_timestamp']}")
        print(f"  Status: {run.data[0]['status']}")
        print(f"  Expected Tickers: {run.data[0].get('total_tickers', 'N/A')}")
        
        # Count signals
        signals = self.client.table('signals').select('id', count='exact').eq('run_id', run_id).execute()
        count = getattr(signals, 'count', 0) if hasattr(signals, 'count') else len(signals.data or [])
        
        print(f"\nSignals Count: {count}")
        
        if count != run.data[0].get('total_tickers'):
            print(f"⚠️  WARNING: Signal count ({count}) doesn't match total_tickers ({run.data[0].get('total_tickers')})")
        else:
            print(f"✅ Signal count matches total_tickers")
    
    def check_backtest_data(self, run_id: Optional[str] = None):
        """Check backtest data completeness"""
        self.print_section(f"BACKTEST DATA{f' (Run: {run_id[:8]}...)' if run_id else ' (Latest Run)'}")
        
        query = self.client.table('signals').select('ticker, backtest_status, return_1d, return_7d, return_30d, return_90d')
        
        if run_id:
            query = query.eq('run_id', run_id)
        
        response = query.execute()
        
        if response.data:
            total = len(response.data)
            with_1d = sum(1 for s in response.data if s.get('return_1d') is not None)
            with_7d = sum(1 for s in response.data if s.get('return_7d') is not None)
            with_30d = sum(1 for s in response.data if s.get('return_30d') is not None)
            with_90d = sum(1 for s in response.data if s.get('return_90d') is not None)
            
            print(f"Total Signals: {total}")
            print(f"\nBacktest Data Completeness:")
            print(f"  1D Returns:  {with_1d:3}/{total} ({with_1d/total*100:.1f}%)")
            print(f"  7D Returns:  {with_7d:3}/{total} ({with_7d/total*100:.1f}%)")
            print(f"  30D Returns: {with_30d:3}/{total} ({with_30d/total*100:.1f}%)")
            print(f"  90D Returns: {with_90d:3}/{total} ({with_90d/total*100:.1f}%)")
            
            # Show status distribution
            statuses = {}
            for s in response.data:
                status = s.get('backtest_status', 'unknown')
                statuses[status] = statuses.get(status, 0) + 1
            
            print(f"\nBacktest Status Distribution:")
            for status, count in statuses.items():
                print(f"  {status}: {count}")
    
    def compare_runs(self, run_id1: str, run_id2: str):
        """Compare two runs side by side"""
        self.print_section(f"COMPARING RUNS")
        
        for i, run_id in enumerate([run_id1, run_id2], 1):
            run = self.client.table('signal_runs').select('*').eq('id', run_id).execute()
            signals = self.client.table('signals').select('id', count='exact').eq('run_id', run_id).execute()
            
            if run.data:
                print(f"\nRun #{i}: {run_id[:8]}...")
                print(f"  Timestamp: {run.data[0]['run_timestamp']}")
                print(f"  Status: {run.data[0]['status']}")
                print(f"  Total Tickers: {run.data[0].get('total_tickers', 'N/A')}")
                count = getattr(signals, 'count', 0) if hasattr(signals, 'count') else len(signals.data or [])
                print(f"  Signals: {count}")
            else:
                print(f"\n❌ Run #{i} not found: {run_id}")
    
    def check_detail_tables(self, signal_id: str):
        """Check all detail tables for a specific signal"""
        self.print_section(f"DETAIL TABLES FOR SIGNAL: {signal_id[:8]}...")
        
        detail_tables = [
            'signals_technical',
            'signals_fundamental', 
            'signals_news_macro',
            'signals_social_alternative',
            'signals_risk_stability',
            'signals_institutional_smart_money'
        ]
        
        for table in detail_tables:
            try:
                response = self.client.table(table).select('*').eq('signal_id', signal_id).execute()
                
                if response.data:
                    print(f"\n✅ {table}:")
                    factors = response.data[0].get('factors', {})
                    print(f"   Factor count: {len(factors)}")
                    print(f"   Sample factors: {list(factors.keys())[:5]}")
                else:
                    print(f"\n❌ {table}: No data")
            except Exception as e:
                print(f"\n❌ {table}: Error - {e}")
    
    def full_diagnostic(self):
        """Run a full diagnostic of the database"""
        print("\n" + "=" * 80)
        print("  SUPABASE DATABASE FULL DIAGNOSTIC")
        print("=" * 80)
        
        # 1. Check all tables exist
        print("\n1. Checking table schemas...")
        for table in ['signal_runs', 'signals']:
            self.check_table_schema(table)
        
        # 2. Check runs
        print("\n2. Checking signal runs...")
        runs = self.check_signal_runs(limit=10)
        
        # 3. Check signals for latest run
        if runs:
            print("\n3. Checking signals for latest run...")
            self.check_signals(run_id=runs[0]['id'], limit=5)
            
            print("\n4. Checking backtest data...")
            self.check_backtest_data(run_id=runs[0]['id'])
            
            # 5. Check detail tables for first signal
            signals = self.client.table('signals').select('id').eq('run_id', runs[0]['id']).limit(1).execute()
            if signals.data:
                print("\n5. Checking detail tables...")
                self.check_detail_tables(signals.data[0]['id'])
        
        print(f"\n{self.separator}")
        print("  DIAGNOSTIC COMPLETE")
        print(f"{self.separator}\n")


def main():
    """Main entry point with interactive menu"""
    debugger = SupabaseDebugger()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'runs':
            debugger.check_signal_runs()
        elif command == 'signals':
            run_id = sys.argv[2] if len(sys.argv) > 2 else None
            debugger.check_signals(run_id=run_id)
        elif command == 'schema':
            table = sys.argv[2] if len(sys.argv) > 2 else 'signal_runs'
            debugger.check_table_schema(table)
        elif command == 'backtest':
            run_id = sys.argv[2] if len(sys.argv) > 2 else None
            debugger.check_backtest_data(run_id=run_id)
        elif command == 'full':
            debugger.full_diagnostic()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  runs - List all signal runs")
            print("  signals [run_id] - List signals (optionally for specific run)")
            print("  schema [table_name] - Show table schema")
            print("  backtest [run_id] - Check backtest data completeness")
            print("  full - Run full diagnostic")
    else:
        # Interactive mode
        debugger.full_diagnostic()


if __name__ == "__main__":
    main()
