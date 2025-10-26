"""
Backfill Script: Import JSON Pipeline Results to Supabase

This script imports historical pipeline results from JSON files into the Supabase database.
Useful for migrating old runs that were only saved as JSON before Phase 5 was working.

Usage:
    python backend/utils/backfill_from_json.py                    # Import all JSON files
    python backend/utils/backfill_from_json.py --limit 5          # Import only 5 most recent
    python backend/utils/backfill_from_json.py --dry-run          # Preview what would be imported
    python backend/utils/backfill_from_json.py --file pipeline_results_20251025_235529.json
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
from backend.storage.database import get_supabase_database
from backend.phases.phase5_persist import Phase5Persist

load_dotenv()


class JSONBackfiller:
    """Import JSON pipeline results into Supabase database"""
    
    def __init__(self, results_dir: str = "frontend/public/results"):
        self.results_dir = Path(results_dir)
        self.db = None
        self.phase5 = None
        
    async def initialize(self):
        """Initialize database connection"""
        self.db = await get_supabase_database()
        await self.db.connect()
        self.phase5 = Phase5Persist(self.db)
        print("✅ Database connection established")
    
    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.disconnect()
            print("✅ Database connection closed")
    
    def find_json_files(self, limit: int = None) -> List[Path]:
        """Find all JSON result files, sorted by timestamp (newest first)"""
        if not self.results_dir.exists():
            print(f"❌ Results directory not found: {self.results_dir}")
            return []
        
        json_files = sorted(
            self.results_dir.glob("pipeline_results_*.json"),
            key=lambda f: f.stem.split('_')[-2:],  # Sort by YYYYMMDD_HHMMSS
            reverse=True
        )
        
        if limit:
            json_files = json_files[:limit]
        
        return json_files
    
    def parse_timestamp_from_filename(self, filepath: Path) -> datetime:
        """Parse timestamp from filename: pipeline_results_YYYYMMDD_HHMMSS.json"""
        parts = filepath.stem.split('_')
        date_str = parts[-2]  # YYYYMMDD
        time_str = parts[-1]  # HHMMSS
        
        # Parse: 20251025_235529 -> 2025-10-25 23:55:29
        timestamp_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    
    async def check_if_run_exists(self, timestamp: datetime) -> bool:
        """Check if a run with this timestamp already exists"""
        # Query for runs within 1 minute of this timestamp
        response = self.db.supabase.table('signal_runs').select('id, run_timestamp').execute()
        
        if response.data:
            for run in response.data:
                # Handle different timestamp formats from Supabase
                ts_str = run['run_timestamp']
                # Remove timezone indicator and parse
                if 'Z' in ts_str:
                    ts_str = ts_str.replace('Z', '')
                elif '+' in ts_str:
                    ts_str = ts_str.split('+')[0]
                
                try:
                    run_time = datetime.fromisoformat(ts_str)
                    time_diff = abs((run_time - timestamp).total_seconds())
                    if time_diff < 60:  # Within 1 minute
                        return True
                except ValueError:
                    continue
        
        return False
    
    def calculate_coverage(self, factors: Dict[str, Any]) -> float:
        """Calculate coverage as ratio of non-null factors"""
        if not factors:
            return 0.0
        
        total = len(factors)
        non_null = sum(1 for v in factors.values() if v is not None)
        return non_null / total if total > 0 else 0.0
    
    async def import_json_file(self, filepath: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Import a single JSON file into Supabase"""
        
        # Parse timestamp from filename
        file_timestamp = self.parse_timestamp_from_filename(filepath)
        
        # Check if already exists
        exists = await self.check_if_run_exists(file_timestamp)
        if exists:
            return {
                'status': 'skipped',
                'reason': 'Run already exists',
                'timestamp': file_timestamp
            }
        
        # Load JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        rankings = data.get('rankings', [])
        
        if not rankings:
            return {
                'status': 'skipped',
                'reason': 'No rankings found',
                'timestamp': file_timestamp
            }
        
        print(f"\n📦 Importing: {filepath.name}")
        print(f"   Timestamp: {file_timestamp}")
        print(f"   Tickers: {len(rankings)}")
        
        if dry_run:
            return {
                'status': 'dry_run',
                'tickers': len(rankings),
                'timestamp': file_timestamp
            }
        
        # Transform rankings to phase4 format
        phase4_results = []
        for rank, item in enumerate(rankings, 1):
            # Extract group scores
            group_scores = item.get('group_scores', {})
            group_coverages = item.get('group_coverages', {})
            
            signal_data = {
                'ticker': item['ticker'],
                'rank': rank,
                'overall_score': item.get('overall_score', 0.0),
                'technical_score': group_scores.get('technical', 0.0),
                'fundamental_score': group_scores.get('fundamental', 0.0),
                'news_macro_score': group_scores.get('news_macro', 0.0),
                'social_alternative_score': group_scores.get('social_alternative', 0.0),
                'risk_stability_score': group_scores.get('risk_stability', 0.0),
                'institutional_smart_money_score': group_scores.get('institutional_smart_money', 0.0),
                'total_coverage': item.get('total_coverage', 0.0),
                'technical_coverage': group_coverages.get('technical', 0.0),
                'fundamental_coverage': group_coverages.get('fundamental', 0.0),
                'news_macro_coverage': group_coverages.get('news_macro', 0.0),
                'social_alternative_coverage': group_coverages.get('social_alternative', 0.0),
                'risk_stability_coverage': group_coverages.get('risk_stability', 0.0),
                'institutional_smart_money_coverage': group_coverages.get('institutional_smart_money', 0.0),
                # Factor data (for detail tables - if available)
                'technical_data': {},
                'fundamental_data': {},
                'news_macro_data': {},
                'social_data': {},
                'risk_data': {},
                'institutional_data': {}
            }
            
            phase4_results.append(signal_data)
        
        # Persist to database using Phase5Persist
        try:
            run_id = await self.phase5.persist_pipeline_run(
                phase4_results=phase4_results
            )
            
            # Update the run timestamp to match the JSON file timestamp
            await self.db.pool.execute(
                "UPDATE signal_runs SET run_timestamp = $1 WHERE id = $2",
                file_timestamp,
                run_id
            )
            
            print(f"   ✅ Imported successfully! Run ID: {run_id}")
            
            return {
                'status': 'success',
                'run_id': run_id,
                'tickers': len(rankings),
                'timestamp': file_timestamp
            }
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': file_timestamp
            }
    
    async def backfill_all(self, limit: int = None, dry_run: bool = False, specific_file: str = None):
        """Backfill all JSON files (or specific file) into Supabase"""
        
        await self.initialize()
        
        print("=" * 80)
        print("  JSON BACKFILL TO SUPABASE")
        print("=" * 80)
        
        if specific_file:
            json_files = [self.results_dir / specific_file]
        else:
            json_files = self.find_json_files(limit=limit)
        
        if not json_files:
            print("\n❌ No JSON files found to import")
            await self.close()
            return
        
        print(f"\nFound {len(json_files)} JSON files to process")
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made\n")
        
        results = {
            'success': 0,
            'skipped': 0,
            'failed': 0,
            'dry_run': 0
        }
        
        for filepath in json_files:
            result = await self.import_json_file(filepath, dry_run=dry_run)
            results[result['status']] += 1
        
        print("\n" + "=" * 80)
        print("  BACKFILL SUMMARY")
        print("=" * 80)
        print(f"  ✅ Successfully imported: {results['success']}")
        print(f"  ⏭️  Skipped (already exist): {results['skipped']}")
        print(f"  ❌ Failed: {results['failed']}")
        if dry_run:
            print(f"  🔍 Dry run (would import): {results['dry_run']}")
        print("=" * 80)
        
        await self.close()


async def main():
    """Main entry point with CLI arguments"""
    parser = argparse.ArgumentParser(description='Backfill JSON pipeline results to Supabase')
    parser.add_argument('--limit', type=int, help='Limit number of files to import (most recent first)')
    parser.add_argument('--dry-run', action='store_true', help='Preview what would be imported without making changes')
    parser.add_argument('--file', type=str, help='Import a specific JSON file')
    
    args = parser.parse_args()
    
    backfiller = JSONBackfiller()
    await backfiller.backfill_all(
        limit=args.limit,
        dry_run=args.dry_run,
        specific_file=args.file
    )


if __name__ == "__main__":
    asyncio.run(main())
