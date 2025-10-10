"""
Clear All Signals - Python Version
===================================
Clears all signal data for fresh run with latest calculations.

Usage:
    python clear_signals.py [--confirm]
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.storage.database import DatabaseManager
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


def show_current_stats(db: DatabaseManager):
    """Show current database statistics."""
    try:
        # Total signals
        result = db.supabase.table('signals') \
            .select('*', count='exact') \
            .execute()
        
        total_count = result.count if hasattr(result, 'count') else 0
        
        if total_count == 0:
            logger.info("📊 Database is already empty")
            return None
        
        # Get detailed stats
        signals = result.data if result.data else []
        
        run_ids = set()
        tickers = set()
        scores = []
        created_times = []
        
        for sig in signals:
            if sig.get('run_id'):
                run_ids.add(sig['run_id'])
            if sig.get('ticker'):
                tickers.add(sig['ticker'])
            if sig.get('signal_score'):
                scores.append(sig['signal_score'])
            if sig.get('created_at'):
                created_times.append(sig['created_at'])
        
        stats = {
            'total': total_count,
            'runs': len(run_ids),
            'tickers': len(tickers),
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'oldest': min(created_times) if created_times else None,
            'newest': max(created_times) if created_times else None
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        return None


def clear_all_signals(db: DatabaseManager, confirm: bool = False):
    """
    Clear all signals from database.
    
    Args:
        db: DatabaseManager instance
        confirm: If True, skip confirmation prompt
    """
    try:
        # Show current stats
        logger.info("=" * 80)
        logger.info("📊 CURRENT DATABASE STATUS")
        logger.info("=" * 80)
        
        stats = show_current_stats(db)
        
        if stats is None:
            logger.info("✅ Nothing to clear")
            return 0
        
        logger.info(f"Total signals: {stats['total']}")
        logger.info(f"Total runs: {stats['runs']}")
        logger.info(f"Unique tickers: {stats['tickers']}")
        logger.info(f"Avg signal score: {stats['avg_score']:.3f}")
        logger.info(f"Max signal score: {stats['max_score']:.3f}")
        logger.info(f"Oldest signal: {stats['oldest']}")
        logger.info(f"Newest signal: {stats['newest']}")
        
        # Confirmation
        if not confirm:
            logger.warning("")
            logger.warning("⚠️  WARNING: This will DELETE ALL signal data!")
            logger.warning("⚠️  This action cannot be undone!")
            logger.warning("")
            response = input("Type 'DELETE ALL' to confirm: ")
            
            if response != 'DELETE ALL':
                logger.info("❌ Deletion cancelled")
                return 0
        
        # Delete all signals
        logger.info("")
        logger.info("🗑️  DELETING ALL SIGNALS...")
        logger.info("=" * 80)
        
        delete_result = db.supabase.table('signals') \
            .delete() \
            .neq('id', '00000000-0000-0000-0000-000000000000') \
            .execute()
        
        logger.info(f"✅ Successfully deleted {stats['total']} signals")
        
        # Verify empty
        verify_result = db.supabase.table('signals') \
            .select('id', count='exact') \
            .execute()
        
        remaining = verify_result.count if hasattr(verify_result, 'count') else 0
        
        if remaining == 0:
            logger.info("✅ Database verified empty")
        else:
            logger.warning(f"⚠️  {remaining} signals still remain")
        
        return stats['total']
        
    except Exception as e:
        logger.error(f"❌ Error clearing signals: {e}")
        raise


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clear all signal data')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🗑️  CLEAR ALL SIGNALS")
    logger.info("=" * 80)
    
    # Initialize database
    db = DatabaseManager()
    
    try:
        deleted_count = clear_all_signals(db, args.confirm)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Deleted: {deleted_count} signals")
        logger.info("")
        logger.info("Next Steps:")
        logger.info("1. Run fresh pipeline: python -m backend.pipeline")
        logger.info("2. All data will use latest calculations")
        logger.info("3. All scores will use current Phase 7 methods")
        
    except Exception as e:
        logger.error(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
