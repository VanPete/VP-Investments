"""
Clean up existing Phase 5 tables before migration

This script drops all Phase 5 tables if they exist.
"""

import psycopg2
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup_tables():
    """Drop all Phase 5 tables if they exist."""
    
    database_url = os.getenv('SUPABASE_DATABASE_URL')
    
    if not database_url:
        logger.error("❌ SUPABASE_DATABASE_URL not found in environment variables")
        return False
    
    try:
        # Connect to database
        logger.info("Connecting to database...")
        conn = psycopg2.connect(database_url, options="-c statement_timeout=60000")
        conn.autocommit = False
        cursor = conn.cursor()
        
        logger.info("✅ Connected to database")
        
        # Tables to drop (in reverse dependency order)
        tables_to_drop = [
            'signals_institutional_smart_money',
            'signals_risk_stability',
            'signals_social_alternative',
            'signals_news_macro',
            'signals_fundamental',
            'signals_technical',
            'signals',
            'signal_runs'
        ]
        
        logger.info("\n" + "="*80)
        logger.info("🗑️  CLEANING UP EXISTING TABLES")
        logger.info("="*80)
        
        for table in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                logger.info(f"  ✅ Dropped table '{table}'")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not drop '{table}': {e}")
        
        conn.commit()
        
        logger.info("="*80)
        logger.info("✅ CLEANUP COMPLETED")
        logger.info("="*80)
        logger.info("\nYou can now run: python migrations/run_migration_psycopg2.py")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        logger.exception("Full error details:")
        return False


if __name__ == "__main__":
    import sys
    try:
        success = cleanup_tables()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Cleanup interrupted by user")
        sys.exit(1)
