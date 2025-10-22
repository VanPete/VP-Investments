"""
Phase 5 Schema Migration (Using psycopg2)

Executes the Phase 5 core schema migration using direct PostgreSQL connection.
"""

import psycopg2
from psycopg2 import sql
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


def run_migration():
    """Execute Phase 5 schema migration using psycopg2."""
    
    # Get database URL from environment
    database_url = os.getenv('SUPABASE_DATABASE_URL')
    
    if not database_url:
        logger.error("❌ SUPABASE_DATABASE_URL not found in environment variables")
        logger.info("Please set SUPABASE_DATABASE_URL in your .env file")
        return False
    
    logger.info(f"📋 Database URL: {database_url[:50]}...")
    
    try:
        # Connect to database
        logger.info("Connecting to Supabase PostgreSQL database...")
        # Disable prepared statements for Supabase Transaction pooler compatibility
        conn = psycopg2.connect(database_url, options="-c statement_timeout=60000")
        conn.autocommit = False  # Use transactions
        cursor = conn.cursor()
        
        logger.info("✅ Connected to database")
        
        # Read migration file
        migration_file = Path(__file__).parent / "001_phase5_core_schema.sql"
        logger.info(f"📂 Reading migration file: {migration_file}")
        
        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        logger.info(f"✅ Migration file loaded ({len(sql_content)} characters)")
        
        # Execute migration
        logger.info("\n" + "="*80)
        logger.info("🚀 EXECUTING PHASE 5 MIGRATION")
        logger.info("="*80)
        logger.info("Creating 8 tables:")
        logger.info("  ✅ signals (core)")
        logger.info("  ✅ signal_runs (core)")
        logger.info("  ✅ signals_technical (~60 factors)")
        logger.info("  ✅ signals_fundamental (~45 factors)")
        logger.info("  ✅ signals_news_macro (~15 factors)")
        logger.info("  ✅ signals_social_alternative (~10 factors)")
        logger.info("  ✅ signals_risk_stability (~25 factors)")
        logger.info("  ✅ signals_institutional_smart_money (~20 factors)")
        logger.info("="*80 + "\n")
        
        try:
            # Execute the entire SQL file
            cursor.execute(sql_content)
            conn.commit()
            logger.info("✅ Migration executed successfully")
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"❌ Migration failed: {e}")
            logger.error(f"Error code: {e.pgcode}")
            logger.error(f"Error message: {e.pgerror}")
            return False
        
        # Verify tables were created
        logger.info("\n📊 Verifying tables...")
        tables_to_verify = [
            'signals',
            'signal_runs',
            'signals_technical',
            'signals_fundamental',
            'signals_news_macro',
            'signals_social_alternative',
            'signals_risk_stability',
            'signals_institutional_smart_money'
        ]
        
        for table in tables_to_verify:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            exists = cursor.fetchone()[0]
            status = "✅" if exists else "❌"
            logger.info(f"  {status} Table '{table}' exists: {exists}")
        
        # Get table row counts
        logger.info("\n📈 Table Statistics:")
        for table in tables_to_verify:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"  📊 {table}: {count} rows")
            except:
                logger.info(f"  ⚠️  {table}: Unable to query")
        
        cursor.close()
        conn.close()
        
        logger.info("\n" + "="*80)
        logger.info("🎉 PHASE 5 MIGRATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("\n✅ Next steps:")
        logger.info("1. Extend SupabaseInterface with Phase 5 methods")
        logger.info("2. Create Phase5Persist class")
        logger.info("3. Integrate with pipeline.py")
        logger.info("4. Test with sample ticker data")
        
        return True
        
    except psycopg2.OperationalError as e:
        logger.error(f"\n❌ Database connection failed: {e}")
        logger.error("\nPossible issues:")
        logger.error("1. Incorrect database URL")
        logger.error("2. Firewall blocking connection")
        logger.error("3. Database credentials expired")
        logger.error("\nPlease check your SUPABASE_DATABASE_URL in .env file")
        return False
        
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        logger.exception("Full error details:")
        return False


if __name__ == "__main__":
    import sys
    try:
        success = run_migration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Migration interrupted by user")
        sys.exit(1)
