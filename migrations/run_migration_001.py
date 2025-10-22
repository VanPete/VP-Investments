"""
Phase 5 Schema Migration Script

Executes the Phase 5 core schema migration on Supabase database.
Creates 8 tables: 2 core + 6 group detail tables.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface
from backend.core.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_migration():
    """Execute Phase 5 schema migration."""
    db = SupabaseInterface()
    
    try:
        # Connect to database
        logger.info("Connecting to Supabase database...")
        await db.connect()
        logger.info("✅ Connected to Supabase")
        
        # Read migration file
        migration_file = Path(__file__).parent.parent / "migrations" / "001_phase5_core_schema.sql"
        logger.info(f"Reading migration file: {migration_file}")
        
        with open(migration_file, 'r', encoding='utf-8') as f:
            sql = f.read()
        
        logger.info("Migration SQL loaded successfully")
        logger.info(f"SQL length: {len(sql)} characters")
        
        # Execute migration
        logger.info("\n" + "="*80)
        logger.info("EXECUTING PHASE 5 MIGRATION")
        logger.info("="*80)
        logger.info("Creating 8 tables:")
        logger.info("  - signals (core)")
        logger.info("  - signal_runs (core)")
        logger.info("  - signals_technical (~60 factors)")
        logger.info("  - signals_fundamental (~45 factors)")
        logger.info("  - signals_news_macro (~15 factors)")
        logger.info("  - signals_social_alternative (~10 factors)")
        logger.info("  - signals_risk_stability (~25 factors)")
        logger.info("  - signals_institutional_smart_money (~20 factors)")
        logger.info("="*80 + "\n")
        
        # Split into individual statements and execute
        # This is safer than executing the entire file at once
        statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
        
        logger.info(f"Executing {len(statements)} SQL statements...")
        
        for i, statement in enumerate(statements, 1):
            # Skip comment-only lines
            if not statement or statement.startswith('--'):
                continue
                
            # Get first line for logging
            first_line = statement.split('\n')[0][:100]
            logger.info(f"[{i}/{len(statements)}] Executing: {first_line}...")
            
            try:
                await db.execute_non_query(statement)
                logger.info(f"  ✅ Success")
            except Exception as e:
                # Log but continue if table already exists
                if "already exists" in str(e).lower():
                    logger.info(f"  ⚠️  Already exists (skipping)")
                else:
                    logger.error(f"  ❌ Error: {e}")
                    raise
        
        logger.info("\n" + "="*80)
        logger.info("✅ PHASE 5 MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        # Verify tables were created
        logger.info("\nVerifying tables...")
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
            result = await db.execute_query(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table}'
                );
            """)
            exists = result[0]['exists'] if result else False
            status = "✅" if exists else "❌"
            logger.info(f"  {status} Table '{table}' exists: {exists}")
        
        logger.info("\n" + "="*80)
        logger.info("🎉 Phase 5 schema is ready for implementation!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("1. Extend SupabaseInterface with Phase 5 methods")
        logger.info("2. Create Phase5Persist class")
        logger.info("3. Integrate with pipeline.py")
        
    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        logger.exception("Full error details:")
        raise
    finally:
        # Disconnect
        await db.disconnect()
        logger.info("\n✅ Database connection closed")


if __name__ == "__main__":
    try:
        asyncio.run(run_migration())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        sys.exit(1)
