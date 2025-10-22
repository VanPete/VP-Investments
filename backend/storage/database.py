"""
Supabase Database Interface - Refactored for Phase 5

Clean, minimal database interface focused on Phase 5 schema.
All Phase 5-specific methods are dynamically added from phase5_persist.py.

Legacy code removed:
- Old schema tables (signals_norm, features, labels, etc.)
- Backtest tables (don't exist in Phase 5 schema)
- Cache operations
- Real-time subscriptions
- Deprecated upsert methods
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import asyncpg
from supabase import create_client, Client

from backend.core.config import get_config

logger = logging.getLogger(__name__)


class DatabaseInterface:
    """Base database interface for VP Investments."""
    
    async def connect(self) -> None:
        """Establish database connection."""
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        """Close database connection.""" 
        raise NotImplementedError
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        raise NotImplementedError


class SupabaseInterface(DatabaseInterface):
    """
    Supabase PostgreSQL database interface for Phase 5 schema.
    
    Core Features:
    - Async connection pooling with asyncpg
    - Direct PostgreSQL access for complex queries
    - Automatic reconnection and retry logic
    - Query performance monitoring
    
    Phase 5 Methods:
    - Dynamically added from backend.phases.phase5_persist
    - See phase5_persist.py for 16 database methods
    """
    
    def __init__(self):
        """Initialize database interface with configuration."""
        config = get_config()
        self.supabase_url = config.get('supabase.url')
        self.supabase_key = config.get('supabase.anon_key')
        self.database_url = config.get('supabase.database_url')
        
        # Validate required configuration
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not self.supabase_key:
            raise ValueError("SUPABASE_ANON_KEY environment variable is required")
        
        self.supabase: Optional[Client] = None
        self.pool: Optional[asyncpg.Pool] = None
        self.connected = False
        
        # Connection settings
        self.max_connections = int(config.get('database.max_connections', 10))
        self.command_timeout = int(config.get('database.command_timeout', 60))
        self.retry_attempts = int(config.get('database.retry_attempts', 3))
        
        logger.info("[SUCCESS] SupabaseInterface initialized with required configuration")
    
    async def connect(self) -> None:
        """Initialize Supabase client and PostgreSQL connection pool."""
        try:
            # Initialize Supabase client
            if not self.supabase_url or not self.supabase_key:
                raise ValueError("Supabase URL and anon key are required")
            
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized")
            
            # Initialize PostgreSQL connection pool for Phase 5 operations
            if self.database_url:
                try:
                    self.pool = await asyncpg.create_pool(
                        self.database_url,
                        min_size=1,
                        max_size=self.max_connections,
                        command_timeout=self.command_timeout,
                        statement_cache_size=0,  # Disable prepared statements for pgbouncer/transaction pooler
                        server_settings={
                            'application_name': 'VP-Investments-Phase5',
                            'timezone': 'UTC'
                        }
                    )
                    logger.info(f"PostgreSQL connection pool created (max: {self.max_connections})")
                except Exception as pool_error:
                    logger.error(f"PostgreSQL pool connection failed: {pool_error}")
                    raise
            
            self.connected = True
            logger.info("Supabase database connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise
    
    @property
    def client(self) -> Client:
        """Get the Supabase client, initializing if needed."""
        if not self.supabase:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        return self.supabase
    
    async def disconnect(self) -> None:
        """Close Supabase client and PostgreSQL connection pool."""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("PostgreSQL connection pool closed")
            
            self.supabase = None
            self.connected = False
            logger.info("Supabase connection closed")
            
        except Exception as e:
            logger.error(f"Error during Supabase disconnect: {e}")
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict]:
        """
        Execute SELECT query and return results.
        
        Args:
            query: SQL SELECT statement with $1, $2, etc. placeholders
            params: List of parameter values
            
        Returns:
            List of dictionaries representing rows
        """
        if not self.connected:
            await self.connect()
        
        for attempt in range(self.retry_attempts):
            try:
                start_time = datetime.now()
                
                if self.pool:
                    async with self.pool.acquire() as conn:
                        if params:
                            rows = await conn.fetch(query, *params)
                        else:
                            rows = await conn.fetch(query)
                        
                        result = [dict(row) for row in rows]
                else:
                    raise ValueError("PostgreSQL pool not initialized")
                
                # Log performance
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Query executed in {execution_time:.3f}s, returned {len(result)} rows")
                
                return result
                
            except Exception as e:
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    logger.error(f"Query failed after {self.retry_attempts} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    async def execute_non_query(self, query: str, params: Optional[List] = None) -> int:
        """
        Execute INSERT, UPDATE, DELETE query.
        
        Args:
            query: SQL statement with $1, $2, etc. placeholders
            params: List of parameter values
            
        Returns:
            Number of affected rows
        """
        if not self.connected:
            await self.connect()
        
        for attempt in range(self.retry_attempts):
            try:
                start_time = datetime.now()
                
                if self.pool:
                    async with self.pool.acquire() as conn:
                        if params:
                            result = await conn.execute(query, *params)
                        else:
                            result = await conn.execute(query)
                        
                        # Extract affected row count from result
                        # Result format: "INSERT 0 5" or "UPDATE 3" or "DELETE 2"
                        if result:
                            try:
                                parts = result.split()
                                if len(parts) > 1 and parts[-1].isdigit():
                                    affected_rows = int(parts[-1])
                                else:
                                    affected_rows = 0
                            except (ValueError, IndexError):
                                affected_rows = 0
                        else:
                            affected_rows = 0
                else:
                    raise ValueError("PostgreSQL pool not initialized")
                
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Non-query executed in {execution_time:.3f}s, affected {affected_rows} rows")
                
                return affected_rows
                
            except Exception as e:
                logger.warning(f"Non-query attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    logger.error(f"Non-query failed after {self.retry_attempts} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)
        
        return 0
    
    async def health_check(self) -> bool:
        """Check database connection health."""
        try:
            if not self.connected:
                return False
            
            # Test with simple query
            result = await self.execute_query("SELECT 1 as health_check")
            if len(result) == 1:
                health_value = result[0].get('health_check')
                return health_value == 1 or health_value == '1'
            return False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Factory functions
async def get_supabase_database() -> SupabaseInterface:
    """Create and return configured Supabase database interface (async)."""
    db = SupabaseInterface()
    await db.connect()
    return db


def get_database() -> SupabaseInterface:
    """Get database interface (synchronous factory)."""
    return SupabaseInterface()
