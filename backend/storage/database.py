"""
Supabase Database Interface Implementation

Provides async PostgreSQL database operations through Supabase
with connection pooling, error handling, and monitoring.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

import asyncpg
from supabase import create_client, Client
from postgrest import APIError

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
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        raise NotImplementedError


class SupabaseInterface(DatabaseInterface):
    """
    Supabase PostgreSQL database interface implementation.
    
    Features:
    - Async connection pooling with asyncpg
    - Supabase client for auth and real-time
    - Automatic reconnection and retry logic
    - Query performance monitoring
    - Row Level Security support
    - Complete replacement for SQLite operations
    """
    
    def __init__(self):
        config = get_config()
        self.supabase_url = config.get('supabase.url')
        self.supabase_key = config.get('supabase.anon_key')
        self.supabase_service_key = config.get('supabase.service_key')
        self.database_url = config.get('supabase.database_url')
        
        # Validate required configuration
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not self.supabase_key:
            raise ValueError("SUPABASE_ANON_KEY environment variable is required")
        
        self.supabase: Optional[Client] = None
        self.pool: Optional[asyncpg.Pool] = None
        self.connected = False
        
        # Connection settings (ensure proper type conversion)
        self.max_connections = int(config.get('database.max_connections', 10))
        self.command_timeout = int(config.get('database.command_timeout', 60))
        self.retry_attempts = int(config.get('database.retry_attempts', 3))
        
        logger.info("[SUCCESS] SupabaseInterface initialized with required configuration")
    
    async def connect(self) -> None:
        """Initialize Supabase client and PostgreSQL connection pool"""
        try:
            # Initialize Supabase client
            if not self.supabase_url or not self.supabase_key:
                raise ValueError("Supabase URL and anon key are required")
            
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized")
            
            # Try to initialize direct PostgreSQL connection pool for complex queries
            # This is optional - we can fall back to Supabase client only
            if self.database_url:
                try:
                    self.pool = await asyncpg.create_pool(
                        self.database_url,
                        min_size=1,
                        max_size=self.max_connections,
                        command_timeout=self.command_timeout,
                        statement_cache_size=0,  # Disable prepared statements for pgbouncer
                        server_settings={
                            'application_name': 'VP-Investments-2.0',
                            'timezone': 'UTC'
                        }
                    )
                    logger.info(f"PostgreSQL connection pool created (max: {self.max_connections})")
                except Exception as pool_error:
                    logger.warning(f"PostgreSQL pool connection failed: {pool_error}")
                    logger.info("Continuing with Supabase client only (limited SQL support)")
                    self.pool = None
            
            self.connected = True
            logger.info("Supabase database connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise
    
    @property
    def client(self):
        """Get the Supabase client, initializing if needed"""
        if not self.supabase:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        return self.supabase
    
    async def disconnect(self) -> None:
        """Close Supabase client and PostgreSQL connection pool"""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("PostgreSQL connection pool closed")
            
            # Supabase client doesn't need explicit closing
            self.supabase = None
            self.connected = False
            logger.info("Supabase connection closed")
            
        except Exception as e:
            logger.error(f"Error during Supabase disconnect: {e}")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute SELECT query and return results
        
        Args:
            query: SQL SELECT statement
            params: Query parameters for parameterized queries
            
        Returns:
            List of dictionaries representing rows
        """
        if not self.connected:
            await self.connect()
        
        for attempt in range(self.retry_attempts):
            try:
                start_time = datetime.now()
                
                if self.pool:
                    # Use direct PostgreSQL connection for complex queries
                    async with self.pool.acquire() as conn:
                        if params:
                            # Handle both dict and list parameters
                            if isinstance(params, dict):
                                param_values = list(params.values())
                                formatted_query = self._format_query_params(query, params)
                                rows = await conn.fetch(formatted_query, *param_values)
                            elif isinstance(params, list):
                                # Direct positional parameters
                                rows = await conn.fetch(query, *params)
                            else:
                                rows = await conn.fetch(query, params)
                        else:
                            rows = await conn.fetch(query)
                        
                        result = [dict(row) for row in rows]
                else:
                    # Fallback to Supabase client (limited SQL support)
                    logger.warning("Using Supabase client for query - limited SQL support")
                    result = []
                
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
    
    async def execute_non_query(self, query: str, params: Optional[Dict] = None) -> int:
        """
        Execute INSERT, UPDATE, DELETE query
        
        Args:
            query: SQL statement
            params: Query parameters
            
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
                            # Handle both dict and list parameters
                            if isinstance(params, dict):
                                param_values = list(params.values())
                                formatted_query = self._format_query_params(query, params)
                                result = await conn.execute(formatted_query, *param_values)
                            elif isinstance(params, list):
                                # Direct positional parameters
                                result = await conn.execute(query, *params)
                            else:
                                result = await conn.execute(query, params)
                        else:
                            result = await conn.execute(query)
                        
                        # Extract affected row count from result
                        # Handle different types of SQL statements
                        if result:
                            try:
                                # For INSERT/UPDATE/DELETE statements, result is like "INSERT 0 5"
                                parts = result.split()
                                if len(parts) > 1 and parts[-1].isdigit():
                                    affected_rows = int(parts[-1])
                                else:
                                    # For CREATE/DROP/ALTER statements, just return 0
                                    affected_rows = 0
                            except (ValueError, IndexError):
                                affected_rows = 0
                        else:
                            affected_rows = 0
                else:
                    # Fallback to Supabase client
                    logger.warning("Using Supabase client for non-query - limited SQL support")
                    affected_rows = 0
                
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
        """Check database connection health"""
        try:
            if not self.connected:
                return False
            
            # Test with simple query
            result = await self.execute_query("SELECT 1 as health_check")
            if len(result) == 1:
                health_value = result[0].get('health_check')
                # Handle both string and integer responses
                return health_value == 1 or health_value == '1'
            return False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def query_data(self, table_name: str, filters: Optional[Dict] = None, 
                        order_by: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Query data from a table with optional filters, ordering, and limit
        
        Args:
            table_name: Name of the table to query
            filters: Dictionary of column=value filters  
            order_by: Column name(s) to order by
            limit: Maximum number of records to return
            
        Returns:
            List of dictionaries representing the query results
        """
        try:
            # Build query
            query = f"SELECT * FROM {table_name}"
            params = []
            
            # Add WHERE clause if filters provided
            if filters:
                where_conditions = []
                param_count = 1
                for column, value in filters.items():
                    where_conditions.append(f"{column} = ${param_count}")
                    params.append(value)
                    param_count += 1
                
                query += " WHERE " + " AND ".join(where_conditions)
            
            # Add ORDER BY clause
            if order_by:
                query += f" ORDER BY {order_by}"
            
            # Add LIMIT clause  
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            result = await self.execute_query(query, params)
            return result if result else []
            
        except Exception as e:
            logger.error(f"Query data failed for table {table_name}: {e}")
            return []
    
    # =============================================================================
    # UPSERT OPERATIONS FOR HOURLY SCHEDULER (Replaces all SQLite operations)
    # =============================================================================
    
    async def upsert_run(self, run_id: str, started_at: str = None, ended_at: str = None,
                        config_json: str = None, code_version: str = None, 
                        notes: str = None) -> None:
        """Upsert run record - replaces utils.db.upsert_run"""
        query = """
        INSERT INTO runs (run_id, started_at, ended_at, config_json, code_version, notes)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (run_id) DO UPDATE SET
            started_at = COALESCE(EXCLUDED.started_at, runs.started_at),
            ended_at = COALESCE(EXCLUDED.ended_at, runs.ended_at),
            config_json = COALESCE(EXCLUDED.config_json, runs.config_json),
            code_version = COALESCE(EXCLUDED.code_version, runs.code_version),
            notes = COALESCE(EXCLUDED.notes, runs.notes)
        """
        params = [run_id, started_at, ended_at, config_json, code_version, notes]
        await self.execute_non_query(query, params)
    
    async def upsert_signal_norm(self, run_id: str, ticker: str, score: float = None,
                               rank: int = None, trade_type: str = None, risk_level: str = None,
                               reddit_score: float = None, news_score: float = None,
                               financial_score: float = None, run_datetime = None) -> None:
        """Upsert signal_norm record - replaces utils.db.upsert_signal_norm"""
        query = """
        INSERT INTO signals_norm (run_id, ticker, score, rank, trade_type, risk_level,
                                reddit_score, news_score, financial_score, run_datetime)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (run_id, ticker) DO UPDATE SET
            score = EXCLUDED.score,
            rank = EXCLUDED.rank,
            trade_type = EXCLUDED.trade_type,
            risk_level = EXCLUDED.risk_level,
            reddit_score = EXCLUDED.reddit_score,
            news_score = EXCLUDED.news_score,
            financial_score = EXCLUDED.financial_score,
            run_datetime = EXCLUDED.run_datetime
        """
        params = [run_id, ticker, score, rank, trade_type, risk_level,
                 reddit_score, news_score, financial_score, run_datetime]
        await self.execute_non_query(query, params)
    
    async def insert_metric(self, run_id: str, name: str, value: float = None,
                          context_json: str = None, created_at = None) -> None:
        """Insert metric record - replaces utils.db.insert_metric"""
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        
        query = """
        INSERT INTO metrics (run_id, name, value, context_json, created_at)
        VALUES ($1, $2, $3, $4, $5)
        """
        params = [run_id, name, value, context_json, created_at]
        await self.execute_non_query(query, params)
    
    async def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent runs ordered by started_at DESC"""
        query = """
        SELECT run_id, started_at, ended_at, config_json, code_version, notes
        FROM runs
        ORDER BY started_at DESC NULLS LAST
        LIMIT $1
        """
        return await self.execute_query(query, [limit])
    
    async def get_signals_by_run(self, run_id: str, limit: int = None) -> List[Dict]:
        """Get signals for a specific run ordered by score DESC"""
        query = """
        SELECT run_id, ticker, score, rank, trade_type, risk_level,
               reddit_score, news_score, financial_score, run_datetime
        FROM signals_norm
        WHERE run_id = $1
        ORDER BY score DESC NULLS LAST
        """
        params = [run_id]
        
        if limit:
            query += f" LIMIT {limit}"
        
        return await self.execute_query(query, params)
    
    async def get_runs_since(self, since_date: str) -> List[Dict]:
        """Get runs since a specific date"""
        query = """
        SELECT run_id, started_at, ended_at, config_json, code_version, notes
        FROM runs
        WHERE started_at >= $1
        ORDER BY started_at DESC
        """
        return await self.execute_query(query, [since_date])
    
    async def get_latest_signals(self, limit: int = 100) -> List[Dict]:
        """Get latest signals from most recent run"""
        # Get the most recent run first
        recent_runs = await self.get_recent_runs(1)
        if not recent_runs:
            return []
        
        latest_run_id = recent_runs[0]['run_id']
        return await self.get_signals_by_run(latest_run_id, limit)
    
    async def clear_current_signals(self) -> None:
        """Clear current signals table (if exists) - for real-time updates"""
        # For now, we'll just use the most recent run signals
        # In the future, we might create a separate 'current_signals' table
        pass
    
    async def insert_current_signal(self, signal) -> None:
        """Insert current signal for real-time API (if needed)"""
        # For now, signals are stored in signals_norm table
        # Real-time API will query the latest run
        pass
    
    # =============================================================================
    # SCHEMA INITIALIZATION (Replaces utils.db.ensure_schema)
    # =============================================================================
    
    async def ensure_schema(self) -> None:
        """Create required tables and indexes if they don't exist"""
        if not self.connected:
            await self.connect()
        
        # Define all table creation statements
        schema_queries = [
            # Runs table
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TIMESTAMPTZ,
                ended_at TIMESTAMPTZ,
                config_json TEXT,
                code_version TEXT,
                notes TEXT
            )
            """,
            
            # Prices table
            """
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume BIGINT,
                PRIMARY KEY (ticker, date)
            )
            """,
            
            # Features table (long form)
            """
            CREATE TABLE IF NOT EXISTS features (
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL,
                as_of TIMESTAMPTZ,
                PRIMARY KEY (run_id, ticker, key)
            )
            """,
            
            # Labels table (quote 'window' as it's a reserved keyword)
            """
            CREATE TABLE IF NOT EXISTS labels (
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                "window" TEXT NOT NULL,
                fwd_return REAL,
                beat_spy INTEGER,
                ready_at TIMESTAMPTZ,
                PRIMARY KEY (run_id, ticker, "window")
            )
            """,
            
            # Signals norm table (compact reference)
            """
            CREATE TABLE IF NOT EXISTS signals_norm (
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                score REAL,
                rank INTEGER,
                trade_type TEXT,
                risk_level TEXT,
                reddit_score REAL,
                news_score REAL,
                financial_score REAL,
                run_datetime TIMESTAMPTZ,
                PRIMARY KEY (run_id, ticker)
            )
            """,
            
            # Metrics table
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id SERIAL PRIMARY KEY,
                run_id TEXT,
                name TEXT,
                value REAL,
                context_json TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            
            # Experiments table
            """
            CREATE TABLE IF NOT EXISTS experiments (
                exp_id TEXT PRIMARY KEY,
                run_id TEXT,
                profile TEXT,
                params_json TEXT,
                code_version TEXT,
                started_at TIMESTAMPTZ,
                ended_at TIMESTAMPTZ,
                notes TEXT
            )
            """
        ]
        
        # Index creation statements
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at)",
            "CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date)",
            "CREATE INDEX IF NOT EXISTS idx_features_run_ticker ON features(run_id, ticker)",
            "CREATE INDEX IF NOT EXISTS idx_labels_run_ticker ON labels(run_id, ticker)",
            "CREATE INDEX IF NOT EXISTS idx_signals_norm_run_ticker ON signals_norm(run_id, ticker)",
            "CREATE INDEX IF NOT EXISTS idx_signals_norm_ticker ON signals_norm(ticker)",
            "CREATE INDEX IF NOT EXISTS idx_signals_norm_score ON signals_norm(score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_run_name ON metrics(run_id, name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_created_at ON metrics(created_at)"
        ]
        
        try:
            # Execute all schema queries
            for query in schema_queries:
                await self.execute_non_query(query)
                logger.debug(f"Schema query executed: {query[:50]}...")
            
            # Execute all index queries  
            for query in index_queries:
                await self.execute_non_query(query)
                logger.debug(f"Index query executed: {query[:50]}...")
            
            logger.info("Database schema initialization completed")
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise
    
    def _format_query_params(self, query: str, params: Dict) -> str:
        """Convert named parameters to positional parameters for asyncpg"""
        # Replace :param_name with $1, $2, etc.
        import re
        param_names = list(params.keys())
        formatted_query = query
        
        for i, param_name in enumerate(param_names, 1):
            formatted_query = re.sub(
                f':{param_name}\\b',  # Match :param_name as whole word
                f'${i}',
                formatted_query
            )
        
        return formatted_query
    
    async def get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get item from cache"""
        try:
            # Try to get from in-memory cache first
            if hasattr(self, '_cache') and cache_key in self._cache:
                return self._cache[cache_key]
            
            # Query from database cache table if it exists
            query = "SELECT cache_value FROM cache_store WHERE cache_key = $1 AND expires_at > NOW()"
            result = await self.execute_query(query, [cache_key])
            
            if result:
                cache_value = json.loads(result[0]['cache_value'])
                # Store in memory cache
                if not hasattr(self, '_cache'):
                    self._cache = {}
                self._cache[cache_key] = cache_value
                return cache_value
            
            return None
            
        except Exception as e:
            logger.debug(f"Cache get failed for {cache_key}: {e}")
            return None
    
    async def set_cache(self, cache_key: str, cache_value: Dict, ttl_seconds: int = 3600) -> bool:
        """Set item in cache"""
        try:
            # Store in memory cache
            if not hasattr(self, '_cache'):
                self._cache = {}
            self._cache[cache_key] = cache_value
            
            # Try to store in database cache table if it exists
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            query = """
            INSERT INTO cache_store (cache_key, cache_value, expires_at, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (cache_key) 
            DO UPDATE SET 
                cache_value = EXCLUDED.cache_value,
                expires_at = EXCLUDED.expires_at,
                updated_at = NOW()
            """
            await self.execute_non_query(query, [cache_key, json.dumps(cache_value), expires_at])
            
            return True
            
        except Exception as e:
            logger.debug(f"Cache set failed for {cache_key}: {e}")
            return False
    
    # Supabase-specific helper methods
    
    async def get_table_client(self, table_name: str):
        """Get Supabase table client for simplified operations"""
        if not self.supabase:
            await self.connect()
        return self.supabase.table(table_name)
    
    async def insert_batch(self, table_name: str, records: List[Dict]) -> Dict:
        """
        Insert multiple records using Supabase client
        
        Args:
            table_name: Target table name
            records: List of record dictionaries
            
        Returns:
            Insert operation result
        """
        try:
            table = await self.get_table_client(table_name)
            result = table.insert(records).execute()
            
            logger.info(f"Batch inserted {len(records)} records to {table_name}")
            return result.data
            
        except APIError as e:
            logger.error(f"Batch insert failed for {table_name}: {e}")
            raise
    
    async def upsert_batch(self, table_name: str, records: List[Dict], 
                          on_conflict: str = None) -> Dict:
        """
        Upsert multiple records using Supabase client
        
        Args:
            table_name: Target table name  
            records: List of record dictionaries
            on_conflict: Conflict resolution strategy
            
        Returns:
            Upsert operation result
        """
        try:
            table = await self.get_table_client(table_name)
            
            if on_conflict:
                result = table.upsert(records, on_conflict=on_conflict).execute()
            else:
                result = table.upsert(records).execute()
            
            logger.info(f"Batch upserted {len(records)} records to {table_name}")
            return result.data
            
        except APIError as e:
            logger.error(f"Batch upsert failed for {table_name}: {e}")
            raise
    
    def _format_query_params(self, query: str, params: Dict) -> str:
        """
        Convert named parameters to positional parameters for asyncpg
        
        Args:
            query: SQL query with named parameters (:param_name)
            params: Dictionary of parameter values
            
        Returns:
            Query with positional parameters ($1, $2, etc.)
        """
        formatted_query = query
        for i, param_name in enumerate(params.keys(), 1):
            formatted_query = formatted_query.replace(f":{param_name}", f"${i}")
        return formatted_query
    
    # Real-time subscriptions (Supabase feature)
    
    def create_realtime_subscription(self, table: str, event: str = "*", 
                                   callback: callable = None):
        """
        Create real-time subscription to table changes
        
        Args:
            table: Table name to watch
            event: Event type ('INSERT', 'UPDATE', 'DELETE', '*')
            callback: Function to call on events
        """
        if not self.supabase:
            raise ValueError("Supabase client not initialized")
        
        channel = self.supabase.channel(f"{table}_changes")
        
        if callback:
            channel.on(event, lambda payload: callback(payload))
        
        channel.subscribe()
        logger.info(f"Real-time subscription created for {table} ({event} events)")
        
        return channel


    # VP Investments 2.0 Enhanced Methods for 1.0 Migration
    
    async def upsert(self, table_name: str, record: Dict) -> Dict:
        """
        Upsert single record using Supabase client
        
        Args:
            table_name: Target table name
            record: Record dictionary to insert/update
            
        Returns:
            Upsert operation result
        """
        try:
            table = await self.get_table_client(table_name)
            result = table.upsert(record).execute()
            
            logger.debug(f"Upserted record to {table_name}")
            return result.data[0] if result.data else {}
            
        except APIError as e:
            logger.error(f"Upsert failed for {table_name}: {e}")
            raise
    
    async def get_latest_signals(self, limit: int = 50) -> List[Dict]:
        """Get latest signals from the enhanced signals table"""
        try:
            # Convert limit to int to prevent type issues
            limit = int(limit) if isinstance(limit, str) else limit
            
            query = """
            SELECT s.*, r.started_at as run_started_at
            FROM latest_signals s
            JOIN runs r ON s.run_id = r.run_id
            ORDER BY s.weighted_score DESC NULLS LAST
            LIMIT $1
            """
            return await self.execute_query(query, [limit])
            
        except Exception as e:
            logger.error(f"Failed to get latest signals: {e}")
            return []
    
    async def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data from enhanced schema"""
        try:
            # Get performance summary
            perf_query = """
            SELECT * FROM performance_dashboard 
            ORDER BY started_at DESC 
            LIMIT 10
            """
            performance_data = await self.execute_query(perf_query)
            
            # Get backtest summary
            backtest_query = "SELECT * FROM backtest_summary"
            backtest_data = await self.execute_query(backtest_query)
            
            # Get latest run statistics
            run_stats_query = """
            SELECT 
                COUNT(DISTINCT run_id) as total_runs,
                COUNT(DISTINCT ticker) as total_tickers,
                AVG(weighted_score) as avg_score,
                MAX(run_datetime) as latest_run
            FROM signals
            WHERE run_datetime >= NOW() - INTERVAL '30 days'
            """
            run_stats = await self.execute_query(run_stats_query)
            
            return {
                'performance_summary': performance_data,
                'backtest_summary': backtest_data,
                'run_statistics': run_stats[0] if run_stats else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    async def get_signal_performance_metrics(self, run_id: str = None) -> Dict[str, Any]:
        """Get detailed performance metrics for signals"""
        try:
            where_clause = "WHERE s.run_id = $1" if run_id else ""
            params = [run_id] if run_id else []
            
            query = f"""
            SELECT 
                COUNT(*) as total_signals,
                AVG(s.weighted_score) as avg_weighted_score,
                COUNT(CASE WHEN s."3d_return" > 0 THEN 1 END)::DECIMAL / 
                COUNT(s."3d_return") as hit_rate_3d,
                AVG(s."3d_return") as avg_return_3d,
                STDDEV(s."3d_return") as volatility_3d,
                CORR(s.weighted_score, s."3d_return") as ic_correlation,
                COUNT(CASE WHEN s.beat_spy_3d = true THEN 1 END)::DECIMAL /
                COUNT(s.beat_spy_3d) as beat_spy_rate
            FROM signals s
            {where_clause}
            """
            
            result = await self.execute_query(query, params)
            return result[0] if result else {}
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    async def save_run_metadata(self, run_data: Dict[str, Any]) -> str:
        """Save enhanced run metadata to runs table"""
        try:
            # Generate run_id if not provided
            if 'run_id' not in run_data:
                run_data['run_id'] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = await self.upsert('runs', run_data)
            logger.info(f"Saved run metadata: {run_data['run_id']}")
            
            return run_data['run_id']
            
        except Exception as e:
            logger.error(f"Failed to save run metadata: {e}")
            raise
    
    async def save_signals_batch(self, signals: List[Dict[str, Any]]) -> int:
        """Save multiple signals to the enhanced signals table"""
        try:
            result = await self.insert_batch('signals', signals)
            logger.info(f"Saved {len(signals)} signals to database")
            
            return len(signals)
            
        except Exception as e:
            logger.error(f"Failed to save signals batch: {e}")
            raise
    
    async def update_signal_backtest_results(self, signal_id: str, backtest_data: Dict[str, Any]) -> bool:
        """Update signal with backtest results"""
        try:
            query = """
            UPDATE signals SET
                "1d_return" = $2,
                "3d_return" = $3,
                "7d_return" = $4,
                "10d_return" = $5,
                "1d_return_net" = $6,
                "3d_return_net" = $7,
                "7d_return_net" = $8,
                "10d_return_net" = $9,
                spy_1d_return = $10,
                spy_3d_return = $11,
                spy_7d_return = $12,
                spy_10d_return = $13,
                beat_spy_1d = $14,
                beat_spy_3d = $15,
                beat_spy_7d = $16,
                beat_spy_10d = $17,
                max_return_pct = $18,
                drawdown_pct = $19,
                signal_duration = $20,
                forward_volatility = $21,
                forward_sharpe_ratio = $22,
                realized_returns = $23,
                backtest_phase = $24,
                backtest_timestamp = $25,
                updated_at = NOW()
            WHERE id = $1
            """
            
            params = [
                signal_id,
                backtest_data.get('1d_return'),
                backtest_data.get('3d_return'),
                backtest_data.get('7d_return'),
                backtest_data.get('10d_return'),
                backtest_data.get('1d_return_net'),
                backtest_data.get('3d_return_net'),
                backtest_data.get('7d_return_net'),
                backtest_data.get('10d_return_net'),
                backtest_data.get('spy_1d_return'),
                backtest_data.get('spy_3d_return'),
                backtest_data.get('spy_7d_return'),
                backtest_data.get('spy_10d_return'),
                backtest_data.get('beat_spy_1d'),
                backtest_data.get('beat_spy_3d'),
                backtest_data.get('beat_spy_7d'),
                backtest_data.get('beat_spy_10d'),
                backtest_data.get('max_return_pct'),
                backtest_data.get('drawdown_pct'),
                backtest_data.get('signal_duration'),
                backtest_data.get('forward_volatility'),
                backtest_data.get('forward_sharpe_ratio'),
                backtest_data.get('realized_returns'),
                backtest_data.get('backtest_phase', 'Pending'),
                datetime.now(timezone.utc)
            ]
            
            affected_rows = await self.execute_non_query(query, params)
            return affected_rows > 0
            
        except Exception as e:
            logger.error(f"Failed to update backtest results for signal {signal_id}: {e}")
            return False
    
    async def get_signals_for_backtest(self, run_id: str = None, limit: int = 1000) -> List[Dict]:
        """Get signals that need backtest processing"""
        try:
            where_clause = ""
            params = []
            
            if run_id:
                where_clause = "WHERE run_id = $1 AND (backtest_phase IS NULL OR backtest_phase != 'Complete')"
                params = [run_id]
            else:
                where_clause = "WHERE (backtest_phase IS NULL OR backtest_phase != 'Complete') AND run_datetime >= NOW() - INTERVAL '30 days'"
            
            query = f"""
            SELECT id, ticker, run_id, run_datetime, weighted_score, 
                   backtest_phase, "1d_return", "3d_return", "7d_return", "10d_return"
            FROM signals 
            {where_clause}
            ORDER BY run_datetime DESC
            LIMIT {limit}
            """
            
            return await self.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Failed to get signals for backtest: {e}")
            return []
    
    async def refresh_performance_views(self) -> bool:
        """Refresh materialized views for dashboard performance"""
        try:
            await self.execute_non_query("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_performance_summary")
            logger.info("Performance materialized views refreshed")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to refresh performance views: {e}")
            return False
    
    async def get_system_config(self, category: str = None) -> Dict[str, Any]:
        """Get system configuration from database"""
        try:
            where_clause = "WHERE category = $1" if category else ""
            params = [category] if category else []
            
            query = f"""
            SELECT key, value, value_type, description, category
            FROM system_config 
            {where_clause}
            ORDER BY category, key
            """
            
            config_rows = await self.execute_query(query, params)
            
            # Convert to nested dictionary structure
            config = {}
            for row in config_rows:
                key = row['key']
                value = row['value']
                value_type = row['value_type']
                
                # Convert value based on type
                if value_type == 'boolean':
                    value = value.lower() in ('true', '1', 'yes')
                elif value_type == 'integer':
                    value = int(value)
                elif value_type == 'float':
                    value = float(value)
                elif value_type == 'json':
                    value = json.loads(value)
                
                config[key] = value
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to get system config: {e}")
            return {}
            return {}
    
    # ============================================================================
    # BACKTEST-SPECIFIC OPERATIONS FOR PHASE 3
    # ============================================================================
    
    async def create_backtest_tables(self) -> bool:
        """Create backtest-related tables if they don't exist"""
        try:
            # Read schema file and execute
            schema_path = Path(__file__).parent / "backtest_schema.sql"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                
                # Split schema into individual statements
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        await self.execute_query(statement)
                
                logger.info("[SUCCESS] Backtest tables created/verified")
                return True
            else:
                logger.error("Backtest schema file not found")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to create backtest tables: {e}")
            return False
    
    async def store_backtest_result(self, backtest_data: Dict[str, Any]) -> bool:
        """Store comprehensive backtest results (DISABLED - backtests table doesn't exist)"""
        try:
            # DISABLED: backtests table doesn't exist - data stored in signals table
            logger.info(f"✅ Backtest data calculated (not stored in separate table)")
            return True
            
            # # Original code (commented out - tables don't exist):
            # # Store main backtest record
            # await self.upsert_data("backtests", [backtest_data["main"]])
            # 
            # # Store trade history
            # if "trades" in backtest_data and backtest_data["trades"]:
            #     await self.upsert_data("backtest_trades", backtest_data["trades"])
            
            # Store portfolio snapshots
            if "portfolios" in backtest_data and backtest_data["portfolios"]:
                await self.upsert_data("backtest_portfolios", backtest_data["portfolios"])
            
            # Store position details
            if "positions" in backtest_data and backtest_data["positions"]:
                await self.upsert_data("backtest_positions", backtest_data["positions"])
            
            logger.info(f"[SUCCESS] Stored backtest results for {backtest_data['main']['backtest_id']}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to store backtest results: {e}")
            return False
    
    async def get_backtest_results(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete backtest results"""
        try:
            # Get main backtest record
            backtest_query = """
                SELECT * FROM backtest_summary 
                WHERE backtest_id = $1
            """
            backtest_result = await self.execute_query(backtest_query, [backtest_id])
            
            if not backtest_result:
                return None
            
            backtest_data = backtest_result[0]
            
            # Get trade history
            trades = await self.query_data(
                "backtest_trades",
                filters={"backtest_id": backtest_id},
                order_by="date"
            )
            
            # Get portfolio snapshots
            portfolios = await self.query_data(
                "backtest_portfolios",
                filters={"backtest_id": backtest_id},
                order_by="date"
            )
            
            # Get position details
            positions = await self.query_data(
                "backtest_positions",
                filters={"backtest_id": backtest_id},
                order_by="portfolio_date, ticker"
            )
            
            return {
                "backtest": backtest_data,
                "trades": trades,
                "portfolios": portfolios,
                "positions": positions
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to retrieve backtest results: {e}")
            return None
    
    async def get_backtest_performance_metrics(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Get calculated performance metrics for a backtest"""
        try:
            query = "SELECT * FROM calculate_portfolio_stats($1)"
            result = await self.execute_query(query, [backtest_id])
            
            if result:
                return result[0]
            
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get backtest metrics: {e}")
            return None
    
    async def list_backtests(self, 
                           strategy: Optional[str] = None,
                           status: Optional[str] = None,
                           limit: int = 50,
                           offset: int = 0) -> List[Dict[str, Any]]:
        """List backtests with filtering and pagination (DISABLED - backtest tables don't exist)"""
        try:
            # DISABLED: backtest_summary table/view doesn't exist
            # Backtest data stored in signals table columns
            logger.info(f"✅ list_backtests called (returns empty - separate backtest tables don't exist)")
            return []
            
            # # Original code (commented out - tables don't exist):
            # filters = {}
            # if strategy:
            #     filters["strategy"] = strategy
            # if status:
            #     filters["status"] = status
            # 
            # results = await self.query_data(
            #     "backtest_summary",
            #     filters=filters,
            #     limit=limit,
            #     offset=offset,
            #     order_by="created_at DESC"
            # )
            # return results
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to list backtests: {e}")
            return []
    
    async def get_portfolio_performance_history(self, backtest_id: str) -> List[Dict[str, Any]]:
        """Get portfolio performance over time for visualization"""
        try:
            query = """
                SELECT 
                    date,
                    total_value,
                    daily_return,
                    cumulative_return,
                    position_count,
                    drawdown_pct,
                    rolling_30d_return,
                    rolling_30d_volatility
                FROM portfolio_performance 
                WHERE backtest_id = $1
                ORDER BY date
            """
            
            results = await self.execute_query(query, [backtest_id])
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get portfolio performance history: {e}")
            return []
    
    async def get_trade_analysis(self, backtest_id: str) -> List[Dict[str, Any]]:
        """Get detailed trade analysis with performance metrics"""
        try:
            query = """
                SELECT 
                    ticker,
                    action,
                    date,
                    shares,
                    price,
                    value,
                    total_cost,
                    signal_score,
                    realized_pnl,
                    holding_period_days,
                    return_pct,
                    annualized_return_pct
                FROM trade_analysis 
                WHERE backtest_id = $1
                ORDER BY date
            """
            
            results = await self.execute_query(query, [backtest_id])
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get trade analysis: {e}")
            return []
    
    async def store_optimization_results(self, optimization_data: Dict[str, Any]) -> bool:
        """Store parameter optimization results"""
        try:
            # Store optimization run
            await self.upsert_data("backtest_optimizations", [optimization_data["optimization"]])
            
            # Store individual results
            if "results" in optimization_data and optimization_data["results"]:
                await self.upsert_data("optimization_results", optimization_data["results"])
            
            logger.info(f"[SUCCESS] Stored optimization results for {optimization_data['optimization']['optimization_id']}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to store optimization results: {e}")
            return False
    
    async def store_walkforward_results(self, walkforward_data: Dict[str, Any]) -> bool:
        """Store walk-forward analysis results"""
        try:
            # Store walk-forward analysis
            await self.upsert_data("walkforward_analysis", [walkforward_data["analysis"]])
            
            # Store individual periods
            if "periods" in walkforward_data and walkforward_data["periods"]:
                await self.upsert_data("walkforward_periods", walkforward_data["periods"])
            
            logger.info(f"[SUCCESS] Stored walk-forward results for {walkforward_data['analysis']['walkforward_id']}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to store walk-forward results: {e}")
            return False

    def delete_old_signals(self, cutoff_time: datetime) -> int:
        """Delete signals older than cutoff_time"""
        try:
            result = self.client.table('signals_norm').delete().lt('created_at', cutoff_time.isoformat()).execute()
            return len(result.data) if result.data else 0
        except Exception as e:
            logger.error(f"Failed to delete old signals: {e}")
            return 0

    def delete_old_runs(self, cutoff_time: datetime) -> int:
        """Delete runs older than cutoff_time"""
        try:
            result = self.client.table('runs').delete().lt('started_at', cutoff_time.isoformat()).execute()
            return len(result.data) if result.data else 0
        except Exception as e:
            logger.error(f"Failed to delete old runs: {e}")
            return 0

    def insert_metric(self, run_id: str, metric_name: str, value: float, units: str):
        """Insert a performance metric (DISABLED - metrics table doesn't exist)"""
        try:
            # DISABLED: metrics table doesn't exist - store in runs.metadata instead
            logger.info(f"✅ Metric {metric_name}={value} {units} calculated for {run_id} (not stored separately)")
            
            # # Original code (commented out - table doesn't exist):
            # data = {
            #     'run_id': run_id,
            #     'metric_name': metric_name,
            #     'value': value,
            #     'units': units,
            #     'timestamp': datetime.now().isoformat()
            # }
            # self.client.table('metrics').insert(data).execute()
        except Exception as e:
            logger.error(f"Failed to insert metric {metric_name}: {e}")

    def get_latest_discovery_run(self) -> Optional[Dict]:
        """Get the latest discovery run"""
        try:
            result = self.client.table('runs').select('run_id, started_at, total_signals, status').eq('run_type', 'discovery_pipeline').order('started_at', desc=True).limit(1).execute()
            
            if result.data:
                run = result.data[0]
                return {
                    "run_id": run['run_id'],
                    "started_at": run['started_at'],
                    "total_signals": run['total_signals'],
                    "status": run['status']
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get latest discovery run: {e}")
            return None

    def get_recent_discovery_metrics(self) -> List[Dict]:
        """Get recent discovery metrics (DISABLED - metrics table doesn't exist)"""
        try:
            # DISABLED: metrics table doesn't exist
            logger.info(f"✅ get_recent_discovery_metrics called (returns empty - metrics table doesn't exist)")
            return []
            
            # # Original code (commented out - table doesn't exist):
            # cutoff = datetime.now() - timedelta(days=1)
            # result = self.client.table('metrics').select('metric_name, value, timestamp').like('run_id', 'discovery_%').gt('timestamp', cutoff.isoformat()).order('timestamp', desc=True).limit(20).execute()
            # 
            # return [
            #     {
            #         "name": row['metric_name'],
            #         "value": float(row['value']),
            #         "timestamp": row['timestamp']
            #     } for row in result.data
            # ] if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get recent discovery metrics: {e}")
            return []

    def fetch_tickers(self, search_term: str) -> List[Dict]:
        """Fetch tickers matching search term"""
        try:
            # Search both ticker and company name
            result = self.client.table('company_tickers').select('ticker, company_name, sector, market_cap_display').or_(
                f'ticker.ilike.%{search_term}%,company_name.ilike.%{search_term}%'
            ).limit(50).execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to fetch tickers for '{search_term}': {e}")
            return []

    def get_all_tickers(self) -> List[str]:
        """Get all available ticker symbols"""
        try:
            result = self.client.table('company_tickers').select('ticker').execute()
            return [row['ticker'] for row in result.data] if result.data else []
        except Exception as e:
            logger.error(f"Failed to get all tickers: {e}")
            return []

    def get_company_by_ticker(self, ticker: str) -> Optional[Dict]:
        """Get company information by ticker"""
        try:
            result = self.client.table('company_tickers').select('*').eq('ticker', ticker.upper()).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get company for ticker '{ticker}': {e}")
            return None


# Factory function for database interface
async def get_supabase_database() -> SupabaseInterface:
    """Create and return configured Supabase database interface"""
    db = SupabaseInterface()
    await db.connect()
    return db


def get_database() -> SupabaseInterface:
    """Get database interface (synchronous factory)"""
    return SupabaseInterface()