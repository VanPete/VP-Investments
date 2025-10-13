"""
VP Investments 3.0 - Phase 1 Cache Layer
=========================================

Eager-but-Bounded data fetching with deterministic caching.

Features:
- Supabase-backed cache (data_cache table)
- Versioned cache keys: cache:{date}:{ticker}:{group}:{provider}:{version}
- TTL-based freshness with configurable expiration
- Provenance tracking (endpoint, params, fetch time, rate limits)
- Circuit breaker pattern for API failures
- Offline mode (cache-only operation)

Phase 1 Rule: ALL external API calls happen here. Phases 2-6 read ONLY from cache.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

class DataGroup(Enum):
    """Signal data groups matching 3.0 architecture"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    NEWS_MACRO = "news_macro"
    SOCIAL_ALTERNATIVE = "social_alternative"
    RISK_STABILITY = "risk_stability"
    INSTITUTIONAL_SMART_MONEY = "institutional_smart_money"


class Provider(Enum):
    """External data providers"""
    YFINANCE = "yfinance"
    REDDIT = "reddit"
    OPENAI = "openai"
    NEWS_API = "news_api"  # PLACEHOLDER
    STOCKTWITS = "stocktwits"  # PLACEHOLDER
    FUTURE_FINANCIAL_API = "future_financial_api"  # PLACEHOLDER


@dataclass
class CacheConfig:
    """TTL configuration for different data types"""
    
    # Technical data (market hours: hourly, after hours: EOD)
    technical_ttl_seconds: int = 3600  # 1 hour
    
    # Fundamental data (slow-moving, refresh weekly)
    fundamental_ttl_seconds: int = 604800  # 7 days
    
    # News (real-time during market hours)
    news_ttl_seconds: int = 1800  # 30 minutes
    
    # Social (Reddit: hourly)
    social_ttl_seconds: int = 3600  # 1 hour
    
    # Risk/Options (intraday updates)
    risk_ttl_seconds: int = 3600  # 1 hour
    
    # Institutional (monthly updates)
    institutional_ttl_seconds: int = 2592000  # 30 days
    
    # Cache version (bump when schema changes)
    cache_version: str = "v1"
    
    def get_ttl(self, group: DataGroup) -> int:
        """Get TTL seconds for a data group"""
        mapping = {
            DataGroup.TECHNICAL: self.technical_ttl_seconds,
            DataGroup.FUNDAMENTAL: self.fundamental_ttl_seconds,
            DataGroup.NEWS_MACRO: self.news_ttl_seconds,
            DataGroup.SOCIAL_ALTERNATIVE: self.social_ttl_seconds,
            DataGroup.RISK_STABILITY: self.risk_ttl_seconds,
            DataGroup.INSTITUTIONAL_SMART_MONEY: self.institutional_ttl_seconds,
        }
        return mapping.get(group, 3600)


# ============================================================================
# CACHE LAYER
# ============================================================================

class CacheLayer:
    """
    Phase 1 cache layer with Supabase backend.
    
    Prevents mid-pipeline API calls and ensures reproducible runs.
    """
    
    def __init__(self, db_interface, config: Optional[CacheConfig] = None):
        """
        Initialize cache layer
        
        Args:
            db_interface: SupabaseInterface instance
            config: Cache configuration (uses defaults if None)
        """
        self.db = db_interface
        self.config = config or CacheConfig()
        self.logger = logger
        
        # Circuit breaker state
        self.circuit_breakers = {}  # {provider: {"failures": 0, "last_failure": None}}
        self.max_failures = 3
        self.circuit_reset_seconds = 300  # 5 minutes
    
    def _generate_cache_key(
        self,
        ticker: str,
        group: DataGroup,
        provider: Provider,
        date: Optional[str] = None
    ) -> str:
        """
        Generate versioned cache key
        
        Format: cache:{date}:{ticker}:{group}:{provider}:{version}
        Example: cache:2025-10-13:AAPL:technical:yfinance:v1
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        return f"cache:{date}:{ticker}:{group.value}:{provider.value}:{self.config.cache_version}"
    
    async def get_cached(
        self,
        ticker: str,
        group: DataGroup,
        provider: Provider,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from cache if fresh
        
        Args:
            ticker: Stock ticker
            group: Data group (technical, fundamental, etc.)
            provider: Data provider (yfinance, reddit, etc.)
            force_refresh: Bypass cache and force new fetch
            
        Returns:
            Cached payload dict or None if cache miss/stale
        """
        if force_refresh:
            self.logger.debug(f"Force refresh - skipping cache for {ticker}/{group.value}")
            return None
        
        cache_key = self._generate_cache_key(ticker, group, provider)
        
        try:
            # Query Supabase cache table
            result = await self.db.execute_query(
                """
                SELECT payload, metadata, expires_at, fetched_at
                FROM data_cache
                WHERE cache_key = $1
                LIMIT 1
                """,
                [cache_key]
            )
            
            if not result:
                self.logger.debug(f"Cache MISS: {cache_key}")
                return None
            
            row = result[0]
            expires_at = row['expires_at']
            now = datetime.now(timezone.utc)
            
            # Check if expired
            if expires_at < now:
                self.logger.debug(f"Cache STALE: {cache_key} (expired {expires_at})")
                return None
            
            self.logger.debug(f"Cache HIT: {cache_key}")
            return row['payload']
            
        except Exception as e:
            self.logger.error(f"Cache retrieval error for {cache_key}: {e}")
            return None
    
    async def set_cached(
        self,
        ticker: str,
        group: DataGroup,
        provider: Provider,
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Store data in cache with TTL
        
        Args:
            ticker: Stock ticker
            group: Data group
            provider: Data provider
            payload: Data to cache
            metadata: Provenance metadata (endpoint, params, etc.)
            ttl_seconds: TTL override (uses group default if None)
            
        Returns:
            True if cached successfully
        """
        cache_key = self._generate_cache_key(ticker, group, provider)
        
        if ttl_seconds is None:
            ttl_seconds = self.config.get_ttl(group)
        
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        
        try:
            # Upsert into cache table
            await self.db.execute_query(
                """
                INSERT INTO data_cache (
                    cache_key, ticker, data_group, provider,
                    payload, metadata, fetched_at, expires_at, ttl_seconds,
                    version, endpoint, response_time_ms, rate_limit_remaining
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (cache_key) 
                DO UPDATE SET
                    payload = EXCLUDED.payload,
                    metadata = EXCLUDED.metadata,
                    fetched_at = EXCLUDED.fetched_at,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = NOW()
                """,
                [
                    cache_key,
                    ticker,
                    group.value,
                    provider.value,
                    json.dumps(payload),
                    json.dumps(metadata),
                    now,
                    expires_at,
                    ttl_seconds,
                    self.config.cache_version,
                    metadata.get('endpoint'),
                    metadata.get('response_time_ms'),
                    metadata.get('rate_limit_remaining')
                ]
            )
            
            self.logger.debug(f"Cache SET: {cache_key} (TTL: {ttl_seconds}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache storage error for {cache_key}: {e}")
            return False
    
    async def get_or_fetch(
        self,
        ticker: str,
        group: DataGroup,
        provider: Provider,
        fetch_func: callable,
        force_refresh: bool = False,
        **fetch_kwargs
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Cache-first fetch with automatic fallback
        
        Args:
            ticker: Stock ticker
            group: Data group
            provider: Data provider
            fetch_func: Async function to fetch data if cache miss
            force_refresh: Bypass cache
            **fetch_kwargs: Arguments to pass to fetch_func
            
        Returns:
            (payload, from_cache) tuple
        """
        # Check circuit breaker
        if self._is_circuit_open(provider):
            self.logger.warning(f"Circuit breaker OPEN for {provider.value} - using cache only")
            cached = await self.get_cached(ticker, group, provider, force_refresh=False)
            return cached, True
        
        # Try cache first
        cached = await self.get_cached(ticker, group, provider, force_refresh)
        if cached is not None:
            return cached, True
        
        # Cache miss - fetch from external API
        self.logger.info(f"Fetching {ticker}/{group.value} from {provider.value}...")
        
        try:
            fetch_start = datetime.now(timezone.utc)
            payload = await fetch_func(ticker, **fetch_kwargs)
            fetch_duration_ms = int((datetime.now(timezone.utc) - fetch_start).total_seconds() * 1000)
            
            if payload is None:
                self.logger.warning(f"Fetch returned None for {ticker}/{group.value}")
                return None, False
            
            # Build metadata
            metadata = {
                "endpoint": fetch_kwargs.get('endpoint', str(fetch_func.__name__)),
                "params": fetch_kwargs,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "response_time_ms": fetch_duration_ms,
                "rate_limit_remaining": None,  # Provider-specific
                "warnings": []
            }
            
            # Cache the result
            await self.set_cached(ticker, group, provider, payload, metadata)
            
            # Reset circuit breaker on success
            self._record_success(provider)
            
            return payload, False
            
        except Exception as e:
            self.logger.error(f"Fetch failed for {ticker}/{group.value}: {e}")
            
            # Record failure for circuit breaker
            self._record_failure(provider)
            
            # Try to return stale cache as fallback
            stale_cached = await self.get_cached(ticker, group, provider, force_refresh=False)
            if stale_cached:
                self.logger.warning(f"Using stale cache for {ticker}/{group.value}")
                return stale_cached, True
            
            return None, False
    
    def _is_circuit_open(self, provider: Provider) -> bool:
        """Check if circuit breaker is open for provider"""
        state = self.circuit_breakers.get(provider.value)
        if not state:
            return False
        
        failures = state.get("failures", 0)
        last_failure = state.get("last_failure")
        
        if failures < self.max_failures:
            return False
        
        # Check if reset time has passed
        if last_failure:
            elapsed = (datetime.now(timezone.utc) - last_failure).total_seconds()
            if elapsed > self.circuit_reset_seconds:
                self.logger.info(f"Circuit breaker RESET for {provider.value}")
                self.circuit_breakers[provider.value] = {"failures": 0, "last_failure": None}
                return False
        
        return True
    
    def _record_failure(self, provider: Provider):
        """Record API failure for circuit breaker"""
        if provider.value not in self.circuit_breakers:
            self.circuit_breakers[provider.value] = {"failures": 0, "last_failure": None}
        
        self.circuit_breakers[provider.value]["failures"] += 1
        self.circuit_breakers[provider.value]["last_failure"] = datetime.now(timezone.utc)
        
        failures = self.circuit_breakers[provider.value]["failures"]
        self.logger.warning(f"Provider {provider.value} failures: {failures}/{self.max_failures}")
    
    def _record_success(self, provider: Provider):
        """Reset circuit breaker on successful fetch"""
        if provider.value in self.circuit_breakers:
            self.circuit_breakers[provider.value] = {"failures": 0, "last_failure": None}
    
    async def bulk_get_cached(
        self,
        tickers: List[str],
        group: DataGroup,
        provider: Provider
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Bulk cache retrieval for multiple tickers
        
        Returns:
            {ticker: payload} dict (payload is None for cache misses)
        """
        cache_keys = [
            self._generate_cache_key(ticker, group, provider)
            for ticker in tickers
        ]
        
        try:
            result = await self.db.execute_query(
                """
                SELECT ticker, payload, expires_at
                FROM data_cache
                WHERE cache_key = ANY($1)
                """,
                [cache_keys]
            )
            
            now = datetime.now(timezone.utc)
            cached_data = {}
            
            for row in result:
                ticker = row['ticker']
                expires_at = row['expires_at']
                
                if expires_at >= now:
                    cached_data[ticker] = row['payload']
            
            # Fill in None for cache misses
            return {ticker: cached_data.get(ticker) for ticker in tickers}
            
        except Exception as e:
            self.logger.error(f"Bulk cache retrieval error: {e}")
            return {ticker: None for ticker in tickers}
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            result = await self.db.execute_query("""
                SELECT 
                    data_group,
                    provider,
                    COUNT(*) as total_entries,
                    COUNT(*) FILTER (WHERE expires_at > NOW()) as fresh_entries,
                    COUNT(*) FILTER (WHERE expires_at <= NOW()) as stale_entries,
                    AVG(response_time_ms) as avg_response_ms
                FROM data_cache
                GROUP BY data_group, provider
                ORDER BY data_group, provider
            """)
            
            return {
                "stats": result,
                "circuit_breakers": self.circuit_breakers
            }
            
        except Exception as e:
            self.logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cache_layer(db_interface, config: Optional[CacheConfig] = None) -> CacheLayer:
    """Factory function to create cache layer"""
    return CacheLayer(db_interface, config)
