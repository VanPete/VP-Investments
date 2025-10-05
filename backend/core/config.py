"""
VP Investments Core - Configuration Management

Centralized configuration management with database backend for admin editability.
Supports hierarchical configs, environment overrides, and validation.
"""
from __future__ import annotations

import os
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from .core import ConfigurationError, LOG_DIR

# Load environment variables from .env file
load_dotenv()


class ConfigManager:
    """
    Centralized configuration manager that supports:
    - Database-backed configurations (editable via web UI)
    - Environment variable overrides
    - Default values with validation
    - Hierarchical configuration structure
    """
    
    def __init__(self, db_client=None):
        self.db_client = db_client
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Load defaults
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values"""
        self._defaults = {
            # System Settings
            "system": {
                "environment": os.getenv("VP_ENV", "development"),
                "debug": os.getenv("DEBUG", "false").lower() == "true",
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "timezone": "UTC",
                "max_concurrent_tasks": int(os.getenv("MAX_CONCURRENT_TASKS", "10")),
                "request_timeout_seconds": int(os.getenv("REQUEST_TIMEOUT", "30")),
                "cache_ttl_seconds": int(os.getenv("CACHE_TTL", "3600")),
            },
            
            # Database Settings
            "database": {
                "supabase_url": os.getenv("SUPABASE_URL"),
                "supabase_key": os.getenv("SUPABASE_ANON_KEY"),
                "supabase_service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
                "connection_pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
                "connection_timeout": int(os.getenv("DB_TIMEOUT", "30")),
                "retry_attempts": int(os.getenv("DB_RETRY_ATTEMPTS", "3")),
            },
            
            # Data Sources
            "data_sources": {
                "yahoo_finance": {
                    "enabled": os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true",
                    "rate_limit_per_second": int(os.getenv("YAHOO_RATE_LIMIT", "10")),
                    "timeout": int(os.getenv("YAHOO_TIMEOUT", "15")),
                },
                "news": {
                    "enabled": os.getenv("NEWS_ENABLED", "true").lower() == "true",
                    "api_key": os.getenv("NEWS_API_KEY"),
                    "sources": ["reuters", "bloomberg", "cnbc", "marketwatch"],
                    "rate_limit_per_day": int(os.getenv("NEWS_RATE_LIMIT", "1000")),
                },
                "reddit": {
                    "enabled": os.getenv("REDDIT_ENABLED", "true").lower() == "true",
                    "client_id": os.getenv("REDDIT_CLIENT_ID"),
                    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
                    "user_agent": os.getenv("REDDIT_USER_AGENT", "VP_Investments_Bot"),
                    "subreddits": ["stocks", "investing", "SecurityAnalysis", "ValueInvesting"],
                    "rate_limit_per_minute": int(os.getenv("REDDIT_RATE_LIMIT", "60")),
                },
                "google_trends": {
                    "enabled": os.getenv("GOOGLE_TRENDS_ENABLED", "true").lower() == "true",
                    "timeout": int(os.getenv("GOOGLE_TRENDS_TIMEOUT", "30")),
                },
            },
            
            # Analysis Settings
            "analysis": {
                "default_lookback_days": int(os.getenv("ANALYSIS_LOOKBACK_DAYS", "30")),
                "min_data_points": int(os.getenv("MIN_DATA_POINTS", "10")),
                "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
                "signal_weights": {
                    "technical": float(os.getenv("TECHNICAL_WEIGHT", "0.4")),
                    "sentiment": float(os.getenv("SENTIMENT_WEIGHT", "0.3")),
                    "news": float(os.getenv("NEWS_WEIGHT", "0.2")),
                    "volume": float(os.getenv("VOLUME_WEIGHT", "0.1")),
                },
                "batch_size": int(os.getenv("ANALYSIS_BATCH_SIZE", "50")),
            },
            
            # Signal Scoring Weights (Configurable)
            "scoring": {
                "weights": {
                    # Primary signal component weights (must sum to 1.0)
                    "reddit": float(os.getenv("SCORING_WEIGHT_REDDIT", "0.5")),  # Default 50%
                    "financial": float(os.getenv("SCORING_WEIGHT_FINANCIAL", "0.5")),  # Default 50%
                    "news": float(os.getenv("SCORING_WEIGHT_NEWS", "0.0")),  # Default 0% (not implemented)
                },
                # Individual component calculation weights
                "reddit_components": {
                    "mention_count": float(os.getenv("REDDIT_WEIGHT_MENTIONS", "0.4")),
                    "sentiment": float(os.getenv("REDDIT_WEIGHT_SENTIMENT", "0.3")),
                    "engagement": float(os.getenv("REDDIT_WEIGHT_ENGAGEMENT", "0.3")),
                },
                "financial_components": {
                    "technical_indicators": float(os.getenv("FINANCIAL_WEIGHT_TECHNICAL", "0.5")),
                    "fundamentals": float(os.getenv("FINANCIAL_WEIGHT_FUNDAMENTALS", "0.3")),
                    "momentum": float(os.getenv("FINANCIAL_WEIGHT_MOMENTUM", "0.2")),
                },
            },
            
            # Database Settings (Supabase only - no SQLite)
            "supabase": {
                "url": os.getenv("SUPABASE_URL"),
                "anon_key": os.getenv("SUPABASE_ANON_KEY"),
                "service_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
                "database_url": os.getenv("SUPABASE_DATABASE_URL"),
            },
            
            "database": {
                "type": "postgres",  # Only PostgreSQL via Supabase
                "max_connections": int(os.getenv("DATABASE_MAX_CONNECTIONS", "10")),
                "command_timeout": int(os.getenv("DATABASE_COMMAND_TIMEOUT", "60")),
                "retry_attempts": int(os.getenv("DATABASE_RETRY_ATTEMPTS", "3")),
            },
            
            # Production Settings
            "production": {
                "enabled": os.getenv("PRODUCTION_MODE", "false").lower() == "true",
                "auto_scaling": {
                    "enabled": os.getenv("AUTO_SCALING", "false").lower() == "true",
                    "min_workers": int(os.getenv("MIN_WORKERS", "2")),
                    "max_workers": int(os.getenv("MAX_WORKERS", "16")),
                    "scaling_threshold": float(os.getenv("SCALING_THRESHOLD", "0.85")),
                },
                "monitoring": {
                    "enabled": os.getenv("MONITORING", "false").lower() == "true",
                    "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
                    "alert_thresholds": {
                        "error_rate": float(os.getenv("ERROR_RATE_THRESHOLD", "0.05")),
                        "response_time": float(os.getenv("RESPONSE_TIME_THRESHOLD", "1.0")),
                    },
                },
                "realtime": {
                    "enabled": os.getenv("REALTIME", "false").lower() == "true",
                    "stream_buffer_size": int(os.getenv("STREAM_BUFFER_SIZE", "1000")),
                    "processing_interval_ms": int(os.getenv("PROCESSING_INTERVAL_MS", "100")),
                },
            },
            
            # API Settings
            "api": {
                "host": os.getenv("API_HOST", "0.0.0.0"),
                "port": int(os.getenv("API_PORT", "8000")),
                "workers": int(os.getenv("API_WORKERS", "4")),
                "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
                "rate_limit": {
                    "requests_per_minute": int(os.getenv("API_RATE_LIMIT", "100")),
                    "burst_limit": int(os.getenv("API_BURST_LIMIT", "200")),
                },
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support
        
        Args:
            key: Configuration key (e.g., 'database.supabase_url')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check cache first
        if self._is_cache_valid() and key in self._cache:
            return self._cache[key]
        
        # Try environment variable override first
        env_key = key.replace(".", "_").upper()
        env_value = os.getenv(env_key)
        if env_value is not None:
            self._cache[key] = env_value
            return env_value
        
        # Try database if available
        if self.db_client:
            try:
                db_value = self._get_from_database(key)
                if db_value is not None:
                    self._cache[key] = db_value
                    return db_value
            except Exception as e:
                logging.warning(f"Failed to get config from database: {e}")
        
        # Fall back to defaults
        value = self._get_nested_value(self._defaults, key.split('.'))
        if value is not None:
            self._cache[key] = value
            return value
        
        return default
    
    def _get_nested_value(self, data: Dict, keys: List[str]) -> Any:
        """Get nested dictionary value"""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if self._cache_timestamp is None:
            return False
        
        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < self.cache_ttl_seconds
    
    def _get_from_database(self, key: str) -> Optional[Any]:
        """Get configuration value from database"""
        # This would be implemented when database integration is added
        return None
    
    def set(self, key: str, value: Any, description: str = None, updated_by: str = 'system') -> bool:
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
            description: Optional description of the setting
            updated_by: Who updated this setting
            
        Returns:
            True if successful
        """
        try:
            # Update cache
            self._cache[key] = value
            self._cache_timestamp = datetime.now()
            
            # Update database if available
            if self.db_client:
                return self._set_in_database(key, value, description, updated_by)
            
            return True
        except Exception as e:
            logging.error(f"Failed to set config {key}: {e}")
            return False
    
    def _set_in_database(self, key: str, value: Any, description: str, updated_by: str) -> bool:
        """Set configuration value in database"""
        # This would be implemented when database integration is added
        return True
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all configuration values for a section"""
        result = {}
        section_data = self._get_nested_value(self._defaults, section.split('.'))
        
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                full_key = f"{section}.{key}"
                result[key] = self.get(full_key, value)
        
        return result
    
    def reload(self):
        """Reload configuration from all sources"""
        self._cache.clear()
        self._cache_timestamp = None
        self._load_defaults()
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate required settings
        required_settings = [
            "database.supabase_url",
            "database.supabase_key"
        ]
        
        for setting in required_settings:
            if not self.get(setting):
                issues.append(f"Missing required setting: {setting}")
        
        # Validate data source API keys if enabled
        if self.get("data_sources.news.enabled"):
            if not self.get("data_sources.news.api_key"):
                issues.append("News API is enabled but api_key is missing")
        
        if self.get("data_sources.reddit.enabled"):
            if not self.get("data_sources.reddit.client_id") or not self.get("data_sources.reddit.client_secret"):
                issues.append("Reddit API is enabled but credentials are missing")
        
        return issues


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration manager instance"""
    return config_manager


def init_config(db_client=None):
    """Initialize configuration manager with database client"""
    global config_manager
    config_manager = ConfigManager(db_client)


# Convenience functions
def get(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config_manager.get(key, default)


def set_config(key: str, value: Any, description: str = None, updated_by: str = 'system') -> bool:
    """Set configuration value"""
    return config_manager.set(key, value, description, updated_by)


def get_section(section: str) -> Dict[str, Any]:
    """Get all configuration values for a section"""
    return config_manager.get_section(section)


def setup_logging():
    """Setup application logging configuration"""
    log_level = get_config().get('system.log_level', 'INFO')
    log_format = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
    
    # Create logs directory if it doesn't exist
    Path(LOG_DIR).mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{LOG_DIR}/vp_investments.log', mode='a')
        ]
    )
    
    # Set third-party library log levels to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")