"""Core module initialization."""

from backend.core.config import get_config, setup_logging
from backend.exceptions import (
    VPInvestmentsError, 
    ConfigurationError,
    DataFetchError as DataError,
    DatabaseError as APIError,
)
from backend.enums import (
    SignalType,
    TradeType,
    RiskLevel,
    FeatureType,
)

# Legacy compatibility - these were in old core.py
class MarketCondition:
    """Market condition placeholder (deprecated)"""
    pass

class DataSource:
    """Data source placeholder (deprecated)"""
    pass

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Moved from legacy core.py

__all__ = [
    "get_config",
    "setup_logging",
    "VPInvestmentsError",
    "ConfigurationError",
    "DataError", 
    "APIError",
    "SignalType",
    "TradeType",
    "RiskLevel", 
    "MarketCondition",
    "DataSource",
    "FeatureType",
    "DEFAULT_TICKERS"
]