"""Core module initialization."""

from vp_investments.core.config import get_config, setup_logging
from vp_investments.core.core import (
    VPInvestmentsError, 
    ConfigurationError, 
    DataError,
    APIError,
    SignalType,
    TradeType,
    RiskLevel,
    MarketCondition,
    DataSource,
    FeatureType,
    DEFAULT_TICKERS
)

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