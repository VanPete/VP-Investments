"""
VP Investments - Enums

Extracted from legacy core.py for v3.1 architecture.
"""
from enum import Enum


class SignalType(Enum):
    """Signal type classification"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeType(Enum):
    """Trade classification"""
    SWING = "swing"
    DAY = "day"
    MOMENTUM = "momentum"
    VALUE = "value"


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FeatureType(Enum):
    """Feature type classification"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
