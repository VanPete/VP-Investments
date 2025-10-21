"""
VP Investments - Custom Exceptions

Extracted from legacy core.py for v3.1 architecture.
"""


class VPInvestmentsError(Exception):
    """Base exception for VP Investments"""
    pass


class ConfigurationError(VPInvestmentsError):
    """Configuration related errors"""
    pass


class DataFetchError(VPInvestmentsError):
    """Data fetching errors"""
    pass


class CalculationError(VPInvestmentsError):
    """Calculation/scoring errors"""
    pass


class DatabaseError(VPInvestmentsError):
    """Database connection/query errors"""
    pass
