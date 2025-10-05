"""
VP Investments - Advanced Investment Analysis Platform

A comprehensive investment analysis platform with production-grade optimizations
including enhanced signal processing, real-time data streams, monitoring,
and auto-scaling capabilities.
"""

__version__ = "3.0.0"
__author__ = "VP Investments Team"
__description__ = "Advanced Investment Analysis Platform"

# Core imports for easy access
from backend.core.config import get_config
from backend.core.core import VPInvestmentsError

__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "get_config",
    "VPInvestmentsError"
]
