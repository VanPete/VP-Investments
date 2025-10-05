# VP Investments Integrations Module
"""
VP Investments integrations for data sources and signal processing.
Consolidated structure following domain-based organization.
"""

# YFinance & Technical Indicators
from .yfinance import get_technical_calculator, get_financial_calculator, technical_calculator

# Signal Processing & ML Analytics  
from .signal_processing import get_signal_classifier, signal_ml_analyzer

# Reddit Analytics
from .reddit import reddit_analytics

# AI Integration
from .ai import ai_analyzer

__all__ = [
    # Factory functions
    'get_technical_calculator',
    'get_financial_calculator', 
    'get_signal_classifier',
    
    # Direct instances
    'technical_calculator',
    'signal_ml_analyzer',
    'reddit_analytics',
    'ai_analyzer'
]