# VP Investments Integrations Module
"""
VP Investments integrations for data sources and signal processing.
Consolidated structure following domain-based organization.

Note: This module exists for backwards compatibility but direct imports
from submodules (reddit, yfinance, ai, etc.) are preferred.
"""

# Available integrations:
# - backend.integrations.reddit (RedditFetcher)
# - backend.integrations.yfinance (YFinanceFetcher)
# - backend.integrations.ai (AIAnalyzer)
# - backend.integrations.news (NewsFetcher)
# - backend.integrations.cache (CacheManager)
# - backend.integrations.backtest (BacktestEngine)

__all__ = []  # Direct imports from submodules are preferred