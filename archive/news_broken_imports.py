"""
VP Investments News Integration
==============================

News data integration with API rate limiting and toggle control.
Integrates with existing news sentiment analysis module.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio

from vp_investments.analysis.news_sentiment import NewsSentimentAnalyzer, NewsArticle
from vp_investments.core.config import ConfigManager

logger = logging.getLogger(__name__)


class NewsIntegrator:
    """
    News data integrator with toggle control and rate limiting.
    Wraps the existing news sentiment analyzer with pipeline integration.
    """
    
    def __init__(self, enabled: bool = None):
        self.config = ConfigManager()
        
        # Check if news integration is enabled (default from env or parameter)
        if enabled is None:
            enabled = os.getenv('DATA_SOURCES_NEWS_ENABLED', 'false').lower() == 'true'
        
        self.enabled = enabled
        
        if self.enabled:
            try:
                self.news_analyzer = NewsSentimentAnalyzer()
                logger.info("News integration initialized and enabled")
            except Exception as e:
                logger.warning(f"News integration failed to initialize: {e}")
                self.enabled = False
                self.news_analyzer = None
        else:
            logger.info("News integration disabled by configuration")
            self.news_analyzer = None
    
    async def get_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get news sentiment data for a ticker.
        
        Returns:
            Dict with news_score, news_sentiment, news_mentions, ai_news_summary
        """
        if not self.enabled or not self.news_analyzer:
            return {
                'news_score': None,
                'news_sentiment': None, 
                'news_mentions': 0,
                'ai_news_summary': None
            }
        
        try:
            # Use existing news sentiment analyzer
            news_data = await self.news_analyzer.analyze_ticker_sentiment(ticker)
            
            if news_data and 'articles' in news_data:
                articles = news_data['articles']
                
                # Calculate aggregate metrics
                news_score = news_data.get('overall_sentiment_score', 0)
                news_sentiment = self._classify_sentiment(news_score)
                news_mentions = len(articles)
                
                # Generate summary from top articles
                ai_news_summary = self._generate_news_summary(articles[:3])
                
                return {
                    'news_score': news_score,
                    'news_sentiment': news_sentiment,
                    'news_mentions': news_mentions, 
                    'ai_news_summary': ai_news_summary
                }
            
            return {
                'news_score': 0.0,
                'news_sentiment': 'neutral',
                'news_mentions': 0,
                'ai_news_summary': None
            }
            
        except Exception as e:
            logger.warning(f"Error getting news sentiment for {ticker}: {e}")
            return {
                'news_score': None,
                'news_sentiment': None,
                'news_mentions': 0, 
                'ai_news_summary': None
            }
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify numerical sentiment score into categories"""
        if score >= 0.1:
            return 'positive'
        elif score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _generate_news_summary(self, articles: List[NewsArticle]) -> Optional[str]:
        """Generate a brief summary from top news articles"""
        if not articles:
            return None
        
        try:
            # Simple summary from top article titles and summaries
            summaries = []
            for article in articles[:3]:  # Top 3 articles
                if hasattr(article, 'title') and hasattr(article, 'summary'):
                    title = article.title[:100] if article.title else ""
                    summary = article.summary[:200] if article.summary else ""
                    if title or summary:
                        summaries.append(f"{title}: {summary}".strip(": "))
            
            if summaries:
                return " | ".join(summaries)[:500]  # Limit to 500 chars
            
        except Exception as e:
            logger.warning(f"Error generating news summary: {e}")
        
        return None


# Singleton instance for reuse
_news_integrator = None

def get_news_integrator() -> NewsIntegrator:
    """Get singleton news integrator instance"""
    global _news_integrator
    if _news_integrator is None:
        _news_integrator = NewsIntegrator()
    return _news_integrator


async def test_news_integration():
    """Test news integration functionality"""
    news = get_news_integrator()
    
    if not news.enabled:
        print("❌ News integration is disabled")
        return
    
    print("🧪 Testing news integration...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n📰 Testing {ticker}...")
        result = await news.get_news_sentiment(ticker)
        print(f"  News Score: {result['news_score']}")
        print(f"  Sentiment: {result['news_sentiment']}")  
        print(f"  Mentions: {result['news_mentions']}")
        if result['ai_news_summary']:
            print(f"  Summary: {result['ai_news_summary'][:100]}...")



# ============================================================================
# 3.0 CACHE-COMPATIBLE METHODS (Phase 1 Integration - PLACEHOLDER)
# ============================================================================

async def fetch_news_bundle(ticker: str, **kwargs) -> Dict[str, Any]:
    """
    Phase 1 compatible: Fetch news/macro data bundle for a ticker.
    
    ⚠️ PLACEHOLDER: News API integration pending.
    Returns empty but valid structure for now.
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: Days of news history (default: 7)
        
    Returns:
        {
            "news_sentiment_score": None,  # 0-1 scale
            "news_mentions": 0,
            "news_sources": [],
            "top_headlines": [],
            "macro_sentiment_score": None,
            "sector_news_sentiment": None,
            "economic_indicator_score": None,
            "available": False,
            "metadata": {...}
        }
    """
    from datetime import datetime, timezone
    
    lookback_days = kwargs.get('lookback_days', 7)
    
    logger.info(f"⚠️  Phase 1: News API placeholder for {ticker} (integration pending)")
    
    # Try to use existing news integrator if available
    try:
        news_integrator = get_news_integrator()
        if news_integrator.enabled:
            result = await news_integrator.get_news_sentiment(ticker)
            return {
                "news_sentiment_score": result.get('news_score'),
                "news_mentions": result.get('news_mentions', 0),
                "news_sources": [],
                "top_headlines": [],
                "macro_sentiment_score": None,
                "sector_news_sentiment": None,
                "economic_indicator_score": None,
                "available": True,
                "metadata": {
                    "ticker": ticker,
                    "lookback_days": lookback_days,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "source": "existing_analyzer"
                }
            }
    except Exception as e:
        logger.debug(f"Existing news analyzer unavailable: {e}")
    
    # Return placeholder
    return {
        "news_sentiment_score": None,
        "news_mentions": 0,
        "news_sources": [],
        "top_headlines": [],
        "macro_sentiment_score": None,
        "sector_news_sentiment": None,
        "economic_indicator_score": None,
        "available": False,
        "metadata": {
            "ticker": ticker,
            "lookback_days": lookback_days,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "status": "placeholder",
            "message": "News API integration pending"
        }
    }


def get_news_fetcher():
    """
    Factory function for 3.0 pipeline.
    Returns news integrator (placeholder mode).
    """
    return get_news_integrator()


if __name__ == "__main__":
    asyncio.run(test_news_integration())