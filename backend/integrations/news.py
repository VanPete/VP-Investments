"""
News Integration (3.0 Architecture - Phase 1 & 3)
==================================================
Fetch news from Yahoo Finance, calculate sentiment

Phase 1: Fetch news headlines (yfinance .news attribute)
Phase 3: Calculate sentiment scores (TextBlob)

This is a placeholder until proper News API is integrated.
"""
import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from backend.utils.logger import get_logger
from backend.utils.metrics import emit_metric

logger = get_logger(__name__)


@dataclass
class NewsArticle:
    """Single news article"""
    title: str
    publisher: str
    link: str
    published_at: str
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"


@dataclass
class NewsBundle:
    """News data bundle (Phase 1)"""
    ticker: str
    articles: List[NewsArticle]
    news_sentiment_score: Optional[float]
    news_mentions: int
    top_headlines: List[str]
    available: bool = True
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "yfinance"


class NewsFetcher:
    """Phase 1: Fetch news from Yahoo Finance"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and YFINANCE_AVAILABLE and TEXTBLOB_AVAILABLE
        
        # Initialize VADER analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Common ticker symbols for pattern matching
        self._common_tickers = None
        
        if not self.enabled:
            if not YFINANCE_AVAILABLE:
                logger.warning("yfinance not available - news disabled")
            if not TEXTBLOB_AVAILABLE:
                logger.warning("textblob not available - sentiment disabled")
        else:
            logger.info("News fetcher initialized (Yahoo Finance)")
            if VADER_AVAILABLE:
                logger.info("VADER sentiment analyzer initialized")
            else:
                logger.warning("VADER not available - using TextBlob only")
    
    async def get_trending_tickers_from_news(self, 
                                            top_n: int = 50,
                                            min_mentions: int = 2) -> Dict[str, int]:
        """
        Discover trending tickers from Yahoo Finance news.
        
        Fetches general market news and extracts ticker mentions to expand
        the ticker universe beyond Reddit-discovered stocks.
        
        Args:
            top_n: Number of top mentioned tickers to return
            min_mentions: Minimum mentions required to include ticker
        
        Returns:
            Dict of ticker -> mention_count for trending tickers
        """
        if not self.enabled:
            logger.info("News fetcher disabled - skipping ticker discovery")
            return {}
        
        try:
            logger.info("Discovering trending tickers from news...")
            emit_metric("news.discovery.start", 1)
            
            # Fetch general market news (top indices/sectors)
            market_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', '^GSPC', '^DJI', '^IXIC']
            all_articles = []
            
            for symbol in market_symbols:
                try:
                    ticker_obj = yf.Ticker(symbol)
                    news = ticker_obj.news
                    if news:
                        all_articles.extend(news[:10])  # Top 10 from each
                except Exception as e:
                    logger.debug(f"Failed to fetch news for {symbol}: {e}")
                    continue
            
            if not all_articles:
                logger.warning("No news articles found for ticker discovery")
                return {}
            
            # Extract ticker mentions from headlines
            ticker_mentions = {}
            import re
            
            # Pattern: Match $TICKER or just TICKER (2-5 uppercase letters)
            ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?=\b)')
            
            # Load common tickers for validation (lazy load)
            if self._common_tickers is None:
                self._common_tickers = self._load_common_tickers()
            
            for article in all_articles:
                # YFinance news structure: article['content']['title'] and article['content']['summary']
                content = article.get('content', {})
                title = content.get('title', '')
                summary = content.get('summary', '')
                
                # Combine title and summary for extraction
                text = f"{title} {summary}"
                
                # Find all ticker-like patterns
                matches = ticker_pattern.findall(text)
                
                for match in matches:
                    # match is tuple: ($TICKER, TICKER), one will be empty
                    ticker = match[0] or match[1]
                    
                    # Filter out common words that look like tickers
                    if ticker in ['A', 'I', 'IT', 'AT', 'TO', 'GO', 'NOW', 'NEW', 
                                 'ALL', 'CEO', 'CFO', 'IPO', 'ETF', 'S&P', 'DOW',
                                 'NASDAQ', 'NYSE', 'SEC', 'FED', 'GDP', 'CPI']:
                        continue
                    
                    # If we have a validation list, check against it
                    if self._common_tickers and ticker not in self._common_tickers:
                        continue
                    
                    ticker_mentions[ticker] = ticker_mentions.get(ticker, 0) + 1
            
            # Filter by minimum mentions and sort
            filtered_mentions = {
                ticker: count 
                for ticker, count in ticker_mentions.items() 
                if count >= min_mentions
            }
            
            # Sort by mention count and take top N
            sorted_tickers = sorted(
                filtered_mentions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            result = dict(sorted_tickers)
            
            logger.info(
                f"Discovered {len(result)} trending tickers from news "
                f"(scanned {len(all_articles)} articles)"
            )
            emit_metric("news.discovery.success", 1, tags={'tickers': len(result)})
            
            return result
            
        except Exception as e:
            logger.error(f"News ticker discovery error: {e}")
            emit_metric("news.discovery.error", 1)
            return {}
    
    def _load_common_tickers(self) -> set:
        """
        Load common ticker symbols for validation.
        
        Returns a set of known tickers to filter out false positives.
        Expanded list to catch more trending stocks from news.
        """
        # Comprehensive ticker list for news discovery
        common_tickers = {
            # Major tech (FAANG+)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
            'NFLX', 'AMD', 'INTC', 'AVGO', 'ORCL', 'CSCO', 'ADBE', 'CRM',
            'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'MU', 'SNPS',
            
            # Finance
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
            'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT', 'SPGI', 'CME', 'ICE',
            
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'ABT',
            'DHR', 'BMY', 'AMGN', 'GILD', 'REGN', 'VRTX', 'ISRG', 'CI',
            
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST',
            'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'EL', 'MDLZ',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX',
            'VLO', 'OXY', 'HAL', 'DVN', 'FANG', 'HES', 'MRO', 'APA',
            
            # Industrials
            'BA', 'CAT', 'UPS', 'RTX', 'GE', 'MMM', 'HON', 'LMT',
            'DE', 'GD', 'NOC', 'UNP', 'CSX', 'NSC', 'FDX', 'DAL',
            
            # Materials & Basic Industries
            'LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'NUE', 'STLD',
            'VMC', 'MLM', 'MP', 'GLDD', 'CLF', 'X', 'AA', 'RS',
            
            # Semiconductors (expanded)
            'TSM', 'ASML', 'ARM', 'MCHP', 'NXPI', 'ADI', 'ON', 'MPWR',
            
            # Uranium & Nuclear
            'UEC', 'CCJ', 'UUUU', 'DNN', 'URG', 'SMR', 'NXE', 'EU',
            
            # Popular meme/growth
            'GME', 'AMC', 'PLTR', 'SOFI', 'RIVN', 'LCID', 'NIO', 'XPEV',
            'COIN', 'HOOD', 'SQ', 'PYPL', 'SHOP', 'SNAP', 'UBER', 'LYFT',
            'DASH', 'ABNB', 'RBLX', 'U', 'SNOW', 'DKNG', 'OPEN', 'WISH',
            
            # Cloud & Software
            'NOW', 'WDAY', 'TEAM', 'ZS', 'CRWD', 'DDOG', 'NET', 'S',
            'OKTA', 'MDB', 'DOCU', 'ZM', 'TWLO', 'FSLY', 'ESTC', 'AI',
            
            # EV & Auto
            'TSLA', 'F', 'GM', 'RIVN', 'LCID', 'FSR', 'GOEV', 'RIDE',
            'TM', 'HMC', 'STLA', 'VWAGY',
            
            # Retail & E-commerce
            'AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'BABA', 'JD',
            'PDD', 'MELI', 'SE', 'CPNG', 'ETSY', 'W', 'CHWY', 'CVNA',
            
            # Biotech
            'MRNA', 'BNTX', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'ALNY',
            'CRSP', 'EDIT', 'NTLA', 'BEAM', 'EXAS', 'TDOC', 'TDOC',
            
            # Communications
            'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'DIS', 'NFLX', 'PARA',
            
            # REITs
            'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'DLR', 'O', 'SPG',
            
            # ETFs (major)
            'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'ARKK', 'ARKG',
            'ARKF', 'ARKW', 'IVV', 'VEA', 'VWO', 'AGG', 'BND', 'TLT',
            
            # China Tech
            'BABA', 'JD', 'BIDU', 'PDD', 'BEKE', 'TME', 'BILI', 'XPEV',
            
            # Crypto-related
            'COIN', 'MARA', 'RIOT', 'HUT', 'BITF', 'MSTR', 'SI', 'SQ',
            
            # SPACs & Recent IPOs
            'SPCE', 'OPEN', 'SKLZ', 'CLOV', 'WISH', 'BODY', 'BARK',
            
            # Defense
            'LMT', 'RTX', 'BA', 'NOC', 'GD', 'LHX', 'HII', 'TXT',
            
            # Banks (regional)
            'KEY', 'FITB', 'HBAN', 'RF', 'CFG', 'ZION', 'MTB', 'SIVB',
        }
        
        return common_tickers
    
    async def fetch_news_bundle(self, ticker: str, lookback_days: int = 7) -> NewsBundle:
        """Fetch news bundle for ticker (Phase 1)"""
        if not self.enabled:
            return self._empty_bundle(ticker, "News fetching disabled")
        
        try:
            logger.debug(f"Fetching news for {ticker}")
            emit_metric("news.fetch.start", 1, tags={'ticker': ticker})
            
            # Fetch from yfinance
            ticker_obj = yf.Ticker(ticker)
            raw_news = ticker_obj.news
            
            if not raw_news:
                return self._empty_bundle(ticker, "No news available")
            
            # Filter by lookback period
            cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            cutoff_unix = int(cutoff_timestamp.timestamp())
            
            filtered_news = [
                article for article in raw_news 
                if article.get('providerPublishTime', 0) >= cutoff_unix
            ]
            
            if not filtered_news:
                return self._empty_bundle(ticker, f"No news in last {lookback_days} days")
            
            # Process articles
            articles = await self._process_articles(filtered_news)
            
            # Calculate aggregate sentiment
            avg_sentiment = sum(a.sentiment_score for a in articles) / len(articles) if articles else 0.0
            
            # Extract top headlines
            top_headlines = [a.title for a in articles[:5]]
            
            bundle = NewsBundle(
                ticker=ticker,
                articles=articles,
                news_sentiment_score=avg_sentiment,
                news_mentions=len(articles),
                top_headlines=top_headlines,
                available=True,
                source="yfinance"
            )
            
            logger.info(f"Fetched {len(articles)} news for {ticker}, sentiment: {avg_sentiment:.3f}")
            emit_metric("news.fetch.success", 1, tags={'ticker': ticker})
            
            return bundle
            
        except Exception as e:
            logger.error(f"News fetch error for {ticker}: {e}")
            emit_metric("news.fetch.error", 1, tags={'ticker': ticker})
            return self._empty_bundle(ticker, f"Error: {str(e)}")
    
    async def _process_articles(self, raw_articles: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Process raw articles and calculate sentiment"""
        articles = []
        
        for raw in raw_articles:
            try:
                # YFinance news structure: raw['content'] contains the actual data
                content = raw.get('content', {})
                title = content.get('title', '')
                
                if not title:
                    continue
                
                provider = content.get('provider', {})
                publisher = provider.get('displayName', 'Unknown')
                
                click_through = content.get('clickThroughUrl', {})
                link = click_through.get('url', '')
                
                # Parse pubDate
                pub_date_str = content.get('pubDate', '')
                if pub_date_str:
                    try:
                        pub_datetime = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        pub_str = pub_datetime.isoformat()
                    except:
                        pub_str = pub_date_str
                else:
                    pub_str = datetime.now(timezone.utc).isoformat()
                
                sentiment_score, sentiment_label = self._analyze_sentiment(title)
                
                article = NewsArticle(
                    title=title,
                    publisher=publisher,
                    link=link,
                    published_at=pub_str,
                    sentiment_score=sentiment_score,
                    sentiment_label=sentiment_label
                )
                
                articles.append(article)
                
            except Exception as e:
                logger.debug(f"Article processing error: {e}")
                continue
        
        return articles
    
    def _analyze_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment using both TextBlob and VADER for better accuracy"""
        if not text:
            return (0.0, "neutral")
        
        scores = []
        
        # TextBlob analysis (general sentiment)
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                textblob_polarity = blob.sentiment.polarity
                scores.append(textblob_polarity)
            except Exception as e:
                logger.debug(f"TextBlob sentiment failed: {e}")
        
        # VADER analysis (social media/news optimized)
        if VADER_AVAILABLE and self.vader_analyzer:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                vader_compound = vader_scores['compound']  # Range: -1 to +1
                scores.append(vader_compound)
            except Exception as e:
                logger.debug(f"VADER sentiment failed: {e}")
        
        # Combine scores (average of available analyzers)
        if scores:
            polarity = sum(scores) / len(scores)
        else:
            return (0.0, "neutral")
        
        # Determine sentiment label
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return (polarity, label)
    
    def _empty_bundle(self, ticker: str, reason: str) -> NewsBundle:
        """Create empty news bundle"""
        return NewsBundle(
            ticker=ticker,
            articles=[],
            news_sentiment_score=None,
            news_mentions=0,
            top_headlines=[],
            available=False,
            source=f"placeholder: {reason}"
        )


class NewsSentimentCalculator:
    """Phase 3: Calculate news scores"""
    
    @staticmethod
    def calculate_news_score(bundle: NewsBundle) -> Dict[str, Any]:
        """Calculate news sentiment score (Phase 3)"""
        if not bundle.available or not bundle.articles:
            return {
                'news_score': 0.0,
                'news_sentiment_normalized': 0.5,
                'news_confidence': 0.0,
                'news_article_count': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
        
        positive = sum(1 for a in bundle.articles if a.sentiment_label == 'positive')
        negative = sum(1 for a in bundle.articles if a.sentiment_label == 'negative')
        neutral = sum(1 for a in bundle.articles if a.sentiment_label == 'neutral')
        total = len(bundle.articles)
        
        pos_ratio = positive / total if total > 0 else 0.0
        neg_ratio = negative / total if total > 0 else 0.0
        neu_ratio = neutral / total if total > 0 else 0.0
        
        raw_sentiment = bundle.news_sentiment_score or 0.0
        normalized_sentiment = (raw_sentiment + 1) / 2
        
        confidence = min(total / 10, 1.0)
        
        return {
            'news_score': raw_sentiment,
            'news_sentiment_normalized': normalized_sentiment,
            'news_confidence': confidence,
            'news_article_count': total,
            'positive_ratio': pos_ratio,
            'negative_ratio': neg_ratio,
            'neutral_ratio': neu_ratio
        }


def create_news_fetcher(enabled: bool = True) -> NewsFetcher:
    """Factory: Create news fetcher"""
    return NewsFetcher(enabled=enabled)


def get_news_fetcher() -> NewsFetcher:
    """Singleton: Get news fetcher"""
    if not hasattr(get_news_fetcher, '_instance'):
        enabled = os.getenv('DATA_SOURCES_NEWS_ENABLED', 'true').lower() == 'true'
        get_news_fetcher._instance = NewsFetcher(enabled=enabled)
    
    return get_news_fetcher._instance


async def fetch_news_bundle(ticker: str, **kwargs) -> Dict[str, Any]:
    """Legacy compatibility for pipeline"""
    fetcher = get_news_fetcher()
    lookback_days = kwargs.get('lookback_days', 7)
    
    bundle = await fetcher.fetch_news_bundle(ticker, lookback_days)
    
    return {
        'ticker': bundle.ticker,
        'news_sentiment_score': bundle.news_sentiment_score,
        'news_mentions': bundle.news_mentions,
        'top_headlines': bundle.top_headlines,
        'articles': [
            {
                'title': a.title,
                'publisher': a.publisher,
                'link': a.link,
                'published_at': a.published_at,
                'sentiment_score': a.sentiment_score,
                'sentiment_label': a.sentiment_label
            }
            for a in bundle.articles
        ],
        'available': bundle.available,
        'fetched_at': bundle.fetched_at,
        'source': bundle.source
    }