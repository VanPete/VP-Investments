"""
Phase 1: Fetch & Cache
=======================

All API calls happen here - ONLY in Phase 1!

This module is responsible for:
- Reddit data fetching
- Yahoo Finance data fetching
- News data fetching (if available)
- Caching all data for downstream phases

NO API calls should happen after Phase 1 completes.
"""

import os
import logging
import asyncio
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Suppress PRAW async environment warnings
warnings.filterwarnings('ignore', message='.*PRAW.*asynchronous environment.*')
logging.getLogger('praw').setLevel(logging.ERROR)  # Only show PRAW errors, not warnings

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Centralized Reddit Settings
# ============================================================================
DEFAULT_SUBREDDITS = [
    # Core - Broad Market Discussion
    'stocks', 
    'investing', 
    'StockMarket', 
    'options',
    
    # Quality Fundamental Analysis
    'SecurityAnalysis',    # Deep fundamental analysis
    'ValueInvesting',      # Long-term value investing
    'dividends',          # Income-focused investing
    
    # Trading & Technical Analysis
    'Daytrading',         # Short-term momentum
    'SwingTrading',       # Medium-term technical
    
    # Sector-Specific
    'technologystock',    # Tech sector analysis
    'biotech_stocks',     # Biotech/pharma
    'RenewableEnergy',    # ESG/clean energy
]

class Phase1Fetcher:
    """
    Phase 1: Fetch & Cache
    
    Fetches all required data from external APIs and caches it for
    downstream phases. This is the ONLY place API calls should occur.
    """
    
    def __init__(self):
        """Initialize Phase 1 fetcher with API clients."""
        self.logger = logger
        self._init_clients()
        self._init_ticker_validation()
    
    def _init_clients(self):
        """Initialize API clients for data fetching."""
        # Reddit client
        try:
            import praw
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'VP_Investments_Bot/1.0')
            )
            self.logger.info("[SUCCESS] Reddit client initialized")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize Reddit client: {e}")
            self.reddit = None
        
        # Yahoo Finance - use new comprehensive fetcher
        try:
            from backend.integrations.yfinance import get_yfinance_fetcher
            self.yfinance_fetcher = get_yfinance_fetcher()
            self.logger.info("[SUCCESS] YFinance v3.1 comprehensive fetcher initialized")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize YFinance fetcher: {e}")
            self.yfinance_fetcher = None
    
    def _init_ticker_validation(self):
        """Initialize ticker validation with local CSV of valid tickers and company names."""
        try:
            import csv
            from pathlib import Path
            
            # Path to CSV file in backend/core/nyse.csv
            csv_path = Path(__file__).parent.parent / 'core' / 'nyse.csv'
            
            if not csv_path.exists():
                self.logger.warning(f"CSV file not found at {csv_path} - using regex-only ticker extraction")
                self.valid_tickers = set()
                self.company_to_ticker = {}
                self.ticker_metadata = {}
                return
            
            # Load tickers from CSV
            self.logger.info(f"Loading valid tickers from CSV: {csv_path}")
            
            all_data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # CSV columns: Symbol, Name, LastSale, MarketCap, IPOyear, Sector, industry
                    if row.get('Symbol'):
                        all_data.append({
                            'ticker': row['Symbol'],
                            'company_name': row.get('Name', ''),
                            'sector': row.get('Sector', ''),
                            'industry': row.get('industry', '')
                        })
            
            if all_data:
                # Create sets and maps for fast lookup
                self.valid_tickers = {row['ticker'].upper() for row in all_data if row.get('ticker')}
                
                # Map company names to tickers (for fuzzy matching)
                self.company_to_ticker = {}
                for row in all_data:
                    if row.get('company_name') and row.get('ticker'):
                        # Store both full name and common variations
                        company_name = row['company_name'].lower()
                        ticker = row['ticker'].upper()
                        self.company_to_ticker[company_name] = ticker
                        
                        # Add common variations (e.g., "Apple Inc." -> "Apple")
                        if company_name.endswith(' inc.'):
                            self.company_to_ticker[company_name.replace(' inc.', '')] = ticker
                        elif company_name.endswith(' corp.'):
                            self.company_to_ticker[company_name.replace(' corp.', '')] = ticker
                        elif company_name.endswith(' corporation'):
                            self.company_to_ticker[company_name.replace(' corporation', '')] = ticker
                        elif company_name.endswith(' ltd.'):
                            self.company_to_ticker[company_name.replace(' ltd.', '')] = ticker
                        elif company_name.endswith(', inc.'):
                            self.company_to_ticker[company_name.replace(', inc.', '')] = ticker
                
                # Store metadata for context-aware validation
                self.ticker_metadata = {
                    row['ticker'].upper(): {
                        'company_name': row.get('company_name'),
                        'sector': row.get('sector'),
                        'industry': row.get('industry')
                    }
                    for row in all_data if row.get('ticker')
                }
                
                self.logger.info(f"[SUCCESS] Loaded {len(self.valid_tickers)} valid tickers from CSV")
                self.logger.info(f"[SUCCESS] Built company name map with {len(self.company_to_ticker)} entries")
            else:
                self.logger.warning("No tickers found in CSV - using regex-only extraction")
                self.valid_tickers = set()
                self.company_to_ticker = {}
                self.ticker_metadata = {}
                
        except Exception as e:
            self.logger.error(f"Failed to load ticker CSV: {e}")
            self.logger.warning("Falling back to regex-only ticker extraction")
            self.valid_tickers = set()
            self.company_to_ticker = {}
            self.ticker_metadata = {}
    
    async def fetch_all_data(self, 
                            tickers: Optional[List[str]] = None,
                            subreddits: Optional[List[str]] = None,
                            post_limit: int = 100) -> Dict[str, Any]:
        """
        Phase 1: Fetch & Cache - Main entry point (v3.1)
        
        FLOW: Reddit → News → YFinance (comprehensive)
        
        1. Fetch Reddit data to discover trending tickers
        2. Fetch news sentiment for discovered tickers
        3. Fetch comprehensive YFinance data (40+ endpoints)
        
        Args:
            tickers: Optional pre-defined ticker list (if None, discovers from Reddit)
            subreddits: List of subreddits to scrape (default: stocks, investing, wallstreetbets)
            post_limit: Number of posts to fetch per subreddit
        
        Returns:
            Dict containing:
                - reddit_data: Reddit mentions and sentiment
                - news_data: News sentiment for discovered tickers
                - raw_cache_by_ticker: Dict[ticker, RawYFinanceData] - comprehensive data
                - discovered_tickers: List of tickers from Reddit
                - metadata: Fetch statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 1: FETCH & CACHE (v3.1 - Comprehensive YFinance Integration)")
        self.logger.info("=" * 80)
        
        phase1_start = datetime.now()
        
        # Use default subreddits if none provided
        if not subreddits:
            subreddits = DEFAULT_SUBREDDITS
        
        # STEP 1: Fetch Reddit data to discover tickers
        self.logger.info(f"\n[REDDIT] Step 1.1: Fetching Reddit data from {len(subreddits)} subreddits...")
        reddit_data = await self._fetch_reddit_data(subreddits, post_limit)
        
        # Extract discovered tickers from Reddit (sorted by mentions)
        discovered_tickers = []
        if reddit_data and 'ticker_mentions' in reddit_data:
            ticker_mentions = reddit_data['ticker_mentions']
            # Sort by mention count
            sorted_tickers = sorted(
                ticker_mentions.items(), 
                key=lambda x: x[1].get('mentions', 0), 
                reverse=True
            )
            discovered_tickers = [t[0] for t in sorted_tickers]
            self.logger.info(f"   [SUCCESS] Discovered {len(discovered_tickers)} tickers from Reddit")
        
        # STEP 1.5: Discover trending tickers from news (NEW!)
        self.logger.info(f"\n[NEWS] Step 1.2: Discovering trending tickers from news...")
        news_tickers = await self._discover_tickers_from_news()
        
        if news_tickers:
            self.logger.info(f"   [SUCCESS] Discovered {len(news_tickers)} trending tickers from news")
            # Merge news-discovered tickers with Reddit tickers
            combined_discovered = list(set(discovered_tickers + list(news_tickers.keys())))
            self.logger.info(f"   [INFO] Combined universe: {len(combined_discovered)} unique tickers")
        else:
            combined_discovered = discovered_tickers
            self.logger.info(f"   [INFO] No tickers from news, using {len(combined_discovered)} from Reddit")
        
        # Merge with pre-defined tickers if provided
        if tickers:
            all_tickers = list(set(tickers + combined_discovered))
            self.logger.info(f"   [INFO] Merged with {len(tickers)} pre-defined tickers -> {len(all_tickers)} total")
        else:
            all_tickers = combined_discovered
        
        # STEP 2: Fetch news sentiment for all tickers
        self.logger.info(f"\n[NEWS] Step 1.3: Fetching news sentiment for {len(all_tickers)} tickers...")
        news_data = await self._fetch_news_data(all_tickers)
        
        # STEP 3: Fetch comprehensive YFinance data (40+ endpoints per ticker)
        self.logger.info(f"\n[STATS] Step 1.3: Fetching comprehensive YFinance data (v3.1)...")
        self.logger.info(f"   Coverage: 40+ endpoints per ticker")
        self.logger.info(f"   - Stock/Meta/News: info, history, news, dividends, splits, etc.")
        self.logger.info(f"   - Financials/Events: statements, earnings, calendar, SEC filings")
        self.logger.info(f"   - Analysis/Holdings: recommendations, estimates, ownership, insiders")
        
        raw_cache_by_ticker = await self._fetch_comprehensive_yfinance_data(all_tickers)
        
        # STEP 4: Fetch market-wide data (SPY, VIX, Treasury yields, etc.)
        self.logger.info(f"\n[MARKET] Step 1.4: Fetching market-wide data (SPY, VIX, Treasuries)...")
        market_data = None
        if self.yfinance_fetcher:
            try:
                # Fetch 2 years of SPY history to match stock history period (needed for downside_capture_1y)
                market_data = self.yfinance_fetcher.fetch_market_data(period='2y')
                if market_data and market_data.is_valid():
                    self.logger.info(f"   [SUCCESS] Market data fetched")
                    if market_data.vix_current:
                        self.logger.info(f"      VIX: {market_data.vix_current:.2f}")
                    if market_data.treasury_yield_10y:
                        self.logger.info(f"      10Y Treasury: {market_data.treasury_yield_10y:.2f}%")
                    if market_data.spy_history is not None and not market_data.spy_history.empty:
                        self.logger.info(f"      SPY history: {len(market_data.spy_history)} days")
                else:
                    self.logger.warning("   [WARNING] Market data fetch returned invalid data")
            except Exception as e:
                self.logger.error(f"   [ERROR] Failed to fetch market data: {e}")
        else:
            self.logger.warning("   [WARNING] YFinance fetcher not initialized")
        
        # STEP 5: Fetch sector ETF data for sector-relative performance tracking
        self.logger.info(f"\n[SECTOR] Step 1.5: Fetching sector ETF data...")
        sector_etf_data = await self._fetch_sector_etf_data(raw_cache_by_ticker)
        self.logger.info(f"   [SUCCESS] Fetched {len(sector_etf_data)} sector ETFs")
        
        # VALIDATION: Validate fetched data before returning
        validated_cache = self._validate_fetched_data(raw_cache_by_ticker)
        
        phase1_end = datetime.now()
        execution_time = (phase1_end - phase1_start).total_seconds()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"[SUCCESS] PHASE 1 COMPLETE - {execution_time:.2f}s")
        self.logger.info(f"   Reddit: {len(reddit_data.get('ticker_mentions', {}))} tickers discovered")
        self.logger.info(f"   News: {len(news_tickers)} tickers discovered from news")
        self.logger.info(f"   News Sentiment: {len(news_data)} tickers with sentiment data")
        self.logger.info(f"   YFinance: {len(validated_cache)} tickers with comprehensive data")
        self.logger.info(f"   Sector ETFs: {len(sector_etf_data)} ETFs fetched")
        self.logger.info("=" * 80)
        
        return {
            'reddit_data': reddit_data,
            'news_data': news_data,
            'raw_cache_by_ticker': validated_cache,  # NEW: RawYFinanceData objects (validated)
            'market_data': market_data,  # NEW: Market-wide data (SPY, VIX, Treasuries)
            'sector_etf_data': sector_etf_data,  # NEW v3.2: Sector ETF historical data
            'discovered_tickers': discovered_tickers,
            'news_discovered_tickers': list(news_tickers.keys()) if news_tickers else [],
            'all_tickers': all_tickers,
            'metadata': {
                'phase': 'Phase 1: Fetch & Cache v3.2',
                'execution_time': execution_time,
                'tickers_count': len(all_tickers),
                'tickers_discovered': len(discovered_tickers),
                'subreddits': subreddits,
                'timestamp': phase1_end.isoformat(),
                'yfinance_version': '3.2_sector_performance'
            }
        }
    
    def _validate_fetched_data(self, raw_cache: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate fetched YFinance data before passing to Phase 2.
        
        Checks:
        - Critical fields present (info, fast_info, history)
        - Price history has minimum rows
        - Data structures are valid
        
        Args:
            raw_cache: Dictionary of ticker -> RawYFinanceData
            
        Returns:
            Validated cache (removing invalid tickers)
        """
        validated = {}
        removed_count = 0
        
        for ticker, data in raw_cache.items():
            is_valid = True
            warnings = []
            
            # Check critical fields
            if data.info is None or len(data.info) == 0:
                self.logger.warning(f"[VALIDATION] {ticker}: Missing 'info' data")
                warnings.append("missing_info")
                is_valid = False
            
            if data.fast_info is None or len(data.fast_info) == 0:
                self.logger.warning(f"[VALIDATION] {ticker}: Missing 'fast_info' data")
                warnings.append("missing_fast_info")
            
            # Check price history
            if data.history is None or data.history.empty:
                self.logger.warning(f"[VALIDATION] {ticker}: Missing or empty price history")
                warnings.append("missing_history")
                is_valid = False
            elif len(data.history) < 5:
                self.logger.warning(f"[VALIDATION] {ticker}: Insufficient price history ({len(data.history)} rows, need 5+)")
                warnings.append("insufficient_history")
                is_valid = False
            
            # Log optional data availability
            optional_checks = {
                'income_stmt': not (data.income_stmt is None or data.income_stmt.empty),
                'balance_sheet': not (data.balance_sheet is None or data.balance_sheet.empty),
                'cashflow': not (data.cashflow is None or data.cashflow.empty),
                'analyst_price_targets': not (data.analyst_price_targets is None or data.analyst_price_targets.empty),
                'institutional_holders': not (data.institutional_holders is None or data.institutional_holders.empty)
            }
            
            missing_optional = [k for k, v in optional_checks.items() if not v]
            if missing_optional and len(missing_optional) > 3:
                self.logger.debug(f"[VALIDATION] {ticker}: Missing {len(missing_optional)} optional datasets")
            
            if is_valid:
                validated[ticker] = data
                if warnings:
                    self.logger.debug(f"[VALIDATION] {ticker}: Valid but with warnings: {', '.join(warnings)}")
            else:
                removed_count += 1
                self.logger.warning(f"[VALIDATION] {ticker}: INVALID - Removed from dataset")
        
        if removed_count > 0:
            self.logger.warning(f"[VALIDATION] Removed {removed_count} invalid tickers from cache")
        
        self.logger.info(f"[VALIDATION] {len(validated)}/{len(raw_cache)} tickers passed validation")
        
        return validated
    
    async def _fetch_sector_etf_data(self, raw_cache_by_ticker: Dict) -> Dict[str, Any]:
        """
        Fetch historical data for sector ETFs based on ticker sectors.
        
        Args:
            raw_cache_by_ticker: Dictionary of ticker -> YFinanceData
            
        Returns:
            Dictionary of ETF ticker -> historical price data (DataFrame)
        """
        from backend.utils.sector_etfs import get_sector_etf
        
        # Extract unique sectors from tickers
        sectors = set()
        for ticker_data in raw_cache_by_ticker.values():
            if ticker_data and ticker_data.info and hasattr(ticker_data.info, 'sector') and ticker_data.info.sector:
                sectors.add(ticker_data.info.sector)
        
        self.logger.info(f"   Discovered {len(sectors)} unique sectors: {sorted(sectors)}")
        
        # Map sectors to ETFs
        sector_etfs = {}
        for sector in sectors:
            etf = get_sector_etf(sector)
            if etf and etf != 'SPY':  # Skip SPY, already fetched in market_data
                sector_etfs[sector] = etf
        
        # Get unique ETF tickers
        unique_etfs = set(sector_etfs.values())
        self.logger.info(f"   Mapped to {len(unique_etfs)} unique sector ETFs: {sorted(unique_etfs)}")
        
        # Fetch historical data for each sector ETF
        etf_data = {}
        if self.yfinance_fetcher and unique_etfs:
            for etf in sorted(unique_etfs):
                try:
                    # Fetch 2 years of history to match SPY
                    import yfinance as yf
                    ticker_obj = yf.Ticker(etf)
                    history = ticker_obj.history(period='2y')
                    
                    if history is not None and not history.empty:
                        etf_data[etf] = history
                        self.logger.info(f"      {etf}: {len(history)} days")
                    else:
                        self.logger.warning(f"      {etf}: No data available")
                except Exception as e:
                    self.logger.error(f"      {etf}: Failed to fetch - {e}")
        
        return etf_data
    
    def _validate_ticker(self, ticker: str, context: str = "") -> bool:
        """
        Validate if a ticker is real using database lookup.
        
        Args:
            ticker: Ticker symbol to validate
            context: Surrounding text for context-aware validation
            
        Returns:
            True if ticker is valid, False otherwise
        """
        ticker = ticker.upper().strip()
        
        # If we have database, validate against it
        if self.valid_tickers:
            is_valid = ticker in self.valid_tickers
            
            # Additional context clues for ambiguous tickers
            if is_valid and len(ticker) == 2:
                # Two-letter tickers often conflict with common words
                # Require stronger evidence ($ prefix or stock-related terms nearby)
                context_lower = context.lower()
                stock_keywords = ['stock', 'share', 'buy', 'sell', 'call', 'put', 'option', 'position', 'trade']
                has_dollar = '$' + ticker in context
                has_keyword = any(kw in context_lower for kw in stock_keywords)
                
                if not (has_dollar or has_keyword):
                    return False
            
            return is_valid
        else:
            # Fallback: if no database, accept all tickers matching pattern
            return True
    
    def _fuzzy_match_companies(self, text: str, threshold: float = 85.0) -> List[tuple]:
        """
        Fuzzy match company names in text to tickers.
        
        Args:
            text: Text to search for company names
            threshold: Minimum similarity score (0-100)
            
        Returns:
            List of (ticker, company_name, score) tuples
        """
        if not self.company_to_ticker:
            return []
        
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            # Fallback to exact matching only
            self.logger.debug("rapidfuzz not installed - using exact matching only")
            text_lower = text.lower()
            matches = []
            for company_name, ticker in self.company_to_ticker.items():
                if company_name in text_lower:
                    matches.append((ticker, company_name, 100.0))
            return matches
        
        # Use rapidfuzz for fuzzy matching
        text_lower = text.lower()
        matches = []
        
        # Extract potential company name phrases (2-4 words)
        words = text_lower.split()
        for i in range(len(words)):
            for length in [2, 3, 4]:  # Try 2, 3, and 4-word phrases
                if i + length <= len(words):
                    phrase = ' '.join(words[i:i+length])
                    
                    # Find best match in company names
                    result = process.extractOne(
                        phrase,
                        self.company_to_ticker.keys(),
                        scorer=fuzz.ratio,
                        score_cutoff=threshold
                    )
                    
                    if result:
                        company_name, score, _ = result
                        ticker = self.company_to_ticker[company_name]
                        matches.append((ticker, company_name, score))
        
        # Remove duplicates, keep highest score for each ticker
        unique_matches = {}
        for ticker, company_name, score in matches:
            if ticker not in unique_matches or score > unique_matches[ticker][2]:
                unique_matches[ticker] = (ticker, company_name, score)
        
        return list(unique_matches.values())
    
    async def _fetch_reddit_data(self, subreddits: List[str], post_limit: int) -> Dict[str, Any]:
        """
        Fetch Reddit data by scraping subreddits for ticker mentions.
        
        IMPROVEMENTS v2.0:
        - ✅ Database-backed ticker validation (eliminates false positives)
        - ✅ Fuzzy matching for company names (e.g., "Apple" → AAPL)
        - ✅ Context-aware validation for ambiguous tickers
        - Enhanced blacklist (crypto, slang, common words)
        - Minimum ticker length (2+ chars)
        - Better spam filtering (min upvotes threshold)
        - Advanced sentiment analysis (TextBlob/VADER if available)
        - Post comment analysis for deeper sentiment
        
        Returns:
            Dict with ticker_mentions and metadata
        """
        try:
            import re
            
            # Try to import sentiment analysis libraries
            try:
                from textblob import TextBlob
                TEXTBLOB_AVAILABLE = True
            except ImportError:
                TEXTBLOB_AVAILABLE = False
            
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                vader_analyzer = SentimentIntensityAnalyzer()
                VADER_AVAILABLE = True
            except ImportError:
                VADER_AVAILABLE = False
                vader_analyzer = None
            
            if VADER_AVAILABLE:
                self.logger.info("Using VADER sentiment analysis")
            elif TEXTBLOB_AVAILABLE:
                self.logger.info("Using TextBlob sentiment analysis")
            else:
                self.logger.info("Using simple sentiment analysis (no NLP libraries)")
            
            if not self.reddit:
                self.logger.warning("Reddit client not available - using fallback data")
                return {
                    'ticker_mentions': {
                        'SPY': {'mentions': 10, 'sentiment': 0.0, 'upvotes': 100},
                    },
                    'metadata': {'error': 'Reddit client unavailable', 'fallback': True}
                }
            
            self.logger.info(f"Scraping Reddit from {len(subreddits)} subreddits (limit: {post_limit} posts each)...")
            
            ticker_data = {}
            total_posts = 0
            total_mentions = 0
            filtered_spam = 0
            
            # Ticker pattern: 2-5 uppercase letters (not 1, to avoid single letter words)
            ticker_pattern = re.compile(r'\b[A-Z]{2,5}\b')
            
            # Enhanced blacklist - common words, crypto, Reddit slang
            blacklist = {
                # Common English words
                'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
                'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW',
                'ITS', 'MAY', 'NEW', 'NOW', 'ONLY', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY',
                'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'WANT', 'WELL', 'VERY',
                'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WILL', 'WITH', 'WOULD', 'MAKE', 'BEEN',
                'GOOD', 'MUCH', 'SOME', 'TIME', 'OVER', 'JUST', 'LIKE', 'THINK', 'INTO',
                'YEAR', 'YOUR', 'KNOW', 'TAKE', 'THAN', 'FIRST', 'WATER', 'OTHER', 'PEOPLE',
                'COULD', 'THESE', 'THEIR', 'EVERY', 'GREAT', 'AFTER', 'NEVER', 'THROUGH',
                'BEING', 'BEFORE', 'ANOTHER', 'STILL', 'SMALL', 'FOUND', 'PLACE', 'WHILE',
                'ASKED', 'GOING', 'WORK', 'THREE', 'WORD', 'MUST', 'DOES', 'PART', 'BECAUSE',
                'EVEN', 'BACK', 'SAID', 'EACH', 'TELL', 'HAND', 'HIGH', 'AWAY', 'LAST',
                'NAME', 'ALSO', 'MADE', 'MOST', 'READ', 'NEXT', 'USED', 'CITY', 'BOTH',
                'KEEP', 'GAVE', 'SHOW', 'HELP', 'CALL', 'MOVE', 'LIVE', 'LINE', 'TURN',
                'SAME', 'DIFFERENT', 'NEED', 'HOUSE', 'POINT', 'PAGE', 'FORM', 'CAME',
                'THIS', 'THAT', 'FROM', 'HAVE', 'MORE', 'THEY', 'THERE', 'ABOUT', 'SHOULD',
                
                # Financial/Trading terms (not tickers)
                'DD', 'YOLO', 'DTE', 'OTM', 'ITM', 'ATM', 'EOD', 'AH', 'PM', 'NAV',
                'YTD', 'MTD', 'QTD', 'YOY', 'MOM', 'ROI', 'ROE', 'EPS', 'PE', 'PEG',
                'EBITDA', 'GAAP', 'SEC', 'FINRA', 'OTC', 'NYSE', 'IPO', 'SPO', 'CEO',
                'CFO', 'CTO', 'COO', 'CIO', 'GDP', 'CPI', 'PCE', 'FOMC', 'FED', 'FED',
                
                # Crypto (we want stocks, not crypto)
                'BTC', 'ETH', 'DOGE', 'SHIB', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX',
                'LINK', 'UNI', 'AAVE', 'SUSHI', 'CAKE', 'XRP', 'XLM', 'ALGO', 'ATOM',
                'LUNA', 'UST', 'USDT', 'USDC', 'DAI', 'BUSD', 'NFT', 'DEFI', 'HODL',
                
                # Reddit/Internet slang
                'TLDR', 'TL', 'DR', 'IMO', 'IMHO', 'FYI', 'AMA', 'NSFW', 'SFW', 'FAQ',
                'LMAO', 'LMFAO', 'ROFL', 'LOL', 'OMG', 'WTF', 'TBH', 'IDK', 'IDC',
                'EDIT', 'UPDATE', 'TLDR', 'ELI5', 'TIL', 'PSA', 'IIRC', 'AFAIK', 'FOMO',
                'FUD', 'REKT', 'BTFD', 'DYOR', 'NFA', 'APE', 'APES', 'MOON', 'GUH',
                
                # Misc
                'USA', 'UK', 'EU', 'US', 'AM', 'PM', 'EST', 'PST', 'GMT', 'UTC',
                'COVID', 'MOASS', 'GME', 'AMC', 'WSB', 'DRS', 'DTCC', 'MIGHT', 'VERY'
            }
            
            # Minimum post score to avoid spam (adjust as needed)
            MIN_POST_SCORE = 2
            MAX_COMMENTS_TO_ANALYZE = 10  # Limit comment analysis for performance
            MAX_POST_AGE_HOURS = 24  # Only consider posts from last 24 hours (1 day)
            # Alternative: MAX_POST_AGE_HOURS = 168  # 7 days for less active trading
            
            from datetime import datetime, timezone
            current_time = datetime.now(timezone.utc)
            
            def analyze_sentiment(text: str, post_score: int, upvote_ratio: float) -> float:
                """
                Advanced sentiment analysis using available libraries.
                Returns sentiment score from -1.0 (negative) to 1.0 (positive)
                """
                sentiment = 0.0
                
                # 1. Use VADER if available (best for social media)
                if VADER_AVAILABLE and vader_analyzer:
                    scores = vader_analyzer.polarity_scores(text)
                    sentiment += scores['compound'] * 0.6  # Weight: 60%
                
                # 2. Use TextBlob if available
                elif TEXTBLOB_AVAILABLE:
                    try:
                        blob = TextBlob(text)
                        # TextBlob returns -1 to 1, similar to VADER
                        sentiment += blob.sentiment.polarity * 0.6  # Weight: 60%
                    except Exception as e:
                        self.logger.debug(f"TextBlob sentiment error: {e}")
                
                # 3. Factor in Reddit metrics (40% weight)
                reddit_sentiment = 0.0
                
                # Upvote ratio component
                if upvote_ratio > 0:
                    ratio_sentiment = (upvote_ratio - 0.5) * 2  # Scale to -1 to 1
                    reddit_sentiment += ratio_sentiment * 0.5
                
                # Post score component (logarithmic to avoid outliers)
                if post_score > 0:
                    score_sentiment = min(1.0, post_score / 200)
                    reddit_sentiment += score_sentiment * 0.5
                elif post_score < 0:
                    reddit_sentiment -= 0.5
                
                sentiment += reddit_sentiment * 0.4  # Weight: 40%
                
                # Normalize to -1 to 1 range
                return max(-1.0, min(1.0, sentiment))
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    posts_processed = 0
                    
                    self.logger.debug(f"Scraping r/{subreddit_name}...")
                    
                    # Get hot posts from subreddit
                    filtered_old_posts = 0
                    for post in subreddit.hot(limit=post_limit):
                        try:
                            # Filter by age: Skip posts older than MAX_POST_AGE_HOURS
                            if hasattr(post, 'created_utc'):
                                post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                                post_age_hours = (current_time - post_time).total_seconds() / 3600
                                
                                if post_age_hours > MAX_POST_AGE_HOURS:
                                    filtered_old_posts += 1
                                    continue
                            
                            # Filter spam: Skip low-score posts
                            if post.score < MIN_POST_SCORE:
                                filtered_spam += 1
                                continue
                            
                            # Combine title and selftext
                            text_content = f"{post.title} {post.selftext if post.selftext else ''}"
                            
                            # TWO-PASS TICKER EXTRACTION:
                            
                            # PASS 1: Extract and validate ticker symbols
                            potential_tickers = ticker_pattern.findall(text_content)
                            
                            # Filter blacklist first (keeps common words out)
                            non_blacklisted = [t for t in potential_tickers if t not in blacklist]
                            
                            # Validate against database
                            validated_tickers = []
                            for ticker in non_blacklisted:
                                if self._validate_ticker(ticker, text_content):
                                    validated_tickers.append(ticker)
                                    
                            # PASS 2: Fuzzy match company names
                            fuzzy_matches = self._fuzzy_match_companies(text_content)
                            fuzzy_tickers = [ticker for ticker, company, score in fuzzy_matches]
                            
                            # Combine both passes (unique tickers only)
                            tickers = list(set(validated_tickers + fuzzy_tickers))
                            
                            # Log fuzzy matches for debugging
                            if fuzzy_matches:
                                for ticker, company, score in fuzzy_matches:
                                    self.logger.debug(f"Fuzzy matched '{company}' → ${ticker} (score: {score:.1f})")
                            
                            if tickers:
                                # Advanced sentiment analysis with NLP libraries
                                post_sentiment = analyze_sentiment(
                                    text_content, 
                                    post.score, 
                                    post.upvote_ratio if hasattr(post, 'upvote_ratio') else 0.5
                                )
                                
                                # Analyze top comments for deeper sentiment (if available)
                                comment_sentiments = []
                                try:
                                    # Load comments (PRAW uses lazy loading)
                                    post.comments.replace_more(limit=0)  # Don't load "more comments"
                                    
                                    # Analyze top comments
                                    for comment in list(post.comments)[:MAX_COMMENTS_TO_ANALYZE]:
                                        if hasattr(comment, 'body') and len(comment.body) > 10:
                                            comment_score = comment.score if hasattr(comment, 'score') else 1
                                            
                                            # Only include comments with valid tickers mentioned
                                            comment_tickers_raw = ticker_pattern.findall(comment.body)
                                            comment_tickers_validated = [
                                                t for t in comment_tickers_raw 
                                                if t not in blacklist and self._validate_ticker(t, comment.body)
                                            ]
                                            relevant_tickers = [t for t in comment_tickers_validated if t in tickers]
                                            
                                            if relevant_tickers:
                                                comment_sent = analyze_sentiment(
                                                    comment.body,
                                                    comment_score,
                                                    0.5  # Comments don't have upvote_ratio
                                                )
                                                comment_sentiments.append(comment_sent)
                                except Exception as e:
                                    self.logger.debug(f"Could not analyze comments: {e}")
                                
                                # Combine post and comment sentiment
                                if comment_sentiments:
                                    # Weight: 70% post, 30% average of comments
                                    avg_comment_sentiment = sum(comment_sentiments) / len(comment_sentiments)
                                    sentiment_value = (post_sentiment * 0.7) + (avg_comment_sentiment * 0.3)
                                else:
                                    sentiment_value = post_sentiment
                                
                                # Store ticker mentions
                                for ticker in set(tickers):  # Unique tickers per post
                                    if ticker not in ticker_data:
                                        ticker_data[ticker] = {
                                            'mentions': 0,
                                            'sentiment': 0.0,
                                            'upvotes': 0,
                                            'posts': [],
                                            'avg_post_score': 0
                                        }
                                    
                                    ticker_data[ticker]['mentions'] += 1
                                    ticker_data[ticker]['sentiment'] += sentiment_value
                                    ticker_data[ticker]['upvotes'] += post.score
                                    ticker_data[ticker]['posts'].append({
                                        'title': post.title[:100],
                                        'score': post.score,
                                        'comments': post.num_comments,
                                        'subreddit': subreddit_name
                                    })
                                    total_mentions += 1
                            
                            posts_processed += 1
                            
                        except Exception as e:
                            self.logger.warning(f"Error processing post: {e}")
                            continue
                    
                    total_posts += posts_processed
                    self.logger.debug(f"  Processed {posts_processed} posts from r/{subreddit_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error scraping r/{subreddit_name}: {e}")
                    continue
            
            # Calculate averages per ticker
            for ticker in ticker_data:
                mention_count = ticker_data[ticker]['mentions']
                if mention_count > 0:
                    # Average sentiment
                    ticker_data[ticker]['sentiment'] = ticker_data[ticker]['sentiment'] / mention_count
                    # Average post score
                    ticker_data[ticker]['avg_post_score'] = ticker_data[ticker]['upvotes'] / mention_count
            
            self.logger.info(f"[SUCCESS] Reddit fetch complete: {len(ticker_data)} unique tickers, {total_mentions} mentions from {total_posts} posts")
            if filtered_old_posts > 0:
                self.logger.info(f"   Filtered {filtered_old_posts} old posts (>{MAX_POST_AGE_HOURS}h)")
            if filtered_spam > 0:
                self.logger.info(f"   Filtered {filtered_spam} low-quality posts (score < {MIN_POST_SCORE})")
            
            return {
                'ticker_mentions': ticker_data,
                'metadata': {
                    'subreddits': subreddits,
                    'post_limit': post_limit,
                    'total_posts': total_posts,
                    'total_mentions': total_mentions,
                    'unique_tickers': len(ticker_data),
                    'filtered_spam': filtered_spam
                }
            }
            
        except Exception as e:
            self.logger.error(f"Reddit fetch failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'ticker_mentions': {
                    'SPY': {'mentions': 10, 'sentiment': 0.0, 'upvotes': 100},
                },
                'metadata': {'error': str(e), 'fallback': True}
            }
    
    async def _fetch_comprehensive_yfinance_data(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Fetch comprehensive YFinance data using v3.1 ComprehensiveYFinanceFetcher.
        
        NEW in v3.1:
        - Uses ComprehensiveYFinanceFetcher with 40+ endpoints
        - Returns RawYFinanceData objects (not dict)
        - Includes all data needed for Phase 2 calculations
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict mapping ticker -> RawYFinanceData bundle
        """
        if not self.yfinance_fetcher:
            self.logger.error("[ERROR] YFinance fetcher not initialized - returning empty cache")
            return {}
        
        self.logger.info(f"[STATS] Fetching comprehensive data for {len(tickers)} tickers...")
        
        # Use the new fetcher's batch method
        loop = asyncio.get_event_loop()
        raw_cache = await loop.run_in_executor(
            None, 
            self.yfinance_fetcher.fetch_batch, 
            tickers
        )
        
        success_count = len(raw_cache)
        success_rate = (success_count / len(tickers) * 100) if tickers else 0
        
        self.logger.info(f"   [SUCCESS] Comprehensive fetch complete: {success_count}/{len(tickers)} tickers ({success_rate:.1f}% success)")
        
        # Log endpoint success statistics
        if raw_cache:
            sample_ticker = list(raw_cache.keys())[0]
            sample_bundle = raw_cache[sample_ticker]
            self.logger.info(f"   [INFO]  Sample ({sample_ticker}): "
                           f"{len(sample_bundle.endpoints_succeeded)}/{len(sample_bundle.endpoints_attempted)} endpoints succeeded")
        
        return raw_cache
    
    # ============================================================================
    # DEPRECATED METHODS (v3.0) - Replaced by ComprehensiveYFinanceFetcher (v3.1)
    # ============================================================================
    # These methods are kept for reference but are no longer used in v3.1
    # The new _fetch_comprehensive_yfinance_data() method replaces all of this
    
    async def _fetch_news_data(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Fetch news sentiment data for tickers.
        
        Gracefully handles if news integration is not available.
        
        Returns:
            Dict mapping ticker -> NewsBundle (from news.py)
        """
        news_data = {}
        
        try:
            # Try to use news integration if available
            from backend.integrations.news import NewsFetcher
            
            news_fetcher = NewsFetcher()
            
            if not news_fetcher.enabled:
                self.logger.info("ℹ️  News fetcher disabled - skipping")
                return {}
            
            for ticker in tickers:
                try:
                    news_bundle = await news_fetcher.fetch_news_bundle(ticker, lookback_days=7)
                    if news_bundle.available:
                        news_data[ticker] = news_bundle
                except Exception as e:
                    self.logger.debug(f"News fetch failed for {ticker}: {e}")
            
            self.logger.info(f"   [SUCCESS] News fetch complete: {len(news_data)}/{len(tickers)} tickers with news")
            
        except ImportError as e:
            self.logger.info(f"[INFO]  News integration not available - skipping ({e})")
        except Exception as e:
            self.logger.warning(f"[WARNING]  News fetch failed: {e}")
        
        return news_data
    
    async def _discover_tickers_from_news(self, 
                                         top_n: int = 30, 
                                         min_mentions: int = 2) -> Dict[str, int]:
        """
        Discover trending tickers from news articles.
        
        Expands the ticker universe beyond Reddit by analyzing news mentions.
        
        Args:
            top_n: Number of top trending tickers to discover
            min_mentions: Minimum mentions required
        
        Returns:
            Dict of ticker -> mention_count
        """
        try:
            from backend.integrations.news import NewsFetcher
            
            news_fetcher = NewsFetcher()
            
            if not news_fetcher.enabled:
                self.logger.debug("News fetcher disabled - skipping ticker discovery")
                return {}
            
            # Discover tickers from news
            news_tickers = await news_fetcher.get_trending_tickers_from_news(
                top_n=top_n, 
                min_mentions=min_mentions
            )
            
            return news_tickers
            
        except ImportError:
            self.logger.debug("News integration not available - skipping ticker discovery")
            return {}
        except Exception as e:
            self.logger.warning(f"News ticker discovery failed: {e}")
            return {}


# ============================================================================
# OPTIMIZED VERSION (Phase 5.6 - Production Optimization)
# ============================================================================

class Phase1FetcherOptimized(Phase1Fetcher):
    """
    Optimized Phase 1 Fetcher with parallel batch processing (Phase 5.6).
    
    Optimizations implemented:
    1. ✅ Parallel batch fetching with asyncio.gather()
    2. ✅ Semaphore-based concurrency limiting
    3. ✅ Optimized ticker batching strategy
    
    Performance targets:
    - Current: ~367s for 500 tickers (Phase 1)
    - Target: ~147s for 500 tickers (60% faster)
    - Method: Process 10-20 tickers concurrently
    
    Inherits all functionality from Phase1Fetcher and adds:
    - Concurrent ticker processing (10-20 at a time)
    - Semaphore-based rate limiting
    - Async batch coordination
    """
    
    def __init__(self, max_concurrent_tickers: int = 10):
        """
        Initialize optimized fetcher.
        
        Args:
            max_concurrent_tickers: Maximum number of tickers to fetch concurrently
                                  (default: 10, increase carefully to avoid rate limits)
        """
        super().__init__()
        self.max_concurrent_tickers = max_concurrent_tickers
        self.semaphore = asyncio.Semaphore(max_concurrent_tickers)
        self.logger.info(f"[OPTIMIZED] Phase1Fetcher initialized with {max_concurrent_tickers} concurrent workers")
    
    async def _fetch_comprehensive_yfinance_data(self, tickers: List[str]) -> Dict[str, Any]:
        """
        OPTIMIZED: Fetch comprehensive YFinance data with parallel batching.
        
        IMPROVEMENT: Process multiple tickers concurrently while respecting rate limits.
        
        Original: Sequential processing via ThreadPoolExecutor
        Optimized: Async coordination + ThreadPoolExecutor batching
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict mapping ticker -> RawYFinanceData bundle
        """
        if not self.yfinance_fetcher:
            self.logger.error("[ERROR] YFinance fetcher not initialized - returning empty cache")
            return {}
        
        self.logger.info(f"[OPTIMIZED] Fetching comprehensive data for {len(tickers)} tickers "
                        f"({self.max_concurrent_tickers} concurrent)...")
        
        # Strategy: Split tickers into batches and process concurrently
        # Each batch goes through yfinance_fetcher which handles its own ThreadPoolExecutor
        batch_size = max(1, len(tickers) // self.max_concurrent_tickers)
        if batch_size < 5:  # Minimum batch size for efficiency
            batch_size = min(5, len(tickers))
        
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        
        self.logger.info(f"[OPTIMIZED] Split into {len(batches)} batches of ~{batch_size} tickers each")
        
        # Process all batches concurrently with semaphore limiting
        tasks = [self._fetch_batch_with_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results from all batches
        raw_cache = {}
        errors = []
        
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                errors.append(f"Batch {idx+1}: {str(result)}")
                self.logger.error(f"[ERROR] Batch {idx+1} failed: {result}")
                continue
            
            if isinstance(result, dict):
                raw_cache.update(result)
        
        success_count = len(raw_cache)
        success_rate = (success_count / len(tickers) * 100) if tickers else 0
        
        self.logger.info(f"   [SUCCESS] Optimized fetch complete: {success_count}/{len(tickers)} tickers "
                        f"({success_rate:.1f}% success, {len(errors)} batch errors)")
        
        # Log endpoint success statistics
        if raw_cache:
            sample_ticker = list(raw_cache.keys())[0]
            sample_bundle = raw_cache[sample_ticker]
            self.logger.info(f"   [INFO]  Sample ({sample_ticker}): "
                           f"{len(sample_bundle.endpoints_succeeded)}/{len(sample_bundle.endpoints_attempted)} endpoints succeeded")
        
        return raw_cache
    
    async def _fetch_batch_with_semaphore(self, batch_tickers: List[str]) -> Dict[str, Any]:
        """
        Fetch a batch of tickers with semaphore-based rate limiting.
        
        This ensures we don't overwhelm the API with too many concurrent requests.
        
        Args:
            batch_tickers: Batch of ticker symbols to fetch
            
        Returns:
            Dict mapping ticker -> RawYFinanceData for this batch
        """
        async with self.semaphore:
            self.logger.debug(f"[BATCH] Processing {len(batch_tickers)} tickers: {batch_tickers[:3]}...")
            
            # Run the synchronous yfinance fetch_batch in executor
            loop = asyncio.get_event_loop()
            try:
                batch_result = await loop.run_in_executor(
                    None,
                    self.yfinance_fetcher.fetch_batch,
                    batch_tickers
                )
                return batch_result
            except Exception as e:
                self.logger.error(f"[BATCH] Error fetching batch: {e}")
                return {}
    
    async def _fetch_news_data(self, tickers: List[str]) -> Dict[str, Any]:
        """
        OPTIMIZED: Fetch news data with parallel processing.
        
        Original method fetches news sequentially.
        This version processes multiple tickers concurrently.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict mapping ticker -> NewsBundle
        """
        news_data = {}
        
        try:
            # Try to use news integration if available
            from backend.integrations.news import NewsFetcher
            
            news_fetcher = NewsFetcher()
            
            if not news_fetcher.enabled:
                self.logger.info("ℹ️  News fetcher disabled - skipping")
                return {}
            
            # Process news fetches in parallel with semaphore
            tasks = [self._fetch_single_news_with_semaphore(news_fetcher, ticker) 
                     for ticker in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            for ticker, result in zip(tickers, results):
                if isinstance(result, Exception):
                    self.logger.debug(f"News fetch failed for {ticker}: {result}")
                    continue
                
                if result and result.available:
                    news_data[ticker] = result
            
            self.logger.info(f"   [SUCCESS] News fetch complete: {len(news_data)}/{len(tickers)} tickers with news")
            
        except ImportError as e:
            self.logger.info(f"[INFO]  News integration not available - skipping ({e})")
        except Exception as e:
            self.logger.warning(f"[WARNING]  News fetch failed: {e}")
        
        return news_data
    
    async def _fetch_single_news_with_semaphore(self, news_fetcher, ticker: str):
        """
        Fetch news for a single ticker with semaphore limiting.
        
        Args:
            news_fetcher: NewsFetcher instance
            ticker: Ticker symbol
            
        Returns:
            NewsBundle or None
        """
        async with self.semaphore:
            try:
                return await news_fetcher.fetch_news_bundle(ticker, lookback_days=7)
            except Exception as e:
                raise  # Re-raise for gather() to catch


# ============================================================================
# Factory Function
# ============================================================================

def get_optimized_phase1_fetcher(max_concurrent: int = 10) -> Phase1FetcherOptimized:
    """
    Factory function to create optimized Phase 1 fetcher.
    
    Args:
        max_concurrent: Maximum concurrent ticker fetches (default: 10)
        
    Returns:
        Phase1FetcherOptimized instance
    """
    return Phase1FetcherOptimized(max_concurrent_tickers=max_concurrent)

