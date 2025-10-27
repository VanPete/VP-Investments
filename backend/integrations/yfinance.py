"""
YFinance Integration v3.1 - Complete Endpoint Coverage
=======================================================
Comprehensive Yahoo Finance data fetcher with ALL applicable endpoints.

Architecture v3.1:
- Phase 1 ONLY: Fetch raw data with maximum coverage
- NO calculations (moved to Phase 2)
- Defensive fetching with graceful fallbacks
- Rate limiting and retry logic
- Returns raw data bundles for caching

Endpoint Coverage (from docs/yfinance_endpoints_full.md):
- Stock/Meta/News: history, info, fast_info, news, dividends, splits, etc.
- Financials & Events: income_stmt, balance_sheet, cashflow, earnings, calendar
- Analysis & Holdings: recommendations, analyst targets, estimates, insider/institutional holders
- Taxonomy: sector/industry context

Dependencies: yfinance, pandas, yaml
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from backend.utils.logger import get_logger
from backend.utils.metrics import emit_metric

logger = get_logger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load endpoint configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "features.yaml"

def load_features_config() -> Dict[str, Any]:
    """Load yfinance endpoint configuration from features.yaml"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded yfinance endpoint config from {CONFIG_PATH}")
                return config
        else:
            logger.warning(f"Config file not found: {CONFIG_PATH}, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

FEATURES_CONFIG = load_features_config()

# Performance configuration (from features.yaml)
PERF_CONFIG = FEATURES_CONFIG.get('performance', {})
PARALLEL_BATCH_SIZE = PERF_CONFIG.get('parallel_fetch_batch_size', 5)
FETCH_TIMEOUT = PERF_CONFIG.get('fetch_timeout_seconds', 30)
DEFAULT_RATE_LIMIT_DELAY = PERF_CONFIG.get('rate_limit_delay_seconds', 0.1)
MAX_RETRIES = PERF_CONFIG.get('max_retries', 3)
RETRY_DELAY = PERF_CONFIG.get('retry_delay_seconds', 1.0)
ENABLE_EXPONENTIAL_BACKOFF = PERF_CONFIG.get('enable_exponential_backoff', True)
BACKOFF_BASE = PERF_CONFIG.get('backoff_base_seconds', 2.0)
BACKOFF_MAX = PERF_CONFIG.get('backoff_max_seconds', 60.0)



# ============================================================================
# DATA STRUCTURES (RAW BUNDLES - NO CALCULATIONS)
# ============================================================================

@dataclass
class RawYFinanceData:
    """
    Complete raw yfinance data bundle for a single ticker.
    Contains ALL fetched data with zero processing.
    
    Structure matches config/features.yaml endpoint categories.
    """
    ticker: str
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Stock/Meta/News
    info: Dict[str, Any] = field(default_factory=dict)
    fast_info: Dict[str, Any] = field(default_factory=dict)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_metadata: Dict[str, Any] = field(default_factory=dict)
    news: List[Dict[str, Any]] = field(default_factory=list)
    dividends: pd.Series = field(default_factory=pd.Series)
    splits: pd.Series = field(default_factory=pd.Series)
    actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    capital_gains: pd.Series = field(default_factory=pd.Series)
    shares: pd.DataFrame = field(default_factory=pd.DataFrame)
    isin: Optional[str] = None
    
    # Financials & Events (Statements)
    income_stmt: pd.DataFrame = field(default_factory=pd.DataFrame)  # annual
    quarterly_income_stmt: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)  # annual
    quarterly_balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)  # annual
    quarterly_cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Financials & Events (Earnings)
    earnings: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_earnings: pd.DataFrame = field(default_factory=pd.DataFrame)
    calendar: Dict[str, Any] = field(default_factory=dict)
    earnings_dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    earnings_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Financials & Events (Other)
    sec_filings: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Analysis & Holdings (Recommendations)
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    upgrades_downgrades: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Analysis & Holdings (Analyst Estimates)
    analyst_price_targets: pd.DataFrame = field(default_factory=pd.DataFrame)
    earnings_estimate: pd.DataFrame = field(default_factory=pd.DataFrame)
    revenue_estimate: pd.DataFrame = field(default_factory=pd.DataFrame)
    eps_trend: pd.DataFrame = field(default_factory=pd.DataFrame)
    eps_revisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    growth_estimates: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Analysis & Holdings (Ownership)
    major_holders: pd.DataFrame = field(default_factory=pd.DataFrame)
    institutional_holders: pd.DataFrame = field(default_factory=pd.DataFrame)
    mutualfund_holders: pd.DataFrame = field(default_factory=pd.DataFrame)
    insider_purchases: pd.DataFrame = field(default_factory=pd.DataFrame)
    insider_transactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    insider_roster_holders: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Analysis & Holdings (Other)
    sustainability: pd.DataFrame = field(default_factory=pd.DataFrame)
    funds_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    fetch_success: bool = True
    fetch_errors: Dict[str, str] = field(default_factory=dict)  # endpoint -> error_msg
    endpoints_attempted: List[str] = field(default_factory=list)
    endpoints_succeeded: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, handling pandas objects"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, pd.DataFrame):
                result[key] = value.to_dict('records') if not value.empty else []
            elif isinstance(value, pd.Series):
                result[key] = value.to_dict() if not value.empty else {}
            else:
                result[key] = value
        return result


@dataclass
class MarketData:
    """Container for market-wide data (SPY, VIX, Treasuries)"""
    spy_history: Optional[pd.DataFrame] = None      # S&P 500 price history
    vix_current: Optional[float] = None             # Current VIX level
    treasury_yield_10y: Optional[float] = None      # 10-year Treasury yield
    treasury_yield_2y: Optional[float] = None       # 2-year Treasury yield
    credit_spread: Optional[float] = None           # Corporate credit spread proxy
    fetch_timestamp: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if market data was successfully fetched"""
        return self.spy_history is not None and not self.spy_history.empty


# ============================================================================
# COMPREHENSIVE YFINANCE FETCHER
# ============================================================================

class ComprehensiveYFinanceFetcher:
    """
    Fetches ALL applicable yfinance endpoints for equity tickers.
    
    Design principles:
    - Maximum coverage: attempt every relevant endpoint
    - Defensive: never fail entire ticker due to one bad endpoint
    - Rate limiting: respect API limits
    - Idempotent: safe to re-fetch same ticker
    - Zero processing: return raw data only
    """
    
    def __init__(self, 
                 rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
                 max_retries: int = MAX_RETRIES,
                 config: Optional[Dict] = None,
                 parallel_batch_size: int = PARALLEL_BATCH_SIZE):
        """
        Initialize fetcher with configuration.
        
        Args:
            rate_limit_delay: Seconds to wait between API calls
            max_retries: Number of retry attempts for failed calls
            config: Features config dict (defaults to FEATURES_CONFIG)
            parallel_batch_size: Number of tickers to fetch concurrently
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance package not installed. Run: pip install yfinance")
        
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.config = config or FEATURES_CONFIG
        self.parallel_batch_size = parallel_batch_size
        self.logger = logger
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=parallel_batch_size)
        self._lock = threading.Lock()  # For thread-safe logging
        
        # Market data caching
        self._market_data_cache: Optional[MarketData] = None
        self._market_data_cache_timestamp: Optional[datetime] = None
        self._market_data_cache_duration = timedelta(hours=1)  # Cache for 1 hour
        
        # Extract priority levels from config
        self.priority_config = self.config.get('priority', {})
        self.critical_endpoints = self.priority_config.get('critical', [])
        self.high_priority_endpoints = self.priority_config.get('high', [])
        
        self.logger.info("ComprehensiveYFinanceFetcher v3.1 initialized")
        self.logger.info(f"  Rate limit: {rate_limit_delay}s")
        self.logger.info(f"  Max retries: {max_retries}")
        self.logger.info(f"  Parallel batch size: {parallel_batch_size}")
        self.logger.info(f"  Critical endpoints: {len(self.critical_endpoints)}")
    
    def fetch_ticker(self, ticker: str, asof: Optional[datetime] = None) -> RawYFinanceData:
        """
        Fetch complete data for a single ticker with ALL endpoints.
        
        This is the main entry point for comprehensive data fetching.
        Attempts every endpoint in config/features.yaml with graceful fallbacks.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            asof: As-of date for point-in-time data (defaults to now)
            
        Returns:
            RawYFinanceData bundle with all fetched data
        """
        ticker = ticker.upper().strip()
        asof = asof or datetime.now()
        
        self.logger.info(f"📊 Fetching comprehensive data for {ticker}")
        start_time = time.time()
        
        # Initialize result bundle
        bundle = RawYFinanceData(ticker=ticker)
        
        try:
            # Create yfinance Ticker object
            stock = yf.Ticker(ticker)
            
            # CATEGORY 1: Stock/Meta/News (CRITICAL - must succeed)
            self._fetch_stock_meta_news(stock, bundle)
            
            # Check if critical endpoints succeeded
            if not self._validate_critical_data(bundle):
                self.logger.error(f"❌ {ticker}: Critical endpoints failed, marking as invalid")
                bundle.fetch_success = False
                return bundle
            
            # CATEGORY 2: Financials & Events (HIGH PRIORITY)
            self._fetch_financials_events(stock, bundle)
            
            # CATEGORY 3: Analysis & Holdings (HIGH PRIORITY)
            self._fetch_analysis_holdings(stock, bundle)
            
            # Calculate success metrics
            total_attempted = len(bundle.endpoints_attempted)
            total_succeeded = len(bundle.endpoints_succeeded)
            success_rate = total_succeeded / total_attempted if total_attempted > 0 else 0
            
            elapsed = time.time() - start_time
            self.logger.info(f"[SUCCESS] {ticker}: Fetched {total_succeeded}/{total_attempted} endpoints "
                           f"({success_rate:.1%} success) in {elapsed:.2f}s")
            
            emit_metric("yfinance.fetch_ticker.success", 1, 
                       tags={'ticker': ticker, 'success_rate': success_rate})
            
            return bundle
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(f"[FAILED] {ticker}: Fatal error during fetch: {e}")
            self.logger.debug(f"Full traceback:\n{error_trace}")
            bundle.fetch_success = False
            bundle.fetch_errors['fatal'] = str(e)
            emit_metric("yfinance.fetch_ticker.fatal_error", 1, tags={'ticker': ticker})
            return bundle
    
    def _fetch_stock_meta_news(self, stock: yf.Ticker, bundle: RawYFinanceData) -> None:
        """
        Fetch Stock/Meta/News category endpoints.
        These are CRITICAL endpoints - ticker is invalid if these fail.
        """
        ticker = bundle.ticker
        
        # 1. info (CRITICAL)
        bundle.info = self._safe_fetch(
            lambda: stock.info,
            endpoint='get_info',
            bundle=bundle,
            critical=True
        ) or {}
        
        # 2. fast_info (CRITICAL)
        try:
            bundle.endpoints_attempted.append('get_fast_info')
            fast_info_obj = stock.fast_info
            bundle.fast_info = {
                'lastPrice': getattr(fast_info_obj, 'lastPrice', None),
                'lastVolume': getattr(fast_info_obj, 'lastVolume', None),
                'marketCap': getattr(fast_info_obj, 'marketCap', None),
                'shares': getattr(fast_info_obj, 'shares', None),
                'yearHigh': getattr(fast_info_obj, 'yearHigh', None),
                'yearLow': getattr(fast_info_obj, 'yearLow', None),
                'yearChange': getattr(fast_info_obj, 'yearChange', None),
                'currency': getattr(fast_info_obj, 'currency', None),
                'timezone': getattr(fast_info_obj, 'timezone', None),
            }
            bundle.endpoints_succeeded.append('get_fast_info')
        except Exception as e:
            self.logger.debug(f"{ticker}: fast_info failed: {e}")
            bundle.fetch_errors['get_fast_info'] = str(e)
        
        # 3. history (CRITICAL - 2 years of price data for risk metrics)
        history_result = self._safe_fetch(
            lambda: stock.history(period="2y", interval="1d"),
            endpoint='history',
            bundle=bundle,
            critical=True
        )
        bundle.history = history_result if history_result is not None else pd.DataFrame()
        
        # 4. history_metadata
        bundle.history_metadata = self._safe_fetch(
            lambda: stock.history_metadata,
            endpoint='get_history_metadata',
            bundle=bundle
        ) or {}
        
        # 5. news
        bundle.news = self._safe_fetch(
            lambda: stock.news,
            endpoint='get_news',
            bundle=bundle
        ) or []
        
        # 6. dividends
        bundle.dividends = self._safe_series(self._safe_fetch(
            lambda: stock.dividends,
            endpoint='get_dividends',
            bundle=bundle
        ))
        
        # 7. splits
        bundle.splits = self._safe_series(self._safe_fetch(
            lambda: stock.splits,
            endpoint='get_splits',
            bundle=bundle
        ))
        
        # 8. actions
        bundle.actions = self._safe_dataframe(self._safe_fetch(
            lambda: stock.actions,
            endpoint='get_actions',
            bundle=bundle
        ))
        
        # 9. capital_gains
        bundle.capital_gains = self._safe_series(self._safe_fetch(
            lambda: stock.capital_gains,
            endpoint='get_capital_gains',
            bundle=bundle
        ))
        
        # 10. shares (full history)
        bundle.shares = self._safe_dataframe(self._safe_fetch(
            lambda: stock.get_shares_full(start="2000-01-01"),
            endpoint='get_shares_full',
            bundle=bundle
        ))
        
        # 11. isin
        bundle.isin = self._safe_fetch(
            lambda: stock.isin,
            endpoint='get_isin',
            bundle=bundle
        )
    
    def _fetch_financials_events(self, stock: yf.Ticker, bundle: RawYFinanceData) -> None:
        """
        Fetch Financials & Events category endpoints.
        These are HIGH PRIORITY for fundamental analysis.
        """
        # Income Statements
        bundle.income_stmt = self._safe_dataframe(self._safe_fetch(
            lambda: stock.income_stmt,
            endpoint='income_stmt',
            bundle=bundle
        ))
        
        bundle.quarterly_income_stmt = self._safe_dataframe(self._safe_fetch(
            lambda: stock.quarterly_income_stmt,
            endpoint='quarterly_income_stmt',
            bundle=bundle
        ))
        
        # Balance Sheets
        bundle.balance_sheet = self._safe_dataframe(self._safe_fetch(
            lambda: stock.balance_sheet,
            endpoint='balance_sheet',
            bundle=bundle
        ))
        
        bundle.quarterly_balance_sheet = self._safe_dataframe(self._safe_fetch(
            lambda: stock.quarterly_balance_sheet,
            endpoint='quarterly_balance_sheet',
            bundle=bundle
        ))
        
        # Cash Flow Statements
        bundle.cashflow = self._safe_dataframe(self._safe_fetch(
            lambda: stock.cashflow,
            endpoint='cashflow',
            bundle=bundle
        ))
        
        bundle.quarterly_cashflow = self._safe_dataframe(self._safe_fetch(
            lambda: stock.quarterly_cashflow,
            endpoint='quarterly_cashflow',
            bundle=bundle
        ))
        
        # Earnings
        bundle.earnings = self._safe_dataframe(self._safe_fetch(
            lambda: stock.earnings,
            endpoint='get_earnings',
            bundle=bundle
        ))
        
        bundle.quarterly_earnings = self._safe_dataframe(self._safe_fetch(
            lambda: stock.quarterly_earnings,
            endpoint='quarterly_earnings',
            bundle=bundle
        ))
        
        bundle.calendar = self._safe_fetch(
            lambda: stock.calendar,
            endpoint='calendar',
            bundle=bundle
        ) or {}
        
        bundle.earnings_dates = self._safe_dataframe(self._safe_fetch(
            lambda: stock.earnings_dates,
            endpoint='get_earnings_dates',
            bundle=bundle
        ))
        
        bundle.earnings_history = self._safe_dataframe(self._safe_fetch(
            lambda: stock.earnings_history,
            endpoint='get_earnings_history',
            bundle=bundle
        ))
        
        # SEC Filings
        bundle.sec_filings = self._safe_dataframe(self._safe_fetch(
            lambda: stock.sec_filings,
            endpoint='get_sec_filings',
            bundle=bundle
        ))
    
    def _fetch_analysis_holdings(self, stock: yf.Ticker, bundle: RawYFinanceData) -> None:
        """
        Fetch Analysis & Holdings category endpoints.
        These are HIGH PRIORITY for smart money / sentiment analysis.
        """
        # Recommendations
        bundle.recommendations = self._safe_dataframe(self._safe_fetch(
            lambda: stock.recommendations,
            endpoint='get_recommendations',
            bundle=bundle
        ))
        
        bundle.recommendations_summary = self._safe_dataframe(self._safe_fetch(
            lambda: stock.recommendations_summary,
            endpoint='get_recommendations_summary',
            bundle=bundle
        ))
        
        bundle.upgrades_downgrades = self._safe_dataframe(self._safe_fetch(
            lambda: stock.upgrades_downgrades,
            endpoint='get_upgrades_downgrades',
            bundle=bundle
        ))
        
        # Analyst Estimates
        bundle.analyst_price_targets = self._safe_dataframe(self._safe_fetch(
            lambda: stock.analyst_price_targets,
            endpoint='get_analyst_price_targets',
            bundle=bundle
        ))
        
        bundle.earnings_estimate = self._safe_dataframe(self._safe_fetch(
            lambda: stock.earnings_estimate,
            endpoint='get_earnings_estimate',
            bundle=bundle
        ))
        
        bundle.revenue_estimate = self._safe_dataframe(self._safe_fetch(
            lambda: stock.revenue_estimate,
            endpoint='get_revenue_estimate',
            bundle=bundle
        ))
        
        bundle.eps_trend = self._safe_dataframe(self._safe_fetch(
            lambda: stock.eps_trend,
            endpoint='get_eps_trend',
            bundle=bundle
        ))
        
        bundle.eps_revisions = self._safe_dataframe(self._safe_fetch(
            lambda: stock.eps_revisions,
            endpoint='get_eps_revisions',
            bundle=bundle
        ))
        
        bundle.growth_estimates = self._safe_dataframe(self._safe_fetch(
            lambda: stock.growth_estimates,
            endpoint='get_growth_estimates',
            bundle=bundle
        ))
        
        # Ownership
        bundle.major_holders = self._safe_dataframe(self._safe_fetch(
            lambda: stock.major_holders,
            endpoint='get_major_holders',
            bundle=bundle
        ))
        
        bundle.institutional_holders = self._safe_dataframe(self._safe_fetch(
            lambda: stock.institutional_holders,
            endpoint='get_institutional_holders',
            bundle=bundle
        ))
        
        bundle.mutualfund_holders = self._safe_dataframe(self._safe_fetch(
            lambda: stock.mutualfund_holders,
            endpoint='get_mutualfund_holders',
            bundle=bundle
        ))
        
        bundle.insider_purchases = self._safe_dataframe(self._safe_fetch(
            lambda: stock.insider_purchases,
            endpoint='get_insider_purchases',
            bundle=bundle
        ))
        
        bundle.insider_transactions = self._safe_dataframe(self._safe_fetch(
            lambda: stock.insider_transactions,
            endpoint='get_insider_transactions',
            bundle=bundle
        ))
        
        bundle.insider_roster_holders = self._safe_dataframe(self._safe_fetch(
            lambda: stock.insider_roster_holders,
            endpoint='get_insider_roster_holders',
            bundle=bundle
        ))
        
        # Other
        bundle.sustainability = self._safe_dataframe(self._safe_fetch(
            lambda: stock.sustainability,
            endpoint='get_sustainability',
            bundle=bundle
        ))
        
        bundle.funds_data = self._safe_dataframe(self._safe_fetch(
            lambda: stock.funds_data,
            endpoint='get_funds_data',
            bundle=bundle
        ))
    
    def _safe_dataframe(self, result: Any) -> pd.DataFrame:
        """
        Safely convert result to DataFrame, avoiding ambiguous truth value error.
        
        Args:
            result: Result from _safe_fetch (could be DataFrame, None, or other)
            
        Returns:
            DataFrame (empty if None)
        """
        if result is None:
            return pd.DataFrame()
        elif isinstance(result, pd.DataFrame):
            return result
        else:
            # Try to convert to DataFrame
            try:
                return pd.DataFrame(result)
            except:
                return pd.DataFrame()
    
    def _safe_series(self, result: Any) -> pd.Series:
        """
        Safely convert result to Series, avoiding ambiguous truth value error.
        
        Args:
            result: Result from _safe_fetch (could be Series, None, or other)
            
        Returns:
            Series (empty if None)
        """
        if result is None:
            return pd.Series(dtype=float)
        elif isinstance(result, pd.Series):
            return result
        else:
            # Try to convert to Series
            try:
                return pd.Series(result)
            except:
                return pd.Series(dtype=float)
    
    def _safe_fetch(self, 
                    fetch_fn, 
                    endpoint: str, 
                    bundle: RawYFinanceData,
                    critical: bool = False) -> Any:
        """
        Safely execute a fetch function with retry logic and error handling.
        
        Args:
            fetch_fn: Function that performs the fetch
            endpoint: Name of the endpoint (for logging)
            bundle: Bundle to track success/failure
            critical: If True, marks entire fetch as failed if this fails
            
        Returns:
            Fetched data or None if failed
        """
        bundle.endpoints_attempted.append(endpoint)
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Execute fetch
                result = fetch_fn()
                
                # Mark success
                bundle.endpoints_succeeded.append(endpoint)
                
                return result
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Retry
                    self.logger.debug(f"{bundle.ticker}: {endpoint} failed (attempt {attempt+1}), retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    # Final failure
                    error_msg = str(e)
                    self.logger.debug(f"{bundle.ticker}: {endpoint} failed after {self.max_retries} attempts: {error_msg}")
                    bundle.fetch_errors[endpoint] = error_msg
                    
                    if critical:
                        self.logger.error(f"{bundle.ticker}: CRITICAL endpoint {endpoint} failed!")
                        bundle.fetch_success = False
                    
                    return None
    
    def _validate_critical_data(self, bundle: RawYFinanceData) -> bool:
        """
        Validate that critical endpoints succeeded.
        
        Returns:
            True if ticker has minimum viable data, False otherwise
        """
        # Check critical endpoints
        has_info = bool(bundle.info)
        has_history = not bundle.history.empty
        
        if not has_info:
            self.logger.error(f"{bundle.ticker}: Missing info data")
            return False
        
        if not has_history:
            self.logger.error(f"{bundle.ticker}: Missing price history")
            return False
        
        # Check for basic required fields in info
        required_fields = ['currentPrice', 'marketCap', 'sector']
        missing_fields = [f for f in required_fields if f not in bundle.info or bundle.info[f] is None]
        
        if missing_fields:
            self.logger.warning(f"{bundle.ticker}: Missing info fields: {missing_fields}")
            # Don't fail for missing fields, just warn
        
        return True
    
    def fetch_batch(self, 
                   tickers: List[str], 
                   asof: Optional[datetime] = None,
                   max_failures: int = 10) -> Dict[str, RawYFinanceData]:
        """
        Fetch comprehensive data for multiple tickers in PARALLEL batches.
        
        NEW in v3.2: Parallel execution with configurable batch size.
        - Processes tickers in batches using ThreadPoolExecutor
        - Respects rate limits via batch processing
        - Timeout protection per ticker
        - Thread-safe logging and error handling
        
        Args:
            tickers: List of ticker symbols
            asof: As-of date for point-in-time data
            max_failures: Stop after this many consecutive failures
            
        Returns:
            Dict mapping ticker to RawYFinanceData bundle
            
        Performance:
            Sequential (old): ~9s per ticker = 288s for 32 tickers
            Parallel (new):   ~9s per batch of 5 = 58s for 32 tickers (4.5x faster)
        """
        results = {}
        consecutive_failures = 0
        
        self.logger.info(f"📊 Fetching comprehensive data for {len(tickers)} tickers (parallel batches of {self.parallel_batch_size})")
        batch_start = time.time()
        
        # Process tickers in batches to respect rate limits
        total_batches = (len(tickers) + self.parallel_batch_size - 1) // self.parallel_batch_size
        
        for batch_idx in range(0, len(tickers), self.parallel_batch_size):
            batch_tickers = tickers[batch_idx:batch_idx + self.parallel_batch_size]
            batch_num = (batch_idx // self.parallel_batch_size) + 1
            
            self.logger.info(f"[Batch {batch_num}/{total_batches}] Processing {len(batch_tickers)} tickers: {', '.join(batch_tickers)}")
            batch_start_time = time.time()
            
            # Submit all tickers in batch to thread pool
            futures = {}
            for ticker in batch_tickers:
                future = self.executor.submit(self._fetch_ticker_with_timeout, ticker, asof)
                futures[ticker] = future
            
            # Wait for all futures in batch to complete
            for ticker, future in futures.items():
                try:
                    # Get result with timeout
                    bundle = future.result(timeout=FETCH_TIMEOUT)
                    
                    if bundle and bundle.fetch_success:
                        results[ticker] = bundle
                        consecutive_failures = 0
                        
                        with self._lock:  # Thread-safe logging
                            self.logger.info(f"  [SUCCESS] {ticker}: {len(bundle.endpoints_succeeded)}/{len(bundle.endpoints_attempted)} endpoints")
                    else:
                        consecutive_failures += 1
                        with self._lock:
                            self.logger.warning(f"  [ERROR] {ticker}: Failed (consecutive failures: {consecutive_failures})")
                        
                        if consecutive_failures >= max_failures:
                            self.logger.error(f"Stopping after {max_failures} consecutive failures")
                            break
                
                except FuturesTimeoutError:
                    consecutive_failures += 1
                    with self._lock:
                        self.logger.error(f"  [TIMEOUT] {ticker}: Timeout after {FETCH_TIMEOUT}s (consecutive failures: {consecutive_failures})")
                    
                    if consecutive_failures >= max_failures:
                        self.logger.error(f"Stopping after {max_failures} consecutive failures")
                        break
                
                except Exception as e:
                    consecutive_failures += 1
                    with self._lock:
                        self.logger.error(f"  [ERROR] {ticker}: Unexpected error: {e} (consecutive failures: {consecutive_failures})")
                    
                    if consecutive_failures >= max_failures:
                        self.logger.error(f"Stopping after {max_failures} consecutive failures")
                        break
            
            # Stop if too many failures
            if consecutive_failures >= max_failures:
                break
            
            batch_elapsed = time.time() - batch_start_time
            self.logger.info(f"[Batch {batch_num}/{total_batches}] Complete in {batch_elapsed:.2f}s")
        
        elapsed = time.time() - batch_start
        success_count = len(results)
        
        self.logger.info(f"[SUCCESS] Batch complete: {success_count}/{len(tickers)} tickers "
                        f"({success_count/len(tickers)*100:.1f}% success) in {elapsed:.2f}s")
        
        # Performance metrics
        if len(tickers) > 0:
            avg_time_per_ticker = elapsed / len(tickers)
            self.logger.info(f"[GAIN] Performance: {avg_time_per_ticker:.2f}s per ticker (parallel batches)")
        
        emit_metric("yfinance.fetch_batch.complete", 1, 
                   tags={'total': len(tickers), 'success': success_count})
        
        return results
    
    def _fetch_ticker_with_timeout(self, ticker: str, asof: Optional[datetime] = None) -> Optional[RawYFinanceData]:
        """
        Wrapper for fetch_ticker with timeout protection.
        
        This method runs in a thread pool and handles per-ticker timeouts.
        
        Args:
            ticker: Ticker symbol
            asof: As-of date
            
        Returns:
            RawYFinanceData bundle or None on failure
        """
        try:
            return self.fetch_ticker(ticker, asof=asof)
        except Exception as e:
            with self._lock:
                self.logger.error(f"[{ticker}] Exception in fetch_ticker: {e}")
            return None
    
    # ========================================================================
    # MARKET DATA FETCHING
    # ========================================================================
    
    def fetch_market_data(self, period: str = '3mo', force_refresh: bool = False) -> MarketData:
        """
        Fetch market-wide indicators (SPY, VIX, Treasuries) for macro analysis.
        
        Data sources:
        - SPY: S&P 500 ETF (market proxy)
        - ^VIX: CBOE Volatility Index
        - ^TNX: 10-year Treasury yield
        - ^IRX: 13-week Treasury bill yield (as proxy for 2-year)
        - LQD: Investment grade corporate bonds (for credit spread)
        
        Args:
            period: Historical period for SPY data (default: 3mo for correlation calcs)
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            MarketData object containing market indicators
        """
        # Check cache
        if not force_refresh and self._is_market_cache_valid():
            self.logger.debug("Using cached market data")
            return self._market_data_cache
        
        self.logger.info("Fetching fresh market data...")
        market_data = MarketData(fetch_timestamp=datetime.now())
        
        try:
            # Fetch SPY (S&P 500)
            spy = yf.Ticker('SPY')
            market_data.spy_history = spy.history(period=period)
            
            if market_data.spy_history.empty:
                self.logger.warning("Could not fetch SPY data")
            else:
                self.logger.debug(f"Fetched SPY: {len(market_data.spy_history)} days")
        
        except Exception as e:
            self.logger.error(f"Error fetching SPY: {e}")
        
        try:
            # Fetch VIX
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='1d')
            if not vix_hist.empty:
                market_data.vix_current = float(vix_hist['Close'].iloc[-1])
                self.logger.debug(f"Fetched VIX: {market_data.vix_current:.2f}")
            else:
                self.logger.warning("Could not fetch VIX data")
        
        except Exception as e:
            self.logger.error(f"Error fetching VIX: {e}")
        
        try:
            # Fetch 10-year Treasury yield (^TNX is in percentage points)
            tnx = yf.Ticker('^TNX')
            tnx_hist = tnx.history(period='1d')
            if not tnx_hist.empty:
                market_data.treasury_yield_10y = float(tnx_hist['Close'].iloc[-1])
                self.logger.debug(f"Fetched 10Y Treasury: {market_data.treasury_yield_10y:.2f}%")
            else:
                self.logger.warning("Could not fetch 10Y Treasury yield")
        
        except Exception as e:
            self.logger.error(f"Error fetching Treasury yield: {e}")
        
        try:
            # Fetch 2-year Treasury yield (^IRX is 13-week T-bill as proxy)
            irx = yf.Ticker('^IRX')
            irx_hist = irx.history(period='1d')
            if not irx_hist.empty:
                market_data.treasury_yield_2y = float(irx_hist['Close'].iloc[-1])
                self.logger.debug(f"Fetched 2Y Treasury (proxy): {market_data.treasury_yield_2y:.2f}%")
            else:
                self.logger.warning("Could not fetch 2Y Treasury yield")
        
        except Exception as e:
            self.logger.error(f"Error fetching 2Y Treasury: {e}")
        
        try:
            # Calculate credit spread (LQD yield - 10Y Treasury)
            # LQD = iShares Investment Grade Corporate Bond ETF
            lqd = yf.Ticker('LQD')
            lqd_info = lqd.info
            
            if lqd_info and 'yield' in lqd_info and market_data.treasury_yield_10y:
                lqd_yield = float(lqd_info.get('yield', 0)) * 100  # Convert to percentage
                market_data.credit_spread = lqd_yield - market_data.treasury_yield_10y
                self.logger.debug(f"Calculated credit spread: {market_data.credit_spread:.2f}%")
            else:
                self.logger.warning("Could not calculate credit spread")
        
        except Exception as e:
            self.logger.error(f"Error calculating credit spread: {e}")
        
        # Cache the result
        self._market_data_cache = market_data
        self._market_data_cache_timestamp = datetime.now()
        
        return market_data
    
    def _is_market_cache_valid(self) -> bool:
        """Check if cached market data is still valid"""
        if self._market_data_cache is None or self._market_data_cache_timestamp is None:
            return False
        
        age = datetime.now() - self._market_data_cache_timestamp
        return age < self._market_data_cache_duration
    
    def get_cached_market_data(self) -> Optional[MarketData]:
        """Get cached market data without refresh"""
        return self._market_data_cache if self._is_market_cache_valid() else None
    
    def shutdown(self):
        """
        Shutdown the thread pool executor.
        
        Call this when done with the fetcher to cleanup resources.
        """
        self.executor.shutdown(wait=True)
        self.logger.info("ComprehensiveYFinanceFetcher thread pool shutdown complete")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

_fetcher_instance = None

def get_yfinance_fetcher() -> ComprehensiveYFinanceFetcher:
    """Get singleton fetcher instance"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = ComprehensiveYFinanceFetcher()
    return _fetcher_instance

def create_yfinance_fetcher(**kwargs) -> ComprehensiveYFinanceFetcher:
    """Create new fetcher instance with custom config"""
    return ComprehensiveYFinanceFetcher(**kwargs)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def fetch_ticker_data(ticker: str, asof: Optional[datetime] = None) -> RawYFinanceData:
    """
    Convenience function to fetch data for a single ticker.
    
    Usage:
        data = fetch_ticker_data('AAPL')
    """
    fetcher = get_yfinance_fetcher()
    return fetcher.fetch_ticker(ticker, asof=asof)

def fetch_tickers_data(tickers: List[str], asof: Optional[datetime] = None) -> Dict[str, RawYFinanceData]:
    """
    Convenience function to fetch data for multiple tickers.
    
    Usage:
        data = fetch_tickers_data(['AAPL', 'MSFT', 'GOOGL'])
    """
    fetcher = get_yfinance_fetcher()
    return fetcher.fetch_batch(tickers, asof=asof)


# ============================================================================
# MARKET DATA HELPER FUNCTIONS
# ============================================================================

def calculate_market_regime(spy_history: pd.DataFrame) -> Optional[float]:
    """
    Calculate market regime indicator.
    
    Simple regime classification:
    - Bull market: Price > 200-day MA and trending up
    - Bear market: Price < 200-day MA and trending down
    - Neutral: Otherwise
    
    Args:
        spy_history: SPY price history DataFrame
        
    Returns:
        Float regime indicator: 1.0 (bull), 0.0 (neutral), -1.0 (bear)
        or None if insufficient data
    """
    try:
        if spy_history is None or spy_history.empty:
            logger.warning("calculate_market_regime: spy_history is None or empty")
            return None
        
        logger.debug(f"calculate_market_regime: SPY history has {len(spy_history)} days")
        
        # Adjust logic for available data
        if len(spy_history) < 200:
            # If less than 200 days, use 50-day MA instead
            if len(spy_history) < 50:
                logger.warning(f"calculate_market_regime: Insufficient data ({len(spy_history)} days, need at least 50)")
                return None
            
            # Use 50-day MA as fallback
            spy_history_copy = spy_history.copy()
            spy_history_copy['MA_50'] = spy_history_copy['Close'].rolling(window=50).mean()
            
            # Get current price and 50-day MA
            current_price = spy_history_copy['Close'].iloc[-1]
            ma_50 = spy_history_copy['MA_50'].iloc[-1]
            
            if pd.isna(ma_50):
                logger.warning("calculate_market_regime: MA_50 is NaN")
                return None
            
            # Calculate 20-day trend (slope)
            if len(spy_history_copy) >= 20:
                recent_20 = spy_history_copy['Close'].iloc[-20:]
                trend = (recent_20.iloc[-1] - recent_20.iloc[0]) / recent_20.iloc[0]
            else:
                trend = 0
            
            # Classify regime (more lenient thresholds for shorter period)
            if current_price > ma_50 and trend > 0.01:  # Price above MA and trending up (>1%)
                logger.debug(f"calculate_market_regime: Bull market (price={current_price:.2f}, MA50={ma_50:.2f}, trend={trend:.4f})")
                return 1.0  # Bull market
            elif current_price < ma_50 and trend < -0.01:  # Price below MA and trending down
                logger.debug(f"calculate_market_regime: Bear market (price={current_price:.2f}, MA50={ma_50:.2f}, trend={trend:.4f})")
                return -1.0  # Bear market
            else:
                logger.debug(f"calculate_market_regime: Neutral (price={current_price:.2f}, MA50={ma_50:.2f}, trend={trend:.4f})")
                return 0.0  # Neutral
        
        else:
            # Standard 200-day MA calculation
            spy_history_copy = spy_history.copy()
            spy_history_copy['MA_200'] = spy_history_copy['Close'].rolling(window=200).mean()
            
            # Get current price and 200-day MA
            current_price = spy_history_copy['Close'].iloc[-1]
            ma_200 = spy_history_copy['MA_200'].iloc[-1]
            
            if pd.isna(ma_200):
                logger.warning("calculate_market_regime: MA_200 is NaN")
                return None
            
            # Calculate 50-day trend (slope)
            if len(spy_history_copy) >= 50:
                recent_50 = spy_history_copy['Close'].iloc[-50:]
                trend = (recent_50.iloc[-1] - recent_50.iloc[0]) / recent_50.iloc[0]
            else:
                trend = 0
            
            # Classify regime
            if current_price > ma_200 and trend > 0.02:  # Price above MA and trending up (>2%)
                logger.debug(f"calculate_market_regime: Bull market (price={current_price:.2f}, MA200={ma_200:.2f}, trend={trend:.4f})")
                return 1.0  # Bull market
            elif current_price < ma_200 and trend < -0.02:  # Price below MA and trending down
                logger.debug(f"calculate_market_regime: Bear market (price={current_price:.2f}, MA200={ma_200:.2f}, trend={trend:.4f})")
                return -1.0  # Bear market
            else:
                logger.debug(f"calculate_market_regime: Neutral (price={current_price:.2f}, MA200={ma_200:.2f}, trend={trend:.4f})")
                return 0.0  # Neutral
    
    except Exception as e:
        logger.error(f"Error calculating market regime: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def calculate_spy_correlation(stock_history: pd.DataFrame, spy_history: pd.DataFrame, 
                               window: int = 60) -> Optional[float]:
    """
    Calculate correlation between stock and SPY returns.
    
    Args:
        stock_history: Stock price history
        spy_history: SPY price history
        window: Rolling window in days (default: 60)
        
    Returns:
        Correlation coefficient (-1 to 1) or None if insufficient data
    """
    try:
        if stock_history is None or spy_history is None:
            return None
        
        if stock_history.empty or spy_history.empty:
            return None
        
        if len(stock_history) < window or len(spy_history) < window:
            return None
        
        # Calculate daily returns
        stock_returns = stock_history['Close'].pct_change().dropna()
        spy_returns = spy_history['Close'].pct_change().dropna()
        
        # Align dates
        common_dates = stock_returns.index.intersection(spy_returns.index)
        
        if len(common_dates) < window:
            return None
        
        # Get last 'window' days
        stock_returns_window = stock_returns.loc[common_dates].iloc[-window:]
        spy_returns_window = spy_returns.loc[common_dates].iloc[-window:]
        
        # Calculate correlation
        import numpy as np
        correlation = stock_returns_window.corr(spy_returns_window)
        
        return float(correlation) if not np.isnan(correlation) else None
    
    except Exception as e:
        logger.error(f"Error calculating SPY correlation: {e}")
        return None


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

if not YFINANCE_AVAILABLE:
    logger.warning("[WARNING] yfinance package not available - install with: pip install yfinance")
else:
    logger.info("[OK] YFinance Integration v3.1 loaded with comprehensive endpoint coverage")
