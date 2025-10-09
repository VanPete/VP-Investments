"""
Yahoo Finance Data Integration

This module integrates real financial data from Yahoo Finance to replace fake/missing data:
1. Real market cap numeric values
2. P/E ratios, debt ratios, revenue growth
3. Sector and industry validation
4. Financial scoring based on real metrics
5. Technical indicators and options data
6. Market structure analysis
"""

import logging
import yfinance as yf
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.storage.database import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceIntegrator:
    """Integrates Yahoo Finance data into company_tickers table"""
    
    def __init__(self, batch_size: int = 50, max_workers: int = 5, rate_limit_delay: float = 0.2):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.db = get_database()
        
        # Cache for financial data to avoid duplicate API calls
        self.financial_cache = {}
        self.processed_tickers = set()
        
    def parse_market_cap_to_numeric(self, market_cap_display: str) -> Optional[int]:
        """Convert market cap display string to numeric value"""
        if not market_cap_display:
            return None
            
        # Remove $ and spaces
        clean_cap = market_cap_display.replace('$', '').replace(',', '').strip()
        
        # Handle B (billions), M (millions), K (thousands)
        multipliers = {'B': 1_000_000_000, 'M': 1_000_000, 'K': 1_000}
        
        for suffix, multiplier in multipliers.items():
            if clean_cap.endswith(suffix):
                try:
                    number = float(clean_cap[:-1])
                    return int(number * multiplier)
                except ValueError:
                    continue
        
        # Try parsing as raw number
        try:
            return int(float(clean_cap))
        except ValueError:
            return None
    
    def _is_valid_ticker_format(self, ticker: str) -> bool:
        """Basic validation for ticker format"""
        if not ticker or len(ticker) < 1 or len(ticker) > 5:
            return False
        # Should be alphanumeric, no spaces or special chars except dots
        if not re.match(r'^[A-Z0-9.]{1,5}$', ticker):
            return False
        # Skip common false positives from Reddit
        invalid_patterns = {'THE', 'AND', 'FOR', 'YOU', 'ARE', 'CAN', 'NOT', 'BUT', 'GET', 'ALL'}
        if ticker in invalid_patterns:
            return False
        return True

    def get_ticker_financials(self, ticker: str) -> Dict:
        """Get financial data for a single ticker from Yahoo Finance"""
        # Basic format validation first
        if not self._is_valid_ticker_format(ticker):
            logger.debug(f"🚫 Skipping invalid ticker format: {ticker}")
            error_result = {
                'ticker': ticker,
                'error': 'Invalid ticker format',
                'is_valid': False,
                'updated_at': datetime.utcnow().isoformat()
            }
            self.financial_cache[ticker] = error_result
            return error_result
        
        # Check cache first to avoid duplicate API calls
        if ticker in self.financial_cache:
            logger.debug(f"📋 Using cached data for {ticker}")
            return self.financial_cache[ticker]
        
        try:
            # Add small delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key financial metrics
            financial_data = {
                'ticker': ticker,
                'market_cap_numeric': info.get('marketCap'),
                'market_cap_display': self._format_market_cap(info.get('marketCap')),
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'enterprise_value': info.get('enterpriseValue'),
                'ebitda': info.get('ebitda'),
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'free_cash_flow': info.get('freeCashflow'),
                'revenue': info.get('totalRevenue'),
                'gross_profit': info.get('grossProfits'),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Cache the result
            self.financial_cache[ticker] = financial_data
            self.processed_tickers.add(ticker)
            
            logger.debug(f"✅ Got data for {ticker}: Market Cap ${financial_data['market_cap_display']}")
            return financial_data
            
        except Exception as e:
            error_msg = str(e)
            # Don't log 404 errors as warnings - they're expected for invalid tickers
            if "404" in error_msg:
                logger.debug(f"🔍 Ticker {ticker} not found (404) - likely invalid or delisted")
            else:
                logger.warning(f"⚠️ Failed to get data for {ticker}: {e}")
            
            error_result = {
                'ticker': ticker,
                'error': str(e),
                'is_valid': False,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Cache errors too to avoid retrying
            self.financial_cache[ticker] = error_result
            return error_result
    
    def _format_market_cap(self, market_cap: Optional[int]) -> Optional[str]:
        """Format market cap as display string (e.g., $4.28B)"""
        if not market_cap:
            return None
            
        if market_cap >= 1_000_000_000:
            return f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            return f"${market_cap / 1_000_000:.2f}M"
        elif market_cap >= 1_000:
            return f"${market_cap / 1_000:.2f}K"
        else:
            return f"${market_cap}"
    
    def get_batch_financials(self, tickers: List[str]) -> List[Dict]:
        """Get financial data for a batch of tickers using threading"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all ticker requests
            future_to_ticker = {
                executor.submit(self.get_ticker_financials, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per ticker
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Timeout/error for {ticker}: {e}")
                    results.append({
                        'ticker': ticker,
                        'error': str(e),
                        'updated_at': datetime.utcnow().isoformat()
                    })
        
        return results
    
    def update_company_tickers_financials(self, limit: Optional[int] = None) -> Dict:
        """Update all company_tickers with real Yahoo Finance data"""
        logger.info("🚀 Starting Yahoo Finance data integration...")
        
        # Get all tickers that need updating
        query = self.db.client.table('company_tickers').select('ticker, market_cap_numeric')
        if limit:
            query = query.limit(limit)
        
        response = query.execute()
        
        if not response.data:
            logger.warning("⚠️ No tickers found in company_tickers table")
            return {'success': False, 'message': 'No tickers found'}
        
        # Filter tickers that need market cap updates (missing numeric values)
        tickers_to_update = [
            row['ticker'] for row in response.data 
            if row['market_cap_numeric'] is None
        ]
        
        if not tickers_to_update:
            logger.info("✅ All tickers already have market cap data")
            return {'success': True, 'message': 'All tickers up to date', 'updated_count': 0}
        
        logger.info(f"📊 Found {len(tickers_to_update)} tickers needing updates")
        
        # Process in batches
        total_updated = 0
        total_errors = 0
        
        for i in range(0, len(tickers_to_update), self.batch_size):
            batch = tickers_to_update[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tickers_to_update) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)")
            
            # Get financial data for batch
            batch_results = self.get_batch_financials(batch)
            
            # Update database with results
            for result in batch_results:
                if 'error' in result:
                    total_errors += 1
                    continue
                
                try:
                    # Prepare update data (only include non-None values)
                    update_data = {}
                    
                    # Core financial metrics
                    if result.get('market_cap_numeric'):
                        update_data['market_cap_numeric'] = result['market_cap_numeric']
                    if result.get('market_cap_display'):
                        update_data['market_cap_display'] = result['market_cap_display']
                    
                    # Additional financial ratios for future scoring
                    financial_metrics = [
                        'pe_ratio', 'pb_ratio', 'debt_to_equity', 'roe', 'revenue_growth',
                        'earnings_growth', 'profit_margin', 'operating_margin', 'current_ratio',
                        'quick_ratio', 'beta', 'dividend_yield', 'peg_ratio', 'price_to_sales'
                    ]
                    
                    # Store additional metrics in metadata JSON
                    metadata = {}
                    for metric in financial_metrics:
                        if result.get(metric) is not None:
                            metadata[metric] = result[metric]
                    
                    if metadata:
                        # Get existing metadata if any
                        existing_response = self.db.client.table('company_tickers').select('id').eq('ticker', result['ticker']).execute()
                        if existing_response.data:
                            # Update with financial metadata
                            self.db.client.table('company_tickers').update({
                                **update_data,
                                'updated_at': datetime.utcnow().isoformat()
                            }).eq('ticker', result['ticker']).execute()
                            
                            total_updated += 1
                            
                            if total_updated % 50 == 0:
                                logger.info(f"✅ Updated {total_updated} tickers so far...")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to update {result['ticker']}: {e}")
                    total_errors += 1
            
            # Small delay between batches
            time.sleep(1)
        
        logger.info(f"🎉 Yahoo Finance integration complete!")
        logger.info(f"✅ Updated: {total_updated} tickers")
        logger.info(f"❌ Errors: {total_errors} tickers")
        
        return {
            'success': True,
            'updated_count': total_updated,
            'error_count': total_errors,
            'total_processed': len(tickers_to_update)
        }
    
    def calculate_financial_score(self, ticker: str) -> float:
        """Calculate financial score based on real Yahoo Finance metrics"""
        try:
            # Get ticker financial data
            response = self.db.client.table('company_tickers').select('*').eq('ticker', ticker).execute()
            
            if not response.data:
                return 0.5  # Default score if no data
            
            ticker_data = response.data[0]
            
            # Get fresh Yahoo Finance data for scoring
            financial_data = self.get_ticker_financials(ticker)
            
            if 'error' in financial_data:
                return 0.5
            
            score = 0.5  # Start with neutral score
            
            # P/E Ratio scoring (lower is better for value)
            pe_ratio = financial_data.get('pe_ratio')
            if pe_ratio and 0 < pe_ratio < 15:
                score += 0.15  # Great P/E
            elif pe_ratio and 15 <= pe_ratio < 25:
                score += 0.10  # Good P/E
            elif pe_ratio and pe_ratio >= 40:
                score -= 0.10  # High P/E (growth or overvalued)
            
            # ROE scoring (higher is better)
            roe = financial_data.get('roe')
            if roe and roe > 0.20:  # 20%+
                score += 0.15
            elif roe and roe > 0.15:  # 15%+
                score += 0.10
            elif roe and roe < 0:
                score -= 0.15  # Negative ROE
            
            # Debt to Equity (lower is better)
            debt_to_equity = financial_data.get('debt_to_equity')
            if debt_to_equity and debt_to_equity < 0.3:
                score += 0.10  # Low debt
            elif debt_to_equity and debt_to_equity > 2.0:
                score -= 0.10  # High debt
            
            # Revenue Growth (higher is better)
            revenue_growth = financial_data.get('revenue_growth')
            if revenue_growth and revenue_growth > 0.20:  # 20%+ growth
                score += 0.15
            elif revenue_growth and revenue_growth > 0.10:  # 10%+ growth
                score += 0.10
            elif revenue_growth and revenue_growth < -0.10:  # Declining revenue
                score -= 0.15
            
            # Profit Margin (higher is better)
            profit_margin = financial_data.get('profit_margin')
            if profit_margin and profit_margin > 0.20:  # 20%+ margin
                score += 0.10
            elif profit_margin and profit_margin < 0:  # Negative margin
                score -= 0.10
            
            # Current Ratio (liquidity - around 2 is ideal)
            current_ratio = financial_data.get('current_ratio')
            if current_ratio and 1.5 <= current_ratio <= 3.0:
                score += 0.05  # Good liquidity
            elif current_ratio and current_ratio < 1.0:
                score -= 0.10  # Liquidity concerns
            
            # Ensure score stays within bounds
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate financial score for {ticker}: {e}")
            return 0.5
    
    def get_comprehensive_financial_data(self, ticker: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive financial data for a ticker with all requested metrics.
        Moved from pipeline.py for better separation of concerns.
        
        Args:
            ticker (str): Stock ticker symbol
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            Optional[Dict[str, Any]]: Comprehensive financial data or None if unavailable
        """
        try:
            # Try to use integrators if available
            try:
                from backend.integrations.signal_processing import get_technical_calculator, get_financial_calculator
                
                technical_calc = get_technical_calculator()
                financial_calc = get_financial_calculator()
                
                # Get comprehensive financial data
                financial_data = financial_calc.get_comprehensive_financial_data(ticker)
                
                # Get technical indicators
                technical_data = technical_calc.calculate_all_indicators(ticker)
                
                # Merge technical data into financial data
                financial_data.update(technical_data)
                
                # Ensure current_price is not limited to 10 (remove artificial limits)
                if financial_data.get('current_price'):
                    financial_data['current_price'] = round(float(financial_data['current_price']), 2)
                
                return financial_data
            except ImportError:
                # Fallback to enhanced basic method
                return self.get_enhanced_financial_data(ticker)
                
        except Exception as e:
            logger.error(f"Comprehensive financial data failed for {ticker}: {e}")
            # Final fallback
            return self.get_basic_financial_data(ticker)
    
    def get_basic_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Basic financial data fallback method.
        Moved from pipeline.py for better separation of concerns.
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            
            # Get recent price data
            hist = stock.history(period='5d')
            
            if hist.empty or not info:
                return None
            
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close * 100) if prev_close and current_price else 0
            
            # Calculate additional metrics
            avg_volume = info.get('averageVolume', 1)
            current_volume = int(hist['Volume'].iloc[-1]) if not hist.empty else 1
            volume_spike_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Extract key financial metrics
            financial_data = {
                'ticker': ticker,
                'company': info.get('shortName', info.get('longName', ticker)),
                'current_price': round(float(current_price), 2) if current_price else None,
                'price_1d_pct': round(float(price_change), 2),
                'market_cap': info.get('marketCap'),
                'volume': current_volume,
                'volume_spike_ratio': round(volume_spike_ratio, 2),
                'pe_ratio': round(info.get('trailingPE'), 2) if info.get('trailingPE') else None,
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': round(info.get('beta'), 2) if info.get('beta') else None,
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': avg_volume,
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                # Financial ratios
                'roe': round(info.get('returnOnEquity') * 100, 2) if info.get('returnOnEquity') else None,
                'debt_equity': round(info.get('debtToEquity'), 2) if info.get('debtToEquity') else None,
                'eps_growth': round(info.get('earningsGrowth') * 100, 2) if info.get('earningsGrowth') else None,
                'short_pct_float': round(info.get('shortPercentOfFloat') * 100, 2) if info.get('shortPercentOfFloat') else None,
                'shares_short': info.get('sharesShort'),
                'timestamp': datetime.now().isoformat()
            }
            
            return financial_data
            
        except Exception as e:
            logger.warning(f"Could not retrieve basic financial data for {ticker}: {e}")
            return None
    
    def get_enhanced_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get enhanced financial data using advanced calculations.
        Moved from pipeline.py for better separation of concerns.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='1y')  # Get more data for technical analysis
            
            if hist.empty or not info:
                return None
            
            # Calculate enhanced metrics
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            
            # Price changes
            price_1d_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            price_7d_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-8]) / hist['Close'].iloc[-8] * 100) if len(hist) > 7 else 0
            
            # Volume analysis
            volume = hist['Volume'].iloc[-1] if not hist.empty else 0
            avg_volume_10d = hist['Volume'].tail(10).mean() if len(hist) >= 10 else volume
            volume_spike_ratio = (volume / avg_volume_10d) if avg_volume_10d > 0 else 1.0
            
            # Technical indicators
            close_prices = hist['Close']
            
            # Moving averages
            ma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else None
            ma_200 = close_prices.rolling(200).mean().iloc[-1] if len(close_prices) >= 200 else None
            
            above_50_ma_pct = ((current_price - ma_50) / ma_50 * 100) if ma_50 else None
            above_200_ma_pct = ((current_price - ma_200) / ma_200 * 100) if ma_200 else None
            
            # RSI calculation (simplified)
            def calculate_rsi(prices, periods=14):
                if len(prices) < periods + 1:
                    return None
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs)).iloc[-1]
            
            rsi = calculate_rsi(close_prices)
            
            # Volatility
            volatility = close_prices.pct_change().rolling(20).std().iloc[-1] * 100 if len(close_prices) >= 20 else None
            
            # Comprehensive financial data
            enhanced_data = {
                'ticker': ticker,
                'company': info.get('longName'),
                'sector': info.get('sector'),
                'current_price': float(current_price) if current_price else None,
                'price_1d_pct': float(price_1d_pct),
                'price_7d_pct': float(price_7d_pct),
                'market_cap': info.get('marketCap'),
                'volume': int(volume) if volume else None,
                'volume_spike_ratio': float(volume_spike_ratio),
                'above_50d_ma_pct': float(above_50_ma_pct) if above_50_ma_pct else None,
                'above_200d_ma_pct': float(above_200_ma_pct) if above_200_ma_pct else None,
                'pe_ratio': info.get('trailingPE'),
                'eps_growth': info.get('earningsGrowth'),
                'roe': info.get('returnOnEquity'),
                'debt_equity': info.get('debtToEquity'),
                'rsi': float(rsi) if rsi else None,
                'volatility': float(volatility) if volatility else None,
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                'short_pct_float': info.get('shortPercentOfFloat'),
                'shares_short': info.get('sharesShort'),
                'timestamp': datetime.now().isoformat()
            }
            
            return enhanced_data
            
        except Exception as e:
            logger.warning(f"Enhanced financial data failed for {ticker}: {e}")
            return None
    
    def calculate_beta(self, ticker_data: Dict[str, Any]) -> Optional[float]:
        """
        Calculate Beta using scipy linear regression with cached data.
        Moved from pipeline.py for better separation of concerns.
        
        Args:
            ticker_data (Dict[str, Any]): Cached ticker data including history_1y
            
        Returns:
            Optional[float]: Beta value or None if calculation fails
        """
        try:
            from scipy.stats import linregress
            
            # Get SPY data for same period
            spy = yf.Ticker("SPY")
            spy_history = spy.history(period="1y", interval="1d")
            
            # Get stock data
            stock_df = ticker_data.get('history_1y', pd.DataFrame())
            if stock_df.empty or spy_history.empty:
                return None
                
            # Align dates and calculate returns
            common_dates = stock_df.index.intersection(spy_history.index)
            if len(common_dates) < 30:  # Need minimum data points
                return None
                
            stock_returns = stock_df.loc[common_dates]['Close'].pct_change().dropna()
            spy_returns = spy_history.loc[common_dates]['Close'].pct_change().dropna()
            
            # Ensure same length
            min_len = min(len(stock_returns), len(spy_returns))
            if min_len < 20:
                return None
                
            stock_returns = stock_returns.iloc[-min_len:]
            spy_returns = spy_returns.iloc[-min_len:]
            
            # Linear regression: stock_returns = alpha + beta * spy_returns
            slope, intercept, r_value, p_value, std_err = linregress(spy_returns, stock_returns)
            
            return float(slope)
            
        except Exception as e:
            logger.debug(f"Beta calculation failed: {e}")
            return None

def main():
    """Run Yahoo Finance integration"""
    integrator = YahooFinanceIntegrator(
        batch_size=50,  # Smaller batches to be nice to Yahoo Finance
        max_workers=5,   # Conservative threading
        rate_limit_delay=0.2  # 200ms delay between requests
    )
    
    # Test with a small batch first
    logger.info("🧪 Testing Yahoo Finance integration with small sample...")
    result = integrator.update_company_tickers_financials(limit=100)
    
    if result['success']:
        logger.info(f"✅ Test successful: {result['updated_count']} tickers updated")
        
        # Ask if user wants to process all tickers
        logger.info("🎯 Test completed successfully!")
        logger.info("To process ALL tickers, run: integrator.update_company_tickers_financials()")
        logger.info("This will take 15-30 minutes depending on your data size.")
    else:
        logger.error(f"❌ Test failed: {result.get('message', 'Unknown error')}")

# ===== ENHANCED TECHNICAL INDICATORS CALCULATOR =====

class TechnicalIndicatorsCalculator:
    """Calculates comprehensive technical indicators for stock analysis."""
    
    def __init__(self):
        self.logger = logger
        
        # Configuration based on v1.0 constants
        self.TECH_VOLATILITY_WINDOW = 14
        self.TECH_RSI_PERIOD = 14
        self.TECH_BB_PERIOD = 20
        self.TECH_MOMENTUM_DAYS = 30
        self.TECH_VOL_SPIKE_WINDOW = 10
        self.BETA_WINDOW = 252  # 1 year
        
    def calculate_all_indicators(self, ticker: str) -> Dict[str, float]:
        """Calculate all technical indicators for a given ticker."""
        try:
            # Get price data
            stock = yf.Ticker(ticker)
            hist = stock.history(period='2y')  # Get 2 years for better calculations
            
            if hist.empty or len(hist) < 30:
                self.logger.warning(f"Insufficient data for {ticker} technical analysis")
                return self._get_empty_indicators()
            
            closes = hist['Close']
            volumes = hist['Volume']
            info = stock.info
            
            # Calculate all indicators
            indicators = {}
            
            # Basic price and moving average indicators
            indicators.update(self._calculate_moving_averages(closes))
            
            # Technical oscillators
            indicators.update(self._calculate_rsi(closes))
            indicators.update(self._calculate_macd(closes))
            indicators.update(self._calculate_bollinger_bands(closes))
            
            # Volatility and momentum
            indicators.update(self._calculate_volatility_metrics(closes))
            indicators.update(self._calculate_momentum_metrics(closes))
            
            # Volume analysis (includes avg_daily_volume, avg_volume_30d)
            indicators.update(self._calculate_volume_metrics(volumes))
            
            # Volume-price correlation
            indicators.update(self._calculate_volume_price_correlation(closes, volumes))
            
            # Relative strength
            indicators.update(self._calculate_relative_strength(ticker, closes))
            
            # Sector relative strength
            sector = info.get('sector', '')
            indicators.update(self._calculate_sector_relative_strength(ticker, closes, sector))
            
            # Risk metrics
            indicators.update(self._calculate_risk_metrics(closes))
            
            # Exit signal strength
            indicators.update(self._calculate_exit_signal_strength(closes, indicators))
            
            # Signal strength percentile (historical ranking)
            indicators.update(self._calculate_signal_strength_percentile(closes, indicators))
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {ticker}: {e}")
            return self._get_empty_indicators()
    
    def _calculate_moving_averages(self, closes: pd.Series) -> Dict[str, float]:
        """Calculate moving average indicators."""
        current = closes.iloc[-1]
        
        # 50-day and 200-day moving averages
        ma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else np.nan
        ma_200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else np.nan
        
        # Calculate percentage above/below moving averages
        above_50_pct = ((current - ma_50) / ma_50 * 100) if ma_50 and not np.isnan(ma_50) else np.nan
        above_200_pct = ((current - ma_200) / ma_200 * 100) if ma_200 and not np.isnan(ma_200) else np.nan
        
        return {
            'above_50d_ma_pct': above_50_pct,
            'above_200d_ma_pct': above_200_pct
        }
    
    def _calculate_rsi(self, closes: pd.Series) -> Dict[str, float]:
        """Calculate Relative Strength Index (RSI)."""
        try:
            if len(closes) < self.TECH_RSI_PERIOD + 1:
                return {'rsi': np.nan}
            
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.TECH_RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.TECH_RSI_PERIOD).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return {'rsi': rsi.iloc[-1] if not rsi.empty else np.nan}
            
        except Exception as e:
            self.logger.warning(f"RSI calculation failed: {e}")
            return {'rsi': np.nan}
    
    def _calculate_macd(self, closes: pd.Series) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            if len(closes) < 26:
                return {
                    'macd_line': np.nan,
                    'macd_signal': np.nan,
                    'macd_histogram': np.nan
                }
            
            # Calculate MACD line
            exp1 = closes.ewm(span=12, adjust=False).mean()
            exp2 = closes.ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            # MACD histogram
            macd_histogram = macd_line - signal_line
            
            return {
                'macd_line': macd_line.iloc[-1] if not macd_line.empty else np.nan,
                'macd_signal': signal_line.iloc[-1] if not signal_line.empty else np.nan,
                'macd_histogram': macd_histogram.iloc[-1] if not macd_histogram.empty else np.nan
            }
            
        except Exception as e:
            self.logger.warning(f"MACD calculation failed: {e}")
            return {
                'macd_line': np.nan,
                'macd_signal': np.nan,
                'macd_histogram': np.nan
            }
    
    def _calculate_bollinger_bands(self, closes: pd.Series) -> Dict[str, float]:
        """Calculate Bollinger Bands (upper, lower, position)."""
        try:
            if len(closes) < self.TECH_BB_PERIOD:
                return {
                    'bollinger_upper': np.nan,
                    'bollinger_lower': np.nan,
                    'bollinger_position': np.nan
                }
            
            bb_mean = closes.rolling(self.TECH_BB_PERIOD).mean()
            bb_std = closes.rolling(self.TECH_BB_PERIOD).std()
            
            upper_band = bb_mean + (2 * bb_std)
            lower_band = bb_mean - (2 * bb_std)
            
            current_price = closes.iloc[-1]
            upper_val = upper_band.iloc[-1]
            lower_val = lower_band.iloc[-1]
            
            # Position within bands (0 = at lower, 1 = at upper)
            position = (current_price - lower_val) / (upper_val - lower_val) if (upper_val != lower_val) else 0.5
            
            return {
                'bollinger_upper': upper_val if not np.isnan(upper_val) else np.nan,
                'bollinger_lower': lower_val if not np.isnan(lower_val) else np.nan,
                'bollinger_position': position if not np.isnan(position) else np.nan
            }
            
        except Exception as e:
            self.logger.warning(f"Bollinger Bands calculation failed: {e}")
            return {
                'bollinger_upper': np.nan,
                'bollinger_lower': np.nan,
                'bollinger_position': np.nan
            }
    
    def _calculate_volatility_metrics(self, closes: pd.Series) -> Dict[str, float]:
        """Calculate volatility metrics."""
        try:
            if len(closes) < self.TECH_VOLATILITY_WINDOW:
                return {'volatility': np.nan, 'volatility_rank': np.nan}
            
            # Calculate volatility as standard deviation of returns
            returns = closes.pct_change().dropna()
            volatility = returns.rolling(self.TECH_VOLATILITY_WINDOW).std().iloc[-1]
            
            # Annualize volatility
            volatility_annualized = volatility * np.sqrt(252)
            
            # Calculate volatility rank (percentile over past year)
            if len(returns) >= 252:
                vol_series = returns.rolling(self.TECH_VOLATILITY_WINDOW).std().tail(252)
                volatility_rank = (vol_series <= volatility).sum() / len(vol_series)
            else:
                volatility_rank = np.nan
            
            return {
                'volatility': volatility_annualized,
                'volatility_rank': volatility_rank
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility calculation failed: {e}")
            return {'volatility': np.nan, 'volatility_rank': np.nan}
    
    def _calculate_momentum_metrics(self, closes: pd.Series) -> Dict[str, float]:
        """Calculate momentum metrics."""
        try:
            if len(closes) < self.TECH_MOMENTUM_DAYS:
                return {'momentum_30d_pct': np.nan}
            
            # 30-day momentum
            current = closes.iloc[-1]
            past_30d = closes.iloc[-self.TECH_MOMENTUM_DAYS]
            momentum_30d = ((current - past_30d) / past_30d * 100)
            
            return {'momentum_30d_pct': momentum_30d}
            
        except Exception as e:
            self.logger.warning(f"Momentum calculation failed: {e}")
            return {'momentum_30d_pct': np.nan}
    
    def _calculate_volume_metrics(self, volumes: pd.Series) -> Dict[str, float]:
        """Calculate volume-based metrics including avg_daily_volume and avg_volume_30d."""
        try:
            if len(volumes) < self.TECH_VOL_SPIKE_WINDOW:
                return {
                    'volume_spike_ratio': 1.0,
                    'avg_daily_volume': np.nan,
                    'avg_volume_30d': np.nan
                }
            
            current_volume = volumes.iloc[-1]
            avg_volume = volumes.rolling(self.TECH_VOL_SPIKE_WINDOW).mean().iloc[-1]
            
            volume_spike_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Average daily volume (30-day)
            avg_volume_30d = volumes.tail(30).mean() if len(volumes) >= 30 else np.nan
            
            # Store as avg_daily_volume (same as avg_volume_30d for consistency)
            avg_daily_volume = avg_volume_30d
            
            return {
                'volume_spike_ratio': volume_spike_ratio,
                'avg_daily_volume': avg_daily_volume,
                'avg_volume_30d': avg_volume_30d
            }
            
        except Exception as e:
            self.logger.warning(f"Volume metrics calculation failed: {e}")
            return {
                'volume_spike_ratio': 1.0,
                'avg_daily_volume': np.nan,
                'avg_volume_30d': np.nan
            }
    
    def _calculate_relative_strength(self, ticker: str, closes: pd.Series) -> Dict[str, float]:
        """Calculate relative strength vs SPY."""
        try:
            # Get SPY data for comparison
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period='1y')
            
            if spy_hist.empty or len(closes) < 7:
                return {'relative_strength': np.nan}
            
            # Calculate 7-day returns for both ticker and SPY
            ticker_7d_return = ((closes.iloc[-1] - closes.iloc[-8]) / closes.iloc[-8] * 100) if len(closes) >= 8 else np.nan
            spy_7d_return = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-8]) / spy_hist['Close'].iloc[-8] * 100) if len(spy_hist) >= 8 else np.nan
            
            relative_strength = ticker_7d_return - spy_7d_return if not np.isnan(ticker_7d_return) and not np.isnan(spy_7d_return) else np.nan
            
            return {'relative_strength': relative_strength}
            
        except Exception as e:
            self.logger.warning(f"Relative strength calculation failed for {ticker}: {e}")
            return {'relative_strength': np.nan}
    
    def _calculate_risk_metrics(self, closes: pd.Series) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        try:
            if len(closes) < self.BETA_WINDOW:
                return {'beta': np.nan}
            
            # Calculate beta vs SPY
            try:
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period='1y')
                
                if not spy_hist.empty and len(spy_hist) >= len(closes):
                    # Align the data
                    min_len = min(len(closes), len(spy_hist))
                    ticker_returns = closes.tail(min_len).pct_change().dropna()
                    spy_returns = spy_hist['Close'].tail(min_len).pct_change().dropna()
                    
                    # Calculate beta using covariance and variance
                    if len(ticker_returns) > 30 and len(spy_returns) > 30:
                        aligned_len = min(len(ticker_returns), len(spy_returns))
                        ticker_aligned = ticker_returns.tail(aligned_len)
                        spy_aligned = spy_returns.tail(aligned_len)
                        
                        covariance = np.cov(ticker_aligned, spy_aligned)[0, 1]
                        spy_variance = np.var(spy_aligned)
                        
                        beta = covariance / spy_variance if spy_variance > 0 else np.nan
                    else:
                        beta = np.nan
                else:
                    beta = np.nan
            except Exception:
                beta = np.nan
            
            return {'beta': beta}
            
        except Exception as e:
            self.logger.warning(f"Risk metrics calculation failed: {e}")
            return {'beta': np.nan}
    
    def _calculate_volume_price_correlation(self, closes: pd.Series, volumes: pd.Series) -> Dict[str, float]:
        """Calculate correlation between volume and price changes."""
        try:
            if len(closes) < 30 or len(volumes) < 30:
                return {'volume_price_correlation': np.nan}
            
            # Calculate price changes
            price_changes = closes.pct_change().tail(30).dropna()
            volume_changes = volumes.pct_change().tail(30).dropna()
            
            # Align the series
            min_len = min(len(price_changes), len(volume_changes))
            if min_len < 20:
                return {'volume_price_correlation': np.nan}
            
            price_aligned = price_changes.tail(min_len)
            volume_aligned = volume_changes.tail(min_len)
            
            # Calculate correlation
            correlation = price_aligned.corr(volume_aligned)
            
            return {'volume_price_correlation': correlation if not np.isnan(correlation) else np.nan}
            
        except Exception as e:
            self.logger.warning(f"Volume-price correlation calculation failed: {e}")
            return {'volume_price_correlation': np.nan}
    
    def _calculate_sector_relative_strength(self, ticker: str, closes: pd.Series, sector: str) -> Dict[str, float]:
        """Calculate relative strength vs sector ETF."""
        try:
            # Map sectors to sector ETFs
            sector_etf_map = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Communication Services': 'XLC',
                'Industrials': 'XLI',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Basic Materials': 'XLB',
                'Consumer Defensive': 'XLP'
            }
            
            sector_etf = sector_etf_map.get(sector)
            if not sector_etf or len(closes) < 30:
                return {'sector_relative_strength': np.nan}
            
            # Get sector ETF data
            etf = yf.Ticker(sector_etf)
            etf_hist = etf.history(period='2mo')
            
            if etf_hist.empty or len(etf_hist) < 30:
                return {'sector_relative_strength': np.nan}
            
            # Calculate 30-day returns
            ticker_30d_return = ((closes.iloc[-1] - closes.iloc[-30]) / closes.iloc[-30] * 100) if len(closes) >= 30 else np.nan
            etf_30d_return = ((etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[-30]) / etf_hist['Close'].iloc[-30] * 100) if len(etf_hist) >= 30 else np.nan
            
            if not np.isnan(ticker_30d_return) and not np.isnan(etf_30d_return):
                sector_rs = ticker_30d_return - etf_30d_return
                return {'sector_relative_strength': sector_rs}
            else:
                return {'sector_relative_strength': np.nan}
                
        except Exception as e:
            self.logger.warning(f"Sector relative strength calculation failed for {ticker}: {e}")
            return {'sector_relative_strength': np.nan}
    
    def _calculate_exit_signal_strength(self, closes: pd.Series, indicators: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate exit signal strength based on multiple factors.
        High values indicate strong exit signals (sell pressure).
        """
        try:
            exit_score = 0.0
            factors = 0
            
            # Factor 1: RSI overbought (>70)
            rsi = indicators.get('rsi', np.nan)
            if not np.isnan(rsi):
                if rsi > 70:
                    exit_score += (rsi - 70) / 30 * 100  # Scale 0-100
                factors += 1
            
            # Factor 2: Price far above 200-day MA (overextended)
            above_200d = indicators.get('above_200d_ma_pct', np.nan)
            if not np.isnan(above_200d) and above_200d > 0:
                if above_200d > 20:
                    exit_score += min((above_200d - 20) / 30 * 100, 100)  # Cap at 100
                factors += 1
            
            # Factor 3: Negative MACD crossover
            macd = indicators.get('macd', np.nan)
            if not np.isnan(macd) and macd < 0:
                exit_score += 50  # Strong exit signal
                factors += 1
            
            # Factor 4: High volatility (risk)
            volatility_rank = indicators.get('volatility_rank', np.nan)
            if not np.isnan(volatility_rank) and volatility_rank > 0.8:
                exit_score += (volatility_rank - 0.8) / 0.2 * 100
                factors += 1
            
            # Calculate weighted average
            if factors > 0:
                exit_signal_strength = exit_score / factors
            else:
                exit_signal_strength = np.nan
            
            return {'exit_signal_strength': exit_signal_strength if not np.isnan(exit_signal_strength) else 0.0}
            
        except Exception as e:
            self.logger.warning(f"Exit signal strength calculation failed: {e}")
            return {'exit_signal_strength': 0.0}
    
    def _calculate_signal_strength_percentile(self, closes: pd.Series, indicators: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate signal strength percentile based on historical data.
        Ranks current signal strength against past 252 days (1 year).
        """
        try:
            if len(closes) < 252:
                return {'signal_strength_percentile': np.nan}
            
            # Calculate signal strength components over time
            strength_scores = []
            
            # Use rolling windows to calculate historical signal strengths
            for i in range(30, len(closes)):
                window_closes = closes.iloc[i-30:i]
                
                # Calculate simple strength score based on momentum
                if len(window_closes) >= 30:
                    momentum = ((window_closes.iloc[-1] - window_closes.iloc[0]) / window_closes.iloc[0] * 100)
                    strength_scores.append(momentum)
            
            if len(strength_scores) < 100:
                return {'signal_strength_percentile': np.nan}
            
            # Current momentum from indicators
            current_momentum = indicators.get('momentum_30d_pct', np.nan)
            
            if np.isnan(current_momentum):
                return {'signal_strength_percentile': np.nan}
            
            # Calculate percentile
            strength_array = np.array(strength_scores)
            percentile = (strength_array <= current_momentum).sum() / len(strength_array)
            
            return {'signal_strength_percentile': percentile * 100}  # Convert to 0-100 scale
            
        except Exception as e:
            self.logger.warning(f"Signal strength percentile calculation failed: {e}")
            return {'signal_strength_percentile': np.nan}
    
    def _get_empty_indicators(self) -> Dict[str, float]:
        """Return empty indicators dictionary with NaN values."""
        return {
            'above_50d_ma_pct': np.nan,
            'above_200d_ma_pct': np.nan,
            'rsi': np.nan,
            'macd': np.nan,
            'bollinger': np.nan,
            'volatility': np.nan,
            'volatility_rank': np.nan,
            'momentum_30d_pct': np.nan,
            'volume_spike_ratio': 1.0,
            'avg_daily_volume': np.nan,
            'avg_volume_30d': np.nan,
            'volume_price_correlation': np.nan,
            'relative_strength': np.nan,
            'sector_relative_strength': np.nan,
            'beta': np.nan,
            'exit_signal_strength': 0.0,
            'signal_strength_percentile': np.nan
        }


# ===== ENHANCED FINANCIAL METRICS CALCULATOR =====

class FinancialMetricsCalculator:
    """Calculates comprehensive financial metrics and fundamental analysis."""
    
    def __init__(self):
        self.logger = logger
        
    def get_comprehensive_financial_data(self, ticker: str) -> Dict[str, any]:
        """Get comprehensive financial data for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='1y')
            
            if not info:
                self.logger.warning(f"No financial data available for {ticker}")
                return self._get_empty_financial_data(ticker)
            
            # Combine all financial metrics
            financial_data = {}
            
            # Basic company information
            financial_data.update(self._get_basic_info(ticker, info))
            
            # Price and market data
            financial_data.update(self._get_price_metrics(hist, info))
            
            # Fundamental ratios
            financial_data.update(self._get_fundamental_ratios(info))
            
            # Earnings and growth metrics
            financial_data.update(self._get_earnings_metrics(stock, info))
            
            # Balance sheet metrics
            financial_data.update(self._get_balance_sheet_metrics(info))
            
            # Short interest and ownership
            financial_data.update(self._get_ownership_metrics(info))
            
            # Volume and liquidity
            financial_data.update(self._get_liquidity_metrics(hist, info))
            
            # PHASE 3: Analyst data
            current_price = financial_data.get('current_price', info.get('previousClose', 0))
            financial_data.update(self._get_analyst_data(stock, info, current_price))
            
            # PHASE 3: Earnings surprise data
            financial_data.update(self._get_earnings_surprise_data(stock))
            
            # PHASE 3: Institutional ownership changes
            financial_data.update(self._get_institutional_ownership_data(stock, info))
            
            # PHASE 3: Insider trading activity
            financial_data.update(self._get_insider_trading_data(stock))
            
            return financial_data
            
        except Exception as e:
            self.logger.error(f"Error getting financial data for {ticker}: {e}")
            return self._get_empty_financial_data(ticker)
    
    def _get_basic_info(self, ticker: str, info: Dict[str, any]) -> Dict[str, any]:
        """Extract basic company information."""
        return {
            'ticker': ticker,
            'company': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'country': info.get('country', ''),
            'website': info.get('website', ''),
            'business_summary': info.get('longBusinessSummary', ''),
        }
    
    def _get_price_metrics(self, hist: pd.DataFrame, info: Dict[str, any]) -> Dict[str, float]:
        """Calculate price-related metrics."""
        try:
            if hist.empty:
                return {
                    'current_price': None,
                    'price_1d_pct': 0.0,
                    'price_7d_pct': 0.0,
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                }
            
            current_price = hist['Close'].iloc[-1]
            
            # 1-day price change
            price_1d_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0.0
            
            # 7-day price change
            price_7d_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-8]) / hist['Close'].iloc[-8] * 100) if len(hist) >= 8 else 0.0
            
            return {
                'current_price': round(float(current_price), 2),
                'price_1d_pct': round(price_1d_pct, 2),
                'price_7d_pct': round(price_7d_pct, 2),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            }
            
        except Exception as e:
            self.logger.warning(f"Price metrics calculation failed: {e}")
            return {
                'current_price': None,
                'price_1d_pct': 0.0,
                'price_7d_pct': 0.0,
                'fifty_two_week_high': None,
                'fifty_two_week_low': None,
            }
    
    def _get_fundamental_ratios(self, info: Dict[str, any]) -> Dict[str, float]:
        """Extract fundamental financial ratios."""
        # Clean and validate PE ratio - remove artificial limits
        pe_ratio = info.get('trailingPE')
        if pe_ratio is not None and (pe_ratio < 0 or pe_ratio > 1000):
            pe_ratio = None  # Remove negative or extremely high PE ratios
        
        return {
            'pe_ratio': round(pe_ratio, 2) if pe_ratio is not None else None,
            'forward_pe': round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else None,
            'peg_ratio': round(info.get('pegRatio', 0), 2) if info.get('pegRatio') else None,
            'price_to_book': round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else None,
            'price_to_sales': round(info.get('priceToSalesTrailing12Months', 0), 2) if info.get('priceToSalesTrailing12Months') else None,
            'enterprise_value': info.get('enterpriseValue'),
            'ev_to_revenue': round(info.get('enterpriseToRevenue', 0), 2) if info.get('enterpriseToRevenue') else None,
            'ev_to_ebitda': round(info.get('enterpriseToEbitda', 0), 2) if info.get('enterpriseToEbitda') else None,
        }
    
    def _get_earnings_metrics(self, stock: yf.Ticker, info: Dict[str, any]) -> Dict[str, any]:
        """Calculate earnings and growth metrics."""
        try:
            # Basic earnings data from info
            eps_growth = info.get('earningsGrowth')
            if eps_growth is not None:
                eps_growth = round(eps_growth * 100, 2)  # Convert to percentage
            
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth is not None:
                revenue_growth = round(revenue_growth * 100, 2)
            
            # Earnings gap calculation
            earnings_gap = self._calculate_earnings_gap(stock)
            
            return {
                'eps_current': round(info.get('trailingEps', 0), 2) if info.get('trailingEps') else None,
                'eps_forward': round(info.get('forwardEps', 0), 2) if info.get('forwardEps') else None,
                'eps_growth': eps_growth,
                'revenue_growth': revenue_growth,
                'earnings_gap': round(earnings_gap, 2) if earnings_gap is not None else None,
                'next_earnings_date': self._get_next_earnings_date(stock),
                'profit_margins': round(info.get('profitMargins', 0) * 100, 2) if info.get('profitMargins') else None,
                'operating_margins': round(info.get('operatingMargins', 0) * 100, 2) if info.get('operatingMargins') else None,
                'gross_margins': round(info.get('grossMargins', 0) * 100, 2) if info.get('grossMargins') else None,
            }
            
        except Exception as e:
            self.logger.warning(f"Earnings metrics calculation failed: {e}")
            return {
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'revenue_growth': None,
                'earnings_gap': None,
                'next_earnings_date': None,
                'profit_margins': None,
                'operating_margins': None,
                'gross_margins': None,
            }
    
    def _get_balance_sheet_metrics(self, info: Dict[str, any]) -> Dict[str, float]:
        """Extract balance sheet and financial health metrics."""
        # ROE calculation and cleanup
        roe = info.get('returnOnEquity')
        if roe is not None:
            roe = round(roe * 100, 2)  # Convert to percentage
        
        # Debt to Equity cleanup
        debt_equity = info.get('debtToEquity')
        if debt_equity is not None:
            debt_equity = round(debt_equity, 2)
        
        # Free cash flow margin
        free_cash_flow = info.get('freeCashflow', 0)
        total_revenue = info.get('totalRevenue', 1)
        fcf_margin = None
        if free_cash_flow and total_revenue and total_revenue > 0:
            fcf_margin = round((free_cash_flow / total_revenue) * 100, 2)
        
        return {
            'roe': roe,
            'roa': round(info.get('returnOnAssets', 0) * 100, 2) if info.get('returnOnAssets') else None,
            'debt_equity': debt_equity,
            'current_ratio': round(info.get('currentRatio', 0), 2) if info.get('currentRatio') else None,
            'quick_ratio': round(info.get('quickRatio', 0), 2) if info.get('quickRatio') else None,
            'fcf_margin': fcf_margin,
            'book_value': round(info.get('bookValue', 0), 2) if info.get('bookValue') else None,
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'working_capital': info.get('totalCash', 0) - info.get('totalDebt', 0) if info.get('totalCash') and info.get('totalDebt') else None,
        }
    
    def _get_ownership_metrics(self, info: Dict[str, any]) -> Dict[str, float]:
        """Calculate ownership and short interest metrics."""
        # Short interest metrics
        short_pct_float = info.get('shortPercentOfFloat')
        if short_pct_float is not None:
            short_pct_float = round(short_pct_float * 100, 2)
        
        short_pct_outstanding = info.get('shortRatio')  # This is actually short % of outstanding from yfinance
        if short_pct_outstanding is not None:
            short_pct_outstanding = round(short_pct_outstanding, 2)
        
        # Institutional ownership
        held_by_institutions = info.get('heldPercentInstitutions')
        if held_by_institutions is not None:
            held_by_institutions = round(held_by_institutions * 100, 2)
        
        held_by_insiders = info.get('heldPercentInsiders')
        if held_by_insiders is not None:
            held_by_insiders = round(held_by_insiders * 100, 2)
        
        return {
            'shares_outstanding': info.get('sharesOutstanding'),
            'float_shares': info.get('floatShares'),
            'shares_short': info.get('sharesShort'),
            'short_pct_float': short_pct_float,
            'short_pct_outstanding': short_pct_outstanding,
            'short_ratio': info.get('shortRatio'),  # Days to cover
            'held_by_institutions_pct': held_by_institutions,
            'held_by_insiders_pct': held_by_insiders,
            'insider_transactions': info.get('lastSplitFactor'),  # Placeholder - would need separate API
        }
    
    def _get_liquidity_metrics(self, hist: pd.DataFrame, info: Dict[str, any]) -> Dict[str, float]:
        """Calculate liquidity and trading metrics."""
        try:
            # Market cap
            market_cap = info.get('marketCap')
            
            # Volume metrics
            current_volume = hist['Volume'].iloc[-1] if not hist.empty else None
            avg_volume = info.get('averageVolume', info.get('averageVolume10days', current_volume))
            
            # Average daily dollar volume
            current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('previousClose', 1)
            avg_daily_value = (avg_volume * current_price) if avg_volume and current_price else None
            
            # Beta
            beta = info.get('beta')
            if beta is not None:
                beta = round(beta, 2)
            
            return {
                'market_cap': market_cap,
                'volume': int(current_volume) if current_volume else None,
                'avg_volume': int(avg_volume) if avg_volume else None,
                'avg_daily_value_traded': int(avg_daily_value) if avg_daily_value else None,
                'beta': beta,
                'dividend_yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else None,
                'payout_ratio': round(info.get('payoutRatio', 0) * 100, 2) if info.get('payoutRatio') else None,
                'ex_dividend_date': info.get('exDividendDate'),
                'dividend_date': info.get('dividendDate'),
            }
            
        except Exception as e:
            self.logger.warning(f"Liquidity metrics calculation failed: {e}")
            return {
                'market_cap': None,
                'volume': None,
                'avg_volume': None,
                'avg_daily_value_traded': None,
                'beta': None,
                'dividend_yield': None,
                'payout_ratio': None,
                'ex_dividend_date': None,
                'dividend_date': None,
            }
    
    def _get_analyst_data(self, stock: yf.Ticker, info: Dict[str, any], current_price: float) -> Dict[str, any]:
        """
        Collect analyst recommendations and price targets.
        
        Returns:
            - target_price_mean: Average analyst price target
            - target_price_high: Highest price target
            - target_price_low: Lowest price target
            - recommendation_mean: Average recommendation (1=Strong Buy, 5=Sell)
            - num_analysts: Number of analysts covering stock
            - target_upside_pct: Potential upside to mean target
        """
        try:
            # Get analyst data from info dict
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            recommendation_mean = info.get('recommendationMean')
            num_analysts = info.get('numberOfAnalystOpinions')
            
            # Calculate target upside
            target_upside_pct = None
            if target_mean and current_price and current_price > 0:
                target_upside_pct = round(((target_mean - current_price) / current_price) * 100, 2)
            
            return {
                'target_price_mean': round(target_mean, 2) if target_mean else None,
                'target_price_high': round(target_high, 2) if target_high else None,
                'target_price_low': round(target_low, 2) if target_low else None,
                'recommendation_mean': round(recommendation_mean, 2) if recommendation_mean else None,
                'num_analysts': int(num_analysts) if num_analysts else None,
                'target_upside_pct': target_upside_pct,
            }
            
        except Exception as e:
            self.logger.warning(f"Analyst data extraction failed: {e}")
            return {
                'target_price_mean': None,
                'target_price_high': None,
                'target_price_low': None,
                'recommendation_mean': None,
                'num_analysts': None,
                'target_upside_pct': None,
            }
    
    def _get_earnings_surprise_data(self, stock: yf.Ticker) -> Dict[str, any]:
        """
        Collect earnings surprise history.
        
        Returns:
            - last_earnings_surprise_pct: Most recent earnings surprise
            - avg_earnings_surprise_pct: Average of last 4 quarters
            - earnings_surprise_trend: Improving/Declining/Stable
        """
        try:
            # Get earnings history
            earnings_history = stock.earnings_dates
            
            if earnings_history is None or earnings_history.empty:
                return {
                    'last_earnings_surprise_pct': None,
                    'avg_earnings_surprise_pct': None,
                    'earnings_surprise_trend': None,
                }
            
            # Calculate surprise percentages
            surprises = []
            for idx, row in earnings_history.head(4).iterrows():
                eps_actual = row.get('Reported EPS')
                eps_estimate = row.get('EPS Estimate')
                
                if eps_actual is not None and eps_estimate is not None and eps_estimate != 0:
                    surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                    surprises.append(surprise_pct)
            
            if not surprises:
                return {
                    'last_earnings_surprise_pct': None,
                    'avg_earnings_surprise_pct': None,
                    'earnings_surprise_trend': None,
                }
            
            last_surprise = surprises[0]
            avg_surprise = sum(surprises) / len(surprises)
            
            # Determine trend
            trend = None
            if len(surprises) >= 3:
                recent_avg = sum(surprises[:2]) / 2
                older_avg = sum(surprises[2:]) / len(surprises[2:])
                
                if recent_avg > older_avg + 5:
                    trend = 'Improving'
                elif recent_avg < older_avg - 5:
                    trend = 'Declining'
                else:
                    trend = 'Stable'
            
            return {
                'last_earnings_surprise_pct': round(last_surprise, 2),
                'avg_earnings_surprise_pct': round(avg_surprise, 2),
                'earnings_surprise_trend': trend,
            }
            
        except Exception as e:
            self.logger.warning(f"Earnings surprise data extraction failed: {e}")
            return {
                'last_earnings_surprise_pct': None,
                'avg_earnings_surprise_pct': None,
                'earnings_surprise_trend': None,
            }
    
    def _get_institutional_ownership_data(self, stock: yf.Ticker, info: Dict[str, any]) -> Dict[str, any]:
        """
        Collect institutional ownership changes.
        
        Returns:
            - institutional_ownership_pct: Current institutional %
            - institutional_change_qoq: Quarter-over-quarter change
            - num_institutions: Number of institutional holders
            - top_10_holders_pct: % held by top 10 institutions
        """
        try:
            # Get current institutional ownership from info
            inst_ownership_pct = info.get('heldPercentInstitutions')
            if inst_ownership_pct is not None:
                inst_ownership_pct = round(inst_ownership_pct * 100, 2)
            
            # Get institutional holders
            institutional_holders = stock.institutional_holders
            
            institutional_change_qoq = None
            num_institutions = None
            top_10_pct = None
            
            if institutional_holders is not None and not institutional_holders.empty:
                num_institutions = len(institutional_holders)
                
                # Calculate top 10 holders percentage
                if 'Shares' in institutional_holders.columns:
                    top_10_shares = institutional_holders.head(10)['Shares'].sum()
                    total_shares = info.get('sharesOutstanding', 1)
                    
                    if total_shares and total_shares > 0:
                        top_10_pct = round((top_10_shares / total_shares) * 100, 2)
                
                # Try to calculate QoQ change if Date Reported is available
                if 'Date Reported' in institutional_holders.columns and len(institutional_holders) >= 2:
                    # Sort by date
                    sorted_holders = institutional_holders.sort_values('Date Reported', ascending=False)
                    
                    # Get most recent and previous quarter data
                    recent_date = sorted_holders['Date Reported'].iloc[0]
                    previous_quarter_data = sorted_holders[sorted_holders['Date Reported'] < recent_date]
                    
                    if not previous_quarter_data.empty:
                        recent_total = sorted_holders[sorted_holders['Date Reported'] == recent_date]['Shares'].sum()
                        previous_total = previous_quarter_data['Shares'].sum()
                        
                        if previous_total > 0:
                            institutional_change_qoq = round(((recent_total - previous_total) / previous_total) * 100, 2)
            
            return {
                'institutional_ownership_pct': inst_ownership_pct,
                'institutional_change_qoq': institutional_change_qoq,
                'num_institutions': num_institutions,
                'top_10_holders_pct': top_10_pct,
            }
            
        except Exception as e:
            self.logger.warning(f"Institutional ownership data extraction failed: {e}")
            return {
                'institutional_ownership_pct': None,
                'institutional_change_qoq': None,
                'num_institutions': None,
                'top_10_holders_pct': None,
            }
    
    def _get_insider_trading_data(self, stock: yf.Ticker) -> Dict[str, any]:
        """
        Collect insider trading activity.
        
        Returns:
            - insider_buy_transactions_3m: Buy transactions in last 3 months
            - insider_sell_transactions_3m: Sell transactions in last 3 months
            - insider_net_shares_3m: Net shares bought (positive) or sold (negative)
            - insider_activity_score: 0-100 score (100 = strong buying)
        """
        try:
            # Get insider transactions
            insider_transactions = stock.insider_transactions
            
            if insider_transactions is None or insider_transactions.empty:
                return {
                    'insider_buy_transactions_3m': 0,
                    'insider_sell_transactions_3m': 0,
                    'insider_net_shares_3m': 0,
                    'insider_activity_score': 50.0,  # Neutral
                }
            
            # Filter to last 3 months
            three_months_ago = datetime.now() - timedelta(days=90)
            
            if 'Start Date' in insider_transactions.columns:
                recent_transactions = insider_transactions[
                    pd.to_datetime(insider_transactions['Start Date']) >= three_months_ago
                ]
            else:
                # If no date column, use all transactions as recent
                recent_transactions = insider_transactions
            
            # Count buy and sell transactions
            buy_transactions = 0
            sell_transactions = 0
            net_shares = 0
            
            if 'Transaction' in recent_transactions.columns and 'Shares' in recent_transactions.columns:
                for _, row in recent_transactions.iterrows():
                    transaction_type = str(row['Transaction']).lower()
                    shares = row['Shares']
                    
                    if pd.notna(shares):
                        if 'buy' in transaction_type or 'purchase' in transaction_type:
                            buy_transactions += 1
                            net_shares += abs(shares)
                        elif 'sell' in transaction_type or 'sale' in transaction_type:
                            sell_transactions += 1
                            net_shares -= abs(shares)
            
            # Calculate insider activity score (0-100)
            # 100 = strong buying, 50 = neutral, 0 = strong selling
            total_transactions = buy_transactions + sell_transactions
            
            if total_transactions > 0:
                buy_ratio = buy_transactions / total_transactions
                insider_score = buy_ratio * 100
            else:
                insider_score = 50.0  # Neutral if no transactions
            
            return {
                'insider_buy_transactions_3m': buy_transactions,
                'insider_sell_transactions_3m': sell_transactions,
                'insider_net_shares_3m': int(net_shares),
                'insider_activity_score': round(insider_score, 2),
            }
            
        except Exception as e:
            self.logger.warning(f"Insider trading data extraction failed: {e}")
            return {
                'insider_buy_transactions_3m': 0,
                'insider_sell_transactions_3m': 0,
                'insider_net_shares_3m': 0,
                'insider_activity_score': 50.0,
            }
    
    def _calculate_earnings_gap(self, stock: yf.Ticker) -> Optional[float]:
        """Calculate earnings reaction (price gap after earnings)."""
        try:
            # Get earnings dates
            calendar = stock.calendar
            if calendar is None:
                return None
            
            # Handle both DataFrame and dict formats
            if isinstance(calendar, dict):
                # If it's a dict, try to extract earnings dates
                earnings_dates = calendar.get('Earnings Date', [])
                if not earnings_dates:
                    return None
                earnings_date = earnings_dates[-1] if earnings_dates else None
            elif hasattr(calendar, 'empty') and calendar.empty:
                return None
            else:
                # Traditional DataFrame format
                earnings_date = calendar.index[-1] if not calendar.empty else None
                
            if earnings_date is None:
                return None
            
            # Get price data around earnings
            hist = stock.history(period='3mo')
            if hist.empty:
                return None
            
            # Find prices before and after earnings
            earnings_datetime = pd.to_datetime(earnings_date).tz_localize(None)
            hist_tz_naive = hist.copy()
            hist_tz_naive.index = hist_tz_naive.index.tz_localize(None) if hist_tz_naive.index.tz is not None else hist_tz_naive.index
            
            pre_earnings = hist_tz_naive[hist_tz_naive.index < earnings_datetime]['Close']
            post_earnings = hist_tz_naive[hist_tz_naive.index >= earnings_datetime]['Close']
            
            if pre_earnings.empty or post_earnings.empty:
                return None
            
            pre_price = pre_earnings.iloc[-1]
            post_price = post_earnings.iloc[0]
            
            earnings_gap = ((post_price - pre_price) / pre_price) * 100
            
            return earnings_gap
            
        except Exception as e:
            self.logger.warning(f"Earnings gap calculation failed: {e}")
            return None
    
    def _get_next_earnings_date(self, stock: yf.Ticker) -> Optional[str]:
        """Get next earnings date."""
        try:
            calendar = stock.calendar
            if calendar is not None and not calendar.empty:
                # Get the most recent (future) earnings date
                next_date = calendar.index[0] if not calendar.empty else None
                return next_date.strftime('%Y-%m-%d') if next_date else None
            return None
        except Exception:
            return None
    
    def _get_empty_financial_data(self, ticker: str) -> Dict[str, any]:
        """Return empty financial data structure."""
        return {
            'ticker': ticker,
            'company': ticker,
            'sector': '',
            'industry': '',
            'country': '',
            'website': '',
            'business_summary': '',
            'current_price': None,
            'price_1d_pct': 0.0,
            'price_7d_pct': 0.0,
            'fifty_two_week_high': None,
            'fifty_two_week_low': None,
            'pe_ratio': None,
            'forward_pe': None,
            'peg_ratio': None,
            'price_to_book': None,
            'price_to_sales': None,
            'enterprise_value': None,
            'ev_to_revenue': None,
            'ev_to_ebitda': None,
            'eps_current': None,
            'eps_forward': None,
            'eps_growth': None,
            'revenue_growth': None,
            'earnings_gap': None,
            'next_earnings_date': None,
            'profit_margins': None,
            'operating_margins': None,
            'gross_margins': None,
            'roe': None,
            'roa': None,
            'debt_equity': None,
            'current_ratio': None,
            'quick_ratio': None,
            'fcf_margin': None,
            'book_value': None,
            'total_cash': None,
            'total_debt': None,
            'working_capital': None,
            'shares_outstanding': None,
            'float_shares': None,
            'shares_short': None,
            'short_pct_float': None,
            'short_pct_outstanding': None,
            'short_ratio': None,
            'held_by_institutions_pct': None,
            'held_by_insiders_pct': None,
            'insider_transactions': None,
            'market_cap': None,
            'volume': None,
            'avg_volume': None,
            'avg_daily_value_traded': None,
            'beta': None,
            'dividend_yield': None,
            'payout_ratio': None,
            'ex_dividend_date': None,
            'dividend_date': None,
            # Phase 3 fields
            'target_price_mean': None,
            'target_price_high': None,
            'target_price_low': None,
            'recommendation_mean': None,
            'num_analysts': None,
            'target_upside_pct': None,
            'last_earnings_surprise_pct': None,
            'avg_earnings_surprise_pct': None,
            'earnings_surprise_trend': None,
            'institutional_ownership_pct': None,
            'institutional_change_qoq': None,
            'num_institutions': None,
            'top_10_holders_pct': None,
            'insider_buy_transactions_3m': 0,
            'insider_sell_transactions_3m': 0,
            'insider_net_shares_3m': 0,
            'insider_activity_score': 50.0,
        }


# ===== TECHNICAL INDICATORS =====

@dataclass
class TechnicalIndicators:
    """Container for technical indicator values"""
    # MACD Indicators
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None  
    macd_histogram: Optional[float] = None
    
    # Bollinger Bands
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_position: Optional[float] = None  # 0-1 position within bands
    bollinger_width: Optional[float] = None     # Band width as % of middle
    
    # Market Structure
    beta: Optional[float] = None
    
    # Options Data
    put_call_oi_ratio: Optional[float] = None
    put_call_vol_ratio: Optional[float] = None
    iv_spike_pct: Optional[float] = None
    options_flow_score: Optional[float] = None
    
    # Advanced Market Structure
    retail_holding_pct: Optional[float] = None
    insider_buy_volume: Optional[float] = None
    float_turnover_ratio: Optional[float] = None
    institutional_flow_direction: Optional[str] = None


class TechnicalIndicatorCalculator:
    """Advanced technical indicator calculator with options and market structure analysis"""
    
    def __init__(self):
        self.db = get_database()
        
    def calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Dict[str, float]:
        """Calculate MACD indicators"""
        try:
            if len(prices) < max(fast, slow, signal) + 10:
                return {}
                
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            macd_signal_line = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal_line
            
            return {
                'macd_line': float(macd_line.iloc[-1]),
                'macd_signal': float(macd_signal_line.iloc[-1]),
                'macd_histogram': float(macd_histogram.iloc[-1])
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {}
    
    def calculate_bollinger_bands(self, prices: pd.Series, period=20, std_dev=2) -> Dict[str, float]:
        """Calculate Bollinger Bands with position and width metrics"""
        try:
            if len(prices) < period + 5:
                return {}
                
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            current_price = prices.iloc[-1]
            
            # Calculate position within bands (0-1 scale)
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]
            latest_sma = sma.iloc[-1]
            
            if latest_upper != latest_lower:
                position = (current_price - latest_lower) / (latest_upper - latest_lower)
            else:
                position = 0.5
            
            # Calculate band width as percentage of middle line
            width = ((latest_upper - latest_lower) / latest_sma) * 100 if latest_sma != 0 else None
            
            return {
                'bollinger_upper': float(latest_upper),
                'bollinger_lower': float(latest_lower),
                'bollinger_position': float(max(0, min(1, position))),  # Clamp to 0-1
                'bollinger_width': float(width) if width else None
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def calculate_beta(self, ticker_prices: pd.Series, market_symbol="SPY", period_days=252) -> Optional[float]:
        """Calculate beta relative to market (SPY)"""
        try:
            # Get market data
            market = yf.Ticker(market_symbol)
            market_hist = market.history(period="1y")
            
            if market_hist.empty or len(ticker_prices) < 50:
                return None
            
            # Align dates
            market_prices = market_hist['Close']
            combined = pd.DataFrame({
                'ticker': ticker_prices,
                'market': market_prices
            }).dropna()
            
            if len(combined) < 30:
                return None
            
            # Calculate returns
            ticker_returns = combined['ticker'].pct_change().dropna()
            market_returns = combined['market'].pct_change().dropna()
            
            if len(ticker_returns) < 20:
                return None
            
            # Calculate beta using covariance
            covariance = np.cov(ticker_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return None
            
            beta = covariance / market_variance
            return float(max(-3.0, min(3.0, beta)))  # Clamp extreme values
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return None
    
    def get_options_data(self, ticker: str) -> Dict[str, float]:
        """Get options data and calculate derived metrics"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get options expiration dates
            exp_dates = stock.options
            if not exp_dates:
                return {}
            
            # Use nearest expiration (typically weekly or monthly)
            nearest_exp = exp_dates[0]
            
            # Get options chain
            options = stock.option_chain(nearest_exp)
            calls = options.calls
            puts = options.puts
            
            if calls.empty or puts.empty:
                return {}
            
            # Calculate put/call ratios
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            total_call_vol = calls['volume'].fillna(0).sum()
            total_put_vol = puts['volume'].fillna(0).sum()
            
            put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None
            put_call_vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else None
            
            # Calculate IV spike (compare current IV to historical)
            avg_iv = calls['impliedVolatility'].median()
            
            # Get historical volatility for comparison
            hist = stock.history(period="3mo")
            if not hist.empty:
                returns = hist['Close'].pct_change().dropna()
                historical_vol = returns.std() * np.sqrt(252)  # Annualized
                iv_spike_pct = ((avg_iv - historical_vol) / historical_vol) * 100 if historical_vol > 0 else None
            else:
                iv_spike_pct = None
            
            # Options flow score (0-100 based on unusual activity)
            options_flow_score = self._calculate_options_flow_score(calls, puts)
            
            return {
                'put_call_oi_ratio': put_call_oi_ratio,
                'put_call_vol_ratio': put_call_vol_ratio,
                'iv_spike_pct': iv_spike_pct,
                'options_flow_score': options_flow_score
            }
            
        except Exception as e:
            logger.error(f"Error getting options data for {ticker}: {e}")
            return {}
    
    def _calculate_options_flow_score(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Optional[float]:
        """Calculate options flow score based on volume and open interest patterns"""
        try:
            # Look for unusual volume vs open interest ratios
            call_vol_oi_ratio = calls['volume'].fillna(0).sum() / max(calls['openInterest'].sum(), 1)
            put_vol_oi_ratio = puts['volume'].fillna(0).sum() / max(puts['openInterest'].sum(), 1)
            
            # Score based on activity levels (normalized 0-100)
            total_activity = call_vol_oi_ratio + put_vol_oi_ratio
            
            # Higher ratios indicate more unusual activity
            if total_activity > 2.0:  # Very high activity
                score = min(100, 70 + (total_activity - 2.0) * 15)
            elif total_activity > 1.0:  # Moderate activity
                score = 40 + (total_activity - 1.0) * 30
            else:  # Normal activity
                score = total_activity * 40
            
            return float(max(0, min(100, score)))
            
        except Exception as e:
            logger.error(f"Error calculating options flow score: {e}")
            return None
    
    def get_market_structure_data(self, ticker: str) -> Dict[str, Any]:
        """Get advanced market structure data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract market structure metrics
            retail_holding_pct = info.get('heldPercentRetail')
            insider_ownership = info.get('heldPercentInsiders', 0)
            
            # Calculate float turnover ratio
            shares_outstanding = info.get('sharesOutstanding', 0)
            float_shares = info.get('floatShares', shares_outstanding)
            avg_volume = info.get('averageVolume', 0)
            
            float_turnover_ratio = (avg_volume * 252) / float_shares if float_shares > 0 else None
            
            # Institutional flow direction (simplified)
            inst_ownership = info.get('heldPercentInstitutions', 0)
            if inst_ownership > 80:
                flow_direction = "Heavy Institutional"
            elif inst_ownership > 60:
                flow_direction = "Institutional Bias"
            elif inst_ownership > 40:
                flow_direction = "Mixed"
            else:
                flow_direction = "Retail Bias"
            
            return {
                'retail_holding_pct': retail_holding_pct,
                'insider_buy_volume': insider_ownership,  # Simplified
                'float_turnover_ratio': float_turnover_ratio,
                'institutional_flow_direction': flow_direction
            }
            
        except Exception as e:
            logger.error(f"Error getting market structure data for {ticker}: {e}")
            return {}
    
    async def enhance_signal_with_technical_indicators(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a signal with comprehensive technical indicators"""
        ticker = signal.get('ticker')
        if not ticker:
            return signal
        
        try:
            # Get historical price data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            
            if hist.empty:
                logger.warning(f"No historical data available for {ticker}")
                return signal
            
            prices = hist['Close']
            
            # Calculate all technical indicators
            indicators = TechnicalIndicators()
            
            # MACD
            macd_data = self.calculate_macd(prices)
            if macd_data:
                indicators.macd_line = macd_data.get('macd_line')
                indicators.macd_signal = macd_data.get('macd_signal')
                indicators.macd_histogram = macd_data.get('macd_histogram')
            
            # Bollinger Bands
            bollinger_data = self.calculate_bollinger_bands(prices)
            if bollinger_data:
                indicators.bollinger_upper = bollinger_data.get('bollinger_upper')
                indicators.bollinger_lower = bollinger_data.get('bollinger_lower')
                indicators.bollinger_position = bollinger_data.get('bollinger_position')
                indicators.bollinger_width = bollinger_data.get('bollinger_width')
            
            # Beta
            indicators.beta = self.calculate_beta(prices)
            
            # Options data
            options_data = self.get_options_data(ticker)
            if options_data:
                indicators.put_call_oi_ratio = options_data.get('put_call_oi_ratio')
                indicators.put_call_vol_ratio = options_data.get('put_call_vol_ratio')
                indicators.iv_spike_pct = options_data.get('iv_spike_pct')
                indicators.options_flow_score = options_data.get('options_flow_score')
            
            # Market structure
            market_data = self.get_market_structure_data(ticker)
            if market_data:
                indicators.retail_holding_pct = market_data.get('retail_holding_pct')
                indicators.insider_buy_volume = market_data.get('insider_buy_volume')
                indicators.float_turnover_ratio = market_data.get('float_turnover_ratio')
                indicators.institutional_flow_direction = market_data.get('institutional_flow_direction')
            
            # Update signal with calculated indicators
            for field, value in indicators.__dict__.items():
                if value is not None:
                    signal[field] = value
            
            logger.info(f"Enhanced {ticker} with technical indicators")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal with technical indicators for {ticker}: {e}")
            return signal
    
    def calculate_all_indicators(self, ticker: str) -> Dict[str, float]:
        """Calculate all technical indicators for a given ticker - compatibility method"""
        try:
            # Get stock data
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            if hist.empty or len(hist) < 30:
                logger.warning(f"Insufficient data for {ticker} technical analysis")
                return {}
            
            closes = hist['Close']
            volumes = hist['Volume']
            
            # Calculate basic indicators using existing methods
            indicators = {}
            
            # MACD
            macd_data = self.calculate_macd(closes)
            indicators.update(macd_data)
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(closes)
            indicators.update(bb_data)
            
            # RSI (calculate manually if not available)
            if len(closes) >= 15:
                delta = closes.diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else 50.0
            
            # Volume indicators
            if len(volumes) >= 20:
                avg_volume = volumes.rolling(window=20).mean()
                volume_ratio = volumes.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
                indicators['volume_spike_ratio'] = float(volume_ratio)
            
            # Price momentum
            if len(closes) >= 30:
                momentum_30d = ((closes.iloc[-1] / closes.iloc[-30]) - 1) * 100
                indicators['momentum_30d_pct'] = float(momentum_30d)
            
            logger.debug(f"Calculated {len(indicators)} technical indicators for {ticker}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {e}")
            return {}


# Export main calculator (use NEW calculator with all Phase B indicators)
technical_calculator = TechnicalIndicatorsCalculator()

# ===== FACTORY FUNCTIONS =====

def get_technical_calculator():
    """Factory function to get technical indicators calculator (with Phase B indicators)."""
    return TechnicalIndicatorsCalculator()

def get_financial_calculator():
    """Factory function to get financial metrics calculator."""
    return FinancialMetricsCalculator()


class PerformanceTracker:
    """Enhanced performance tracking with return calculations and SPY benchmarks."""
    
    def __init__(self):
        self.logger = logger
        
        # Performance tracking settings
        self.return_windows = [1, 3, 7, 10]  # Days to track returns
        self.benchmark_ticker = "SPY"
        self.cost_basis_points = 10  # Transaction costs in basis points
        
        self.logger.info("Performance tracker initialized")
    
    async def calculate_signal_performance(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate comprehensive performance metrics for signals.
        
        Args:
            signals: List of signal dictionaries to enhance with performance data
            
        Returns:
            Enhanced signals with performance metrics
        """
        try:
            self.logger.info(f"Calculating performance metrics for {len(signals)} signals")
            
            # Get SPY benchmark data once
            spy_data = await self._get_benchmark_data()
            
            # Process signals in batches for efficiency
            enhanced_signals = []
            batch_size = 20
            
            for i in range(0, len(signals), batch_size):
                batch = signals[i:i + batch_size]
                enhanced_batch = await self._process_signal_batch(batch, spy_data)
                enhanced_signals.extend(enhanced_batch)
                
                # Log progress
                self.logger.info(f"Processed {min(i + batch_size, len(signals))}/{len(signals)} signals")
            
            self.logger.info("Performance calculation complete")
            return enhanced_signals
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            return signals  # Return original signals if enhancement fails
    
    async def _process_signal_batch(self, signals: List[Dict], spy_data: pd.Series) -> List[Dict]:
        """Process a batch of signals for performance calculations."""
        enhanced_signals = []
        
        for signal in signals:
            try:
                enhanced_signal = signal.copy()
                ticker = signal['ticker']
                
                # Get historical performance if signal has a timestamp
                if 'run_datetime' in signal:
                    performance_data = await self._calculate_historical_returns(
                        ticker, signal['run_datetime'], spy_data
                    )
                    enhanced_signal.update(performance_data)
                
                # Add forward-looking performance calculations
                forward_data = await self._calculate_forward_metrics(ticker)
                enhanced_signal.update(forward_data)
                
                enhanced_signals.append(enhanced_signal)
                
            except Exception as e:
                self.logger.warning(f"Performance calculation failed for {signal.get('ticker', 'unknown')}: {e}")
                enhanced_signals.append(signal)  # Keep original if enhancement fails
        
        return enhanced_signals
    
    async def _get_benchmark_data(self) -> pd.Series:
        """Get SPY benchmark data for performance comparisons."""
        try:
            # Get 1 year of SPY data for comprehensive benchmarking
            spy = yf.Ticker(self.benchmark_ticker)
            hist = spy.history(period="1y")
            
            if hist.empty:
                self.logger.warning("Could not retrieve SPY benchmark data")
                return pd.Series(dtype=float)
            
            # Return adjusted close prices with timezone-aware index
            spy_prices = hist['Close']
            spy_prices.index = pd.to_datetime(spy_prices.index, utc=True)
            
            return spy_prices
            
        except Exception as e:
            self.logger.warning(f"Failed to get benchmark data: {e}")
            return pd.Series(dtype=float)
    
    async def _calculate_historical_returns(self, ticker: str, run_datetime: str, spy_data: pd.Series) -> Dict[str, Any]:
        """Calculate historical returns for a signal from its run date."""
        try:
            # Parse run datetime
            run_date = pd.to_datetime(run_datetime, utc=True)
            
            # Calculate end date (today or max available data)
            end_date = min(
                pd.Timestamp.now(tz='UTC'),
                run_date + timedelta(days=max(self.return_windows) + 5)
            )
            
            # Skip if signal is too recent (need at least 1 day)
            if (pd.Timestamp.now(tz='UTC') - run_date).days < 1:
                return self._get_null_performance_data()
            
            # Get ticker price data
            ticker_prices = await self._get_price_data(ticker, run_date, end_date)
            
            if ticker_prices is None or ticker_prices.empty:
                return self._get_null_performance_data()
            
            # Calculate returns for each window
            performance_data = {}
            base_price = ticker_prices.iloc[0]
            
            # Calculate realized returns
            max_return = -float('inf')
            max_drawdown = 0
            peak_price = base_price
            
            for i, price in enumerate(ticker_prices):
                # Track max return and drawdown
                return_pct = (price - base_price) / base_price * 100
                max_return = max(max_return, return_pct)
                
                peak_price = max(peak_price, price)
                drawdown = (price - peak_price) / peak_price * 100
                max_drawdown = min(max_drawdown, drawdown)
            
            # Calculate returns for specific windows
            for window in self.return_windows:
                if len(ticker_prices) > window:
                    end_price = ticker_prices.iloc[window]
                    
                    # Gross return
                    gross_return = (end_price - base_price) / base_price * 100
                    performance_data[f'{window}d_return'] = self._safe_round(gross_return, 2)
                    
                    # Net return (after transaction costs)
                    net_return = gross_return - (self.cost_basis_points / 100)
                    performance_data[f'{window}d_return_net'] = self._safe_round(net_return, 2)
                    
                    # SPY benchmark comparison
                    spy_return = self._calculate_spy_return(run_date, window, spy_data)
                    if spy_return is not None:
                        performance_data[f'spy_{window}d_return'] = self._safe_round(spy_return, 2)
                        performance_data[f'beat_spy_{window}d'] = gross_return > spy_return
                else:
                    # Not enough data for this window
                    performance_data[f'{window}d_return'] = None
                    performance_data[f'{window}d_return_net'] = None
                    performance_data[f'spy_{window}d_return'] = None
                    performance_data[f'beat_spy_{window}d'] = None
            
            # Additional performance metrics
            performance_data.update({
                'max_return_pct': self._safe_round(max_return, 2),
                'drawdown_pct': self._safe_round(max_drawdown, 2),
                'signal_duration': len(ticker_prices),
                'forward_volatility': self._calculate_volatility(ticker_prices),
                'forward_sharpe_ratio': self._calculate_sharpe_ratio(ticker_prices),
                'realized_returns': ','.join([str(w) for w in self.return_windows if f'{w}d_return' in performance_data and performance_data[f'{w}d_return'] is not None]),
                'backtest_phase': self._determine_backtest_phase(performance_data),
                'backtest_timestamp': datetime.now().isoformat(),
                'backtest_notes': f"Performance calculated for {len(ticker_prices)} days"
            })
            
            return performance_data
            
        except Exception as e:
            self.logger.warning(f"Historical return calculation failed for {ticker}: {e}")
            return self._get_null_performance_data()
    
    async def _calculate_forward_metrics(self, ticker: str) -> Dict[str, Any]:
        """Calculate forward-looking performance metrics."""
        try:
            # Get recent price data for technical analysis
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="3mo")  # 3 months for volatility analysis
            
            if hist.empty:
                return {}
            
            prices = hist['Close']
            returns = prices.pct_change().dropna()
            
            # Calculate forward volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized percentage
            
            # Calculate forward Sharpe ratio estimate (using risk-free rate of 4%)
            risk_free_rate = 0.04
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else None
            
            return {
                'forward_volatility': self._safe_round(volatility, 2),
                'forward_sharpe_ratio': self._safe_round(sharpe_ratio, 2)
            }
            
        except Exception as e:
            self.logger.warning(f"Forward metrics calculation failed for {ticker}: {e}")
            return {}
    
    async def _get_price_data(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.Series]:
        """Get price data for a ticker within a date range."""
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # Convert timestamps to date strings for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get historical data
            hist = ticker_obj.history(start=start_str, end=end_str)
            
            if hist.empty:
                return None
            
            # Return adjusted close prices
            prices = hist['Adj Close'] if 'Adj Close' in hist.columns else hist['Close']
            return prices
            
        except Exception as e:
            self.logger.warning(f"Price data retrieval failed for {ticker}: {e}")
            return None
    
    def _calculate_spy_return(self, run_date: pd.Timestamp, window: int, spy_data: pd.Series) -> Optional[float]:
        """Calculate SPY return for a specific window from run date."""
        try:
            if spy_data.empty:
                return None
            
            # Find the closest SPY price to run date
            start_idx = spy_data.index.searchsorted(run_date, side='left')
            end_idx = start_idx + window
            
            if start_idx >= len(spy_data) or end_idx >= len(spy_data):
                return None
            
            start_price = spy_data.iloc[start_idx]
            end_price = spy_data.iloc[end_idx]
            
            spy_return = (end_price - start_price) / start_price * 100
            return spy_return
            
        except Exception:
            return None
    
    def _calculate_volatility(self, prices: pd.Series) -> Optional[float]:
        """Calculate annualized volatility from price series."""
        try:
            if len(prices) < 2:
                return None
            
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                return None
            
            # Annualized volatility
            volatility = returns.std() * np.sqrt(252) * 100
            return self._safe_round(volatility, 2)
            
        except Exception:
            return None
    
    def _calculate_sharpe_ratio(self, prices: pd.Series) -> Optional[float]:
        """Calculate Sharpe ratio from price series."""
        try:
            if len(prices) < 2:
                return None
            
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                return None
            
            # Annualized return and volatility
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            
            if annual_vol == 0:
                return None
            
            # Assuming 4% risk-free rate
            risk_free_rate = 0.04
            sharpe = (annual_return - risk_free_rate) / annual_vol
            
            return self._safe_round(sharpe, 2)
            
        except Exception:
            return None
    
    def _get_null_performance_data(self) -> Dict[str, Any]:
        """Return null performance data structure."""
        data = {}
        
        for window in self.return_windows:
            data[f'{window}d_return'] = None
            data[f'{window}d_return_net'] = None
            data[f'spy_{window}d_return'] = None
            data[f'beat_spy_{window}d'] = None
        
        data.update({
            'max_return_pct': None,
            'drawdown_pct': None,
            'signal_duration': None,
            'forward_volatility': None,
            'forward_sharpe_ratio': None,
            'realized_returns': None,
            'backtest_phase': 'Pending',
            'backtest_timestamp': None,
            'backtest_notes': None
        })
        
        return data
    
    def _determine_backtest_phase(self, performance_data: Dict[str, Any]) -> str:
        """Determine the current backtest phase based on available data."""
        realized_windows = [w for w in self.return_windows 
                          if performance_data.get(f'{w}d_return') is not None]
        
        if len(realized_windows) == len(self.return_windows):
            return 'Complete'
        elif len(realized_windows) > 0:
            return 'Partial'
        else:
            return 'Pending'
    
    def _safe_round(self, value: Optional[float], decimals: int = 2) -> Optional[float]:
        """Safely round a value, handling NaN, infinity, and None."""
        if value is None:
            return None
        
        try:
            if np.isnan(value) or np.isinf(value):
                return None
            return round(float(value), decimals)
        except (ValueError, TypeError, OverflowError):
            return None


def get_performance_tracker():
    """Factory function to get performance tracker."""
    return PerformanceTracker()


# Integration function for use in pipeline
async def enhance_signals_with_performance(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance signals with performance tracking data.
    
    Args:
        signals: List of signals to enhance
        
    Returns:
        Enhanced signals with performance metrics
    """
    try:
        tracker = PerformanceTracker()
        return await tracker.calculate_signal_performance(signals)
    except Exception as e:
        logger.error(f"Performance enhancement failed: {e}")
        return signals


async def enhance_signals_with_technical_data(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance signals with comprehensive technical data including missing fields.
    
    Populates fields that are calculated but missing from database:
    - relative_strength, volatility_rank, above_200d_ma_pct
    - beta, macd_signal, macd_line
    - sector_relative_strength, float_turnover_ratio, institutional_flow_direction
    
    Args:
        signals: List of signals to enhance
        
    Returns:
        Enhanced signals with complete technical data
    """
    if not signals:
        logger.info("No signals provided for technical enhancement")
        return signals
    
    logger.info(f"Enhancing {len(signals)} signals with technical data...")
    
    # Initialize calculators
    tech_calc = TechnicalIndicatorsCalculator()
    financial_calc = FinancialMetricsCalculator()
    
    enhanced_signals = []
    
    for signal in signals:
        try:
            ticker = signal.get('ticker', '')
            if not ticker:
                enhanced_signals.append(signal)
                continue
            
            # Get comprehensive technical data
            tech_data = tech_calc.calculate_all_indicators(ticker)
            financial_data = financial_calc.get_comprehensive_financial_data(ticker)
            
            # Calculate medium priority indicators
            medium_priority_data = await _calculate_medium_priority_indicators(ticker, tech_data, financial_data)
            
            # Merge all technical data into signal
            signal.update(tech_data)
            signal.update(medium_priority_data)
            
            # Ensure beta is included from financial data
            if 'beta' in financial_data:
                signal['beta'] = financial_data['beta']
            
            enhanced_signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Failed to enhance technical data for {signal.get('ticker', 'Unknown')}: {e}")
            enhanced_signals.append(signal)  # Keep original signal
    
    logger.info(f"Technical enhancement completed for {len(enhanced_signals)} signals")
    return enhanced_signals


async def _calculate_medium_priority_indicators(ticker: str, tech_data: Dict[str, Any], financial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate medium priority technical indicators.
    
    Args:
        ticker: Stock ticker
        tech_data: Technical indicators data
        financial_data: Financial data
        
    Returns:
        Dictionary with medium priority indicators
    """
    indicators = {}
    
    try:
        # Sector relative strength - compare to sector ETF
        sector = financial_data.get('sector', '')
        sector_rs = await _calculate_sector_relative_strength(ticker, sector)
        indicators['sector_relative_strength'] = sector_rs
        
        # Float turnover ratio - Volume vs float calculation
        volume = financial_data.get('volume', 0)
        float_shares = financial_data.get('float_shares', 0)
        if volume and float_shares and float_shares > 0:
            float_turnover = (volume / float_shares) * 100
            indicators['float_turnover_ratio'] = round(float_turnover, 4)
        else:
            indicators['float_turnover_ratio'] = None
        
        # Institutional flow direction - Trend in institutional ownership
        institutional_pct = financial_data.get('held_by_institutions_pct', 0)
        institutional_flow = await _calculate_institutional_flow_direction(ticker, institutional_pct)
        indicators['institutional_flow_direction'] = institutional_flow
        
    except Exception as e:
        logger.warning(f"Error calculating medium priority indicators for {ticker}: {e}")
        indicators.update({
            'sector_relative_strength': None,
            'float_turnover_ratio': None,
            'institutional_flow_direction': None
        })
    
    return indicators


async def _calculate_sector_relative_strength(ticker: str, sector: str) -> Optional[float]:
    """Calculate relative strength vs sector ETF."""
    try:
        # Map sectors to common sector ETFs
        sector_etf_map = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financials': 'XLF',
            'Consumer Cyclical': 'XLY',
            'Communication Services': 'XLC',
            'Industrials': 'XLI',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB',
            'Consumer Defensive': 'XLP'
        }
        
        sector_etf = sector_etf_map.get(sector)
        if not sector_etf:
            return None
        
        # Get 30-day returns for both ticker and sector ETF
        import yfinance as yf
        
        ticker_data = yf.download(ticker, period='2mo', interval='1d', auto_adjust=True)
        etf_data = yf.download(sector_etf, period='2mo', interval='1d', auto_adjust=True)
        
        if len(ticker_data) < 30 or len(etf_data) < 30:
            return None
        
        # Calculate 30-day returns
        ticker_return = ((ticker_data['Close'].iloc[-1] - ticker_data['Close'].iloc[-30]) / ticker_data['Close'].iloc[-30]) * 100
        etf_return = ((etf_data['Close'].iloc[-1] - etf_data['Close'].iloc[-30]) / etf_data['Close'].iloc[-30]) * 100
        
        relative_strength = ticker_return - etf_return
        return round(float(relative_strength), 2)
        
    except Exception as e:
        logger.warning(f"Error calculating sector relative strength for {ticker}: {e}")
        return None


async def _calculate_institutional_flow_direction(ticker: str, current_institutional_pct: float) -> Optional[str]:
    """
    Calculate institutional flow direction trend.
    
    Since we don't have historical institutional data, we'll use current levels
    to estimate flow direction based on typical patterns.
    """
    try:
        if not current_institutional_pct:
            return None
        
        # Rough heuristic based on institutional ownership levels
        if current_institutional_pct > 80:
            return "high_institutional"  # High institutional interest
        elif current_institutional_pct > 60:
            return "moderate_institutional"  # Moderate institutional interest  
        elif current_institutional_pct > 30:
            return "mixed_ownership"  # Mixed retail/institutional
        else:
            return "retail_dominated"  # Retail dominated
            
    except Exception as e:
        logger.warning(f"Error calculating institutional flow for {ticker}: {e}")
        return None


if __name__ == "__main__":
    main()