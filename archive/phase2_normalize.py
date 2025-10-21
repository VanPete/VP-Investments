"""
Phase 2: Parse & Normalize
===========================

Convert raw data from Phase 1 into standardized signal format.

This module is responsible for:
- Converting Reddit data into normalized signals
- Converting financial data into normalized signals
- Converting news data into normalized signals
- NO scoring logic (that's Phase 3)
- NO API calls (all data from Phase 1 cache)

Output: List of normalized signals ready for Phase 3 scoring
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class Phase2Normalizer:
    """
    Phase 2: Parse & Normalize
    
    Converts raw data from Phase 1 into standardized signal structures.
    NO scoring happens here - that's Phase 3's job.
    """
    
    def __init__(self):
        """Initialize Phase 2 normalizer."""
        self.logger = logger
    
    def normalize_all_signals(self, phase1_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main entry point for Phase 2.
        
        Normalizes all data types into unified signal format.
        
        Args:
            phase1_data: Output from Phase 1 containing reddit_data, financial_data, news_data
        
        Returns:
            List of normalized signals (no scores yet)
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: PARSE & NORMALIZE")
        self.logger.info("=" * 60)
        
        phase2_start = datetime.now()
        
        reddit_data = phase1_data.get('reddit_data', {})
        financial_data = phase1_data.get('financial_data', {})
        news_data = phase1_data.get('news_data', {})
        
        # 1. Normalize Reddit signals
        self.logger.info("Step 2.1: Normalizing Reddit signals...")
        reddit_signals = self.normalize_reddit_signals(reddit_data)
        
        # 2. Normalize Financial signals  
        self.logger.info("Step 2.2: Normalizing Financial signals...")
        financial_signals = self.normalize_financial_signals(financial_data)
        
        # 3. Normalize News signals
        self.logger.info("Step 2.3: Normalizing News signals...")
        news_signals = self.normalize_news_signals(news_data)
        
        # Combine all normalized signals
        all_signals = {
            'reddit_signals': reddit_signals,
            'financial_signals': financial_signals,
            'news_signals': news_signals
        }
        
        phase2_end = datetime.now()
        execution_time = (phase2_end - phase2_start).total_seconds()
        
        total_signals = len(reddit_signals) + len(financial_signals) + len(news_signals)
        
        self.logger.info("=" * 60)
        self.logger.info(f"PHASE 2 COMPLETE - {execution_time:.2f}s")
        self.logger.info(f"  Reddit signals: {len(reddit_signals)}")
        self.logger.info(f"  Financial signals: {len(financial_signals)}")
        self.logger.info(f"  News signals: {len(news_signals)}")
        self.logger.info(f"  Total signals: {total_signals}")
        self.logger.info("=" * 60)
        
        return all_signals
    
    def normalize_reddit_signals(self, reddit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize Reddit data into signal format.
        
        Args:
            reddit_data: Raw Reddit data from Phase 1
        
        Returns:
            List of normalized Reddit signals (NO SCORES - just data)
        """
        normalized_signals = []
        
        ticker_mentions = reddit_data.get('ticker_mentions', {})
        
        for ticker, data in ticker_mentions.items():
            try:
                # Extract Reddit metrics (no scoring - just normalization)
                mention_count = data.get('mention_count', 0)
                avg_sentiment = data.get('avg_sentiment', 0.0)
                avg_score = data.get('avg_score', 0.0)
                mentions = data.get('mentions', [])
                
                # Create normalized signal structure
                signal = {
                    'ticker': ticker.upper(),
                    'source': 'reddit',
                    'data_type': 'sentiment',
                    'metrics': {
                        'mention_count': mention_count,
                        'avg_sentiment': avg_sentiment,
                        'avg_score': avg_score,
                        'confidence': self._calculate_confidence(mention_count)
                    },
                    'raw_data': {
                        'mentions': mentions,
                        'total_mentions': mention_count
                    },
                    'normalized_at': datetime.now().isoformat()
                }
                
                normalized_signals.append(signal)
                
            except Exception as e:
                self.logger.warning(f"Failed to normalize Reddit data for {ticker}: {e}")
                continue
        
        self.logger.info(f"✅ Normalized {len(normalized_signals)} Reddit signals")
        return normalized_signals
    
    def normalize_financial_signals(self, financial_data: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Normalize financial data into signal format.
        
        Args:
            financial_data: Ticker cache from Phase 1
        
        Returns:
            List of normalized financial signals (NO SCORES - just data)
        """
        normalized_signals = []
        
        for ticker, ticker_data in financial_data.items():
            try:
                # Skip if no data
                if not ticker_data or ticker_data.get('stock') is None:
                    continue
                
                # Convert cached data to financial metrics
                financial_metrics = self._extract_financial_metrics(ticker_data)
                
                if not financial_metrics:
                    continue
                
                # Create normalized signal structure
                signal = {
                    'ticker': ticker.upper(),
                    'source': 'yahoo_finance',
                    'data_type': 'financial',
                    'metrics': financial_metrics,
                    'raw_data': {
                        'info': ticker_data.get('info', {}),
                        'phase3_data': ticker_data.get('phase3_data', {}),
                        'fetched_at': ticker_data.get('fetched_at')
                    },
                    'normalized_at': datetime.now().isoformat()
                }
                
                normalized_signals.append(signal)
                
            except Exception as e:
                self.logger.warning(f"Failed to normalize financial data for {ticker}: {e}")
                continue
        
        self.logger.info(f"✅ Normalized {len(normalized_signals)} financial signals")
        return normalized_signals
    
    def normalize_news_signals(self, news_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize news data into signal format.
        
        Args:
            news_data: News data from Phase 1
        
        Returns:
            List of normalized news signals (NO SCORES - just data)
        """
        normalized_signals = []
        
        for ticker, ticker_news in news_data.items():
            try:
                # Skip if no news data
                if not ticker_news or ticker_news.get('news_mentions', 0) == 0:
                    continue
                
                # Create normalized signal structure
                signal = {
                    'ticker': ticker.upper(),
                    'source': 'news',
                    'data_type': 'sentiment',
                    'metrics': {
                        'news_sentiment_score': ticker_news.get('news_sentiment_score'),
                        'news_mentions': ticker_news.get('news_mentions', 0),
                        'confidence': min(ticker_news.get('news_mentions', 0) / 5, 1.0)
                    },
                    'raw_data': {
                        'articles': ticker_news.get('articles', []),
                        'ai_summary': ticker_news.get('ai_news_summary')
                    },
                    'normalized_at': datetime.now().isoformat()
                }
                
                normalized_signals.append(signal)
                
            except Exception as e:
                self.logger.warning(f"Failed to normalize news data for {ticker}: {e}")
                continue
        
        self.logger.info(f"✅ Normalized {len(normalized_signals)} news signals")
        return normalized_signals
    
    def _calculate_confidence(self, mention_count: int) -> float:
        """
        Calculate confidence based on mention count.
        
        More mentions = higher confidence (capped at 1.0)
        """
        return min(mention_count / 10.0, 1.0)
    
    def _extract_financial_metrics(self, ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract financial metrics from cached ticker data.
        
        Converts yfinance data structure into normalized metrics.
        
        Args:
            ticker_data: Cached data from Phase 1
        
        Returns:
            Dict of financial metrics (or None if extraction fails)
        """
        try:
            info = ticker_data.get('info', {})
            history_1m = ticker_data.get('history_1m')
            history_3m = ticker_data.get('history_3m')
            history_1y = ticker_data.get('history_1y')
            phase3_data = ticker_data.get('phase3_data', {})
            
            # Extract basic metrics
            metrics = {
                # Price metrics
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'market_cap': info.get('marketCap'),
                'volume': info.get('volume'),
                'avg_volume': info.get('averageVolume'),
                
                # Financial metrics
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                
                # Growth metrics
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'profit_margin': info.get('profitMargin'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                
                # Analyst data (Phase 3)
                'analyst_target_price': phase3_data.get('analyst_target_price'),
                'analyst_recommendation': phase3_data.get('analyst_recommendation'),
                'num_analyst_opinions': phase3_data.get('num_analyst_opinions'),
                
                # Earnings data (Phase 3)
                'earnings_surprise_pct': phase3_data.get('earnings_surprise_pct'),
                'earnings_beat_rate': phase3_data.get('earnings_beat_rate'),
                
                # Institutional data (Phase 3)
                'institutional_ownership_pct': phase3_data.get('institutional_ownership_pct'),
                'num_institutions': phase3_data.get('num_institutions'),
                
                # Insider data (Phase 3)
                'insider_ownership_pct': phase3_data.get('insider_ownership_pct'),
                'insider_net_bought_value': phase3_data.get('insider_net_bought_value'),
                
                # Additional info
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'company_name': info.get('longName', info.get('shortName')),
            }
            
            # Calculate price momentum if we have history
            if not history_1m.empty:
                try:
                    current_price = float(history_1m['Close'].iloc[-1])
                    price_1d_ago = float(history_1m['Close'].iloc[-2]) if len(history_1m) > 1 else current_price
                    price_1w_ago = float(history_1m['Close'].iloc[-5]) if len(history_1m) > 5 else current_price
                    price_1m_ago = float(history_1m['Close'].iloc[0])
                    
                    metrics['price_change_1d'] = ((current_price - price_1d_ago) / price_1d_ago * 100)
                    metrics['price_change_1w'] = ((current_price - price_1w_ago) / price_1w_ago * 100)
                    metrics['price_change_1m'] = ((current_price - price_1m_ago) / price_1m_ago * 100)
                    
                    # Update current price with actual latest price
                    metrics['current_price'] = current_price
                except Exception as e:
                    self.logger.debug(f"Failed to calculate price momentum: {e}")
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to extract financial metrics: {e}")
            return None
