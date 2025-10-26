"""
Phase 7: Performance Analytics
================================

Calculates advanced performance metrics from historical return data.

This phase runs AFTER Phase 6 (Performance Tracking) to provide:
- Risk-adjusted metrics (Sharpe, Sortino)
- Drawdown analysis
- Win rate statistics
- Profit factor calculation
- Signal quality assessment

Key Features:
1. Sharpe Ratio - Risk-adjusted return vs volatility
2. Sortino Ratio - Downside risk-adjusted return
3. Max Drawdown - Worst peak-to-trough decline
4. Win Rate - Percentage of profitable signals
5. Profit Factor - Total profits / total losses ratio
6. Average returns by interval
7. Best/worst performing signals

Architecture:
- Reads from performance table (created by Phase 6)
- Calculates metrics on-demand (no new database tables)
- Provides aggregate statistics across all signals
- Can filter by ticker, date range, score threshold
- Frontend-ready JSON output

Design Decisions:
- Calculate metrics on-demand (flexible, no schema changes)
- Use numpy for efficient statistical calculations
- Support multiple time intervals (7d, 30d, 90d)
- Cache results to avoid redundant calculations
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Calculate advanced performance metrics from historical signal data.
    
    Provides risk-adjusted metrics, win rates, and drawdown analysis
    for signal performance evaluation.
    """
    
    def __init__(self, db=None, risk_free_rate: float = 0.02):
        """
        Initialize performance analytics.
        
        Args:
            db: SupabaseInterface instance (optional)
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default: 2%)
        """
        self.db = db
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        self._cache = {}  # Simple cache for expensive calculations
        
    async def set_database(self):
        """Initialize database connection if not provided."""
        if self.db is None:
            from ..storage.database import get_supabase_database
            self.db = await get_supabase_database()
    
    async def calculate_all_metrics(
        self, 
        interval: str = '7d',
        min_signals: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for all signals.
        
        Args:
            interval: Time interval to analyze ('7d', '30d', '90d')
            min_signals: Minimum number of signals required for valid stats
            
        Returns:
            Dict with all performance metrics
        """
        try:
            await self.set_database()
            
            self.logger.info(f"📊 Calculating performance analytics for {interval} interval...")
            
            # Get all performance data for the interval
            return_col = f'return_{interval}'
            spy_return_col = f'spy_return_{interval}'
            alpha_col = f'alpha_{interval}'
            
            result = self.db.client.table('performance').select(
                f'id, signal_id, baseline_price, baseline_date, {return_col}, {spy_return_col}, {alpha_col}, signals!inner(ticker, overall_score, created_at)'
            ).not_.is_(return_col, 'null').execute()
            
            performance_data = result.data if result.data else []
            
            if len(performance_data) < min_signals:
                self.logger.warning(
                    f"⚠️ Only {len(performance_data)} signals with {interval} data "
                    f"(minimum {min_signals} required)"
                )
                return self._empty_metrics()
            
            # Extract returns for calculations
            returns = [float(p[return_col]) for p in performance_data]
            spy_returns = [float(p[spy_return_col]) for p in performance_data if p.get(spy_return_col)]
            alphas = [float(p[alpha_col]) for p in performance_data if p.get(alpha_col)]
            
            # Calculate metrics
            sharpe = self.calculate_sharpe_ratio(returns)
            sortino = self.calculate_sortino_ratio(returns)
            win_stats = self.calculate_win_rate(returns)
            drawdown_stats = self.calculate_max_drawdown(performance_data, return_col)
            
            # Aggregate statistics
            metrics = {
                # Risk-adjusted metrics
                'sharpe_ratio': round(sharpe, 3),
                'sortino_ratio': round(sortino, 3),
                
                # Drawdown analysis
                'max_drawdown_pct': round(drawdown_stats['max_drawdown_pct'], 2),
                'max_drawdown_duration_days': drawdown_stats['max_drawdown_duration_days'],
                
                # Win rate statistics
                'win_rate_pct': round(win_stats['win_rate_pct'], 1),
                'win_count': win_stats['win_count'],
                'loss_count': win_stats['loss_count'],
                'profit_factor': round(win_stats['profit_factor'], 2),
                
                # Return statistics
                'avg_return_pct': round(np.mean(returns), 2),
                'median_return_pct': round(np.median(returns), 2),
                'std_return_pct': round(np.std(returns), 2),
                'min_return_pct': round(min(returns), 2),
                'max_return_pct': round(max(returns), 2),
                
                # Alpha statistics (vs SPY)
                'avg_alpha_pct': round(np.mean(alphas), 2) if alphas else 0.0,
                'positive_alpha_rate_pct': round(len([a for a in alphas if a > 0]) / len(alphas) * 100, 1) if alphas else 0.0,
                
                # Sample info
                'total_signals': len(performance_data),
                'interval': interval,
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(
                f"✅ Analytics complete: "
                f"Sharpe {metrics['sharpe_ratio']:.2f}, "
                f"Win Rate {metrics['win_rate_pct']:.1f}%, "
                f"Max DD {metrics['max_drawdown_pct']:.1f}%"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return self._empty_metrics()
    
    def calculate_sharpe_ratio(
        self, 
        returns: List[float], 
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Sharpe = (Return - RiskFreeRate) / Volatility
        
        Args:
            returns: List of return percentages
            periods_per_year: Trading periods per year (252 for daily)
            
        Returns:
            Sharpe ratio (higher is better, >1 is good, >2 is excellent)
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        try:
            returns_array = np.array(returns) / 100  # Convert % to decimal
            
            # Calculate annualized metrics
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)
            
            if std_return == 0:
                return 0.0
            
            # Annualize
            annualized_return = avg_return * periods_per_year
            annualized_std = std_return * np.sqrt(periods_per_year)
            
            # Sharpe ratio
            sharpe = (annualized_return - self.risk_free_rate) / annualized_std
            
            return float(sharpe)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(
        self, 
        returns: List[float], 
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).
        
        Like Sharpe but only penalizes DOWNSIDE volatility (losses).
        Better for asymmetric returns (big wins, small losses).
        
        Args:
            returns: List of return percentages
            periods_per_year: Trading periods per year
            
        Returns:
            Sortino ratio (higher is better)
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        try:
            returns_array = np.array(returns) / 100  # Convert % to decimal
            
            # Only consider downside volatility (negative returns)
            downside_returns = returns_array[returns_array < 0]
            
            if len(downside_returns) == 0:
                # No losses - infinite Sortino (cap at 999)
                return 999.0
            
            avg_return = np.mean(returns_array)
            downside_std = np.std(downside_returns, ddof=1)
            
            if downside_std == 0:
                return 999.0
            
            # Annualize
            annualized_return = avg_return * periods_per_year
            annualized_downside_std = downside_std * np.sqrt(periods_per_year)
            
            # Sortino ratio
            sortino = (annualized_return - self.risk_free_rate) / annualized_downside_std
            
            return float(sortino)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(
        self, 
        performance_data: List[Dict[str, Any]], 
        return_col: str
    ) -> Dict[str, Any]:
        """
        Calculate maximum drawdown from performance data.
        
        Drawdown = peak-to-trough decline in cumulative returns.
        
        Args:
            performance_data: List of performance records with returns
            return_col: Column name for returns (e.g., 'return_7d')
            
        Returns:
            Dict with max_drawdown_pct and max_drawdown_duration_days
        """
        if not performance_data or len(performance_data) < 2:
            return {'max_drawdown_pct': 0.0, 'max_drawdown_duration_days': 0}
        
        try:
            # Sort by baseline_date to simulate chronological order
            sorted_data = sorted(
                performance_data, 
                key=lambda x: x.get('baseline_date', '')
            )
            
            # Build equity curve (cumulative returns)
            equity_curve = [100.0]  # Start with $100
            for record in sorted_data:
                return_pct = float(record.get(return_col, 0))
                new_equity = equity_curve[-1] * (1 + return_pct / 100)
                equity_curve.append(new_equity)
            
            # Find maximum drawdown
            max_dd = 0.0
            max_dd_duration = 0
            peak = equity_curve[0]
            peak_idx = 0
            
            for i, equity in enumerate(equity_curve):
                if equity > peak:
                    # New peak
                    peak = equity
                    peak_idx = i
                else:
                    # In drawdown
                    drawdown = (equity - peak) / peak * 100  # As percentage
                    if drawdown < max_dd:
                        max_dd = drawdown
                        max_dd_duration = i - peak_idx
            
            return {
                'max_drawdown_pct': abs(max_dd),
                'max_drawdown_duration_days': max_dd_duration
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return {'max_drawdown_pct': 0.0, 'max_drawdown_duration_days': 0}
    
    def calculate_win_rate(self, returns: List[float]) -> Dict[str, Any]:
        """
        Calculate win rate and profit factor.
        
        Args:
            returns: List of return percentages
            
        Returns:
            Dict with win_count, loss_count, win_rate_pct, profit_factor
        """
        if not returns:
            return {
                'win_count': 0,
                'loss_count': 0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0
            }
        
        try:
            winners = [r for r in returns if r > 0]
            losers = [r for r in returns if r <= 0]
            
            win_rate = (len(winners) / len(returns) * 100) if returns else 0.0
            
            # Profit factor = total wins / total losses
            total_wins = sum(winners) if winners else 0
            total_losses = abs(sum(losers)) if losers else 0
            
            profit_factor = (total_wins / total_losses) if total_losses > 0 else 999.99
            
            return {
                'win_count': len(winners),
                'loss_count': len(losers),
                'win_rate_pct': win_rate,
                'profit_factor': min(profit_factor, 999.99)  # Cap at 999.99
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return {
                'win_count': 0,
                'loss_count': 0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0
            }
    
    async def get_best_performers(
        self, 
        interval: str = '7d', 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top performing signals by return.
        
        Args:
            interval: Time interval ('7d', '30d', '90d')
            limit: Number of top performers to return
            
        Returns:
            List of top performing signals with details
        """
        try:
            await self.set_database()
            
            return_col = f'return_{interval}'
            
            result = self.db.client.table('performance').select(
                f'id, baseline_price, baseline_date, {return_col}, signals!inner(ticker, overall_score, created_at)'
            ).not_.is_(
                return_col, 'null'
            ).order(
                return_col, desc=True
            ).limit(limit).execute()
            
            performers = result.data if result.data else []
            
            return [
                {
                    'ticker': p['signals']['ticker'],
                    'overall_score': p['signals']['overall_score'],
                    'return_pct': round(float(p[return_col]), 2),
                    'signal_date': p['signals']['created_at'],
                    'baseline_price': p['baseline_price']
                }
                for p in performers
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting best performers: {e}")
            return []
    
    async def get_worst_performers(
        self, 
        interval: str = '7d', 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get worst performing signals by return.
        
        Args:
            interval: Time interval ('7d', '30d', '90d')
            limit: Number of worst performers to return
            
        Returns:
            List of worst performing signals with details
        """
        try:
            await self.set_database()
            
            return_col = f'return_{interval}'
            
            result = self.db.client.table('performance').select(
                f'id, baseline_price, baseline_date, {return_col}, signals!inner(ticker, overall_score, created_at)'
            ).not_.is_(
                return_col, 'null'
            ).order(
                return_col, desc=False
            ).limit(limit).execute()
            
            performers = result.data if result.data else []
            
            return [
                {
                    'ticker': p['signals']['ticker'],
                    'overall_score': p['signals']['overall_score'],
                    'return_pct': round(float(p[return_col]), 2),
                    'signal_date': p['signals']['created_at'],
                    'baseline_price': p['baseline_price']
                }
                for p in performers
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting worst performers: {e}")
            return []
    
    async def get_performance_by_score_range(
        self, 
        interval: str = '7d',
        score_ranges: List[tuple] = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by signal score ranges.
        
        Useful for answering: "Do higher-scored signals perform better?"
        
        Args:
            interval: Time interval to analyze
            score_ranges: List of (min_score, max_score) tuples
            
        Returns:
            Dict mapping score range to performance metrics
        """
        try:
            await self.set_database()
            
            results = {}
            
            for min_score, max_score in score_ranges:
                return_col = f'return_{interval}'
                
                result = self.db.client.table('performance').select(
                    f'{return_col}, signals!inner(overall_score)'
                ).not_.is_(
                    return_col, 'null'
                ).gte(
                    'signals.overall_score', min_score
                ).lt(
                    'signals.overall_score', max_score
                ).execute()
                
                data = result.data if result.data else []
                
                if data:
                    returns = [float(d[return_col]) for d in data]
                    range_key = f"{min_score:.1f}-{max_score:.1f}"
                    
                    results[range_key] = {
                        'signal_count': len(returns),
                        'avg_return_pct': round(np.mean(returns), 2),
                        'median_return_pct': round(np.median(returns), 2),
                        'win_rate_pct': round(len([r for r in returns if r > 0]) / len(returns) * 100, 1)
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance by score: {e}")
            return {}
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when insufficient data."""
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'max_drawdown_duration_days': 0,
            'win_rate_pct': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'profit_factor': 0.0,
            'avg_return_pct': 0.0,
            'median_return_pct': 0.0,
            'std_return_pct': 0.0,
            'min_return_pct': 0.0,
            'max_return_pct': 0.0,
            'avg_alpha_pct': 0.0,
            'positive_alpha_rate_pct': 0.0,
            'total_signals': 0,
            'interval': '',
            'calculation_timestamp': datetime.now().isoformat(),
            'error': 'Insufficient data'
        }


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def get_performance_analytics(db=None, risk_free_rate: float = 0.02):
    """
    Factory function to create PerformanceAnalytics instance.
    
    Args:
        db: SupabaseInterface instance (optional)
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        PerformanceAnalytics instance
    """
    return PerformanceAnalytics(db=db, risk_free_rate=risk_free_rate)
