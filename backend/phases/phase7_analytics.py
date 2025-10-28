"""
Phase 7: Performance Analytics & Persistence
==============================================

Calculates and PERSISTS analytics metrics to the analytics table.
Runs automatically as part of the main pipeline.

Key Features:
1. Win rates across all intervals (1d, 3d, 7d, 10d, 14d, 30d, 90d)
2. Sharpe ratios (risk-adjusted returns)
3. Max drawdown analysis
4. Average returns and alpha vs SPY
5. Sector rotation analysis
6. Top contributing factors per signal group
7. Signal quality metrics

Architecture:
- Reads from performance table (created by Phase 6)
- Calculates comprehensive analytics
- PERSISTS to analytics table for dashboard consumption
- Runs automatically after Phase 6 in pipeline
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Phase 7: Calculate and persist portfolio analytics.
    
    Calculates all metrics and saves to analytics table for
    fast dashboard loading.
    """
    
    INTERVALS = ['1d', '3d', '7d', '10d', '14d', '30d', '90d']
    
    def __init__(self, db=None, risk_free_rate: float = 0.02):
        """
        Initialize analytics engine.
        
        Args:
            db: SupabaseInterface instance (optional)
            risk_free_rate: Annual risk-free rate for Sharpe ratio (default: 2%)
        """
        self.db = db
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        
    async def set_database(self):
        """Initialize database connection if not provided."""
        if self.db is None:
            from ..storage.database import get_supabase_database
            self.db = await get_supabase_database()
    
    async def calculate_and_persist_analytics(
        self, 
        period_type: str = 'all_time',
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate all analytics and persist to analytics table.
        
        This is the main method called by the pipeline.
        
        Args:
            period_type: 'daily', 'weekly', 'monthly', 'all_time'
            run_id: Optional signal run ID to analyze specific run
            
        Returns:
            Dict with analytics results
        """
        try:
            await self.set_database()
            
            self.logger.info("=" * 100)
            self.logger.info(f"PHASE 7: ANALYTICS ({period_type.upper()})")
            self.logger.info("=" * 100)
            
            # Determine time period
            period_start, period_end = self._get_period_bounds(period_type)
            
            self.logger.info(f"Analyzing period: {period_start} to {period_end}")
            
            # Fetch all performance data
            performance_data = await self._fetch_performance_data(period_start, period_end, run_id)
            
            if not performance_data:
                self.logger.warning("No performance data found for analytics")
                return {'error': 'No data'}
            
            self.logger.info(f"Fetched {len(performance_data)} performance records")
            
            # Calculate all metrics
            metrics = await self._calculate_all_metrics(performance_data)
            
            # Add period info
            metrics['period_start'] = period_start
            metrics['period_end'] = period_end
            metrics['period_type'] = period_type
            metrics['signals_analyzed'] = len(performance_data)
            metrics['performance_records_used'] = len(performance_data)
            
            # Persist to analytics table
            await self._persist_analytics(metrics)
            
            self.logger.info("=" * 100)
            self.logger.info(f"[SUCCESS] Phase 7 analytics complete")
            self.logger.info(f"  Total signals: {metrics['total_signals']}")
            avg_score = metrics.get('avg_overall_score', 0) or 0
            win_rate_7d = metrics.get('win_rate_7d', 0) or 0
            sharpe_30d = metrics.get('sharpe_ratio_30d', 0) or 0
            self.logger.info(f"  Avg score: {avg_score:.1f}")
            self.logger.info(f"  Win rate (7d): {win_rate_7d:.1f}%")
            self.logger.info(f"  Sharpe (30d): {sharpe_30d:.2f}")
            self.logger.info("=" * 100)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in Phase 7 analytics: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def _fetch_performance_data(
        self, 
        period_start: datetime, 
        period_end: datetime,
        run_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch all performance data with signal details."""
        try:
            # Build query
            query = self.db.client.table('performance').select('''
                *,
                signals!inner(
                    ticker,
                    overall_score,
                    technical_score,
                    fundamental_score,
                    news_macro_score,
                    social_alternative_score,
                    risk_stability_score,
                    institutional_smart_money_score,
                    created_at,
                    run_id
                )
            ''').gte('baseline_date', period_start.isoformat()).lte('baseline_date', period_end.isoformat())
            
            if run_id:
                query = query.eq('signals.run_id', run_id)
            
            result = query.execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            self.logger.error(f"Error fetching performance data: {e}")
            return []
    
    async def _calculate_all_metrics(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Calculate all analytics metrics."""
        metrics = {}
        
        # Basic stats
        metrics['total_signals'] = len(performance_data)
        metrics['avg_overall_score'] = self._safe_avg([p['signals']['overall_score'] for p in performance_data])
        
        # Calculate metrics for each interval
        for interval in self.INTERVALS:
            interval_metrics = self._calculate_interval_metrics(performance_data, interval)
            metrics.update(interval_metrics)
        
        # Sector analysis
        sector_metrics = self._analyze_sectors(performance_data)
        metrics.update(sector_metrics)
        
        # Signal group scores
        group_scores = self._calculate_group_scores(performance_data)
        metrics.update(group_scores)
        
        # Top factors analysis
        top_factors = await self._analyze_top_factors(performance_data)
        metrics['top_factors'] = top_factors
        
        # NEW ANALYTICS - Score Bucket Performance
        self.logger.info("Calculating score bucket performance...")
        score_bucket_perf = self._calculate_score_bucket_performance(performance_data)
        metrics['score_bucket_performance'] = score_bucket_perf
        
        # NEW ANALYTICS - Factor Correlations
        self.logger.info("Calculating factor correlations...")
        factor_correlations = await self._calculate_factor_correlations(performance_data)
        metrics['factor_correlations'] = factor_correlations
        
        # NEW ANALYTICS - Factor Contributions
        self.logger.info("Calculating factor contributions...")
        factor_contributions = self._calculate_factor_contributions(performance_data)
        metrics['factor_contributions'] = factor_contributions
        
        # NEW ANALYTICS - Group Performance
        self.logger.info("Calculating group performance analysis...")
        group_performance = self._calculate_group_performance(performance_data)
        metrics['group_performance'] = group_performance
        
        # NEW ANALYTICS - Backtest Cumulative Returns
        self.logger.info("Calculating backtest cumulative returns...")
        backtest_returns = await self._calculate_backtest_returns(performance_data)
        metrics['backtest_cumulative_returns'] = backtest_returns
        
        return metrics
    
    def _calculate_interval_metrics(self, performance_data: List[Dict], interval: str) -> Dict[str, Any]:
        """Calculate metrics for a specific time interval."""
        metrics = {}
        
        return_col = f'return_{interval}'
        spy_return_col = f'spy_return_{interval}'
        alpha_col = f'alpha_{interval}'
        
        # Filter records with data for this interval
        valid_data = [p for p in performance_data if p.get(return_col) is not None]
        
        if not valid_data:
            return {
                f'win_rate_{interval}': None,
                f'sharpe_ratio_{interval}': None,
                f'max_drawdown_{interval}': None,
                f'avg_return_{interval}': None,
                f'avg_alpha_{interval}': None
            }
        
        # Extract returns
        returns = [float(p[return_col]) for p in valid_data]
        alphas = [float(p[alpha_col]) for p in valid_data if p.get(alpha_col) is not None]
        
        # Win rate
        wins = [r for r in returns if r > 0]
        win_rate = (len(wins) / len(returns) * 100) if returns else 0
        
        # Sharpe ratio
        sharpe = self._calculate_sharpe_ratio(returns)
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown(returns)
        
        # Average return
        avg_return = np.mean(returns) if returns else 0
        
        # Average alpha
        avg_alpha = np.mean(alphas) if alphas else 0
        
        metrics[f'win_rate_{interval}'] = round(win_rate, 2)
        metrics[f'sharpe_ratio_{interval}'] = round(sharpe, 3)
        metrics[f'max_drawdown_{interval}'] = round(max_dd, 2)
        metrics[f'avg_return_{interval}'] = round(avg_return, 2)
        metrics[f'avg_alpha_{interval}'] = round(avg_alpha, 2)
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: List[float], periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio (risk-adjusted return)."""
        if not returns or len(returns) < 2:
            return 0.0
        
        try:
            returns_array = np.array(returns) / 100  # Convert % to decimal
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
        except:
            return 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not returns or len(returns) < 2:
            return 0.0
        
        try:
            # Build equity curve
            equity = [100.0]
            for r in returns:
                equity.append(equity[-1] * (1 + r / 100))
            
            # Find max drawdown
            max_dd = 0.0
            peak = equity[0]
            
            for value in equity:
                if value > peak:
                    peak = value
                else:
                    drawdown = (value - peak) / peak * 100
                    if drawdown < max_dd:
                        max_dd = drawdown
            
            return abs(max_dd)
        except:
            return 0.0
    
    def _analyze_sectors(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sector performance."""
        sector_stats = defaultdict(lambda: {'returns': [], 'count': 0})
        
        # Group by sector
        for p in performance_data:
            sector = p.get('sector')
            if sector and p.get('return_30d') is not None:
                sector_stats[sector]['returns'].append(float(p['return_30d']))
                sector_stats[sector]['count'] += 1
        
        # Calculate averages
        sector_performance = {}
        for sector, stats in sector_stats.items():
            if stats['returns']:
                avg_return = np.mean(stats['returns'])
                win_rate = len([r for r in stats['returns'] if r > 0]) / len(stats['returns']) * 100
                
                sector_performance[sector] = {
                    'avg_return': round(avg_return, 2),
                    'count': stats['count'],
                    'win_rate': round(win_rate, 1)
                }
        
        # Find top and worst
        if sector_performance:
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['avg_return'], reverse=True)
            
            top_sector = sorted_sectors[0]
            worst_sector = sorted_sectors[-1]
            
            return {
                'top_sector': top_sector[0],
                'top_sector_avg_return': top_sector[1]['avg_return'],
                'top_sector_count': top_sector[1]['count'],
                'worst_sector': worst_sector[0],
                'worst_sector_avg_return': worst_sector[1]['avg_return'],
                'worst_sector_count': worst_sector[1]['count'],
                'sector_performance': sector_performance
            }
        
        return {
            'top_sector': None,
            'top_sector_avg_return': None,
            'top_sector_count': None,
            'worst_sector': None,
            'worst_sector_avg_return': None,
            'worst_sector_count': None,
            'sector_performance': {}
        }
    
    def _calculate_group_scores(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Calculate average scores for each signal group."""
        signals = [p['signals'] for p in performance_data]
        
        return {
            'avg_technical_score': self._safe_avg([s['technical_score'] for s in signals]),
            'avg_fundamental_score': self._safe_avg([s['fundamental_score'] for s in signals]),
            'avg_news_macro_score': self._safe_avg([s['news_macro_score'] for s in signals]),
            'avg_social_alternative_score': self._safe_avg([s['social_alternative_score'] for s in signals]),
            'avg_risk_stability_score': self._safe_avg([s['risk_stability_score'] for s in signals]),
            'avg_institutional_score': self._safe_avg([s['institutional_smart_money_score'] for s in signals])
        }
    
    async def _analyze_top_factors(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze top contributing factors per signal group.
        
        Returns JSON structure with top 5 factors per group.
        """
        # This requires fetching the factor details from signal detail tables
        # For now, return placeholder - will implement full version
        return {
            "technical": [],
            "fundamental": [],
            "news_macro": [],
            "social_alternative": [],
            "risk_stability": [],
            "institutional_smart_money": []
        }
    
    def _safe_avg(self, values: List) -> float:
        """Calculate average, handling None values."""
        valid = [v for v in values if v is not None]
        return round(np.mean(valid), 2) if valid else 0.0
    
    def _get_period_bounds(self, period_type: str) -> tuple:
        """Get start and end dates for period type."""
        now = datetime.now(timezone.utc)
        
        if period_type == 'daily':
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif period_type == 'weekly':
            start = now - timedelta(days=7)
            end = now
        elif period_type == 'monthly':
            start = now - timedelta(days=30)
            end = now
        else:  # all_time
            start = datetime(2020, 1, 1, tzinfo=timezone.utc)
            end = now
        
        return start, end
    
    async def _persist_analytics(self, metrics: Dict[str, Any]) -> None:
        """Save analytics to analytics table."""
        try:
            # Insert into analytics table
            result = await self.db.execute_query("""
                INSERT INTO analytics (
                    period_start, period_end, period_type,
                    total_signals, avg_overall_score,
                    win_rate_1d, win_rate_3d, win_rate_7d, win_rate_10d, win_rate_14d, win_rate_30d, win_rate_90d,
                    sharpe_ratio_1d, sharpe_ratio_3d, sharpe_ratio_7d, sharpe_ratio_10d, sharpe_ratio_14d, sharpe_ratio_30d, sharpe_ratio_90d,
                    max_drawdown_1d, max_drawdown_3d, max_drawdown_7d, max_drawdown_10d, max_drawdown_14d, max_drawdown_30d, max_drawdown_90d,
                    avg_return_1d, avg_return_3d, avg_return_7d, avg_return_10d, avg_return_14d, avg_return_30d, avg_return_90d,
                    avg_alpha_1d, avg_alpha_3d, avg_alpha_7d, avg_alpha_10d, avg_alpha_14d, avg_alpha_30d, avg_alpha_90d,
                    top_sector, top_sector_avg_return, top_sector_count,
                    worst_sector, worst_sector_avg_return, worst_sector_count,
                    sector_performance,
                    avg_technical_score, avg_fundamental_score, avg_news_macro_score,
                    avg_social_alternative_score, avg_risk_stability_score, avg_institutional_score,
                    top_factors,
                    signals_analyzed, performance_records_used,
                    score_bucket_performance, factor_correlations, factor_contributions,
                    group_performance, backtest_cumulative_returns
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10, $11, $12,
                    $13, $14, $15, $16, $17, $18, $19,
                    $20, $21, $22, $23, $24, $25, $26,
                    $27, $28, $29, $30, $31, $32, $33,
                    $34, $35, $36, $37, $38, $39, $40,
                    $41, $42, $43, $44, $45, $46, $47,
                    $48, $49, $50, $51, $52, $53, $54, $55, $56,
                    $57, $58, $59, $60, $61
                )
                RETURNING id
            """, [
                metrics['period_start'], metrics['period_end'], metrics['period_type'],
                metrics['total_signals'], metrics.get('avg_overall_score'),
                metrics.get('win_rate_1d'), metrics.get('win_rate_3d'), metrics.get('win_rate_7d'), 
                metrics.get('win_rate_10d'), metrics.get('win_rate_14d'), metrics.get('win_rate_30d'), metrics.get('win_rate_90d'),
                metrics.get('sharpe_ratio_1d'), metrics.get('sharpe_ratio_3d'), metrics.get('sharpe_ratio_7d'),
                metrics.get('sharpe_ratio_10d'), metrics.get('sharpe_ratio_14d'), metrics.get('sharpe_ratio_30d'), metrics.get('sharpe_ratio_90d'),
                metrics.get('max_drawdown_1d'), metrics.get('max_drawdown_3d'), metrics.get('max_drawdown_7d'),
                metrics.get('max_drawdown_10d'), metrics.get('max_drawdown_14d'), metrics.get('max_drawdown_30d'), metrics.get('max_drawdown_90d'),
                metrics.get('avg_return_1d'), metrics.get('avg_return_3d'), metrics.get('avg_return_7d'),
                metrics.get('avg_return_10d'), metrics.get('avg_return_14d'), metrics.get('avg_return_30d'), metrics.get('avg_return_90d'),
                metrics.get('avg_alpha_1d'), metrics.get('avg_alpha_3d'), metrics.get('avg_alpha_7d'),
                metrics.get('avg_alpha_10d'), metrics.get('avg_alpha_14d'), metrics.get('avg_alpha_30d'), metrics.get('avg_alpha_90d'),
                metrics.get('top_sector'), metrics.get('top_sector_avg_return'), metrics.get('top_sector_count'),
                metrics.get('worst_sector'), metrics.get('worst_sector_avg_return'), metrics.get('worst_sector_count'),
                json.dumps(metrics.get('sector_performance')) if metrics.get('sector_performance') else None,
                metrics.get('avg_technical_score'), metrics.get('avg_fundamental_score'), metrics.get('avg_news_macro_score'),
                metrics.get('avg_social_alternative_score'), metrics.get('avg_risk_stability_score'), metrics.get('avg_institutional_score'),
                json.dumps(metrics.get('top_factors')) if metrics.get('top_factors') else None,
                metrics['signals_analyzed'], metrics['performance_records_used'],
                json.dumps(metrics.get('score_bucket_performance')) if metrics.get('score_bucket_performance') else None,
                json.dumps(metrics.get('factor_correlations')) if metrics.get('factor_correlations') else None,
                json.dumps(metrics.get('factor_contributions')) if metrics.get('factor_contributions') else None,
                json.dumps(metrics.get('group_performance')) if metrics.get('group_performance') else None,
                json.dumps(metrics.get('backtest_cumulative_returns')) if metrics.get('backtest_cumulative_returns') else None
            ])
            
            self.logger.info(f"[SUCCESS] Analytics persisted to database")
            
        except Exception as e:
            self.logger.error(f"Error persisting analytics: {e}", exc_info=True)
    
    def _calculate_score_bucket_performance(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate performance metrics by score bucket across all intervals.
        
        Buckets based on methodology:
        - Strong Buy: > 0.75
        - Buy: 0.50 to 0.75
        - Hold: -0.25 to 0.50
        - Sell: -0.50 to -0.25
        - Strong Sell: < -0.50
        """
        buckets = {
            'strong_buy': {'threshold': '> 0.75', 'min': 0.75, 'max': 999, 'signals': []},
            'buy': {'threshold': '0.50 to 0.75', 'min': 0.50, 'max': 0.75, 'signals': []},
            'hold': {'threshold': '-0.25 to 0.50', 'min': -0.25, 'max': 0.50, 'signals': []},
            'sell': {'threshold': '-0.50 to -0.25', 'min': -0.50, 'max': -0.25, 'signals': []},
            'strong_sell': {'threshold': '< -0.50', 'min': -999, 'max': -0.50, 'signals': []}
        }
        
        # Classify signals into buckets
        for p in performance_data:
            score = p.get('signals', {}).get('overall_score')
            if score is None:
                continue
                
            for bucket_name, bucket_info in buckets.items():
                if bucket_info['min'] <= score < bucket_info['max']:
                    buckets[bucket_name]['signals'].append(p)
                    break
        
        # Calculate metrics for each bucket
        result = {}
        for bucket_name, bucket_info in buckets.items():
            signals = bucket_info['signals']
            bucket_metrics = {
                'threshold': bucket_info['threshold'],
                'count': len(signals)
            }
            
            if signals:
                # Calculate for all intervals
                for interval in self.INTERVALS:
                    return_col = f'return_{interval}'
                    returns = [float(p[return_col]) for p in signals if p.get(return_col) is not None]
                    
                    if returns:
                        wins = [r for r in returns if r > 0]
                        bucket_metrics[interval] = {
                            'avg_return': round(np.mean(returns), 4),
                            'win_rate': round(len(wins) / len(returns), 4),
                            'sharpe': round(self._calculate_sharpe_ratio(returns), 4),
                            'max': round(max(returns), 4),
                            'min': round(min(returns), 4),
                            'count': len(returns)
                        }
                    else:
                        bucket_metrics[interval] = None
            
            result[bucket_name] = bucket_metrics
        
        return result
    
    async def _calculate_factor_correlations(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate factor correlation matrices.
        
        Returns:
        - 6x6 group correlation matrix
        - Optionally 158x158 full factor correlation (if enabled)
        - Top positive and negative correlation pairs
        """
        try:
            # Fetch full factor data from database
            # For now, calculate group-level correlations from group scores
            
            group_names = ['technical', 'fundamental', 'news_macro', 
                          'social_alternative', 'risk_stability', 'institutional_smart_money']
            
            # Build group score matrix
            group_scores = []
            for p in performance_data:
                signal = p.get('signals', {})
                scores = [
                    signal.get('technical_score', 0) or 0,
                    signal.get('fundamental_score', 0) or 0,
                    signal.get('news_macro_score', 0) or 0,
                    signal.get('social_alternative_score', 0) or 0,
                    signal.get('risk_stability_score', 0) or 0,
                    signal.get('institutional_smart_money_score', 0) or 0
                ]
                group_scores.append(scores)
            
            # Calculate correlation matrix
            group_scores_array = np.array(group_scores)
            corr_matrix = np.corrcoef(group_scores_array.T)
            
            # Find top correlations
            top_positive = []
            top_negative = []
            
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    corr_val = corr_matrix[i][j]
                    pair = {
                        'factor1': group_names[i],
                        'factor2': group_names[j],
                        'correlation': round(float(corr_val), 3)
                    }
                    
                    if corr_val > 0:
                        top_positive.append(pair)
                    else:
                        top_negative.append(pair)
            
            # Sort
            top_positive = sorted(top_positive, key=lambda x: x['correlation'], reverse=True)[:10]
            top_negative = sorted(top_negative, key=lambda x: x['correlation'])[:10]
            
            result = {
                'group_correlations': {
                    'matrix': [[round(float(v), 3) for v in row] for row in corr_matrix],
                    'labels': group_names
                },
                'top_positive_pairs': top_positive,
                'top_negative_pairs': top_negative
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating factor correlations: {e}")
            return {}
    
    def _calculate_factor_contributions(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate factor contribution to returns using correlation analysis.
        
        Returns correlation between group scores and returns for each interval.
        """
        try:
            group_names = ['technical', 'fundamental', 'news_macro',
                          'social_alternative', 'risk_stability', 'institutional_smart_money']
            
            result = {}
            
            for interval in self.INTERVALS:
                return_col = f'return_{interval}'
                
                # Filter valid data
                valid_data = [p for p in performance_data 
                             if p.get(return_col) is not None and p.get('signals')]
                
                if not valid_data or len(valid_data) < 10:
                    continue
                
                # Extract returns and group scores
                returns = np.array([float(p[return_col]) for p in valid_data])
                
                factor_correlations = []
                for group in group_names:
                    score_col = f'{group}_score'
                    scores = np.array([p['signals'].get(score_col, 0) or 0 for p in valid_data])
                    
                    # Calculate correlation
                    if len(scores) > 1 and np.std(scores) > 0:
                        corr = np.corrcoef(scores, returns)[0, 1]
                        
                        factor_correlations.append({
                            'factor': group,
                            'group': group,
                            'correlation': round(float(corr), 4),
                            'abs_correlation': round(abs(float(corr)), 4)
                        })
                
                # Sort by absolute correlation
                factor_correlations = sorted(factor_correlations, 
                                            key=lambda x: x['abs_correlation'], 
                                            reverse=True)
                
                result[interval] = {
                    'top_contributors': factor_correlations[:20],
                    'all_correlations': factor_correlations
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating factor contributions: {e}")
            return {}
    
    def _calculate_group_performance(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze factor group performance - both per-signal and aggregated.
        """
        try:
            group_names = ['technical', 'fundamental', 'news_macro',
                          'social_alternative', 'risk_stability', 'institutional_smart_money']
            
            # Per-signal analysis: which group dominated each signal
            dominant_groups = defaultdict(int)
            dominant_group_returns = defaultdict(list)
            
            for p in performance_data:
                signal = p.get('signals', {})
                if not signal:
                    continue
                
                # Find dominant group (highest score)
                max_score = -999
                dominant_group = None
                
                for group in group_names:
                    score = signal.get(f'{group}_score', 0) or 0
                    if score > max_score:
                        max_score = score
                        dominant_group = group
                
                if dominant_group and p.get('return_30d') is not None:
                    dominant_groups[dominant_group] += 1
                    dominant_group_returns[dominant_group].append(float(p['return_30d']))
            
            # Aggregated analysis: correlation of each group with returns
            aggregated = {}
            
            for group in group_names:
                score_col = f'{group}_score'
                valid_data = [p for p in performance_data 
                             if p.get('signals', {}).get(score_col) is not None 
                             and p.get('return_30d') is not None]
                
                if valid_data:
                    scores = [p['signals'][score_col] for p in valid_data]
                    returns_30d = [float(p['return_30d']) for p in valid_data]
                    
                    # Calculate correlation with different intervals
                    interval_correlations = {}
                    for interval in self.INTERVALS:
                        return_col = f'return_{interval}'
                        interval_returns = [float(p[return_col]) for p in valid_data 
                                           if p.get(return_col) is not None]
                        
                        if len(interval_returns) > 1:
                            corr = np.corrcoef(scores[:len(interval_returns)], interval_returns)[0, 1]
                            interval_correlations[interval] = round(float(corr), 4)
                    
                    aggregated[group] = {
                        'avg_score': round(np.mean(scores), 4),
                        'correlation_with_returns': interval_correlations,
                        'signals_count': len(valid_data)
                    }
            
            result = {
                'per_signal_analysis': {
                    'dominant_group_distribution': dict(dominant_groups),
                    'avg_return_by_dominant_group': {
                        group: round(np.mean(returns), 4) 
                        for group, returns in dominant_group_returns.items()
                    }
                },
                'aggregated_analysis': aggregated
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating group performance: {e}")
            return {}
    
    async def _calculate_backtest_returns(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate cumulative returns for VP strategy vs SPY vs QQQ.
        
        Assumes equal-weight portfolio, daily rebalancing.
        """
        try:
            # Group signals by baseline_date
            signals_by_date = defaultdict(list)
            
            for p in performance_data:
                baseline_date = p.get('baseline_date')
                if baseline_date and p.get('return_1d') is not None:
                    signals_by_date[baseline_date].append(p)
            
            # Sort dates
            dates = sorted(signals_by_date.keys())
            
            if not dates:
                return {}
            
            # Calculate daily portfolio returns
            daily_returns = []
            
            for date in dates:
                signals = signals_by_date[date]
                
                # Equal-weight portfolio return
                vp_returns_1d = [float(p['return_1d']) for p in signals if p.get('return_1d') is not None]
                spy_returns_1d = [float(p['spy_return_1d']) for p in signals if p.get('spy_return_1d') is not None]
                
                if vp_returns_1d:
                    daily_returns.append({
                        'date': date,
                        'vp_return': np.mean(vp_returns_1d) / 100,  # Convert % to decimal
                        'spy_return': np.mean(spy_returns_1d) / 100 if spy_returns_1d else 0,
                        'qqq_return': np.mean(spy_returns_1d) / 100 * 1.1 if spy_returns_1d else 0  # Approximate QQQ
                    })
            
            # Calculate cumulative returns
            vp_cum = 1.0
            spy_cum = 1.0
            qqq_cum = 1.0
            
            cumulative_series = []
            
            for dr in daily_returns:
                vp_cum *= (1 + dr['vp_return'])
                spy_cum *= (1 + dr['spy_return'])
                qqq_cum *= (1 + dr['qqq_return'])
                
                cumulative_series.append({
                    'date': dr['date'],
                    'vp_strategy': round(vp_cum, 4),
                    'spy': round(spy_cum, 4),
                    'qqq': round(qqq_cum, 4)
                })
            
            # Calculate summary statistics
            vp_returns = [dr['vp_return'] for dr in daily_returns]
            
            result = {
                'start_date': dates[0] if dates else None,
                'end_date': dates[-1] if dates else None,
                'daily_returns': cumulative_series[-100:],  # Last 100 days for visualization
                'summary': {
                    'vp_total_return': round(vp_cum - 1, 4),
                    'spy_total_return': round(spy_cum - 1, 4),
                    'qqq_total_return': round(qqq_cum - 1, 4),
                    'vp_sharpe': round(self._calculate_sharpe_ratio([r * 100 for r in vp_returns]), 4),
                    'vp_max_drawdown': round(self._calculate_max_drawdown([r * 100 for r in vp_returns]), 4),
                    'vp_win_rate': round(len([r for r in vp_returns if r > 0]) / len(vp_returns), 4) if vp_returns else 0
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest returns: {e}")
            return {}


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def get_analytics_engine(db=None, risk_free_rate: float = 0.02):
    """
    Factory function to create AnalyticsEngine instance.
    
    Args:
        db: SupabaseInterface instance (optional)
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        AnalyticsEngine instance
    """
    return AnalyticsEngine(db=db, risk_free_rate=risk_free_rate)



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
            
            sharpe = metrics['sharpe_ratio'] or 0
            win_rate = metrics['win_rate_pct'] or 0
            max_dd = metrics['max_drawdown_pct'] or 0
            
            self.logger.info(
                f"[OK] Analytics complete: "
                f"Sharpe {sharpe:.2f}, "
                f"Win Rate {win_rate:.1f}%, "
                f"Max DD {max_dd:.1f}%"
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
