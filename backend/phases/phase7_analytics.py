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
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


def sanitize_for_json(obj):
    """
    Recursively sanitize NaN/Inf values for JSON serialization.
    Converts NaN/Inf to None (null in JSON).
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


class AnalyticsEngine:
    """
    Phase 7: Calculate and persist portfolio analytics.
    
    Calculates all metrics and saves to analytics table for
    fast dashboard loading.
    """
    
    INTERVALS = ['1d', '3d', '7d', '10d', '14d', '30d', '90d', 'all_time']
    
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
        self.score_ranges = self._load_score_ranges()
    
    def _load_score_ranges(self) -> Dict[str, Dict[str, float]]:
        """Load score interpretation ranges from methodology.yaml"""
        try:
            import yaml
            from pathlib import Path
            
            methodology_path = Path(__file__).parent.parent.parent / 'config' / 'methodology.yaml'
            
            with open(methodology_path, 'r') as f:
                methodology = yaml.safe_load(f)
            
            ranges_str = methodology['interpretation']['overall_score']['ranges']
            
            # Parse ranges into numeric thresholds
            ranges = {
                'strong_buy': {'min': 0.75, 'max': 999, 'label': ranges_str['strong_buy']},
                'buy': {'min': 0.50, 'max': 0.75, 'label': ranges_str['buy']},
                'hold': {'min': -0.25, 'max': 0.50, 'label': ranges_str['hold']},
                'sell': {'min': -0.50, 'max': -0.25, 'label': ranges_str['sell']},
                'strong_sell': {'min': -999, 'max': -0.50, 'label': ranges_str['strong_sell']}
            }
            
            return ranges
            
        except Exception as e:
            self.logger.warning(f"Could not load score ranges from methodology.yaml: {e}")
            # Fallback to hardcoded values
            return {
                'strong_buy': {'min': 0.75, 'max': 999, 'label': '> 0.75'},
                'buy': {'min': 0.50, 'max': 0.75, 'label': '0.50 to 0.75'},
                'hold': {'min': -0.25, 'max': 0.50, 'label': '-0.25 to 0.50'},
                'sell': {'min': -0.50, 'max': -0.25, 'label': '-0.50 to -0.25'},
                'strong_sell': {'min': -999, 'max': -0.50, 'label': '< -0.50'}
            }
    
    def _load_factor_definitions(self) -> Dict[str, List[str]]:
        """
        Load factor definitions from factor_to_group.yaml.
        Returns dict mapping group names to list of factor names.
        
        Returns:
            {
                'technical': ['rsi_14', 'macd_value', ...],
                'fundamental': ['pe_ratio', 'roe', ...],
                ...
            }
        """
        try:
            import yaml
            from pathlib import Path
            
            factor_path = Path(__file__).parent.parent.parent / 'config' / 'factor_to_group.yaml'
            
            with open(factor_path, 'r') as f:
                factor_config = yaml.safe_load(f)
            
            # Extract factor names from each group
            factor_groups = {}
            for group_name, factors in factor_config.items():
                if group_name == 'validation':
                    continue  # Skip validation section
                
                if isinstance(factors, dict):
                    # Extract factor names (keys in the dict)
                    factor_groups[group_name] = list(factors.keys())
            
            self.logger.info(f"Loaded {sum(len(f) for f in factor_groups.values())} factors from {len(factor_groups)} groups")
            
            return factor_groups
            
        except Exception as e:
            self.logger.error(f"Error loading factor definitions: {e}")
            return {}
        
    async def set_database(self):
        """Initialize database connection if not provided."""
        if self.db is None:
            from ..storage.database import get_supabase_database
            self.db = await get_supabase_database()
    
    async def calculate_and_persist_analytics(
        self, 
        run_id: str,
        period_type: str = 'all_time'
    ) -> Dict[str, Any]:
        """
        Calculate analytics for all holding periods and persist to analytics table.
        
        v3.6: Multi-period approach
        - Calculates 7 analytics rows: 1d, 3d, 7d, 10d, 14d, 30d, 90d
        - Uses ALL historical performance data (not just current run)
        - Each row focuses on a specific holding period
        - Frontend can easily switch between holding periods
        
        Args:
            run_id: Signal run ID to link analytics to (for reference)
            period_type: Deprecated, kept for compatibility
            
        Returns:
            Dict with analytics results for all holding periods
        """
        try:
            await self.set_database()
            
            self.logger.info("=" * 100)
            self.logger.info(f"PHASE 7: ANALYTICS (Multi-Period v3.6)")
            self.logger.info(f"  Latest Run ID: {run_id}")
            self.logger.info(f"  Holding Periods: {', '.join(self.INTERVALS)}")
            self.logger.info("=" * 100)
            
            # Fetch ALL performance data once (all-time)
            period_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
            period_end = datetime.now(timezone.utc)
            
            self.logger.info(f"\nFetching all performance data...")
            self.logger.info(f"  Period: {period_start.date()} to {period_end.date()}")
            
            performance_data = await self._fetch_performance_data(
                period_start, 
                period_end, 
                run_id=None  # Fetch all runs, not just current
            )
            
            if not performance_data:
                self.logger.warning(f"  No performance data found!")
                return {}
            
            self.logger.info(f"  Fetched {len(performance_data)} performance records")
            
            results = {}
            
            # Calculate analytics for each holding period
            for interval in self.INTERVALS:
                self.logger.info(f"\nCalculating analytics for interval: {interval}")
                
                # Calculate metrics focused on this specific holding period
                metrics = await self._calculate_interval_analytics(performance_data, interval)
                
                # Add metadata
                metrics['run_id'] = run_id  # Reference to latest run
                metrics['period_type'] = interval  # Use interval as period_type (1d, 3d, etc.)
                metrics['period_start'] = period_start
                metrics['period_end'] = period_end
                metrics['signals_analyzed'] = len(performance_data)
                metrics['performance_records_used'] = len(performance_data)
                
                # Persist to analytics table (one row per interval)
                await self._persist_analytics(metrics, run_id, interval)
                
                results[interval] = metrics
                
                self.logger.info(f"  Completed {interval}: {metrics.get('total_signals', 0)} signals")
            
            self.logger.info("=" * 100)
            self.logger.info(f"[SUCCESS] Phase 7 analytics complete")
            self.logger.info(f"  Intervals calculated: {len(results)}")
            for interval in self.INTERVALS:
                if interval in results:
                    self.logger.info(f"  {interval}: {results[interval].get('total_signals', 0)} signals")
            self.logger.info("=" * 100)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Phase 7 analytics: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def _fetch_performance_data(
        self, 
        period_start: datetime, 
        period_end: datetime,
        run_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch all performance data with signal details and factor values."""
        try:
            # Fetch performance data with signals
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
            
            if not result.data:
                return []
            
            # Get signal IDs for fetching factor values
            signal_ids = [record['signal_id'] for record in result.data if record.get('signal_id')]
            self.logger.debug(f"Fetching factor values for {len(signal_ids)} signals")
            
            if signal_ids:
                # Fetch factor values from signals_* tables
                signal_tables = {
                    'signals_technical': 'signals_technical',
                    'signals_fundamental': 'signals_fundamental',
                    'signals_news_macro': 'signals_news_macro',
                    'signals_social_alternative': 'signals_social_alternative',
                    'signals_risk_stability': 'signals_risk_stability',
                    'signals_institutional_smart_money': 'signals_institutional_smart_money'
                }
                
                # Fetch all signal tables in batches to avoid JSON size limits
                signal_data = {}
                batch_size = 100  # Fetch 100 signal_ids at a time
                
                for table_name in signal_tables.keys():
                    try:
                        # Fetch in batches - select all columns (factor values are individual columns)
                        for i in range(0, len(signal_ids), batch_size):
                            batch = signal_ids[i:i+batch_size]
                            table_result = self.db.client.table(table_name).select('*').in_('signal_id', batch).execute()
                            
                            if table_result.data:
                                # Log first batch sample
                                if i == 0:
                                    first_row = table_result.data[0]
                                    if isinstance(first_row, dict):
                                        # All columns except signal_id, created_at, updated_at are factor columns
                                        factor_cols = {k: v for k, v in first_row.items() if k not in ['signal_id', 'created_at', 'updated_at']}
                                        self.logger.info(f"[DEBUG FETCH] {table_name} first row has {len(factor_cols)} factor columns")
                                        if factor_cols:
                                            self.logger.info(f"[DEBUG FETCH] {table_name} sample factor names: {list(factor_cols.keys())[:5]}")
                                
                                # Create lookup by signal_id - store all factor columns in a dict
                                for row in table_result.data:
                                    if isinstance(row, dict):
                                        sig_id = row.get('signal_id')
                                        if sig_id:
                                            if sig_id not in signal_data:
                                                signal_data[sig_id] = {}
                                            # Extract the 'factors' JSONB column which contains all factor values
                                            factor_dict = row.get('factors', {})
                                            signal_data[sig_id][table_name] = factor_dict if isinstance(factor_dict, dict) else {}
                    
                    except Exception as e:
                        self.logger.warning(f"Could not fetch {table_name}: {e}")
                
                # Merge factor values into performance data
                for record in result.data:
                    signal_id = record.get('signal_id')
                    if signal_id and signal_id in signal_data:
                        for table_name in signal_tables.keys():
                            record[table_name] = signal_data[signal_id].get(table_name, {})
                    else:
                        # Initialize empty dicts if no data
                        for table_name in signal_tables.keys():
                            record[table_name] = {}
            
            return result.data
            
        except Exception as e:
            self.logger.error(f"Error fetching performance data: {e}")
            return []
    
    async def _calculate_interval_analytics(self, performance_data: List[Dict], interval: str) -> Dict[str, Any]:
        """
        Calculate analytics focused on a specific holding period (interval).
        
        Args:
            performance_data: All performance records
            interval: Specific holding period (e.g., '1d', '7d', '30d', '90d', 'all_time')
            
        Returns:
            Dict with analytics metrics focused on this interval
        """
        metrics = {}
        
        # Basic stats
        metrics['total_signals'] = len(performance_data)
        metrics['avg_overall_score'] = self._safe_avg([p['signals']['overall_score'] for p in performance_data])
        
        # Special handling for "all_time" - aggregate across all holding periods
        if interval == 'all_time':
            # Aggregate metrics from all 7 holding periods
            all_intervals = ['1d', '3d', '7d', '10d', '14d', '30d', '90d']
            aggregated_metrics = self._aggregate_all_time_metrics(performance_data, all_intervals)
            metrics.update(aggregated_metrics)
        else:
            # Calculate metrics specifically for this interval (primary focus)
            interval_metrics = self._calculate_interval_metrics(performance_data, interval)
            metrics.update(interval_metrics)
        
        # Sector analysis (using this interval's returns)
        sector_metrics = self._analyze_sectors(performance_data, interval)
        metrics.update(sector_metrics)
        
        # Signal group scores
        group_scores = self._calculate_group_scores(performance_data)
        metrics.update(group_scores)
        
        # Top factors analysis
        top_factors = await self._analyze_top_factors(performance_data)
        metrics['top_factors'] = top_factors
        
        # Score Bucket Performance (for this interval)
        score_bucket_perf = self._calculate_score_bucket_performance(performance_data, interval)
        metrics['score_bucket_performance'] = score_bucket_perf
        
        # Group Performance - performance metrics by each score group (technical, fundamental, etc.)
        self.logger.info(f"  Calculating group performance for {interval}...")
        group_performance = self._calculate_group_performance(performance_data, interval)
        metrics['group_performance'] = group_performance
        
        # Factor Correlations (using this interval's returns)
        factor_correlations = await self._calculate_factor_correlations(performance_data, interval)
        metrics['factor_correlations'] = factor_correlations
        
        # Factor Contributions (using this interval's returns)
        factor_contributions = self._calculate_factor_contributions(performance_data, interval)
        metrics['factor_contributions'] = factor_contributions
        
        # Backtest Cumulative Returns (for this interval)
        backtest_returns = await self._calculate_backtest_returns(performance_data, interval)
        metrics['backtest_cumulative_returns'] = backtest_returns
        
        # Predictive Strength (using this interval's returns)
        ic_series = self._calculate_rank_ic(performance_data, interval)
        metrics['ic_series'] = ic_series
        metrics['ic_mean'] = self._safe_avg([x['ic'] for x in ic_series]) if ic_series else 0.0
        metrics['ic_std'] = float(np.std([x['ic'] for x in ic_series])) if ic_series else 0.0
        metrics['hit_rate_top_decile'] = self._calculate_hit_rate_top_decile(performance_data, interval)
        metrics['profit_factor'] = self._calculate_profit_factor(performance_data, interval)
        metrics['win_loss_ratio'] = self._calculate_win_loss_ratio(performance_data, interval)
        
        # Benchmark Analysis (SPY and QQQ) - using interval-specific returns
        self.logger.info(f"  Calculating benchmark analysis for {interval}...")
        spy_metrics = self._calculate_benchmark_metrics(performance_data, 'SPY', interval)
        qqq_metrics = self._calculate_benchmark_metrics(performance_data, 'QQQ', interval)
        metrics.update(spy_metrics)
        metrics.update(qqq_metrics)
        
        # Additional Performance Metrics (interval-specific)
        self.logger.info(f"  Calculating performance metrics for {interval}...")
        metrics['cagr'] = self._calculate_cagr_for_interval(performance_data, interval)
        metrics['volatility'] = self._calculate_volatility_for_interval(performance_data, interval)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio_for_interval(performance_data, interval)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio_for_interval(performance_data, interval)
        
        # Benchmark Correlations (interval-specific)
        metrics['benchmark_correlations'] = self._calculate_benchmark_correlation_for_interval(performance_data, interval)
        
        # Factor-Return Correlations (individual factor correlations for ML feature importance)
        # Only calculate for intervals with sufficient data (1d, 3d)
        if interval in ['1d', '3d']:
            self.logger.info(f"  Calculating factor-return correlations for {interval}...")
            metrics['factor_return_correlations'] = await self._calculate_factor_return_correlations(
                performance_data, 
                interval,
                min_samples=50
            )
        else:
            metrics['factor_return_correlations'] = {}
        
        return metrics
    
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
        
        # VanPiQ ANALYTICS - Predictive Strength
        self.logger.info("Calculating predictive strength metrics...")
        ic_series = self._calculate_rank_ic(performance_data)
        metrics['ic_series'] = ic_series
        metrics['ic_mean'] = self._safe_avg([x['ic'] for x in ic_series]) if ic_series else 0.0
        metrics['ic_std'] = float(np.std([x['ic'] for x in ic_series])) if ic_series else 0.0
        metrics['hit_rate_top_decile'] = self._calculate_hit_rate_top_decile(performance_data)
        metrics['profit_factor'] = self._calculate_profit_factor(performance_data)
        metrics['win_loss_ratio'] = self._calculate_win_loss_ratio(performance_data)
        
        # VanPiQ ANALYTICS - Global Performance Summary
        self.logger.info("Calculating global performance metrics...")
        metrics['cagr'] = self._calculate_cagr(performance_data)
        metrics['volatility'] = self._calculate_volatility(performance_data)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(performance_data)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(performance_data)
        
        # For all_time, aggregate benchmark metrics across all intervals
        spy_metrics = self._calculate_all_time_benchmark_metrics(performance_data, 'SPY', ['1d', '3d', '7d', '10d', '14d', '30d', '90d'])
        qqq_metrics = self._calculate_all_time_benchmark_metrics(performance_data, 'QQQ', ['1d', '3d', '7d', '10d', '14d', '30d', '90d'])
        metrics.update(spy_metrics)
        metrics.update(qqq_metrics)
        
        # Benchmark Correlations
        metrics['benchmark_correlations'] = self._calculate_benchmark_correlation(performance_data)
        
        return metrics
    
    def _calculate_interval_metrics(self, performance_data: List[Dict], interval: str) -> Dict[str, Any]:
        """Calculate metrics for a specific time interval."""
        metrics = {}
        
        return_col = f'return_{interval}'
        spy_return_col = f'spy_return_{interval}'
        alpha_col = f'alpha_{interval}'
        
        # Extract interval days from string (e.g., "1d" -> 1)
        interval_days = int(interval.replace('d', ''))
        
        # Current time for age calculation
        now = datetime.now(timezone.utc)
        
        # Filter records with data for this interval AND sufficient time elapsed
        valid_data = []
        for p in performance_data:
            # Must have return data
            if p.get(return_col) is None:
                continue
            
            # Check if enough time has passed since baseline_date
            baseline_date = p.get('baseline_date')
            if baseline_date:
                # Parse baseline_date (ISO format string from database)
                try:
                    if isinstance(baseline_date, str):
                        baseline_dt = datetime.fromisoformat(baseline_date.replace('Z', '+00:00'))
                    else:
                        baseline_dt = baseline_date
                    
                    # Calculate days elapsed
                    days_elapsed = (now - baseline_dt).total_seconds() / 86400
                    
                    # Only include if sufficient time has passed
                    if days_elapsed >= interval_days:
                        valid_data.append(p)
                except (ValueError, AttributeError) as e:
                    # If date parsing fails, skip this record
                    self.logger.debug(f"Failed to parse baseline_date {baseline_date}: {e}")
                    continue
        
        if not valid_data:
            return {
                'win_rate': None,
                'sharpe_ratio': None,
                'max_drawdown': None,
                'avg_return': None,
                'avg_alpha': None
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
        
        # Store without interval suffix since each row is one interval
        metrics['win_rate'] = round(win_rate, 2)
        metrics['sharpe_ratio'] = round(sharpe, 3)
        metrics['max_drawdown'] = round(max_dd, 2)
        metrics['avg_return'] = round(avg_return, 2)
        metrics['avg_alpha'] = round(avg_alpha, 2)
        
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
    
    def _calculate_sortino_ratio_from_returns(self, returns: List[float], periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio from a list of returns (for score buckets)."""
        if not returns or len(returns) < 2:
            return 0.0
        
        try:
            returns_array = np.array(returns) / 100  # Convert % to decimal
            avg_return = np.mean(returns_array)
            
            # Downside returns only (negative returns)
            downside_returns = returns_array[returns_array < 0]
            
            if len(downside_returns) == 0:
                return 0.0
            
            downside_std = np.std(downside_returns, ddof=1)
            
            if downside_std == 0:
                return 0.0
            
            # Annualize
            annualized_return = avg_return * periods_per_year
            annualized_downside_std = downside_std * np.sqrt(periods_per_year)
            
            # Sortino ratio
            sortino = (annualized_return - self.risk_free_rate) / annualized_downside_std
            
            return float(sortino)
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
    
    def _aggregate_all_time_metrics(self, performance_data: List[Dict], all_intervals: List[str]) -> Dict[str, Any]:
        """
        Aggregate metrics across all holding periods for 'all_time' interval.
        
        Args:
            performance_data: All performance records
            all_intervals: List of intervals to aggregate (['1d', '3d', '7d', '10d', '14d', '30d', '90d'])
            
        Returns:
            Dict with aggregated metrics
        """
        all_returns = []
        all_alphas = []
        
        # Collect returns from all intervals
        for interval in all_intervals:
            return_col = f'return_{interval}'
            alpha_col = f'alpha_{interval}'
            
            for p in performance_data:
                if p.get(return_col) is not None:
                    all_returns.append(float(p[return_col]))
                if p.get(alpha_col) is not None:
                    all_alphas.append(float(p[alpha_col]))
        
        if not all_returns:
            return {
                'win_rate': None,
                'sharpe_ratio': None,
                'max_drawdown': None,
                'avg_return': None,
                'avg_alpha': None
            }
        
        # Win rate across all periods
        wins = [r for r in all_returns if r > 0]
        win_rate = (len(wins) / len(all_returns) * 100) if all_returns else 0
        
        # Sharpe ratio across all periods
        sharpe = self._calculate_sharpe_ratio(all_returns)
        
        # Max drawdown across all periods
        max_dd = self._calculate_max_drawdown(all_returns)
        
        # Average return across all periods
        avg_return = np.mean(all_returns) if all_returns else 0
        
        # Average alpha across all periods
        avg_alpha = np.mean(all_alphas) if all_alphas else 0
        
        return {
            'win_rate': round(win_rate, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_dd, 2),
            'avg_return': round(avg_return, 2),
            'avg_alpha': round(avg_alpha, 2)
        }
    
    def _analyze_sectors(self, performance_data: List[Dict], interval: str = '30d') -> Dict[str, Any]:
        """Analyze sector performance for a specific interval."""
        sector_stats = defaultdict(lambda: {'returns': [], 'count': 0})
        
        # Use the specified interval's return column
        return_col = f'return_{interval}'
        
        # Group by sector
        for p in performance_data:
            sector = p.get('sector')
            if sector and p.get(return_col) is not None:
                sector_stats[sector]['returns'].append(float(p[return_col]))
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
        # Use 4 decimal places to avoid rounding small scores to 0
        return round(np.mean(valid), 4) if valid else 0.0
    
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
    
    async def _persist_analytics(self, metrics: Dict[str, Any], run_id: str, period_type: str) -> None:
        """
        Save analytics to analytics table using period-based UPSERT (v3.5).
        
        Args:
            metrics: Calculated analytics metrics
            run_id: Signal run ID to link analytics to (for reference)
            period_type: Window type (all_time, 90d, 30d)
        """
        try:
            # v3.5: Period-based UPSERT - one row per window type
            result = await self.db.execute_query("""
                INSERT INTO analytics (
                    run_id,
                    period_type, period_start, period_end,
                    total_signals, avg_overall_score,
                    avg_technical_score, avg_fundamental_score, avg_news_macro_score,
                    avg_social_alternative_score, avg_risk_stability_score, avg_institutional_score,
                    top_sector, top_sector_avg_return, top_sector_count,
                    worst_sector, worst_sector_avg_return, worst_sector_count,
                    signals_analyzed, performance_records_used,
                    win_rate, sharpe_ratio, max_drawdown, avg_return, avg_alpha,
                    score_bucket_performance, group_performance, factor_correlations, factor_contributions,
                    backtest_cumulative_returns,
                    ic_series, ic_mean, ic_std, hit_rate_top_decile, profit_factor, win_loss_ratio,
                    cagr, volatility, sortino_ratio, calmar_ratio,
                    alpha_vs_spy, beta_vs_spy, alpha_vs_qqq, beta_vs_qqq,
                    benchmark_correlations,
                    factor_return_correlations
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7, $8, $9, $10, $11, $12,
                    $13, $14, $15, $16, $17, $18,
                    $19, $20, $21, $22, $23, $24, $25,
                    $26, $27, $28, $29, $30,
                    $31, $32, $33, $34, $35, $36,
                    $37, $38, $39, $40,
                    $41, $42, $43, $44,
                    $45, $46
                )
                ON CONFLICT (period_type)
                WHERE period_type IS NOT NULL
                DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    period_start = EXCLUDED.period_start,
                    period_end = EXCLUDED.period_end,
                    total_signals = EXCLUDED.total_signals,
                    avg_overall_score = EXCLUDED.avg_overall_score,
                    avg_technical_score = EXCLUDED.avg_technical_score,
                    avg_fundamental_score = EXCLUDED.avg_fundamental_score,
                    avg_news_macro_score = EXCLUDED.avg_news_macro_score,
                    avg_social_alternative_score = EXCLUDED.avg_social_alternative_score,
                    avg_risk_stability_score = EXCLUDED.avg_risk_stability_score,
                    avg_institutional_score = EXCLUDED.avg_institutional_score,
                    top_sector = EXCLUDED.top_sector,
                    top_sector_avg_return = EXCLUDED.top_sector_avg_return,
                    top_sector_count = EXCLUDED.top_sector_count,
                    worst_sector = EXCLUDED.worst_sector,
                    worst_sector_avg_return = EXCLUDED.worst_sector_avg_return,
                    worst_sector_count = EXCLUDED.worst_sector_count,
                    signals_analyzed = EXCLUDED.signals_analyzed,
                    performance_records_used = EXCLUDED.performance_records_used,
                    win_rate = EXCLUDED.win_rate,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    max_drawdown = EXCLUDED.max_drawdown,
                    avg_return = EXCLUDED.avg_return,
                    avg_alpha = EXCLUDED.avg_alpha,
                    score_bucket_performance = EXCLUDED.score_bucket_performance,
                    group_performance = EXCLUDED.group_performance,
                    factor_correlations = EXCLUDED.factor_correlations,
                    factor_contributions = EXCLUDED.factor_contributions,
                    backtest_cumulative_returns = EXCLUDED.backtest_cumulative_returns,
                    ic_series = EXCLUDED.ic_series,
                    ic_mean = EXCLUDED.ic_mean,
                    ic_std = EXCLUDED.ic_std,
                    hit_rate_top_decile = EXCLUDED.hit_rate_top_decile,
                    profit_factor = EXCLUDED.profit_factor,
                    win_loss_ratio = EXCLUDED.win_loss_ratio,
                    cagr = EXCLUDED.cagr,
                    volatility = EXCLUDED.volatility,
                    sortino_ratio = EXCLUDED.sortino_ratio,
                    calmar_ratio = EXCLUDED.calmar_ratio,
                    alpha_vs_spy = EXCLUDED.alpha_vs_spy,
                    beta_vs_spy = EXCLUDED.beta_vs_spy,
                    alpha_vs_qqq = EXCLUDED.alpha_vs_qqq,
                    beta_vs_qqq = EXCLUDED.beta_vs_qqq,
                    benchmark_correlations = EXCLUDED.benchmark_correlations,
                    factor_return_correlations = EXCLUDED.factor_return_correlations,
                    updated_at = NOW()
                RETURNING id
            """, [
                run_id,
                metrics['period_type'],
                metrics['period_start'],
                metrics['period_end'],
                metrics['total_signals'], metrics.get('avg_overall_score'),
                metrics.get('avg_technical_score'), metrics.get('avg_fundamental_score'), metrics.get('avg_news_macro_score'),
                metrics.get('avg_social_alternative_score'), metrics.get('avg_risk_stability_score'), metrics.get('avg_institutional_score'),
                metrics.get('top_sector'), metrics.get('top_sector_avg_return'), metrics.get('top_sector_count'),
                metrics.get('worst_sector'), metrics.get('worst_sector_avg_return'), metrics.get('worst_sector_count'),
                metrics['signals_analyzed'], metrics['performance_records_used'],
                metrics.get('win_rate'), metrics.get('sharpe_ratio'), metrics.get('max_drawdown'), metrics.get('avg_return'), metrics.get('avg_alpha'),
                json.dumps(sanitize_for_json(metrics.get('score_bucket_performance'))) if metrics.get('score_bucket_performance') else None,
                json.dumps(sanitize_for_json(metrics.get('group_performance'))) if metrics.get('group_performance') else None,
                json.dumps(sanitize_for_json(metrics.get('factor_correlations'))) if metrics.get('factor_correlations') else None,
                json.dumps(sanitize_for_json(metrics.get('factor_contributions'))) if metrics.get('factor_contributions') else None,
                json.dumps(sanitize_for_json(metrics.get('backtest_cumulative_returns'))) if metrics.get('backtest_cumulative_returns') else None,
                json.dumps(sanitize_for_json(metrics.get('ic_series'))) if metrics.get('ic_series') else None,
                metrics.get('ic_mean'),
                metrics.get('ic_std'),
                metrics.get('hit_rate_top_decile'),
                metrics.get('profit_factor'),
                metrics.get('win_loss_ratio'),
                metrics.get('cagr'),
                metrics.get('volatility'),
                metrics.get('sortino_ratio'),
                metrics.get('calmar_ratio'),
                metrics.get('alpha_vs_spy'),
                metrics.get('beta_vs_spy'),
                metrics.get('alpha_vs_qqq'),
                metrics.get('beta_vs_qqq'),
                json.dumps(sanitize_for_json(metrics.get('benchmark_correlations'))) if metrics.get('benchmark_correlations') else None,
                json.dumps(sanitize_for_json(metrics.get('factor_return_correlations'))) if metrics.get('factor_return_correlations') else None
            ])
            
            self.logger.info(f"[SUCCESS] Analytics persisted to database")
            
        except Exception as e:
            self.logger.error(f"Error persisting analytics: {e}", exc_info=True)
    
    def _calculate_score_bucket_performance(self, performance_data: List[Dict], interval: str) -> Dict[str, Any]:
        """
        Calculate performance metrics by score bucket for a specific interval.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
        
        Buckets loaded dynamically from methodology.yaml
        """
        # Use score ranges from methodology.yaml
        buckets = {}
        for bucket_name, range_info in self.score_ranges.items():
            buckets[bucket_name] = {
                'threshold': range_info['label'],
                'min': range_info['min'],
                'max': range_info['max'],
                'signals': []
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
        
        # Calculate metrics for each bucket using the specified interval
        result = {}
        return_col = f'return_{interval}'
        
        for bucket_name, bucket_info in buckets.items():
            signals = bucket_info['signals']
            bucket_metrics = {
                'threshold': bucket_info['threshold'],
                'count': len(signals)
            }
            
            if signals:
                returns = [float(p[return_col]) for p in signals if p.get(return_col) is not None]
                
                if returns and len(returns) > 1:
                    wins = [r for r in returns if r > 0]
                    losses = [r for r in returns if r < 0]
                    
                    # Basic metrics
                    bucket_metrics['avg_return'] = round(np.mean(returns), 4)
                    bucket_metrics['win_rate'] = round(len(wins) / len(returns) * 100, 4)  # Store as percentage
                    bucket_metrics['max'] = round(max(returns), 4)
                    bucket_metrics['min'] = round(min(returns), 4)
                    bucket_metrics['count'] = len(returns)
                    
                    # Risk-adjusted metrics
                    bucket_metrics['sharpe_ratio'] = round(self._calculate_sharpe_ratio(returns), 4)
                    bucket_metrics['sortino_ratio'] = round(self._calculate_sortino_ratio_from_returns(returns), 4)
                    
                    # Volatility (store as percentage)
                    volatility = np.std(returns, ddof=1) * np.sqrt(252) * 100
                    bucket_metrics['volatility'] = round(volatility, 4)
                    
                    # CAGR calculation (store as percentage)
                    avg_return = np.mean(returns)
                    if interval == '1d':
                        periods_per_year = 252
                    elif interval == '3d':
                        periods_per_year = 252 / 3
                    elif interval == '7d':
                        periods_per_year = 52
                    elif interval == '10d':
                        periods_per_year = 252 / 10
                    elif interval == '14d':
                        periods_per_year = 26
                    elif interval == '30d':
                        periods_per_year = 12
                    elif interval == '90d':
                        periods_per_year = 4
                    else:  # all_time
                        periods_per_year = 1
                    
                    cagr = ((1 + avg_return / 100) ** periods_per_year - 1) * 100
                    bucket_metrics['cagr'] = round(cagr, 4)
                    
                    # Calmar Ratio
                    max_dd = self._calculate_max_drawdown(returns)
                    bucket_metrics['max_drawdown'] = round(max_dd, 4)
                    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0
                    bucket_metrics['calmar_ratio'] = round(calmar, 4)
                    
                    # Win/Loss metrics
                    if losses:
                        avg_win = np.mean(wins) if wins else 0
                        avg_loss = abs(np.mean(losses))
                        bucket_metrics['win_loss_ratio'] = round(avg_win / avg_loss if avg_loss != 0 else 0, 4)
                        bucket_metrics['profit_factor'] = round(sum(wins) / abs(sum(losses)) if losses else 0, 4)
                    else:
                        bucket_metrics['win_loss_ratio'] = 0
                        bucket_metrics['profit_factor'] = 0
            
            result[bucket_name] = bucket_metrics
        
        return result
    
    def _calculate_group_performance(self, performance_data: List[Dict], interval: str) -> Dict[str, Any]:
        """
        Calculate performance metrics by score group for a specific interval.
        Shows win_rate, sharpe_ratio, avg_return, etc. for each signal group.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
            
        Returns:
            Dict with metrics for each group (technical, fundamental, news_macro, etc.)
        """
        result = {}
        return_col = f'return_{interval}'
        
        # Define score groups
        groups = {
            'technical': 'technical_score',
            'fundamental': 'fundamental_score',
            'news_macro': 'news_macro_score',
            'social_alternative': 'social_alternative_score',
            'risk_stability': 'risk_stability_score',
            'institutional': 'institutional_smart_money_score'
        }
        
        # For each group, segment by score quartiles and calculate metrics
        for group_name, score_field in groups.items():
            # Get all scores for this group
            scores = []
            for p in performance_data:
                score = p.get('signals', {}).get(score_field)
                if score is not None:
                    scores.append((score, p))
            
            if not scores:
                result[group_name] = {'error': 'No data'}
                continue
            
            # Sort by score
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Split into quintiles (top 20%, 20-40%, 40-60%, 60-80%, bottom 20%)
            n = len(scores)
            quintiles = {
                'top_20pct': scores[:n//5],
                'q2': scores[n//5:2*n//5],
                'q3': scores[2*n//5:3*n//5],
                'q4': scores[3*n//5:4*n//5],
                'bottom_20pct': scores[4*n//5:]
            }
            
            group_metrics = {}
            for quintile_name, quintile_data in quintiles.items():
                returns = []
                for score, perf in quintile_data:
                    ret = perf.get(return_col)
                    if ret is not None:
                        returns.append(float(ret))
                
                if returns:
                    wins = [r for r in returns if r > 0]
                    avg_return = np.mean(returns)
                    sharpe = self._calculate_sharpe_ratio(returns)
                    max_dd = self._calculate_max_drawdown(returns)
                    
                    # Calculate additional risk metrics
                    volatility = self._calculate_volatility_from_returns(returns, interval)
                    sortino = self._calculate_sortino_from_returns(returns, interval)
                    calmar = self._calculate_calmar_from_returns(returns, interval, max_dd)
                    
                    group_metrics[quintile_name] = {
                        'count': len(returns),
                        'avg_return': round(avg_return, 4),
                        'win_rate': round(len(wins) / len(returns) * 100, 2),
                        'sharpe': round(sharpe, 4),
                        'max_drawdown': round(max_dd, 2),
                        'volatility': round(volatility, 4) if volatility is not None else None,
                        'sortino': round(sortino, 4) if sortino is not None else None,
                        'calmar': round(calmar, 4) if calmar is not None else None
                    }
                else:
                    group_metrics[quintile_name] = {
                        'count': 0, 
                        'avg_return': 0, 
                        'win_rate': 0, 
                        'sharpe': 0, 
                        'max_drawdown': 0,
                        'volatility': None,
                        'sortino': None,
                        'calmar': None
                    }
            
            result[group_name] = group_metrics
        
        return result
    
    async def _calculate_factor_correlations(self, performance_data: List[Dict], interval: str) -> Dict[str, Any]:
        """
        Calculate factor correlation matrices.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
        
        Returns:
        - 6x6 group correlation matrix
        - Optionally 158x158 full factor correlation (if enabled)
        - Top positive and negative correlation pairs
        
        Note: Currently calculates group-level correlations from group scores.
        In the future, we could correlate factors with interval-specific returns.
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
    
    def _calculate_factor_contributions(self, performance_data: List[Dict], interval: str) -> Dict[str, Any]:
        """
        Calculate factor contribution to returns using correlation analysis.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
        
        Returns correlation between group scores and returns for the specified interval.
        """
        try:
            group_names = ['technical', 'fundamental', 'news_macro',
                          'social_alternative', 'risk_stability', 'institutional_smart_money']
            
            return_col = f'return_{interval}'
            
            # Filter valid data
            valid_data = [p for p in performance_data 
                         if p.get(return_col) is not None and p.get('signals')]
            
            if not valid_data or len(valid_data) < 10:
                return {}
            
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
            
            result = {
                'top_contributors': factor_correlations[:20],
                'all_correlations': factor_correlations
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating factor contributions: {e}")
            return {}
    
    def _calculate_group_performance_old(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """
        OLD METHOD - Analyze factor group performance - both per-signal and aggregated.
        DEPRECATED: Use _calculate_group_performance(performance_data, interval) instead.
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
    
    async def _calculate_backtest_returns(self, performance_data: List[Dict], interval: str) -> Dict[str, Any]:
        """
        Calculate cumulative returns for VP strategy vs SPY vs QQQ for a specific interval.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
        
        Assumes equal-weight portfolio, rebalancing at each interval.
        """
        try:
            # Group signals by baseline_date
            signals_by_date = defaultdict(list)
            return_col = f'return_{interval}'
            spy_return_col = f'spy_return_{interval}'
            qqq_return_col = f'qqq_return_{interval}'
            
            for p in performance_data:
                baseline_date = p.get('baseline_date')
                if baseline_date and p.get(return_col) is not None:
                    signals_by_date[baseline_date].append(p)
            
            # Sort dates
            dates = sorted(signals_by_date.keys())
            
            if not dates:
                return {}
            
            # Calculate portfolio returns for each date
            period_returns = []
            
            for date in dates:
                signals = signals_by_date[date]
                
                # Equal-weight portfolio return
                vp_returns = [float(p[return_col]) for p in signals if p.get(return_col) is not None]
                spy_returns = [float(p[spy_return_col]) for p in signals if p.get(spy_return_col) is not None]
                qqq_returns = [float(p[qqq_return_col]) for p in signals if p.get(qqq_return_col) is not None]
                
                if vp_returns:
                    period_returns.append({
                        'date': date,
                        'vp_return': np.mean(vp_returns) / 100,  # Convert % to decimal
                        'spy_return': np.mean(spy_returns) / 100 if spy_returns else 0,
                        'qqq_return': np.mean(qqq_returns) / 100 if qqq_returns else 0  # Real QQQ data
                    })
            
            # Calculate cumulative returns
            vp_cum = 1.0
            spy_cum = 1.0
            qqq_cum = 1.0
            
            cumulative_series = []
            
            for pr in period_returns:
                vp_cum *= (1 + pr['vp_return'])
                spy_cum *= (1 + pr['spy_return'])
                qqq_cum *= (1 + pr['qqq_return'])
                
                cumulative_series.append({
                    'date': pr['date'],
                    'vp_strategy': round(vp_cum, 4),
                    'spy': round(spy_cum, 4),
                    'qqq': round(qqq_cum, 4)
                })
            
            # Calculate summary statistics
            vp_returns = [pr['vp_return'] for pr in period_returns]
            
            result = {
                'start_date': dates[0] if dates else None,
                'end_date': dates[-1] if dates else None,
                'period_returns': cumulative_series[-100:],  # Last 100 periods for visualization
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

    # ==========================================================================
    # NEW ANALYTICS METHODS (VanPiQ Spec Additions)
    # ==========================================================================
    
    def _calculate_rank_ic(self, performance_data: List[Dict], interval: str) -> List[Dict[str, Any]]:
        """
        Calculate Rank Information Coefficient (IC) time series for a specific interval.
        IC measures correlation between signal scores and forward returns.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
        
        Returns list of {date, ic} for each day with sufficient data.
        """
        try:
            from scipy.stats import spearmanr
            
            # Group by baseline date
            return_col = f'return_{interval}'
            date_groups = defaultdict(list)
            
            for p in performance_data:
                baseline = p.get('baseline_date')
                score = p.get('signals', {}).get('overall_score')
                return_val = p.get(return_col)
                
                if baseline and score is not None and return_val is not None:
                    date_groups[baseline].append({
                        'score': score,
                        'return': return_val
                    })
            
            # Calculate IC for each date
            ic_series = []
            for date_str in sorted(date_groups.keys()):
                signals = date_groups[date_str]
                
                if len(signals) >= 10:  # Need minimum sample size
                    scores = [s['score'] for s in signals]
                    returns = [s['return'] for s in signals]
                    
                    # Spearman rank correlation
                    ic, p_value = spearmanr(scores, returns)
                    
                    if not math.isnan(ic):
                        ic_series.append({
                            'date': date_str[:10] if isinstance(date_str, str) else date_str.strftime('%Y-%m-%d'),
                            'ic': round(float(ic), 4)
                        })
            
            return ic_series[-90:] if ic_series else []  # Last 90 periods
            
        except ImportError:
            self.logger.warning("scipy not installed, skipping IC calculation")
            return []
        except Exception as e:
            self.logger.error(f"Error calculating Rank IC: {e}")
            return []
    
    def _calculate_hit_rate_top_decile(self, performance_data: List[Dict], interval: str) -> float:
        """
        Calculate hit rate for top decile (top 10% of signals by score) for a specific interval.
        Hit rate = fraction of top decile signals with positive returns.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
        """
        try:
            # Get signals with scores and returns for the specified interval
            return_col = f'return_{interval}'
            signals = []
            
            for p in performance_data:
                score = p.get('signals', {}).get('overall_score')
                return_val = p.get(return_col)
                
                if score is not None and return_val is not None:
                    signals.append({'score': score, 'return': return_val})
            
            if len(signals) < 10:
                return 0.0
            
            # Sort by score descending
            signals.sort(key=lambda x: x['score'], reverse=True)
            
            # Get top decile
            top_10_pct_count = max(1, len(signals) // 10)
            top_decile = signals[:top_10_pct_count]
            
            # Calculate hit rate
            wins = sum(1 for s in top_decile if s['return'] > 0)
            hit_rate = wins / len(top_decile)
            
            return round(hit_rate, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating hit rate: {e}")
            return 0.0
    
    def _calculate_profit_factor(self, performance_data: List[Dict], interval: str) -> float:
        """
        Calculate profit factor = total wins / abs(total losses) for a specific interval.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
        """
        try:
            return_col = f'return_{interval}'
            returns = [p.get(return_col) for p in performance_data if p.get(return_col) is not None]
            
            if not returns:
                return 0.0
            
            total_wins = sum(r for r in returns if r > 0)
            total_losses = abs(sum(r for r in returns if r < 0))
            
            if total_losses == 0:
                return float('inf') if total_wins > 0 else 0.0
            
            profit_factor = total_wins / total_losses
            return round(profit_factor, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    def _calculate_win_loss_ratio(self, performance_data: List[Dict], interval: str) -> float:
        """
        Calculate win/loss ratio = avg win / abs(avg loss) for a specific interval.
        
        Args:
            performance_data: List of performance records
            interval: The specific interval to analyze (e.g., '1d', '7d', '30d')
        """
        try:
            return_col = f'return_{interval}'
            returns = [p.get(return_col) for p in performance_data if p.get(return_col) is not None]
            
            if not returns:
                return 0.0
            
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            
            if not wins or not losses:
                return 0.0
            
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            if avg_loss == 0:
                return 0.0
            
            win_loss_ratio = avg_win / avg_loss
            return round(win_loss_ratio, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating win/loss ratio: {e}")
            return 0.0
    
    def _calculate_cagr(self, performance_data: List[Dict]) -> float:
        """
        Calculate Compound Annual Growth Rate from cumulative returns.
        Uses 90D returns annualized.
        """
        try:
            returns_90d = [p.get('return_90d') for p in performance_data if p.get('return_90d') is not None]
            
            if not returns_90d:
                return 0.0
            
            # Average 90D return
            avg_90d = np.mean(returns_90d)
            
            # Annualize: (1 + r_90d)^(365/90) - 1
            cagr = ((1 + avg_90d) ** (365/90)) - 1
            
            return round(cagr, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating CAGR: {e}")
            return 0.0
    
    def _calculate_volatility(self, performance_data: List[Dict]) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).
        Uses 7D returns.
        """
        try:
            returns_7d = [p.get('return_7d') for p in performance_data if p.get('return_7d') is not None]
            
            if not returns_7d:
                return 0.0
            
            # Calculate standard deviation
            vol = np.std(returns_7d)
            
            # Annualize: vol * sqrt(252/7) for 7-day returns
            annualized_vol = vol * np.sqrt(252 / 7)
            
            return round(annualized_vol, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, performance_data: List[Dict]) -> float:
        """
        Calculate Sortino ratio (excess return / downside deviation).
        Uses 7D returns.
        """
        try:
            returns_7d = [p.get('return_7d') for p in performance_data if p.get('return_7d') is not None]
            
            if not returns_7d:
                return 0.0
            
            avg_return = np.mean(returns_7d)
            
            # Downside returns only (negative returns)
            downside_returns = [r for r in returns_7d if r < 0]
            
            if not downside_returns:
                return 0.0
            
            downside_std = np.std(downside_returns)
            
            if downside_std == 0:
                return 0.0
            
            # Annualize
            sortino = (avg_return * np.sqrt(252/7)) / (downside_std * np.sqrt(252/7))
            
            return round(sortino, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, performance_data: List[Dict]) -> float:
        """
        Calculate Calmar ratio = CAGR / abs(max drawdown).
        Uses 90D returns.
        """
        try:
            returns_90d = [p.get('return_90d') for p in performance_data if p.get('return_90d') is not None]
            
            if not returns_90d:
                return 0.0
            
            cagr = self._calculate_cagr(performance_data)
            max_dd = self._calculate_max_drawdown(returns_90d)
            
            if max_dd == 0:
                return 0.0
            
            calmar = cagr / abs(max_dd)
            return round(calmar, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    # ========================================
    # INTERVAL-SPECIFIC METRIC CALCULATIONS
    # ========================================
    
    def _calculate_cagr_for_interval(self, performance_data: List[Dict], interval: str) -> float:
        """
        Calculate CAGR for specific interval.
        Annualizes the average return based on interval length.
        """
        try:
            return_col = f'return_{interval}'
            returns = [p.get(return_col) for p in performance_data if p.get(return_col) is not None]
            
            if not returns:
                return None
            
            # Average return for this interval
            avg_return = np.mean(returns)
            
            # Annualization factor based on interval
            interval_days = self._get_interval_days(interval)
            periods_per_year = 365 / interval_days
            
            # CAGR = (1 + avg_return) ^ periods_per_year - 1
            cagr = ((1 + avg_return / 100) ** periods_per_year - 1) * 100
            
            return round(cagr, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating CAGR for {interval}: {e}")
            return None
    
    def _calculate_volatility_for_interval(self, performance_data: List[Dict], interval: str) -> float:
        """
        Calculate annualized volatility for specific interval.
        """
        try:
            return_col = f'return_{interval}'
            returns = [p.get(return_col) for p in performance_data if p.get(return_col) is not None]
            
            if not returns:
                return None
            
            # Standard deviation of returns
            vol = np.std(returns)
            
            # Annualize based on interval
            interval_days = self._get_interval_days(interval)
            periods_per_year = 365 / interval_days
            annualized_vol = vol * np.sqrt(periods_per_year)
            
            return round(annualized_vol, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {interval}: {e}")
            return None
    
    def _calculate_sortino_ratio_for_interval(self, performance_data: List[Dict], interval: str) -> float:
        """
        Calculate Sortino ratio for specific interval.
        Uses downside deviation (only negative returns).
        """
        try:
            return_col = f'return_{interval}'
            returns = [p.get(return_col) for p in performance_data if p.get(return_col) is not None]
            
            if not returns:
                return None
            
            avg_return = np.mean(returns)
            
            # Downside returns (below zero)
            downside_returns = [r for r in returns if r < 0]
            
            if not downside_returns:
                return None
            
            downside_std = np.std(downside_returns)
            
            if downside_std == 0:
                return None
            
            # Annualize
            interval_days = self._get_interval_days(interval)
            periods_per_year = 365 / interval_days
            sortino = (avg_return * np.sqrt(periods_per_year)) / (downside_std * np.sqrt(periods_per_year))
            
            return round(sortino, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio for {interval}: {e}")
            return None
    
    def _calculate_calmar_ratio_for_interval(self, performance_data: List[Dict], interval: str) -> float:
        """
        Calculate Calmar ratio for specific interval.
        Calmar = CAGR / abs(max drawdown)
        """
        try:
            return_col = f'return_{interval}'
            returns = [p.get(return_col) for p in performance_data if p.get(return_col) is not None]
            
            if not returns:
                return None
            
            cagr = self._calculate_cagr_for_interval(performance_data, interval)
            if cagr is None:
                return None
            
            max_dd = self._calculate_max_drawdown(returns)
            
            if max_dd == 0:
                return None
            
            calmar = cagr / abs(max_dd)
            return round(calmar, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating Calmar ratio for {interval}: {e}")
            return None
    

    def _calculate_benchmark_correlation_for_interval(self, performance_data: List[Dict], interval: str) -> Dict[str, float]:
        """
        Calculate correlation with SPY and QQQ for specific interval.
        """
        try:
            return_col = f'return_{interval}'
            spy_return_col = f'spy_return_{interval}'
            qqq_return_col = f'qqq_return_{interval}'
            
            vp_returns = []
            spy_returns = []
            qqq_returns = []
            
            for p in performance_data:
                vp_ret = p.get(return_col)
                spy_ret = p.get(spy_return_col)
                qqq_ret = p.get(qqq_return_col)
                
                if vp_ret is not None and spy_ret is not None:
                    vp_returns.append(vp_ret)
                    spy_returns.append(spy_ret)
                
                if vp_ret is not None and qqq_ret is not None:
                    if len(qqq_returns) < len(vp_returns):
                        qqq_returns.append(qqq_ret)
            
            correlations = {}
            
            if len(vp_returns) >= 2 and len(spy_returns) >= 2:
                spy_corr = np.corrcoef(vp_returns, spy_returns)[0, 1]
                correlations['spy'] = round(float(spy_corr), 4) if not np.isnan(spy_corr) else 0.0
            else:
                correlations['spy'] = None
            
            if len(vp_returns) >= 2 and len(qqq_returns) >= 2:
                qqq_corr = np.corrcoef(vp_returns[:len(qqq_returns)], qqq_returns)[0, 1]
                correlations['qqq'] = round(float(qqq_corr), 4) if not np.isnan(qqq_corr) else 0.0
            else:
                correlations['qqq'] = None
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark correlation for {interval}: {e}")
            return {'spy': None, 'qqq': None}
    
    async def _calculate_factor_return_correlations(
        self, 
        performance_data: List[Dict], 
        interval: str,
        min_samples: int = 50
    ) -> Dict[str, List[Dict]]:
        """
        Calculate correlations between individual factor values and returns.
        
        This differs from factor_correlations (which correlates 6 group scores).
        Here we correlate each of 158 individual factors with returns for ML feature importance.
        
        Args:
            performance_data: List of dicts with returns and factor values
            interval: Time interval (e.g., '1d', '3d')
            min_samples: Minimum sample size for valid correlation (default 50)
            
        Returns:
            Dict mapping group names to list of factor correlations:
            {
                'technical': [
                    {
                        'factor': 'rsi_14',
                        'correlation': 0.65,
                        'p_value': 0.001,
                        'n': 214,
                        'confidence': 'high'
                    },
                    ...
                ],
                'fundamental': [...],
                ...
            }
        """
        try:
            from scipy import stats
            
            # Load factor definitions (158 factors from 6 groups)
            factor_groups = self._load_factor_definitions()
            if not factor_groups:
                self.logger.warning("No factor groups loaded, skipping factor-return correlations")
                return {}
            
            return_col = f'return_{interval}'
            
            # Extract returns (filter out None values)
            returns = []
            indices = []
            for i, p in enumerate(performance_data):
                ret = p.get(return_col)
                if ret is not None:
                    returns.append(ret)
                    indices.append(i)
            
            if len(returns) < min_samples:
                self.logger.info(
                    f"Insufficient samples for {interval} factor correlations: "
                    f"{len(returns)} < {min_samples}"
                )
                return {}
            
            # Group mappings to signal table names
            group_to_signals_table = {
                'technical': 'signals_technical',
                'fundamental': 'signals_fundamental',
                'news_macro': 'signals_news_macro',
                'social_alternative': 'signals_social_alternative',
                'risk_stability': 'signals_risk_stability',
                'institutional_smart_money': 'signals_institutional_smart_money'
            }
            
            result = {}
            
            # Calculate correlations for each group
            for group_name, factor_names in factor_groups.items():
                group_correlations = []
                
                # Get signals key based on group
                signals_key = group_to_signals_table.get(group_name)
                if not signals_key:
                    self.logger.warning(f"Unknown group: {group_name}, skipping")
                    continue
                
                # Calculate correlation for each factor in this group
                factors_processed = 0
                factors_with_data = 0
                for factor_name in factor_names:
                    factors_processed += 1
                    # Extract factor values (aligned with returns)
                    factor_values = []
                    for idx in indices:
                        p = performance_data[idx]
                        
                        # Get signals JSONB from performance data
                        signals = p.get(signals_key, {})
                        
                        # Handle both dict and string (parse if string)
                        if isinstance(signals, str):
                            import json
                            try:
                                signals = json.loads(signals)
                            except:
                                signals = {}
                        
                        # Get factor value from signals
                        factor_value = signals.get(factor_name) if signals else None
                        
                        # Factor values are stored as dicts with 'raw', 'normalized', 'percentile' keys
                        # Extract the normalized value for correlation calculation
                        if factor_value is not None:
                            if isinstance(factor_value, dict):
                                # Extract normalized value from factor dict
                                numeric_value = factor_value.get('normalized')
                                if numeric_value is None:
                                    numeric_value = factor_value.get('raw')
                            else:
                                # If not a dict, try to use directly
                                numeric_value = factor_value
                            
                            # Convert to float
                            try:
                                factor_values.append(float(numeric_value))
                            except (ValueError, TypeError):
                                # Cannot convert to float, treat as missing
                                factor_values.append(None)
                        else:
                            # If factor value is missing, skip this data point
                            factor_values.append(None)                    # Remove indices where factor value is None
                    valid_pairs = [
                        (r, f) for r, f in zip(returns, factor_values)
                        if f is not None
                    ]
                    
                    if len(valid_pairs) < min_samples:
                        # Not enough data for this factor
                        continue
                    
                    factors_with_data += 1
                    
                    valid_returns, valid_factors = zip(*valid_pairs)
                    
                    # Calculate Pearson correlation
                    try:
                        corr, p_value = stats.pearsonr(valid_factors, valid_returns)
                        
                        # Determine confidence level
                        n = len(valid_pairs)
                        if n < 50:
                            confidence = 'low'
                        elif n < 100:
                            confidence = 'medium'
                        else:
                            confidence = 'high'
                        
                        group_correlations.append({
                            'factor': factor_name,
                            'correlation': round(float(corr), 4) if not np.isnan(corr) else 0.0,
                            'p_value': round(float(p_value), 4),
                            'n': n,
                            'confidence': confidence
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error calculating correlation for {factor_name}: {e}")
                        continue
                
                # Log group summary
                self.logger.debug(
                    f"Group {group_name} complete: "
                    f"{factors_processed} factors processed, "
                    f"{factors_with_data} had sufficient data, "
                    f"{len(group_correlations)} correlations calculated"
                )
                
                # Add group results if any correlations were calculated
                if group_correlations:
                    result[group_name] = group_correlations
            
            self.logger.info(
                f"Calculated factor-return correlations for {interval}: "
                f"{sum(len(g) for g in result.values())} factors across {len(result)} groups"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating factor-return correlations for {interval}: {e}")
            return {}
    
    def _get_interval_days(self, interval: str) -> int:
        """Convert interval string to approximate days."""
        interval_map = {
            '1d': 1,
            '3d': 3,
            '7d': 7,
            '10d': 10,
            '14d': 14,
            '30d': 30,
            '90d': 90,
            'all_time': 365  # Default to 1 year for all_time
        }
        return interval_map.get(interval, 7)
    
    def _calculate_volatility_from_returns(self, returns: List[float], interval: str) -> Optional[float]:
        """Calculate annualized volatility from a list of returns."""
        try:
            if not returns or len(returns) < 2:
                return None
            
            vol = np.std(returns)
            interval_days = self._get_interval_days(interval)
            periods_per_year = 365 / interval_days
            annualized_vol = vol * np.sqrt(periods_per_year)
            
            return float(annualized_vol)
        except Exception as e:
            self.logger.error(f"Error calculating volatility from returns: {e}")
            return None
    
    def _calculate_sortino_from_returns(self, returns: List[float], interval: str) -> Optional[float]:
        """Calculate Sortino ratio from a list of returns."""
        try:
            if not returns or len(returns) < 2:
                return None
            
            avg_return = np.mean(returns)
            downside_returns = [r for r in returns if r < 0]
            
            if not downside_returns:
                return None
            
            downside_std = np.std(downside_returns)
            
            if downside_std == 0:
                return None
            
            interval_days = self._get_interval_days(interval)
            periods_per_year = 365 / interval_days
            sortino = (avg_return * np.sqrt(periods_per_year)) / (downside_std * np.sqrt(periods_per_year))
            
            return float(sortino)
        except Exception as e:
            self.logger.error(f"Error calculating Sortino from returns: {e}")
            return None
    
    def _calculate_calmar_from_returns(self, returns: List[float], interval: str, max_dd: float) -> Optional[float]:
        """Calculate Calmar ratio from returns and max drawdown."""
        try:
            if not returns or len(returns) < 2 or max_dd == 0:
                return None
            
            # Calculate annualized return
            avg_return = np.mean(returns)
            interval_days = self._get_interval_days(interval)
            periods_per_year = 365 / interval_days
            cagr = ((1 + avg_return / 100) ** periods_per_year - 1) * 100
            
            calmar = cagr / abs(max_dd)
            return float(calmar)
        except Exception as e:
            self.logger.error(f"Error calculating Calmar from returns: {e}")
            return None
    
    def _calculate_benchmark_metrics(self, performance_data: List[Dict], benchmark: str, interval: str = '7d') -> Dict[str, float]:
        """
        Calculate alpha and beta vs benchmark (SPY or QQQ) for specific interval.
        Returns dict with alpha_vs_{benchmark} and beta_vs_{benchmark}.
        """
        try:
            # Get return columns based on benchmark and interval
            vp_returns = []
            bm_returns = []
            
            return_col = f'return_{interval}'
            bm_return_col = f'{benchmark.lower()}_return_{interval}'
            
            for p in performance_data:
                vp_ret = p.get(return_col)
                bm_ret = p.get(bm_return_col)
                
                if vp_ret is not None and bm_ret is not None:
                    vp_returns.append(vp_ret)
                    bm_returns.append(bm_ret)
            
            if len(vp_returns) < 2:
                return {
                    f'alpha_vs_{benchmark.lower()}': 0.0,
                    f'beta_vs_{benchmark.lower()}': 0.0
                }
            
            # Calculate beta using covariance
            covariance = np.cov(vp_returns, bm_returns)[0][1]
            bm_variance = np.var(bm_returns)
            
            beta = covariance / bm_variance if bm_variance != 0 else 0.0
            
            # Calculate alpha = avg(VP returns) - beta * avg(BM returns)
            alpha = np.mean(vp_returns) - beta * np.mean(bm_returns)
            
            return {
                f'alpha_vs_{benchmark.lower()}': round(alpha, 4),
                f'beta_vs_{benchmark.lower()}': round(beta, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark metrics for {benchmark}: {e}")
            return {
                f'alpha_vs_{benchmark.lower()}': 0.0,
                f'beta_vs_{benchmark.lower()}': 0.0
            }
    
    def _calculate_all_time_benchmark_metrics(self, performance_data: List[Dict], benchmark: str, intervals: List[str]) -> Dict[str, Optional[float]]:
        """
        Calculate alpha and beta vs benchmark aggregated across ALL intervals.
        Used for 'all_time' period which doesn't have a single interval.
        
        Args:
            performance_data: Performance records
            benchmark: 'SPY' or 'QQQ'
            intervals: List of intervals to aggregate ['1d', '3d', '7d', ...]
            
        Returns:
            Dict with alpha_vs_{benchmark} and beta_vs_{benchmark}
        """
        try:
            vp_returns = []
            bm_returns = []
            
            # Collect returns from all intervals
            for interval in intervals:
                return_col = f'return_{interval}'
                bm_return_col = f'{benchmark.lower()}_return_{interval}'
                
                for p in performance_data:
                    vp_ret = p.get(return_col)
                    bm_ret = p.get(bm_return_col)
                    
                    if vp_ret is not None and bm_ret is not None:
                        vp_returns.append(vp_ret)
                        bm_returns.append(bm_ret)
            
            # Need minimum 10 datapoints for meaningful calculation
            if len(vp_returns) < 10:
                self.logger.warning(
                    f"all_time {benchmark} benchmark: Insufficient data "
                    f"({len(vp_returns)} datapoints). Need at least 10. "
                    f"System may be too new - returns will mature over time."
                )
                return {
                    f'alpha_vs_{benchmark.lower()}': None,
                    f'beta_vs_{benchmark.lower()}': None
                }
            
            # Calculate beta using covariance
            covariance = np.cov(vp_returns, bm_returns)[0][1]
            bm_variance = np.var(bm_returns)
            
            beta = covariance / bm_variance if bm_variance != 0 else 0.0
            
            # Calculate alpha = avg(VP returns) - beta * avg(BM returns)
            alpha = np.mean(vp_returns) - beta * np.mean(bm_returns)
            
            self.logger.info(
                f"all_time {benchmark} benchmark calculated with {len(vp_returns)} datapoints: "
                f"alpha={alpha:.4f}, beta={beta:.4f}"
            )
            
            return {
                f'alpha_vs_{benchmark.lower()}': round(alpha, 4),
                f'beta_vs_{benchmark.lower()}': round(beta, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating all_time benchmark metrics for {benchmark}: {e}")
            return {
                f'alpha_vs_{benchmark.lower()}': None,
                f'beta_vs_{benchmark.lower()}': None
            }
    
    def _calculate_rolling_sharpe(self, performance_data: List[Dict], window: int = 30) -> List[Dict[str, Any]]:
        """
        Calculate 30-day rolling Sharpe ratio time series.
        Returns list of {date, sharpe}.
        """
        try:
            # Group by baseline date
            date_returns = defaultdict(list)
            for p in performance_data:
                baseline = p.get('baseline_date')
                ret_7d = p.get('return_7d')
                
                if baseline and ret_7d is not None:
                    date_str = baseline[:10] if isinstance(baseline, str) else baseline.strftime('%Y-%m-%d')
                    date_returns[date_str].append(ret_7d)
            
            # Calculate daily average returns
            sorted_dates = sorted(date_returns.keys())
            daily_avg_returns = [(date, np.mean(date_returns[date])) for date in sorted_dates]
            
            if len(daily_avg_returns) < window:
                return []
            
            # Calculate rolling Sharpe
            rolling_sharpe = []
            for i in range(window - 1, len(daily_avg_returns)):
                window_returns = [r for _, r in daily_avg_returns[i - window + 1:i + 1]]
                
                avg_ret = np.mean(window_returns)
                std_ret = np.std(window_returns)
                
                if std_ret != 0:
                    sharpe = (avg_ret * np.sqrt(252/7)) / (std_ret * np.sqrt(252/7))
                    rolling_sharpe.append({
                        'date': daily_avg_returns[i][0],
                        'sharpe': round(sharpe, 4)
                    })
            
            return rolling_sharpe[-90:] if rolling_sharpe else []  # Last 90 days
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling Sharpe: {e}")
            return []
    
    def _calculate_benchmark_correlation(self, performance_data: List[Dict]) -> Dict[str, float]:
        """
        Calculate Pearson correlation between VP returns and benchmark returns.
        Returns dict with SPY and QQQ correlations.
        """
        try:
            vp_returns = []
            spy_returns = []
            qqq_returns = []
            
            for p in performance_data:
                vp_ret = p.get('return_7d')
                spy_ret = p.get('spy_return_7d')
                qqq_ret = p.get('qqq_return_7d')
                
                if vp_ret is not None and spy_ret is not None and qqq_ret is not None:
                    vp_returns.append(vp_ret)
                    spy_returns.append(spy_ret)
                    qqq_returns.append(qqq_ret)
            
            if len(vp_returns) < 2:
                return {'SPY': 0.0, 'QQQ': 0.0}
            
            # Calculate Pearson correlation
            spy_corr = np.corrcoef(vp_returns, spy_returns)[0][1]
            qqq_corr = np.corrcoef(vp_returns, qqq_returns)[0][1]
            
            return {
                'SPY': round(float(spy_corr), 4) if not math.isnan(spy_corr) else 0.0,
                'QQQ': round(float(qqq_corr), 4) if not math.isnan(qqq_corr) else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark correlations: {e}")
            return {'SPY': 0.0, 'QQQ': 0.0}
    


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
