"""
Phase 3: Normalize (v3.1)
=========================

Normalize calculated factors from Phase 2 using robust z-score normalization.

This module is responsible for:
- Converting raw factor values into normalized z-scores
- Using robust normalization (median/MAD instead of mean/std)
- Applying winsorization to handle outliers
- Preserving group structure from Phase 2

Input: Dict[ticker, GroupFactors] from Phase 2
Output: Dict[ticker, GroupFactors] with normalized values
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

from backend.utils.metrics import emit_metric

# Import GroupFactors from Phase 2
import sys
sys.path.insert(0, str(Path(__file__).parent))
from phase2_calculate import GroupFactors

logger = logging.getLogger(__name__)


# ============================================================================
# NORMALIZED GROUP FACTORS
# ============================================================================

@dataclass
class NormalizedGroupFactors:
    """
    Normalized factors organized by 6 signal groups (v3.1).
    
    ALL values are z-scores (mean=0, std=1) after robust normalization.
    """
    ticker: str
    
    # Six signal groups (matches GroupFactors from Phase 2)
    technical: Dict[str, float] = field(default_factory=dict)
    fundamental: Dict[str, float] = field(default_factory=dict)
    news_macro: Dict[str, float] = field(default_factory=dict)
    social_alternative: Dict[str, float] = field(default_factory=dict)
    risk_stability: Dict[str, float] = field(default_factory=dict)
    institutional_smart_money: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    normalized_at: str = field(default_factory=lambda: datetime.now().isoformat())
    normalization_method: str = "robust_zscore"
    winsorize_pct: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def get_all_factors(self) -> Dict[str, float]:
        """Get flattened dict of all normalized factors"""
        all_factors = {}
        all_factors.update(self.technical)
        all_factors.update(self.fundamental)
        all_factors.update(self.news_macro)
        all_factors.update(self.social_alternative)
        all_factors.update(self.risk_stability)
        all_factors.update(self.institutional_smart_money)
        return all_factors


# ============================================================================
# PHASE 3 NORMALIZER
# ============================================================================

class Phase3Normalizer:
    """
    Phase 3: Normalize calculated factors using robust z-scores.
    
    Design principles:
    - Robust normalization (median/MAD instead of mean/std)
    - Winsorization to handle outliers
    - Cross-sectional normalization (normalize across tickers, not time)
    - Config-driven parameters
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize Phase 3 normalizer.
        
        Args:
            config_path: Path to weights.yaml config (optional)
        """
        self.logger = logger
        
        # Load normalization parameters from config
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'weights.yaml'
        
        self.config = self._load_config(config_path)
        
        # Extract normalization parameters
        scoring_params = self.config.get('scoring_parameters', {})
        self.method = scoring_params.get('normalization_method', 'robust_zscore')
        self.winsorize_pct = scoring_params.get('winsorize_percentile', 0.01)
        self.min_tickers = scoring_params.get('min_tickers_for_normalization', 3)
        
        self.logger.info(f"[SUCCESS] Phase3Normalizer initialized")
        self.logger.info(f"   Method: {self.method}")
        self.logger.info(f"   Winsorization: {self.winsorize_pct*100}%")
        self.logger.info(f"   Min tickers: {self.min_tickers}")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load normalization config from weights.yaml"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"[SUCCESS] Loaded normalization config from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            self.logger.info("Using default normalization parameters")
            return {
                'scoring_parameters': {
                    'normalization_method': 'robust_zscore',
                    'winsorize_percentile': 0.01,
                    'min_tickers_for_normalization': 3
                }
            }
    
    def normalize_all_factors(self, 
                             calculated_by_ticker: Dict[str, GroupFactors]) -> Dict[str, NormalizedGroupFactors]:
        """
        Main entry point for Phase 3.
        
        Normalize all calculated factors from Phase 2 using robust z-scores.
        
        Args:
            calculated_by_ticker: Dict mapping ticker -> GroupFactors from Phase 2
        
        Returns:
            Dict mapping ticker -> NormalizedGroupFactors
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 3: NORMALIZE (v3.1 - Robust Z-Score)")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        if len(calculated_by_ticker) < self.min_tickers:
            self.logger.warning(f"Only {len(calculated_by_ticker)} tickers - need {self.min_tickers} minimum for normalization")
            self.logger.warning("Returning identity normalization (z-score = 0 for all factors)")
            return self._identity_normalization(calculated_by_ticker)
        
        self.logger.info(f"[STATS] Normalizing {len(calculated_by_ticker)} tickers...")
        
        # Step 1: Extract all factors into cross-sectional DataFrames (one per group)
        self.logger.info("Step 3.1: Extracting factors into cross-sectional DataFrames...")
        group_dataframes = self._build_cross_sectional_dataframes(calculated_by_ticker)
        
        # Step 2: Normalize each group's factors using robust z-score
        self.logger.info("Step 3.2: Applying robust z-score normalization...")
        normalized_dataframes = self._normalize_cross_sectional(group_dataframes)
        
        # Step 3: Reconstruct NormalizedGroupFactors for each ticker
        self.logger.info("Step 3.3: Reconstructing normalized factors by ticker...")
        normalized_by_ticker = self._reconstruct_by_ticker(
            normalized_dataframes, 
            list(calculated_by_ticker.keys())
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.logger.info("=" * 80)
        self.logger.info(f"[SUCCESS] PHASE 3 COMPLETE - {elapsed:.2f}s")
        self.logger.info(f"   Normalized {len(normalized_by_ticker)} tickers")
        self.logger.info(f"   Method: {self.method}")
        self.logger.info("=" * 80)
        
        emit_metric("phase3.normalize.success", len(normalized_by_ticker), 
                   tags={'method': self.method})
        
        return normalized_by_ticker
    
    def normalize_batch(self, 
                       calculated_by_ticker: Dict[str, GroupFactors]) -> Dict[str, NormalizedGroupFactors]:
        """Alias for normalize_all_factors (for consistency with Phase 2 API)"""
        return self.normalize_all_factors(calculated_by_ticker)
    
    def _build_cross_sectional_dataframes(self, 
                                         calculated_by_ticker: Dict[str, GroupFactors]) -> Dict[str, pd.DataFrame]:
        """
        Build cross-sectional DataFrames for each group.
        
        Each DataFrame has:
        - Rows: tickers
        - Columns: factors in that group
        - Values: raw calculated values from Phase 2
        
        Args:
            calculated_by_ticker: Dict[ticker, GroupFactors]
        
        Returns:
            Dict[group_name, DataFrame] with cross-sectional data
        """
        group_names = ['technical', 'fundamental', 'news_macro', 
                      'social_alternative', 'risk_stability', 'institutional_smart_money']
        
        group_dataframes = {}
        
        for group_name in group_names:
            # Extract factors for this group across all tickers
            data_dict = {}
            
            for ticker, group_factors in calculated_by_ticker.items():
                group_dict = getattr(group_factors, group_name, {})
                data_dict[ticker] = group_dict
            
            # Convert to DataFrame (tickers as rows, factors as columns)
            df = pd.DataFrame.from_dict(data_dict, orient='index')
            
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            group_dataframes[group_name] = df
            
            self.logger.info(f"   {group_name}: {df.shape[0]} tickers × {df.shape[1]} factors")
        
        return group_dataframes
    
    def _normalize_cross_sectional(self, 
                                   group_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Normalize each group's DataFrame using robust z-score.
        
        Robust z-score formula:
            z = (x - median(x)) / MAD(x)
        
        Where MAD = median absolute deviation = median(|x - median(x)|)
        
        Args:
            group_dataframes: Dict[group_name, DataFrame] with raw values
        
        Returns:
            Dict[group_name, DataFrame] with normalized z-scores
        """
        normalized_dataframes = {}
        
        for group_name, df in group_dataframes.items():
            self.logger.info(f"   Normalizing {group_name}...")
            
            # Create output DataFrame
            normalized_df = pd.DataFrame(index=df.index, columns=df.columns)
            
            # Normalize each factor (column) independently
            for factor_name in df.columns:
                values = df[factor_name].dropna()
                
                # VALIDATION: Check for sufficient data
                if len(values) < self.min_tickers:
                    self.logger.debug(f"      [{factor_name}] Insufficient tickers ({len(values)} < {self.min_tickers}), setting to 0.0")
                    normalized_df[factor_name] = 0.0
                    continue
                
                # VALIDATION: Check for zero variance
                if values.std() == 0:
                    self.logger.debug(f"      [{factor_name}] Zero variance (all values identical), setting to 0.0")
                    normalized_df[factor_name] = 0.0
                    continue
                
                # Apply winsorization (clip extreme values)
                if self.winsorize_pct > 0:
                    lower = values.quantile(self.winsorize_pct)
                    upper = values.quantile(1 - self.winsorize_pct)
                    values_winsorized = values.clip(lower=lower, upper=upper)
                else:
                    values_winsorized = values
                
                # Compute robust z-score
                median = values_winsorized.median()
                mad = np.median(np.abs(values_winsorized - median))
                
                # VALIDATION: Check for zero MAD
                if mad == 0 or np.isnan(mad):
                    self.logger.debug(f"      [{factor_name}] Zero MAD (no variation after winsorization), setting to 0.0")
                    normalized_df[factor_name] = 0.0
                else:
                    # Robust z-score: (x - median) / (1.4826 * MAD)
                    # 1.4826 is the scaling factor to make MAD comparable to std dev
                    z_scores = (df[factor_name] - median) / (1.4826 * mad)
                    
                    # VALIDATION: Check for infinite or extreme values
                    if np.any(np.isinf(z_scores)):
                        self.logger.warning(f"      [{factor_name}] Infinite z-scores detected, clipping")
                        z_scores = z_scores.replace([np.inf, -np.inf], np.nan)
                    
                    # Clip extreme z-scores (beyond ±5 is very unusual)
                    if np.any(np.abs(z_scores) > 10):
                        extreme_count = (np.abs(z_scores) > 10).sum()
                        self.logger.warning(f"      [{factor_name}] {extreme_count} extreme z-scores (>10), clipping to ±5")
                        z_scores = z_scores.clip(lower=-5, upper=5)
                    
                    normalized_df[factor_name] = z_scores
            
            normalized_dataframes[group_name] = normalized_df
            
            # Log normalization stats
            non_nan_count = normalized_df.notna().sum().sum()
            total_count = normalized_df.size
            self.logger.info(f"      -> {non_nan_count}/{total_count} values normalized")
        
        return normalized_dataframes
    
    def _reconstruct_by_ticker(self, 
                              normalized_dataframes: Dict[str, pd.DataFrame],
                              tickers: List[str]) -> Dict[str, NormalizedGroupFactors]:
        """
        Reconstruct NormalizedGroupFactors for each ticker from DataFrames.
        
        Args:
            normalized_dataframes: Dict[group_name, DataFrame] with normalized values
            tickers: List of ticker symbols
        
        Returns:
            Dict[ticker, NormalizedGroupFactors]
        """
        normalized_by_ticker = {}
        
        for ticker in tickers:
            try:
                # Extract normalized factors for this ticker from each group
                technical = normalized_dataframes['technical'].loc[ticker].to_dict() if ticker in normalized_dataframes['technical'].index else {}
                fundamental = normalized_dataframes['fundamental'].loc[ticker].to_dict() if ticker in normalized_dataframes['fundamental'].index else {}
                news_macro = normalized_dataframes['news_macro'].loc[ticker].to_dict() if ticker in normalized_dataframes['news_macro'].index else {}
                social_alternative = normalized_dataframes['social_alternative'].loc[ticker].to_dict() if ticker in normalized_dataframes['social_alternative'].index else {}
                risk_stability = normalized_dataframes['risk_stability'].loc[ticker].to_dict() if ticker in normalized_dataframes['risk_stability'].index else {}
                institutional_smart_money = normalized_dataframes['institutional_smart_money'].loc[ticker].to_dict() if ticker in normalized_dataframes['institutional_smart_money'].index else {}
                
                # Create NormalizedGroupFactors
                normalized = NormalizedGroupFactors(
                    ticker=ticker,
                    technical=technical,
                    fundamental=fundamental,
                    news_macro=news_macro,
                    social_alternative=social_alternative,
                    risk_stability=risk_stability,
                    institutional_smart_money=institutional_smart_money,
                    normalization_method=self.method,
                    winsorize_pct=self.winsorize_pct
                )
                
                normalized_by_ticker[ticker] = normalized
                
            except Exception as e:
                self.logger.error(f"{ticker}: Failed to reconstruct normalized factors: {e}")
                continue
        
        return normalized_by_ticker
    
    def _identity_normalization(self, 
                               calculated_by_ticker: Dict[str, GroupFactors]) -> Dict[str, NormalizedGroupFactors]:
        """
        Return identity normalization (all z-scores = 0) when not enough tickers.
        
        Used as fallback when we don't have enough tickers for cross-sectional normalization.
        """
        normalized_by_ticker = {}
        
        for ticker, group_factors in calculated_by_ticker.items():
            # Create empty dicts (z-score = 0 for all factors)
            technical = {k: 0.0 for k in group_factors.technical.keys()}
            fundamental = {k: 0.0 for k in group_factors.fundamental.keys()}
            news_macro = {k: 0.0 for k in group_factors.news_macro.keys()}
            social_alternative = {k: 0.0 for k in group_factors.social_alternative.keys()}
            risk_stability = {k: 0.0 for k in group_factors.risk_stability.keys()}
            institutional_smart_money = {k: 0.0 for k in group_factors.institutional_smart_money.keys()}
            
            normalized = NormalizedGroupFactors(
                ticker=ticker,
                technical=technical,
                fundamental=fundamental,
                news_macro=news_macro,
                social_alternative=social_alternative,
                risk_stability=risk_stability,
                institutional_smart_money=institutional_smart_money,
                normalization_method="identity",
                winsorize_pct=0.0
            )
            
            normalized_by_ticker[ticker] = normalized
        
        return normalized_by_ticker


# ============================================================================
# CONVENIENCE API
# ============================================================================

def normalize_factors(calculated_by_ticker: Dict[str, GroupFactors]) -> Dict[str, NormalizedGroupFactors]:
    """
    Convenience function to normalize calculated factors.
    
    Args:
        calculated_by_ticker: Dict[ticker, GroupFactors] from Phase 2
    
    Returns:
        Dict[ticker, NormalizedGroupFactors] with normalized z-scores
    """
    normalizer = Phase3Normalizer()
    return normalizer.normalize_all_factors(calculated_by_ticker)
