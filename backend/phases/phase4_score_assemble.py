"""
Phase 4: Score & Assemble (v3.1)
==================================

Calculate weighted scores from normalized factors and assemble final signal scores.

This module is responsible for:
1. Loading factor weights and group weights from config/weights.yaml
2. Computing per-group scores by weighted aggregation of normalized factors
3. Computing overall signal_score by weighted aggregation of group scores
4. NO API calls
5. NO database operations (that's Phase 5)

Architecture:
- Input: Dict[ticker, NormalizedGroupFactors] from Phase 3
- Processing: 
  * Step 1: Compute group scores = Σ(factor_weight × z_score) for each group
  * Step 2: Compute overall score = Σ(group_weight × group_score)
- Output: Dict[ticker, ScoreResult] with group scores + overall score

Formula:
  group_score = Σ(factor_weight_i × z_score_i) for i in group
  overall_score = Σ(group_weight_j × group_score_j) for j in all groups
  
Example:
  technical_score = 0.10 × RSI_z + 0.10 × MACD_z + ... (35 factors)
  overall_score = 0.20 × technical_score + 0.25 × fundamental_score + ...
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import yaml
import math

from backend.phases.phase3_normalize import NormalizedGroupFactors
from backend.utils.metrics import emit_metric

logger = logging.getLogger(__name__)


@dataclass
class GroupScore:
    """Individual group score with metadata."""
    score: float              # Weighted score for this group (unbounded)
    factor_count: int         # Number of factors in this group
    populated_count: int      # Number of non-NaN factors
    coverage: float           # populated_count / factor_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ScoreResult:
    """Complete scoring result for one ticker."""
    ticker: str
    overall_score: float      # Final weighted score (unbounded)
    
    # Group scores
    technical: GroupScore
    fundamental: GroupScore
    news_macro: GroupScore
    social_alternative: GroupScore
    risk_stability: GroupScore
    institutional_smart_money: GroupScore
    
    # Metadata
    scored_at: str           # ISO timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'overall_score': self.overall_score,
            'technical': self.technical.to_dict(),
            'fundamental': self.fundamental.to_dict(),
            'news_macro': self.news_macro.to_dict(),
            'social_alternative': self.social_alternative.to_dict(),
            'risk_stability': self.risk_stability.to_dict(),
            'institutional_smart_money': self.institutional_smart_money.to_dict(),
            'scored_at': self.scored_at
        }


class Phase4ScoreAssembler:
    """
    Phase 4: Score & Assemble (v3.1)
    
    Computes weighted scores from normalized factors.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Phase 4 score assembler.
        
        Args:
            config_path: Path to weights.yaml (default: config/weights.yaml)
        """
        self.logger = logger
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "weights.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
        
        self.logger.info(f"[SUCCESS] Loaded scoring config from {self.config_path}")
        self.logger.info(f"[SUCCESS] Phase4ScoreAssembler initialized")
        self.logger.info(f"   Group weights loaded: {len(self.group_weights)} groups")
        self.logger.info(f"   Factor weights loaded: {sum(len(fw) for fw in self.factor_weights.values())} factors")
    
    def _load_config(self):
        """Load scoring configuration from weights.yaml."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load group weights
        self.group_weights = config['group_weights']
        
        # Validate group weights sum to 1.0
        group_weight_sum = sum(self.group_weights.values())
        if abs(group_weight_sum - 1.0) > 0.001:
            self.logger.warning(f"Group weights sum to {group_weight_sum:.4f}, normalizing...")
            total = sum(self.group_weights.values())
            self.group_weights = {k: v / total for k, v in self.group_weights.items()}
        
        # Load factor weights for each group
        self.factor_weights = {
            'technical': config['factor_weights_technical'],
            'fundamental': config['factor_weights_fundamental'],
            'news_macro': config['factor_weights_news_macro'],
            'social_alternative': config['factor_weights_social_alternative'],
            'risk_stability': config['factor_weights_risk_stability'],
            'institutional_smart_money': config['factor_weights_institutional_smart_money']
        }
        
        # Validate factor weights within each group sum to 1.0
        for group_name, weights in self.factor_weights.items():
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                self.logger.warning(f"{group_name} factor weights sum to {weight_sum:.4f}, normalizing...")
                total = sum(weights.values())
                self.factor_weights[group_name] = {k: v / total for k, v in weights.items()}
        
        # Load scoring parameters
        self.scoring_params = config.get('scoring_parameters', {})
        self.missing_penalty = self.scoring_params.get('missing_factor_penalty', 0.0)
    
    def score_all_tickers(self, 
                          normalized_by_ticker: Dict[str, NormalizedGroupFactors]) -> Dict[str, ScoreResult]:
        """
        Main entry point: Calculate scores for all tickers.
        
        Args:
            normalized_by_ticker: Dict[ticker, NormalizedGroupFactors] from Phase 3
        
        Returns:
            Dict[ticker, ScoreResult] with group scores and overall score
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 4: SCORE & ASSEMBLE (v3.1 - Weighted Scoring)")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        self.logger.info(f"[STATS] Scoring {len(normalized_by_ticker)} tickers...")
        
        # Score each ticker
        results = {}
        for ticker, normalized_factors in normalized_by_ticker.items():
            try:
                score_result = self._score_ticker(ticker, normalized_factors)
                results[ticker] = score_result
                
                emit_metric("phase4.ticker_scored", 1, {"ticker": ticker})
            except Exception as e:
                self.logger.error(f"Failed to score {ticker}: {e}", exc_info=True)
                emit_metric("phase4.ticker_failed", 1, {"ticker": ticker})
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        if results:
            avg_overall = sum(r.overall_score for r in results.values()) / len(results)
            
            self.logger.info("=" * 80)
            self.logger.info(f"[SUCCESS] PHASE 4 COMPLETE - {execution_time:.2f}s")
            self.logger.info(f"   Scored tickers: {len(results)}")
            self.logger.info(f"   Avg overall score: {avg_overall:.3f}")
            self.logger.info("=" * 80)
        else:
            self.logger.warning("[WARNING] No tickers scored successfully")
        
        emit_metric("phase4.execution_time", execution_time)
        emit_metric("phase4.tickers_scored", len(results))
        
        return results
    
    def _score_ticker(self, ticker: str, normalized: NormalizedGroupFactors) -> ScoreResult:
        """
        Calculate scores for a single ticker.
        
        Args:
            ticker: Ticker symbol
            normalized: NormalizedGroupFactors from Phase 3
        
        Returns:
            ScoreResult with group scores and overall score
        """
        # Score each group
        technical_score = self._score_group(
            'technical',
            normalized.technical,
            self.factor_weights['technical']
        )
        
        fundamental_score = self._score_group(
            'fundamental',
            normalized.fundamental,
            self.factor_weights['fundamental']
        )
        
        news_macro_score = self._score_group(
            'news_macro',
            normalized.news_macro,
            self.factor_weights['news_macro']
        )
        
        social_alternative_score = self._score_group(
            'social_alternative',
            normalized.social_alternative,
            self.factor_weights['social_alternative']
        )
        
        risk_stability_score = self._score_group(
            'risk_stability',
            normalized.risk_stability,
            self.factor_weights['risk_stability']
        )
        
        institutional_smart_money_score = self._score_group(
            'institutional_smart_money',
            normalized.institutional_smart_money,
            self.factor_weights['institutional_smart_money']
        )
        
        # Compute overall score as weighted sum of group scores
        overall_score = (
            self.group_weights['technical'] * technical_score.score +
            self.group_weights['fundamental'] * fundamental_score.score +
            self.group_weights['news_macro'] * news_macro_score.score +
            self.group_weights['social_alternative'] * social_alternative_score.score +
            self.group_weights['risk_stability'] * risk_stability_score.score +
            self.group_weights['institutional_smart_money'] * institutional_smart_money_score.score
        )
        
        # VALIDATION: Check overall score is finite and reasonable
        if np.isnan(overall_score) or np.isinf(overall_score):
            self.logger.warning(f"[{ticker}] Invalid overall score ({overall_score}), setting to 0.0")
            overall_score = 0.0
        
        if abs(overall_score) > 10:
            self.logger.warning(f"[{ticker}] Extreme overall score ({overall_score:.2f}), may indicate issue")
        
        # Calculate overall coverage
        total_factors = sum([
            technical_score.factor_count,
            fundamental_score.factor_count,
            news_macro_score.factor_count,
            social_alternative_score.factor_count,
            risk_stability_score.factor_count,
            institutional_smart_money_score.factor_count
        ])
        
        total_populated = sum([
            technical_score.populated_count,
            fundamental_score.populated_count,
            news_macro_score.populated_count,
            social_alternative_score.populated_count,
            risk_stability_score.populated_count,
            institutional_smart_money_score.populated_count
        ])
        
        total_coverage = total_populated / total_factors if total_factors > 0 else 0.0
        
        # VALIDATION: Check coverage is reasonable
        if total_coverage < 0.3:
            self.logger.warning(f"[{ticker}] Low factor coverage ({total_coverage:.1%}), scores may be unreliable")
        
        return ScoreResult(
            ticker=ticker,
            overall_score=overall_score,
            technical=technical_score,
            fundamental=fundamental_score,
            news_macro=news_macro_score,
            social_alternative=social_alternative_score,
            risk_stability=risk_stability_score,
            institutional_smart_money=institutional_smart_money_score,
            scored_at=datetime.now().isoformat()
        )
    
    def _score_group(self, 
                     group_name: str,
                     factor_dict: Dict[str, float],
                     factor_weights: Dict[str, float]) -> GroupScore:
        """
        Calculate weighted score for a single group.
        
        Formula: group_score = Σ(factor_weight_i × z_score_i)
        
        Args:
            group_name: Name of the group (for logging)
            factor_dict: Dict of factor_name -> z_score (from Phase 3)
            factor_weights: Dict of factor_name -> weight (from config)
        
        Returns:
            GroupScore with score and metadata
        """
        weighted_sum = 0.0
        populated_count = 0
        factor_count = len(factor_weights)
        
        for factor_name, weight in factor_weights.items():
            z_score = factor_dict.get(factor_name)
            
            if z_score is not None and not math.isnan(z_score):
                # Valid z-score: apply weight
                weighted_sum += weight * z_score
                populated_count += 1
            else:
                # Missing factor: apply penalty (default 0.0)
                weighted_sum += weight * self.missing_penalty
        
        coverage = populated_count / factor_count if factor_count > 0 else 0.0
        
        return GroupScore(
            score=weighted_sum,
            factor_count=factor_count,
            populated_count=populated_count,
            coverage=coverage
        )


# Convenience API
def score_tickers(normalized_by_ticker: Dict[str, NormalizedGroupFactors],
                  config_path: Optional[str] = None) -> Dict[str, ScoreResult]:
    """
    Convenience function to score tickers.
    
    Args:
        normalized_by_ticker: Dict[ticker, NormalizedGroupFactors] from Phase 3
        config_path: Optional path to weights.yaml
    
    Returns:
        Dict[ticker, ScoreResult] with scores
    """
    scorer = Phase4ScoreAssembler(config_path=config_path)
    return scorer.score_all_tickers(normalized_by_ticker)
