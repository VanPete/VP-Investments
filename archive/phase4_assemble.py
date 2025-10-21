"""
Phase 4: Assemble Scores
=========================

Combine group scores from Phase 3 into final signal_score.

This module is responsible for:
- Combining 3.0 signal groups: Technical, Fundamental, News/Macro, 
  Social/Alternative, Risk/Stability, Institutional/Smart Money
- Applying configurable weights
- Calculating final signal_score
- Calculating confidence metrics
- NO API calls
- NO database operations

Output: List of signals with final signal_score
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class Phase4Assembler:
    """
    Phase 4: Assemble Scores
    
    Combines group scores from Phase 3 into final weighted signal_score.
    """
    
    def __init__(self, config=None):
        """Initialize Phase 4 assembler."""
        self.logger = logger
        self.config = config
    
    def assemble_final_scores(self, phase3_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main entry point for Phase 4.
        
        Assembles group scores into final signal_score.
        
        Args:
            phase3_scores: Output from Phase 3 containing scored signals by group
        
        Returns:
            List of signals with final signal_score
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 4: ASSEMBLE SCORES")
        self.logger.info("=" * 60)
        
        phase4_start = datetime.now()
        
        # Extract group scores (3.0 signal groups)
        technical_scores = phase3_scores.get('technical_scores', [])
        fundamental_scores = phase3_scores.get('fundamental_scores', [])
        news_macro_scores = phase3_scores.get('news_macro_scores', [])
        social_alternative_scores = phase3_scores.get('social_alternative_scores', [])
        risk_stability_scores = phase3_scores.get('risk_stability_scores', [])
        institutional_smart_money_scores = phase3_scores.get('institutional_smart_money_scores', [])
        
        # Get scoring weights from config
        scoring_weights = self._get_scoring_weights()
        
        self.logger.info(f"📊 Scoring weights:")
        for group, weight in scoring_weights.items():
            self.logger.info(f"  {group}: {weight:.1%}")
        
        # Combine all signals by ticker
        self.logger.info("Step 4.1: Indexing signals by ticker...")
        ticker_signals = self._index_signals_by_ticker(
            technical_scores,
            fundamental_scores,
            news_macro_scores,
            social_alternative_scores,
            risk_stability_scores,
            institutional_smart_money_scores
        )
        
        # Assemble final scores
        self.logger.info("Step 4.2: Assembling final signal scores...")
        final_signals = self._assemble_ticker_scores(ticker_signals, scoring_weights)
        
        # Sort by final signal_score
        final_signals.sort(key=lambda x: x.get('signal_score', 0), reverse=True)
        
        phase4_end = datetime.now()
        execution_time = (phase4_end - phase4_start).total_seconds()
        
        self.logger.info("=" * 60)
        self.logger.info(f"PHASE 4 COMPLETE - {execution_time:.2f}s")
        self.logger.info(f"  Final signals: {len(final_signals)}")
        if final_signals:
            avg_score = sum(s.get('signal_score', 0) for s in final_signals) / len(final_signals)
            self.logger.info(f"  Avg signal_score: {avg_score:.3f}")
        else:
            self.logger.info(f"  Avg signal_score: N/A (no signals)")
        self.logger.info("=" * 60)
        
        return final_signals
    
    def _get_scoring_weights(self) -> Dict[str, float]:
        """
        Get scoring weights from config or use defaults.
        
        Returns:
            Dict of normalized scoring weights
        """
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Get weights from environment or use defaults (3.0 signal groups)
        weights = {
            'technical': float(os.getenv('SCORE_WEIGHT_TECHNICAL', '0.20')),
            'fundamental': float(os.getenv('SCORE_WEIGHT_FUNDAMENTAL', '0.25')),
            'news_macro': float(os.getenv('SCORE_WEIGHT_NEWS_MACRO', '0.15')),
            'social_alternative': float(os.getenv('SCORE_WEIGHT_SOCIAL_ALTERNATIVE', '0.10')),
            'risk_stability': float(os.getenv('SCORE_WEIGHT_RISK_STABILITY', '0.15')),
            'institutional_smart_money': float(os.getenv('SCORE_WEIGHT_INSTITUTIONAL_SMART_MONEY', '0.15')),
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Fallback if all weights are 0 (3.0 defaults)
            weights = {
                'technical': 0.20,
                'fundamental': 0.25,
                'news_macro': 0.15,
                'social_alternative': 0.10,
                'risk_stability': 0.15,
                'institutional_smart_money': 0.15
            }
        
        return weights
    
    def _index_signals_by_ticker(self,
                                 technical_scores: List[Dict],
                                 fundamental_scores: List[Dict],
                                 news_macro_scores: List[Dict],
                                 social_alternative_scores: List[Dict],
                                 risk_stability_scores: List[Dict],
                                 institutional_smart_money_scores: List[Dict]) -> Dict[str, Dict]:
        """
        Index all group scores by ticker (3.0 signal groups).
        
        Returns:
            Dict mapping ticker -> {group: score_data}
        """
        ticker_signals = {}
        
        # Index Technical scores
        for score in technical_scores:
            ticker = score.get('ticker', '').upper()
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {
                    'technical': None,
                    'fundamental': None,
                    'news_macro': None,
                    'social_alternative': None,
                    'risk_stability': None,
                    'institutional_smart_money': None
                }
            ticker_signals[ticker]['technical'] = score
        
        # Index Fundamental scores
        for score in fundamental_scores:
            ticker = score.get('ticker', '').upper()
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {
                    'technical': None,
                    'fundamental': None,
                    'news_macro': None,
                    'social_alternative': None,
                    'risk_stability': None,
                    'institutional_smart_money': None
                }
            ticker_signals[ticker]['fundamental'] = score
        
        # Index News/Macro scores
        for score in news_macro_scores:
            ticker = score.get('ticker', '').upper()
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {
                    'technical': None,
                    'fundamental': None,
                    'news_macro': None,
                    'social_alternative': None,
                    'risk_stability': None,
                    'institutional_smart_money': None
                }
            ticker_signals[ticker]['news_macro'] = score
        
        # Index Social/Alternative scores
        for score in social_alternative_scores:
            ticker = score.get('ticker', '').upper()
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {
                    'technical': None,
                    'fundamental': None,
                    'news_macro': None,
                    'social_alternative': None,
                    'risk_stability': None,
                    'institutional_smart_money': None
                }
            ticker_signals[ticker]['social_alternative'] = score
        
        # Index Risk/Stability scores
        for score in risk_stability_scores:
            ticker = score.get('ticker', '').upper()
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {
                    'technical': None,
                    'fundamental': None,
                    'news_macro': None,
                    'social_alternative': None,
                    'risk_stability': None,
                    'institutional_smart_money': None
                }
            ticker_signals[ticker]['risk_stability'] = score
        
        # Index Institutional/Smart Money scores
        for score in institutional_smart_money_scores:
            ticker = score.get('ticker', '').upper()
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {
                    'technical': None,
                    'fundamental': None,
                    'news_macro': None,
                    'social_alternative': None,
                    'risk_stability': None,
                    'institutional_smart_money': None
                }
            ticker_signals[ticker]['institutional_smart_money'] = score
        
        self.logger.info(f"✅ Indexed {len(ticker_signals)} unique tickers")
        return ticker_signals
    
    def _assemble_ticker_scores(self, 
                                ticker_signals: Dict[str, Dict],
                                scoring_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Assemble final scores for each ticker.
        
        Args:
            ticker_signals: Dict mapping ticker -> group scores
            scoring_weights: Dict of group weights
        
        Returns:
            List of signals with final signal_score
        """
        final_signals = []
        
        for ticker, group_scores in ticker_signals.items():
            try:
                # Extract individual group scores (default 0 if missing) - 3.0 groups
                technical_score = group_scores['technical'].get('score', 0.0) if group_scores['technical'] else 0.0
                fundamental_score = group_scores['fundamental'].get('score', 0.0) if group_scores['fundamental'] else 0.0
                news_macro_score = group_scores['news_macro'].get('score', 0.0) if group_scores['news_macro'] else 0.0
                social_alternative_score = group_scores['social_alternative'].get('score', 0.0) if group_scores['social_alternative'] else 0.0
                risk_stability_score = group_scores['risk_stability'].get('score', 0.0) if group_scores['risk_stability'] else 0.0
                institutional_smart_money_score = group_scores['institutional_smart_money'].get('score', 0.0) if group_scores['institutional_smart_money'] else 0.0
                
                # Calculate weighted final signal_score
                signal_score = (
                    technical_score * scoring_weights['technical'] +
                    fundamental_score * scoring_weights['fundamental'] +
                    news_macro_score * scoring_weights['news_macro'] +
                    social_alternative_score * scoring_weights['social_alternative'] +
                    risk_stability_score * scoring_weights['risk_stability'] +
                    institutional_smart_money_score * scoring_weights['institutional_smart_money']
                )
                
                # Calculate confidence based on available scores
                active_scores = sum(1 for score in [
                    group_scores['technical'],
                    group_scores['fundamental'],
                    group_scores['news_macro'],
                    group_scores['social_alternative'],
                    group_scores['risk_stability'],
                    group_scores['institutional_smart_money']
                ] if score is not None)
                
                confidence = active_scores / 6.0  # 6 groups total in 3.0
                
                # Assemble final signal (3.0 signal groups)
                final_signal = {
                    'ticker': ticker,
                    'signal_score': signal_score,
                    
                    # Group scores (3.0)
                    'technical_score': technical_score,
                    'fundamental_score': fundamental_score,
                    'news_macro_score': news_macro_score,
                    'social_alternative_score': social_alternative_score,
                    'risk_stability_score': risk_stability_score,
                    'institutional_smart_money_score': institutional_smart_money_score,
                    
                    # Metadata
                    'confidence': confidence,
                    'active_scores': active_scores,
                    'scoring_weights': scoring_weights,
                    
                    # Group data (3.0)
                    'technical_data': group_scores['technical'].get('data', {}) if group_scores['technical'] else {},
                    'fundamental_data': group_scores['fundamental'].get('data', {}) if group_scores['fundamental'] else {},
                    'news_macro_data': group_scores['news_macro'].get('data', {}) if group_scores['news_macro'] else {},
                    'social_alternative_data': group_scores['social_alternative'].get('data', {}) if group_scores['social_alternative'] else {},
                    'risk_stability_data': group_scores['risk_stability'].get('data', {}) if group_scores['risk_stability'] else {},
                    'institutional_smart_money_data': group_scores['institutional_smart_money'].get('data', {}) if group_scores['institutional_smart_money'] else {},
                    
                    # Phase 4 metadata
                    'assembled_at': datetime.now().isoformat(),
                    'phase': 'Phase 4: Assemble'
                }
                
                final_signals.append(final_signal)
                
            except Exception as e:
                self.logger.warning(f"Failed to assemble scores for {ticker}: {e}")
                continue
        
        return final_signals
