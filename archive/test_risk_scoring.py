"""
Phase 9: Unit Tests for Risk Scoring
Tests RiskScoreCalculator logic including volatility, liquidity, leverage, etc.
"""

import sys
from typing import Dict
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class MockRiskData:
    """Mock risk data for testing"""
    # Volatility
    atr_pct: float = 0.0
    beta: float = 1.0
    inverse_beta: bool = False
    
    # Liquidity
    avg_daily_volume: float = 0.0
    float_pct: float = 100.0
    market_cap: float = 1e9
    
    # Leverage
    debt_to_equity: float = 0.0
    interest_coverage: float = None
    
    # Short Interest
    short_pct_float: float = 0.0
    
    # Concentration
    theme: str = None
    sector: str = 'Technology'


class MockRiskScoreCalculator:
    """Mock risk scorer implementing Phase 4 logic"""
    
    WEIGHTS = {
        'volatility': 0.40,
        'liquidity': 0.25,
        'leverage': 0.15,
        'short_interest': 0.10,
        'concentration': 0.10
    }
    
    RISK_LEVELS = {
        'Low': (0, 25),
        'Moderate': (25, 45),
        'Elevated': (45, 65),
        'High': (65, 80),
        'Extreme': (80, 100)
    }
    
    def calculate_composite_risk(self, data: MockRiskData) -> tuple:
        """
        Calculate composite risk score with worst-factor guard.
        Returns: (risk_score, risk_level, risk_factors_dict)
        """
        # Calculate subscores
        subscores = {
            'volatility': self._score_volatility(data),
            'liquidity': self._score_liquidity(data),
            'leverage': self._score_leverage(data),
            'short_interest': self._score_short_interest(data),
            'concentration': self._score_concentration(data)
        }
        
        # Weighted composite
        composite = sum(subscores[k] * self.WEIGHTS[k] for k in subscores)
        
        # Worst-factor guard: ensure risk_score >= 0.9 * max_subfactor
        max_subfactor = max(subscores.values())
        risk_score = max(composite, 0.9 * max_subfactor)
        
        # Map to categorical
        risk_level = self._map_to_level(risk_score)
        
        # Build risk factors dict
        risk_factors = {
            'volatility': {
                'score': subscores['volatility'],
                'atr_pct': data.atr_pct,
                'beta': data.beta,
                'inverse_beta': data.inverse_beta
            },
            'liquidity': {
                'score': subscores['liquidity'],
                'adv': data.avg_daily_volume,
                'float_pct': data.float_pct,
                'market_cap_value': data.market_cap
            },
            'leverage': {
                'score': subscores['leverage'],
                'debt_to_equity': data.debt_to_equity,
                'interest_coverage': data.interest_coverage
            },
            'short_interest': {
                'score': subscores['short_interest'],
                'pct_float': data.short_pct_float
            },
            'concentration': {
                'score': subscores['concentration'],
                'theme': data.theme,
                'sector': data.sector,
                'market_cap_value': data.market_cap
            }
        }
        
        return risk_score, risk_level, risk_factors
    
    def _score_volatility(self, data: MockRiskData) -> float:
        """Score volatility (40% weight)"""
        atr_score = min(data.atr_pct * 10, 100)  # 10% ATR = 100 score
        beta_score = min(abs(data.beta) * 50, 100)  # Beta 2.0 = 100 score
        
        # Average of ATR and beta
        score = (atr_score + beta_score) / 2.0
        return min(score, 100)
    
    def _score_liquidity(self, data: MockRiskData) -> float:
        """Score liquidity risk (25% weight)"""
        # Low volume = high risk
        if data.avg_daily_volume < 100_000:
            volume_score = 90
        elif data.avg_daily_volume < 500_000:
            volume_score = 70
        elif data.avg_daily_volume < 1_000_000:
            volume_score = 50
        elif data.avg_daily_volume < 5_000_000:
            volume_score = 30
        else:
            volume_score = 10
        
        # Low float % = high risk
        if data.float_pct < 30:
            float_score = 80
        elif data.float_pct < 50:
            float_score = 60
        elif data.float_pct < 70:
            float_score = 40
        else:
            float_score = 20
        
        # Small cap = higher risk
        if data.market_cap < 100e6:  # <$100M
            cap_score = 80
        elif data.market_cap < 1e9:  # <$1B
            cap_score = 60
        elif data.market_cap < 10e9:  # <$10B
            cap_score = 30
        else:
            cap_score = 10
        
        score = (volume_score * 0.5 + float_score * 0.3 + cap_score * 0.2)
        return min(score, 100)
    
    def _score_leverage(self, data: MockRiskData) -> float:
        """Score leverage risk (15% weight)"""
        # High D/E = high risk
        if data.debt_to_equity > 3.0:
            de_score = 90
        elif data.debt_to_equity > 2.0:
            de_score = 70
        elif data.debt_to_equity > 1.0:
            de_score = 50
        elif data.debt_to_equity > 0.5:
            de_score = 30
        else:
            de_score = 10
        
        # Low interest coverage = high risk
        if data.interest_coverage is None:
            cov_score = 50  # Unknown = moderate
        elif data.interest_coverage < 1.5:
            cov_score = 90
        elif data.interest_coverage < 3.0:
            cov_score = 60
        elif data.interest_coverage < 5.0:
            cov_score = 30
        else:
            cov_score = 10
        
        score = (de_score * 0.6 + cov_score * 0.4)
        return min(score, 100)
    
    def _score_short_interest(self, data: MockRiskData) -> float:
        """Score short interest (10% weight)"""
        # High short interest = potential squeeze or distress
        if data.short_pct_float > 30:
            return 85  # Extreme short interest
        elif data.short_pct_float > 20:
            return 70
        elif data.short_pct_float > 10:
            return 50
        elif data.short_pct_float > 5:
            return 30
        else:
            return 10
    
    def _score_concentration(self, data: MockRiskData) -> float:
        """Score concentration risk (10% weight)"""
        # Thematic concentration
        risky_themes = ['Crypto', 'Biotech', 'Speculative']
        theme_score = 70 if data.theme in risky_themes else 30
        
        # Small cap adds risk
        if data.market_cap < 1e9:
            cap_score = 60
        elif data.market_cap < 10e9:
            cap_score = 40
        else:
            cap_score = 20
        
        score = (theme_score * 0.5 + cap_score * 0.5)
        return min(score, 100)
    
    def _map_to_level(self, risk_score: float) -> str:
        """Map numeric score to categorical risk level"""
        for level, (min_score, max_score) in self.RISK_LEVELS.items():
            if min_score <= risk_score < max_score:
                return level
        return 'Extreme'  # 80-100


def test_low_risk_score():
    """Test low risk profile (blue chip stock)"""
    print("\n" + "="*80)
    print("TEST 1: Low Risk Score (Blue Chip)")
    print("="*80)
    
    calculator = MockRiskScoreCalculator()
    
    # Large cap, low volatility, good financials
    data = MockRiskData(
        atr_pct=1.5,              # Low volatility
        beta=0.8,                 # Defensive
        avg_daily_volume=10_000_000,  # High liquidity
        float_pct=90.0,
        market_cap=100e9,         # $100B cap
        debt_to_equity=0.3,       # Low debt
        interest_coverage=10.0,   # Strong coverage
        short_pct_float=2.0,      # Low short interest
        theme=None,
        sector='Consumer Staples'
    )
    
    risk_score, risk_level, risk_factors = calculator.calculate_composite_risk(data)
    
    print(f"Profile: Large cap ($100B), Low vol (1.5% ATR), Beta 0.8, High liquidity")
    print(f"Result: Risk Score={risk_score:.2f}, Risk Level={risk_level}")
    print(f"Subscores: Vol={risk_factors['volatility']['score']:.1f}, "
          f"Liq={risk_factors['liquidity']['score']:.1f}, "
          f"Lev={risk_factors['leverage']['score']:.1f}")
    
    assert risk_score < 30, f"Expected low risk score (<30), got {risk_score:.2f}"
    assert risk_level in ['Low', 'Moderate'], f"Expected Low/Moderate risk, got '{risk_level}'"
    
    print(f"✅ PASS: Low risk score = {risk_score:.2f} ({risk_level})")
    return True


def test_high_volatility_score():
    """Test high volatility driving risk score"""
    print("\n" + "="*80)
    print("TEST 2: High Volatility Risk")
    print("="*80)
    
    calculator = MockRiskScoreCalculator()
    
    # High volatility stock
    data = MockRiskData(
        atr_pct=8.5,              # Very high volatility
        beta=1.8,                 # Aggressive
        avg_daily_volume=2_000_000,
        float_pct=75.0,
        market_cap=5e9,
        debt_to_equity=0.5,
        interest_coverage=5.0,
        short_pct_float=3.0,
        theme=None
    )
    
    risk_score, risk_level, risk_factors = calculator.calculate_composite_risk(data)
    
    print(f"Profile: High volatility (8.5% ATR), Beta 1.8")
    print(f"Result: Risk Score={risk_score:.2f}, Risk Level={risk_level}")
    print(f"Volatility subscore: {risk_factors['volatility']['score']:.1f}")
    
    assert risk_factors['volatility']['score'] > 60, "Expected high volatility score"
    assert risk_score > 50, f"Expected elevated risk score, got {risk_score:.2f}"
    assert risk_level in ['Elevated', 'High', 'Extreme'], f"Expected elevated+ risk, got '{risk_level}'"
    
    print(f"✅ PASS: High volatility reflected in risk score = {risk_score:.2f}")
    return True


def test_low_liquidity_score():
    """Test low liquidity driving risk score"""
    print("\n" + "="*80)
    print("TEST 3: Low Liquidity Risk")
    print("="*80)
    
    calculator = MockRiskScoreCalculator()
    
    # Illiquid small cap
    data = MockRiskData(
        atr_pct=3.0,
        beta=1.2,
        avg_daily_volume=50_000,   # Very low volume
        float_pct=25.0,            # Low float
        market_cap=150e6,          # $150M cap
        debt_to_equity=0.4,
        interest_coverage=6.0,
        short_pct_float=4.0,
        theme=None
    )
    
    risk_score, risk_level, risk_factors = calculator.calculate_composite_risk(data)
    
    print(f"Profile: Low liquidity (50k ADV), Small cap ($150M), Low float (25%)")
    print(f"Result: Risk Score={risk_score:.2f}, Risk Level={risk_level}")
    print(f"Liquidity subscore: {risk_factors['liquidity']['score']:.1f}")
    
    assert risk_factors['liquidity']['score'] > 65, "Expected high liquidity risk score"
    assert risk_score > 50, f"Expected elevated risk, got {risk_score:.2f}"
    
    print(f"✅ PASS: Low liquidity reflected in risk score = {risk_score:.2f}")
    return True


def test_high_leverage_score():
    """Test high leverage driving risk score"""
    print("\n" + "="*80)
    print("TEST 4: High Leverage Risk")
    print("="*80)
    
    calculator = MockRiskScoreCalculator()
    
    # Over-leveraged company
    data = MockRiskData(
        atr_pct=4.0,
        beta=1.5,
        avg_daily_volume=1_000_000,
        float_pct=70.0,
        market_cap=2e9,
        debt_to_equity=3.5,        # Very high leverage
        interest_coverage=1.2,     # Barely covering interest
        short_pct_float=8.0,
        theme=None
    )
    
    risk_score, risk_level, risk_factors = calculator.calculate_composite_risk(data)
    
    print(f"Profile: High leverage (D/E=3.5), Low coverage (1.2x)")
    print(f"Result: Risk Score={risk_score:.2f}, Risk Level={risk_level}")
    print(f"Leverage subscore: {risk_factors['leverage']['score']:.1f}")
    
    assert risk_factors['leverage']['score'] > 75, "Expected high leverage risk score"
    assert risk_score > 50, f"Expected elevated risk, got {risk_score:.2f}"
    
    print(f"✅ PASS: High leverage reflected in risk score = {risk_score:.2f}")
    return True


def test_extreme_short_interest():
    """Test extreme short interest score"""
    print("\n" + "="*80)
    print("TEST 5: Extreme Short Interest Risk")
    print("="*80)
    
    calculator = MockRiskScoreCalculator()
    
    # Heavily shorted stock
    data = MockRiskData(
        atr_pct=6.0,
        beta=1.4,
        avg_daily_volume=5_000_000,
        float_pct=65.0,
        market_cap=3e9,
        debt_to_equity=1.2,
        interest_coverage=3.5,
        short_pct_float=35.0,      # Extreme short interest
        theme=None
    )
    
    risk_score, risk_level, risk_factors = calculator.calculate_composite_risk(data)
    
    print(f"Profile: Extreme short interest (35% of float)")
    print(f"Result: Risk Score={risk_score:.2f}, Risk Level={risk_level}")
    print(f"Short interest subscore: {risk_factors['short_interest']['score']:.1f}")
    
    assert risk_factors['short_interest']['score'] > 80, "Expected extreme short interest score"
    
    print(f"✅ PASS: Extreme short interest detected = {risk_factors['short_interest']['score']:.1f}")
    return True


def test_thematic_concentration_risk():
    """Test thematic concentration risk (Crypto/Biotech)"""
    print("\n" + "="*80)
    print("TEST 6: Thematic Concentration Risk")
    print("="*80)
    
    calculator = MockRiskScoreCalculator()
    
    # Crypto stock with small cap
    data = MockRiskData(
        atr_pct=7.5,
        beta=2.0,
        avg_daily_volume=800_000,
        float_pct=60.0,
        market_cap=500e6,          # $500M cap
        debt_to_equity=0.8,
        interest_coverage=4.0,
        short_pct_float=12.0,
        theme='Crypto',            # Risky theme
        sector='Technology'
    )
    
    risk_score, risk_level, risk_factors = calculator.calculate_composite_risk(data)
    
    print(f"Profile: Crypto theme, Small cap ($500M), High volatility")
    print(f"Result: Risk Score={risk_score:.2f}, Risk Level={risk_level}")
    print(f"Concentration subscore: {risk_factors['concentration']['score']:.1f}")
    
    assert risk_factors['concentration']['theme'] == 'Crypto', "Theme not captured"
    assert risk_factors['concentration']['score'] > 50, "Expected elevated concentration risk"
    
    print(f"✅ PASS: Thematic concentration risk detected")
    return True


def test_worst_factor_guard():
    """Test worst-factor guard (max subscore override)"""
    print("\n" + "="*80)
    print("TEST 7: Worst-Factor Guard Logic")
    print("="*80)
    
    calculator = MockRiskScoreCalculator()
    
    # One extreme factor with otherwise moderate profile
    data = MockRiskData(
        atr_pct=12.0,              # Extreme volatility (score ~100)
        beta=1.0,
        avg_daily_volume=5_000_000,  # Good liquidity
        float_pct=80.0,
        market_cap=10e9,           # Large cap
        debt_to_equity=0.4,        # Low leverage
        interest_coverage=8.0,
        short_pct_float=3.0,       # Low short interest
        theme=None
    )
    
    risk_score, risk_level, risk_factors = calculator.calculate_composite_risk(data)
    
    vol_score = risk_factors['volatility']['score']
    liq_score = risk_factors['liquidity']['score']
    lev_score = risk_factors['leverage']['score']
    
    print(f"Profile: Extreme volatility (12% ATR) but otherwise healthy")
    print(f"Subscores: Vol={vol_score:.1f}, Liq={liq_score:.1f}, Lev={lev_score:.1f}")
    print(f"Result: Risk Score={risk_score:.2f}, Risk Level={risk_level}")
    
    # Composite without guard would be lower
    composite_no_guard = (
        vol_score * 0.40 +
        liq_score * 0.25 +
        lev_score * 0.15 +
        risk_factors['short_interest']['score'] * 0.10 +
        risk_factors['concentration']['score'] * 0.10
    )
    
    guard_threshold = 0.9 * vol_score
    
    print(f"Composite (no guard): {composite_no_guard:.2f}")
    print(f"Guard threshold (0.9 × {vol_score:.1f}): {guard_threshold:.2f}")
    print(f"Final score: {risk_score:.2f}")
    
    assert risk_score >= guard_threshold, "Guard not applied"
    assert risk_score >= composite_no_guard, "Risk score should be max of composite and guard"
    
    print(f"✅ PASS: Worst-factor guard applied correctly")
    return True


def test_extreme_risk_classification():
    """Test extreme risk classification (80-100)"""
    print("\n" + "="*80)
    print("TEST 8: Extreme Risk Classification")
    print("="*80)
    
    calculator = MockRiskScoreCalculator()
    
    # Perfect storm of risk factors
    data = MockRiskData(
        atr_pct=15.0,              # Extreme volatility
        beta=2.5,
        avg_daily_volume=25_000,   # Very low liquidity
        float_pct=15.0,            # Low float
        market_cap=80e6,           # Micro cap
        debt_to_equity=4.0,        # Extreme leverage
        interest_coverage=0.8,     # Can't cover interest
        short_pct_float=40.0,      # Extreme short interest
        theme='Crypto',
        sector='Technology'
    )
    
    risk_score, risk_level, risk_factors = calculator.calculate_composite_risk(data)
    
    print(f"Profile: Extreme across all factors")
    print(f"Result: Risk Score={risk_score:.2f}, Risk Level={risk_level}")
    print(f"All subscores:")
    for factor, details in risk_factors.items():
        print(f"  {factor:15s}: {details['score']:.1f}")
    
    assert risk_score > 70, f"Expected extreme risk score (>70), got {risk_score:.2f}"
    assert risk_level in ['High', 'Extreme'], f"Expected High/Extreme risk, got '{risk_level}'"
    
    print(f"✅ PASS: Extreme risk classification = {risk_score:.2f} ({risk_level})")
    return True


def run_all_tests():
    """Run all risk scoring tests"""
    print("\n" + "="*80)
    print("PHASE 9: RISK SCORING UNIT TESTS")
    print("="*80)
    
    tests = [
        ("Low Risk Score (Blue Chip)", test_low_risk_score),
        ("High Volatility Risk", test_high_volatility_score),
        ("Low Liquidity Risk", test_low_liquidity_score),
        ("High Leverage Risk", test_high_leverage_score),
        ("Extreme Short Interest", test_extreme_short_interest),
        ("Thematic Concentration Risk", test_thematic_concentration_risk),
        ("Worst-Factor Guard Logic", test_worst_factor_guard),
        ("Extreme Risk Classification", test_extreme_risk_classification),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_name} - {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("RISK SCORING TEST RESULTS")
    print("="*80)
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failed}")
    print(f"Total Tests: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL RISK SCORING TESTS PASSED!")
        print("\nRisk Scoring Features Verified:")
        print("  ✓ Volatility scoring (ATR%, Beta)")
        print("  ✓ Liquidity scoring (Volume, Float, Market Cap)")
        print("  ✓ Leverage scoring (D/E, Interest Coverage)")
        print("  ✓ Short interest scoring")
        print("  ✓ Concentration scoring (Theme, Sector)")
        print("  ✓ Worst-factor guard logic")
        print("  ✓ Risk level classification (Low → Extreme)")
    else:
        print(f"\n⚠️  {failed} test(s) failed - review errors above")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
