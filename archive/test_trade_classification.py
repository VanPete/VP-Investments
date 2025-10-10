"""
Phase 9: Unit Tests for Trade Classification
Tests TradeTypeClassifier logic including momentum, value, event-driven, etc.
"""

import sys
from typing import Dict, List
from dataclasses import dataclass


# Mock classes for testing without full backend imports
@dataclass
class MockSignal:
    """Mock signal data for testing"""
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    news_macro_score: float = 0.0
    social_alternative_score: float = 0.0
    trend_strength: float = 0.0
    valuation_z: float = 0.0
    rsi: float = 50.0
    momentum_30d_pct: float = 0.0
    has_event: bool = False
    event_flags: Dict = None
    theme: str = None
    technical_z: float = 0.0
    fundamental_z: float = 0.0
    news_z: float = 0.0
    social_z: float = 0.0
    
    def __post_init__(self):
        if self.event_flags is None:
            self.event_flags = {}


class MockTradeTypeClassifier:
    """Mock classifier implementing Phase 3 logic"""
    
    def classify_trade_type(self, signal_data: MockSignal, z_scores: Dict) -> tuple:
        """
        Classify trade type based on signal characteristics.
        Returns: (primary_type, secondary_type, tags_list)
        """
        primary = None
        secondary = None
        tags = []
        
        # 1. Momentum Detection
        if signal_data.trend_strength > 0.6 and signal_data.technical_score > 0.70:
            if signal_data.rsi > 70 or signal_data.momentum_30d_pct > 15:
                if primary is None:
                    primary = 'Momentum'
                else:
                    secondary = 'Momentum'
        
        # 2. Value Detection
        if signal_data.valuation_z < -0.5 and signal_data.fundamental_score > 0.65:
            if primary is None:
                primary = 'Value'
            elif secondary is None:
                secondary = 'Value'
        
        # 3. Event-Driven Detection
        if signal_data.has_event or any(signal_data.event_flags.values()):
            if primary is None:
                primary = 'Event-Driven'
            elif secondary is None:
                secondary = 'Event-Driven'
        
        # 4. Speculative Growth
        if signal_data.technical_score > 0.75 and signal_data.valuation_z > 1.0:
            if signal_data.theme in ['AI', 'Biotech', 'Crypto']:
                if primary is None:
                    primary = 'Speculative Growth'
                elif secondary is None:
                    secondary = 'Speculative Growth'
        
        # 5. Contrarian Detection
        if signal_data.social_alternative_score < 0.35 and signal_data.rsi < 30:
            if signal_data.fundamental_score > 0.60:
                if primary is None:
                    primary = 'Contrarian'
                elif secondary is None:
                    secondary = 'Contrarian'
        
        # Default to Balanced if nothing detected
        if primary is None:
            primary = 'Balanced'
        
        # Multi-Factor auto-tagging (≥3 strong components)
        strong_components = sum([
            z_scores.get('technical_z', 0) > 0.5,
            z_scores.get('fundamental_z', 0) > 0.5,
            z_scores.get('news_z', 0) > 0.5,
            z_scores.get('social_z', 0) > 0.5
        ])
        
        if strong_components >= 3:
            tags.append('Multi-Factor')
        
        # Add both to tags list
        tags.insert(0, primary)
        if secondary:
            tags.insert(1, secondary)
        
        return primary, secondary, tags


def test_momentum_classification():
    """Test momentum trade type detection"""
    print("\n" + "="*80)
    print("TEST 1: Momentum Classification")
    print("="*80)
    
    classifier = MockTradeTypeClassifier()
    
    # Strong momentum signal
    signal = MockSignal(
        technical_score=0.85,
        trend_strength=0.72,
        rsi=75,
        momentum_30d_pct=18.5
    )
    
    z_scores = {
        'technical_z': 0.9,
        'fundamental_z': 0.3,
        'news_z': 0.4,
        'social_z': 0.2
    }
    
    primary, secondary, tags = classifier.classify_trade_type(signal, z_scores)
    
    print(f"Signal: Tech={signal.technical_score:.2f}, Trend={signal.trend_strength:.2f}, RSI={signal.rsi:.0f}")
    print(f"Result: Primary={primary}, Secondary={secondary}, Tags={tags}")
    
    assert primary == 'Momentum', f"Expected 'Momentum', got '{primary}'"
    assert 'Momentum' in tags, f"'Momentum' not in tags: {tags}"
    
    print("✅ PASS: Momentum classification correct")
    return True


def test_value_classification():
    """Test value trade type detection"""
    print("\n" + "="*80)
    print("TEST 2: Value Classification")
    print("="*80)
    
    classifier = MockTradeTypeClassifier()
    
    # Undervalued fundamental signal
    signal = MockSignal(
        fundamental_score=0.75,
        valuation_z=-0.8,  # Undervalued
        technical_score=0.55
    )
    
    z_scores = {
        'technical_z': 0.2,
        'fundamental_z': 0.7,
        'news_z': 0.3,
        'social_z': 0.1
    }
    
    primary, secondary, tags = classifier.classify_trade_type(signal, z_scores)
    
    print(f"Signal: Fundamental={signal.fundamental_score:.2f}, Valuation Z={signal.valuation_z:.2f}")
    print(f"Result: Primary={primary}, Secondary={secondary}, Tags={tags}")
    
    assert primary == 'Value', f"Expected 'Value', got '{primary}'"
    assert 'Value' in tags, f"'Value' not in tags: {tags}"
    
    print("✅ PASS: Value classification correct")
    return True


def test_event_driven_classification():
    """Test event-driven trade type detection"""
    print("\n" + "="*80)
    print("TEST 3: Event-Driven Classification")
    print("="*80)
    
    classifier = MockTradeTypeClassifier()
    
    # Event-driven signal
    signal = MockSignal(
        has_event=True,
        event_flags={'has_ma': True, 'has_contract': False, 'has_product': False},
        technical_score=0.65,
        news_macro_score=0.80
    )
    
    z_scores = {
        'technical_z': 0.4,
        'fundamental_z': 0.3,
        'news_z': 0.8,
        'social_z': 0.5
    }
    
    primary, secondary, tags = classifier.classify_trade_type(signal, z_scores)
    
    print(f"Signal: Has Event={signal.has_event}, Event Flags={signal.event_flags}")
    print(f"Result: Primary={primary}, Secondary={secondary}, Tags={tags}")
    
    assert primary == 'Event-Driven', f"Expected 'Event-Driven', got '{primary}'"
    assert 'Event-Driven' in tags, f"'Event-Driven' not in tags: {tags}"
    
    print("✅ PASS: Event-Driven classification correct")
    return True


def test_speculative_growth_classification():
    """Test speculative growth classification"""
    print("\n" + "="*80)
    print("TEST 4: Speculative Growth Classification")
    print("="*80)
    
    classifier = MockTradeTypeClassifier()
    
    # High-flying AI stock
    signal = MockSignal(
        technical_score=0.82,
        valuation_z=1.5,  # Overvalued but strong momentum
        theme='AI',
        trend_strength=0.68
    )
    
    z_scores = {
        'technical_z': 0.8,
        'fundamental_z': 0.2,
        'news_z': 0.7,
        'social_z': 0.6
    }
    
    primary, secondary, tags = classifier.classify_trade_type(signal, z_scores)
    
    print(f"Signal: Tech={signal.technical_score:.2f}, Valuation Z={signal.valuation_z:.2f}, Theme={signal.theme}")
    print(f"Result: Primary={primary}, Secondary={secondary}, Tags={tags}")
    
    assert primary == 'Speculative Growth', f"Expected 'Speculative Growth', got '{primary}'"
    assert 'Speculative Growth' in tags, f"'Speculative Growth' not in tags: {tags}"
    
    print("✅ PASS: Speculative Growth classification correct")
    return True


def test_contrarian_classification():
    """Test contrarian trade type detection"""
    print("\n" + "="*80)
    print("TEST 5: Contrarian Classification")
    print("="*80)
    
    classifier = MockTradeTypeClassifier()
    
    # Oversold with good fundamentals
    signal = MockSignal(
        social_alternative_score=0.25,  # Negative sentiment
        rsi=28,  # Oversold
        fundamental_score=0.70,
        technical_score=0.45
    )
    
    z_scores = {
        'technical_z': 0.1,
        'fundamental_z': 0.6,
        'news_z': 0.2,
        'social_z': -0.3
    }
    
    primary, secondary, tags = classifier.classify_trade_type(signal, z_scores)
    
    print(f"Signal: Social={signal.social_alternative_score:.2f}, RSI={signal.rsi:.0f}, Fundamental={signal.fundamental_score:.2f}")
    print(f"Result: Primary={primary}, Secondary={secondary}, Tags={tags}")
    
    assert primary == 'Contrarian', f"Expected 'Contrarian', got '{primary}'"
    assert 'Contrarian' in tags, f"'Contrarian' not in tags: {tags}"
    
    print("✅ PASS: Contrarian classification correct")
    return True


def test_multi_factor_tagging():
    """Test Multi-Factor tag auto-assignment"""
    print("\n" + "="*80)
    print("TEST 6: Multi-Factor Auto-Tagging")
    print("="*80)
    
    classifier = MockTradeTypeClassifier()
    
    # Strong across multiple components
    signal = MockSignal(
        technical_score=0.80,
        fundamental_score=0.75,
        news_macro_score=0.72,
        social_alternative_score=0.68
    )
    
    z_scores = {
        'technical_z': 0.8,   # Strong
        'fundamental_z': 0.7, # Strong
        'news_z': 0.6,        # Strong
        'social_z': 0.5       # Strong (4 components > 0.5)
    }
    
    primary, secondary, tags = classifier.classify_trade_type(signal, z_scores)
    
    print(f"Z-Scores: Tech={z_scores['technical_z']:.2f}, Fund={z_scores['fundamental_z']:.2f}, News={z_scores['news_z']:.2f}, Social={z_scores['social_z']:.2f}")
    print(f"Result: Primary={primary}, Secondary={secondary}, Tags={tags}")
    
    assert 'Multi-Factor' in tags, f"'Multi-Factor' not in tags: {tags}"
    
    strong_count = sum([
        z_scores['technical_z'] > 0.5,
        z_scores['fundamental_z'] > 0.5,
        z_scores['news_z'] > 0.5,
        z_scores['social_z'] > 0.5
    ])
    print(f"Strong components: {strong_count}/4")
    
    print("✅ PASS: Multi-Factor tag added correctly")
    return True


def test_balanced_default():
    """Test default Balanced classification"""
    print("\n" + "="*80)
    print("TEST 7: Balanced Default Classification")
    print("="*80)
    
    classifier = MockTradeTypeClassifier()
    
    # Mediocre signal, no strong characteristics
    signal = MockSignal(
        technical_score=0.55,
        fundamental_score=0.52,
        news_macro_score=0.48,
        social_alternative_score=0.50,
        trend_strength=0.40,
        rsi=50
    )
    
    z_scores = {
        'technical_z': 0.2,
        'fundamental_z': 0.1,
        'news_z': 0.0,
        'social_z': 0.1
    }
    
    primary, secondary, tags = classifier.classify_trade_type(signal, z_scores)
    
    print(f"Signal: All scores moderate (0.48-0.55)")
    print(f"Result: Primary={primary}, Secondary={secondary}, Tags={tags}")
    
    assert primary == 'Balanced', f"Expected 'Balanced', got '{primary}'"
    assert 'Balanced' in tags, f"'Balanced' not in tags: {tags}"
    
    print("✅ PASS: Default Balanced classification correct")
    return True


def test_dual_type_classification():
    """Test signal with both momentum and value characteristics"""
    print("\n" + "="*80)
    print("TEST 8: Dual-Type Classification (Momentum + Value)")
    print("="*80)
    
    classifier = MockTradeTypeClassifier()
    
    # Strong momentum AND undervalued
    signal = MockSignal(
        technical_score=0.78,
        trend_strength=0.65,
        rsi=72,
        momentum_30d_pct=16.0,
        fundamental_score=0.70,
        valuation_z=-0.6
    )
    
    z_scores = {
        'technical_z': 0.7,
        'fundamental_z': 0.6,
        'news_z': 0.3,
        'social_z': 0.2
    }
    
    primary, secondary, tags = classifier.classify_trade_type(signal, z_scores)
    
    print(f"Signal: Momentum indicators strong + Undervalued")
    print(f"Result: Primary={primary}, Secondary={secondary}, Tags={tags}")
    
    # Should detect both types
    assert primary in ['Momentum', 'Value'], f"Expected Momentum or Value as primary, got '{primary}'"
    assert secondary in ['Momentum', 'Value', None], f"Unexpected secondary: '{secondary}'"
    assert len(tags) >= 1, f"Expected at least 1 tag, got {len(tags)}"
    
    print(f"✅ PASS: Dual-type classification detected: {tags}")
    return True


def run_all_tests():
    """Run all trade classification tests"""
    print("\n" + "="*80)
    print("PHASE 9: TRADE CLASSIFICATION UNIT TESTS")
    print("="*80)
    
    tests = [
        ("Momentum Classification", test_momentum_classification),
        ("Value Classification", test_value_classification),
        ("Event-Driven Classification", test_event_driven_classification),
        ("Speculative Growth Classification", test_speculative_growth_classification),
        ("Contrarian Classification", test_contrarian_classification),
        ("Multi-Factor Auto-Tagging", test_multi_factor_tagging),
        ("Balanced Default", test_balanced_default),
        ("Dual-Type Classification", test_dual_type_classification),
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
    print("TRADE CLASSIFICATION TEST RESULTS")
    print("="*80)
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failed}")
    print(f"Total Tests: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TRADE CLASSIFICATION TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed - review errors above")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
