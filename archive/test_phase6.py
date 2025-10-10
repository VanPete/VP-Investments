"""
Phase 6 Verification Test
Tests risk narrative generation from structured risk factors
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.signals import RiskScoreCalculator


def test_phase6_narrative_generation():
    """Test Phase 6 risk narrative generation"""
    
    print("=" * 70)
    print("PHASE 6 VERIFICATION TEST")
    print("=" * 70)
    
    # Initialize calculator
    print("\n1. Initializing RiskScoreCalculator...")
    calc = RiskScoreCalculator()
    print("   ✅ Calculator initialized")
    
    # Verify method exists
    print("\n2. Verifying generate_risk_narrative method...")
    assert hasattr(calc, 'generate_risk_narrative'), "❌ generate_risk_narrative method missing"
    print("   ✅ Method exists")
    
    # Test Case 1: Moderate Risk with Liquidity Concern
    print("\n3. Test Case 1: Moderate Risk (Liquidity Primary Concern)")
    print("   " + "-" * 66)
    
    risk_factors_1 = {
        'volatility_subscore': 45.2,
        'liquidity_subscore': 78.5,
        'leverage_subscore': 12.3,
        'short_interest_subscore': 30.1,
        'concentration_subscore': 55.0,
        'worst_factor': 'liquidity',
        'max_subscore': 78.5
    }
    
    narrative_1 = calc.generate_risk_narrative(
        risk_score=52.0,
        risk_level='Moderate',
        risk_factors=risk_factors_1,
        theme='Tech Rally'
    )
    
    print(f"\n   Risk Score: 52.0/100")
    print(f"   Risk Level: Moderate")
    print(f"   Worst Factor: liquidity (78.5)")
    print(f"\n   Generated Narrative:")
    print(f"   {narrative_1}")
    
    # Verify narrative components
    assert 'MODERATE RISK (52.0/100)' in narrative_1, "❌ Risk level header missing"
    assert 'liquidity' in narrative_1.lower(), "❌ Primary concern not mentioned"
    assert '78.5' in narrative_1, "❌ Liquidity score not mentioned"
    assert 'leverage is reasonable' in narrative_1.lower() or 'short interest is low' in narrative_1.lower(), "❌ Positive notes not mentioned"
    assert 'medium-risk tolerance' in narrative_1.lower(), "❌ Suitability not mentioned"
    assert 'Tech Rally' in narrative_1, "❌ Theme not mentioned"
    print("\n   ✅ Narrative correctly structured")
    
    # Test Case 2: High Risk with Volatility Concern
    print("\n4. Test Case 2: High Risk (Volatility Primary Concern)")
    print("   " + "-" * 66)
    
    risk_factors_2 = {
        'volatility_subscore': 85.0,
        'liquidity_subscore': 35.0,
        'leverage_subscore': 72.0,
        'short_interest_subscore': 65.0,
        'concentration_subscore': 40.0,
        'worst_factor': 'volatility',
        'max_subscore': 85.0
    }
    
    narrative_2 = calc.generate_risk_narrative(
        risk_score=75.0,
        risk_level='High',
        risk_factors=risk_factors_2,
        theme=None
    )
    
    print(f"\n   Risk Score: 75.0/100")
    print(f"   Risk Level: High")
    print(f"   Worst Factor: volatility (85.0)")
    print(f"\n   Generated Narrative:")
    print(f"   {narrative_2}")
    
    # Verify narrative components
    assert 'HIGH RISK (75.0/100)' in narrative_2, "❌ Risk level header missing"
    assert 'volatility' in narrative_2.lower(), "❌ Primary concern not mentioned"
    assert '85.0' in narrative_2, "❌ Volatility score not mentioned"
    assert 'leverage' in narrative_2.lower(), "❌ Leverage concern not mentioned"
    assert 'short interest' in narrative_2.lower(), "❌ Short interest concern not mentioned"
    assert 'aggressive' in narrative_2.lower(), "❌ Aggressive suitability not mentioned"
    print("\n   ✅ Narrative correctly structured")
    
    # Test Case 3: Low Risk (Conservative)
    print("\n5. Test Case 3: Low Risk (Conservative Portfolio)")
    print("   " + "-" * 66)
    
    risk_factors_3 = {
        'volatility_subscore': 18.0,
        'liquidity_subscore': 15.0,
        'leverage_subscore': 25.0,
        'short_interest_subscore': 12.0,
        'concentration_subscore': 22.0,
        'worst_factor': 'leverage',
        'max_subscore': 25.0
    }
    
    narrative_3 = calc.generate_risk_narrative(
        risk_score=20.0,
        risk_level='Low',
        risk_factors=risk_factors_3,
        theme='Blue Chip Value'
    )
    
    print(f"\n   Risk Score: 20.0/100")
    print(f"   Risk Level: Low")
    print(f"   Worst Factor: leverage (25.0)")
    print(f"\n   Generated Narrative:")
    print(f"   {narrative_3}")
    
    # Verify narrative components
    assert 'LOW RISK (20.0/100)' in narrative_3, "❌ Risk level header missing"
    assert 'volatility is manageable' in narrative_3.lower(), "❌ Positive note missing"
    assert 'liquidity is adequate' in narrative_3.lower(), "❌ Positive note missing"
    assert 'conservative' in narrative_3.lower(), "❌ Conservative suitability not mentioned"
    assert 'Blue Chip Value' in narrative_3, "❌ Theme not mentioned"
    print("\n   ✅ Narrative correctly structured")
    
    # Test Case 4: Extreme Risk (Speculative)
    print("\n6. Test Case 4: Extreme Risk (High Speculation)")
    print("   " + "-" * 66)
    
    risk_factors_4 = {
        'volatility_subscore': 92.0,
        'liquidity_subscore': 88.0,
        'leverage_subscore': 85.0,
        'short_interest_subscore': 78.0,
        'concentration_subscore': 95.0,
        'worst_factor': 'concentration',
        'max_subscore': 95.0
    }
    
    narrative_4 = calc.generate_risk_narrative(
        risk_score=92.0,
        risk_level='Extreme',
        risk_factors=risk_factors_4,
        theme='Penny Stock'
    )
    
    print(f"\n   Risk Score: 92.0/100")
    print(f"   Risk Level: Extreme")
    print(f"   Worst Factor: concentration (95.0)")
    print(f"\n   Generated Narrative:")
    print(f"   {narrative_4}")
    
    # Verify narrative components
    assert 'EXTREME RISK (92.0/100)' in narrative_4, "❌ Risk level header missing"
    assert 'concentration' in narrative_4.lower(), "❌ Primary concern not mentioned"
    assert 'volatility is elevated' in narrative_4.lower(), "❌ Secondary concern not mentioned"
    assert 'high-risk speculators' in narrative_4.lower(), "❌ Extreme warning not mentioned"
    print("\n   ✅ Narrative correctly structured")
    
    # Test narrative length
    print("\n7. Testing narrative characteristics...")
    assert len(narrative_1) > 50, "❌ Narrative too short"
    assert len(narrative_1) < 500, "❌ Narrative too long"
    assert narrative_1.endswith('.'), "❌ Narrative doesn't end with period"
    assert narrative_1.count('.') >= 2, "❌ Narrative should have multiple sentences"
    print(f"   ✅ Narrative length: {len(narrative_1)} characters")
    print(f"   ✅ Sentence count: {narrative_1.count('.')}")
    print(f"   ✅ Proper punctuation")
    
    # Test all risk levels have suitability
    print("\n8. Testing suitability recommendations for all levels...")
    levels = ['Low', 'Moderate', 'Elevated', 'High', 'Extreme']
    for level in levels:
        narrative = calc.generate_risk_narrative(
            50.0, level, risk_factors_1, None
        )
        assert any(word in narrative.lower() for word in ['suitable', 'tolerance', 'only']), \
            f"❌ No suitability recommendation for {level}"
    print("   ✅ All risk levels have suitability recommendations")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 6 VERIFICATION COMPLETE - ALL TESTS PASSED")
    print("=" * 70)
    print("\n📊 Summary:")
    print("   • generate_risk_narrative() method working")
    print("   • Risk level and score included in narrative")
    print("   • Primary concern (worst factor) highlighted")
    print("   • Secondary concerns mentioned when > 60")
    print("   • Positive notes added when < 40")
    print("   • Theme context integrated when available")
    print("   • Suitability recommendations provided")
    print("   • Narratives are concise and readable")
    print("\n🚀 Ready for Phase 7: Backtesting Integration")
    print()


if __name__ == "__main__":
    test_phase6_narrative_generation()
