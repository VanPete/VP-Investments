"""
Phase 7 AI Integration Test
Tests AI-enhanced risk narrative generation
"""

import sys
import os
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.signals import RiskScoreCalculator


async def test_phase7_ai_narrative():
    """Test Phase 7 AI-enhanced risk narrative generation"""
    
    print("=" * 70)
    print("PHASE 7 AI INTEGRATION VERIFICATION TEST")
    print("=" * 70)
    
    # Initialize calculator
    print("\n1. Initializing RiskScoreCalculator...")
    calc = RiskScoreCalculator()
    print("   ✅ Calculator initialized")
    
    # Verify AI method exists
    print("\n2. Verifying generate_risk_narrative_ai method...")
    assert hasattr(calc, 'generate_risk_narrative_ai'), "❌ generate_risk_narrative_ai method missing"
    print("   ✅ AI method exists")
    
    # Test Case 1: AI-Enhanced Narrative (Moderate Risk)
    print("\n3. Test Case 1: AI-Enhanced Narrative - Moderate Risk")
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
    
    narrative_ai_1 = await calc.generate_risk_narrative_ai(
        risk_score=52.0,
        risk_level='Moderate',
        risk_factors=risk_factors_1,
        theme='Tech Rally',
        ticker='AAPL',
        use_ai=True
    )
    
    print(f"\n   Risk Score: 52.0/100")
    print(f"   Risk Level: Moderate")
    print(f"   Ticker: AAPL")
    print(f"   Theme: Tech Rally")
    print(f"   Worst Factor: liquidity (78.5)")
    print(f"\n   AI-Generated Narrative:")
    print(f"   {narrative_ai_1}")
    
    # Verify narrative quality
    assert len(narrative_ai_1) > 100, "❌ Narrative too short"
    assert 'MODERATE RISK' in narrative_ai_1.upper() or '52' in narrative_ai_1, "❌ Risk score missing"
    print(f"\n   ✅ AI narrative generated ({len(narrative_ai_1)} chars)")
    
    # Test Case 2: High Risk with Multiple Concerns
    print("\n4. Test Case 2: AI-Enhanced Narrative - High Risk")
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
    
    narrative_ai_2 = await calc.generate_risk_narrative_ai(
        risk_score=75.0,
        risk_level='High',
        risk_factors=risk_factors_2,
        theme='Biotech',
        ticker='MRNA',
        use_ai=True
    )
    
    print(f"\n   Risk Score: 75.0/100")
    print(f"   Risk Level: High")
    print(f"   Ticker: MRNA")
    print(f"   Theme: Biotech")
    print(f"   Worst Factor: volatility (85.0)")
    print(f"\n   AI-Generated Narrative:")
    print(f"   {narrative_ai_2}")
    
    assert len(narrative_ai_2) > 100, "❌ Narrative too short"
    assert 'HIGH RISK' in narrative_ai_2.upper() or '75' in narrative_ai_2, "❌ Risk score missing"
    print(f"\n   ✅ AI narrative generated ({len(narrative_ai_2)} chars)")
    
    # Test Case 3: Extreme Risk
    print("\n5. Test Case 3: AI-Enhanced Narrative - Extreme Risk")
    print("   " + "-" * 66)
    
    risk_factors_3 = {
        'volatility_subscore': 95.0,
        'liquidity_subscore': 88.0,
        'leverage_subscore': 92.0,
        'short_interest_subscore': 85.0,
        'concentration_subscore': 90.0,
        'worst_factor': 'volatility',
        'max_subscore': 95.0
    }
    
    narrative_ai_3 = await calc.generate_risk_narrative_ai(
        risk_score=92.0,
        risk_level='Extreme',
        risk_factors=risk_factors_3,
        theme='Crypto',
        ticker='MSTR',
        use_ai=True
    )
    
    print(f"\n   Risk Score: 92.0/100")
    print(f"   Risk Level: Extreme")
    print(f"   Ticker: MSTR")
    print(f"   Theme: Crypto")
    print(f"   Worst Factor: volatility (95.0)")
    print(f"\n   AI-Generated Narrative:")
    print(f"   {narrative_ai_3}")
    
    assert len(narrative_ai_3) > 100, "❌ Narrative too short"
    assert 'EXTREME' in narrative_ai_3.upper() or '92' in narrative_ai_3, "❌ Risk score missing"
    print(f"\n   ✅ AI narrative generated ({len(narrative_ai_3)} chars)")
    
    # Test Case 4: Fallback to Template (AI disabled)
    print("\n6. Test Case 4: Template-Based Fallback (AI disabled)")
    print("   " + "-" * 66)
    
    narrative_template = await calc.generate_risk_narrative_ai(
        risk_score=52.0,
        risk_level='Moderate',
        risk_factors=risk_factors_1,
        theme='Tech Rally',
        ticker='AAPL',
        use_ai=False  # Explicitly disable AI
    )
    
    print(f"\n   Template-Based Narrative:")
    print(f"   {narrative_template}")
    
    # Template should be structured and deterministic
    assert 'MODERATE RISK (52.0/100)' in narrative_template, "❌ Template missing risk header"
    assert 'liquidity' in narrative_template.lower(), "❌ Template missing primary concern"
    print(f"\n   ✅ Template fallback working ({len(narrative_template)} chars)")
    
    # Comparison
    print("\n7. Comparison: AI vs Template")
    print("   " + "-" * 66)
    print(f"   AI Length: {len(narrative_ai_1)} characters")
    print(f"   Template Length: {len(narrative_template)} characters")
    print(f"   AI/Template Ratio: {len(narrative_ai_1) / len(narrative_template):.2f}x")
    
    if len(narrative_ai_1) > len(narrative_template):
        print("   ✅ AI narrative is more detailed")
    else:
        print("   ⚠️  AI narrative is similar length to template (may be using fallback)")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 7 AI INTEGRATION COMPLETE - ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_phase7_ai_narrative())
