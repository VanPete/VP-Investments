"""
Phase 5 Verification Test
Tests that SignalScorer integrates all Phase 2-5 calculators correctly
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.signals import SignalScorer


async def test_phase5_integration():
    """Test Phase 5 integration of all calculators"""
    
    print("=" * 70)
    print("PHASE 5 VERIFICATION TEST")
    print("=" * 70)
    
    # Initialize SignalScorer
    print("\n1. Initializing SignalScorer...")
    scorer = SignalScorer(profile="ml_optimized", db_manager=None)
    
    # Verify calculator initialization
    print("\n2. Verifying calculator initialization...")
    assert hasattr(scorer, 'z_calc'), "❌ ZScoreCalculator not initialized"
    assert hasattr(scorer, 'trend_calc'), "❌ TrendStrengthCalculator not initialized"
    assert hasattr(scorer, 'val_calc'), "❌ ValuationCalculator not initialized"
    assert hasattr(scorer, 'trade_classifier'), "❌ TradeTypeClassifier not initialized"
    assert hasattr(scorer, 'risk_calc'), "❌ RiskScoreCalculator not initialized"
    assert hasattr(scorer, 'data_cache'), "❌ Data cache not initialized"
    print("   ✅ All calculators initialized")
    
    # Verify cache methods
    print("\n3. Verifying cache methods...")
    assert hasattr(scorer, 'clear_cache'), "❌ clear_cache method missing"
    assert hasattr(scorer, '_get_enhanced_data'), "❌ _get_enhanced_data method missing"
    print("   ✅ Cache methods available")
    
    # Verify weight adjustment methods
    print("\n4. Verifying weight adjustment methods...")
    assert hasattr(scorer, '_adjust_weights_by_trade_type'), "❌ _adjust_weights_by_trade_type missing"
    assert hasattr(scorer, '_renormalize_weights'), "❌ _renormalize_weights missing"
    assert hasattr(scorer, '_calculate_contrarian_bonus'), "❌ _calculate_contrarian_bonus missing"
    print("   ✅ Weight adjustment methods available")
    
    # Test weight adjustment
    print("\n5. Testing weight adjustment...")
    
    # Test Momentum adjustment
    momentum_weights = scorer._adjust_weights_by_trade_type(['Momentum'])
    assert momentum_weights['technical'] > scorer.weights['technical'], "❌ Momentum boost not applied"
    assert abs(sum(momentum_weights.values()) - 1.0) < 0.001, "❌ Weights don't sum to 1.0"
    assert max(momentum_weights.values()) <= 0.35, "❌ Weight cap not applied"
    print(f"   ✅ Momentum: technical boosted to {momentum_weights['technical']:.4f}")
    
    # Test Value adjustment
    value_weights = scorer._adjust_weights_by_trade_type(['Value'])
    assert value_weights['fundamental'] > scorer.weights['fundamental'], "❌ Value boost not applied"
    print(f"   ✅ Value: fundamental boosted to {value_weights['fundamental']:.4f}")
    
    # Test Event-Driven adjustment
    event_weights = scorer._adjust_weights_by_trade_type(['Event-Driven'])
    assert event_weights['news_macro'] > scorer.weights['news_macro'], "❌ Event boost not applied"
    print(f"   ✅ Event-Driven: news_macro boosted to {event_weights['news_macro']:.4f}")
    
    # Test contrarian bonus
    print("\n6. Testing contrarian bonus...")
    
    # Test with valid contrarian conditions
    contrarian_details = {
        'is_oversold': True,
        'scores': {'social_z': -1.5}
    }
    bonus = scorer._calculate_contrarian_bonus(['Contrarian'], contrarian_details)
    assert bonus > 0, "❌ Contrarian bonus not calculated"
    assert abs(bonus - 0.06) < 0.001, f"❌ Contrarian bonus incorrect: {bonus}"
    print(f"   ✅ Contrarian bonus (oversold + negative sentiment): +{bonus:.4f}")
    
    # Test without contrarian tag
    no_bonus = scorer._calculate_contrarian_bonus(['Momentum'], contrarian_details)
    assert no_bonus == 0, "❌ Bonus applied to non-contrarian"
    print("   ✅ No bonus for non-contrarian trades: 0.0000")
    
    # Test cache functionality
    print("\n7. Testing cache functionality...")
    initial_cache_size = len(scorer.data_cache)
    scorer.data_cache['TEST'] = {'test': 'data'}
    assert len(scorer.data_cache) == initial_cache_size + 1, "❌ Cache not storing data"
    scorer.clear_cache()
    assert len(scorer.data_cache) == 0, "❌ Cache not clearing"
    print("   ✅ Cache store and clear working")
    
    # Test score_ticker method exists and has correct signature
    print("\n8. Verifying score_ticker method...")
    assert hasattr(scorer, 'score_ticker'), "❌ score_ticker method missing"
    import inspect
    sig = inspect.signature(scorer.score_ticker)
    assert 'ticker_data' in sig.parameters, "❌ score_ticker missing ticker_data parameter"
    print("   ✅ score_ticker method signature correct")
    
    # Verify SignalResult has Phase 5 fields
    print("\n9. Verifying SignalResult Phase 5 fields...")
    from backend.core.signals import SignalResult
    from dataclasses import fields
    
    field_names = [f.name for f in fields(SignalResult)]
    phase5_fields = [
        'trade_tags', 'risk_score', 'risk_factors', 'theme', 'event_flags',
        'technical_z', 'fundamental_z', 'news_z', 'social_z',
        'trend_strength_z', 'valuation_z',
        'ma_slope_50', 'ma_slope_200', 'volume_trend_z', 'price_z_20day',
        'atr_pct', 'float_pct', 'interest_coverage'
    ]
    
    for field in phase5_fields:
        assert field in field_names, f"❌ SignalResult missing field: {field}"
    print(f"   ✅ All {len(phase5_fields)} Phase 5 fields present in SignalResult")
    
    # Verify to_dict method exists
    print("\n10. Verifying SignalResult.to_dict() method...")
    assert hasattr(SignalResult, 'to_dict'), "❌ to_dict method missing"
    print("   ✅ to_dict method available")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 5 VERIFICATION COMPLETE - ALL TESTS PASSED")
    print("=" * 70)
    print("\n📊 Summary:")
    print("   • All 5 calculators initialized")
    print("   • Data caching system operational")
    print("   • Weight adjustment working (Momentum/Value/Event-Driven)")
    print("   • Contrarian bonus calculation correct")
    print("   • SignalResult enhanced with 18 new fields")
    print("   • to_dict() method for database storage")
    print("\n🚀 Ready for Phase 6: Narrative Generation")
    print()


if __name__ == "__main__":
    asyncio.run(test_phase5_integration())
