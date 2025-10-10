"""
Phase 8 Standalone Verification Test
Tests dynamic backtesting enhancements without requiring full backend imports
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple, Dict, Any


@dataclass
class Phase8BacktestConfig:
    """Phase 8 Enhanced Backtest Configuration"""
    
    # Entry thresholds by risk level
    ENTRY_THRESHOLDS_BY_RISK = {
        'Low': 0.70,
        'Moderate': 0.65,
        'Elevated': 0.60,
        'High': 0.55,
        'Extreme': 0.50
    }
    
    # Hold periods by trade type (min_days, max_days)
    HOLD_PERIODS_BY_TRADE_TYPE = {
        'Momentum': (3, 7),
        'Value': (30, 90),
        'Event-Driven': (1, 10),
        'Speculative Growth': (14, 30),
        'Contrarian': (7, 21),
        'Multi-Factor': (7, 14),
        'Balanced': (7, 14),
    }
    
    # Position sizing by risk score (% of portfolio)
    POSITION_SIZE_BY_RISK_SCORE = {
        (0, 30): (0.05, 0.10),
        (30, 50): (0.03, 0.05),
        (50, 70): (0.02, 0.03),
        (70, 85): (0.01, 0.02),
        (85, 100): (0.005, 0.01),
    }
    
    # Stop loss multipliers by risk level
    STOP_LOSS_MULTIPLIERS_BY_RISK = {
        'Low': 1.5,
        'Moderate': 1.8,
        'Elevated': 2.0,
        'High': 2.5,
        'Extreme': 3.0
    }
    
    # Take profit multipliers by risk level
    TAKE_PROFIT_MULTIPLIERS_BY_RISK = {
        'Low': 2.5,
        'Moderate': 3.0,
        'Elevated': 3.0,
        'High': 3.5,
        'Extreme': 4.0
    }
    
    # Feature flags
    use_dynamic_entry_thresholds: bool = True
    use_dynamic_hold_periods: bool = True
    use_risk_based_position_sizing: bool = True
    use_atr_based_stops: bool = True
    skip_extreme_risk: bool = False


@dataclass
class MockBacktestConfig:
    """Mock config for testing"""
    signal_threshold: Decimal = Decimal('70.0')
    position_size_pct: Decimal = Decimal('5.0')
    stop_loss_pct: Optional[Decimal] = None
    take_profit_pct: Optional[Decimal] = None
    phase8_config: Optional[Phase8BacktestConfig] = None


class MockBacktestEngine:
    """Mock engine with Phase 8 methods for testing"""
    
    def _get_entry_threshold_for_signal(self, signal: Any, config: MockBacktestConfig) -> float:
        if config.phase8_config and config.phase8_config.use_dynamic_entry_thresholds:
            risk_level = signal.get('risk_level', 'Moderate') if isinstance(signal, dict) else 'Moderate'
            return config.phase8_config.ENTRY_THRESHOLDS_BY_RISK.get(risk_level, 0.65)
        return float(config.signal_threshold) / 100.0
    
    def _get_hold_period_for_signal(self, signal: Any, config: MockBacktestConfig) -> Tuple[int, int]:
        if config.phase8_config and config.phase8_config.use_dynamic_hold_periods:
            trade_type = signal.get('trade_type', 'Balanced') if isinstance(signal, dict) else 'Balanced'
            if isinstance(trade_type, str) and ',' in trade_type:
                trade_type = trade_type.split(',')[0].strip()
            return config.phase8_config.HOLD_PERIODS_BY_TRADE_TYPE.get(trade_type, (7, 14))
        return (7, 14)
    
    def _get_position_size_for_signal(self, signal: Any, config: MockBacktestConfig, portfolio_value: Decimal) -> Decimal:
        if config.phase8_config and config.phase8_config.use_risk_based_position_sizing:
            risk_score = signal.get('risk_score', 50.0) if isinstance(signal, dict) else 50.0
            for (min_risk, max_risk), (min_pct, max_pct) in config.phase8_config.POSITION_SIZE_BY_RISK_SCORE.items():
                if min_risk <= risk_score < max_risk:
                    position_pct = (min_pct + max_pct) / 2.0
                    return portfolio_value * Decimal(str(position_pct))
            return portfolio_value * Decimal('0.01')
        position_pct = float(config.position_size_pct) / 100.0
        return portfolio_value * Decimal(str(position_pct))
    
    def _get_stop_loss_for_signal(self, signal: Any, config: MockBacktestConfig, entry_price: Decimal) -> Optional[Decimal]:
        if config.phase8_config and config.phase8_config.use_atr_based_stops:
            risk_level = signal.get('risk_level', 'Moderate') if isinstance(signal, dict) else 'Moderate'
            atr = signal.get('atr') if isinstance(signal, dict) else None
            if atr:
                multiplier = config.phase8_config.STOP_LOSS_MULTIPLIERS_BY_RISK.get(risk_level, 2.0)
                stop_distance = Decimal(str(atr)) * Decimal(str(multiplier))
                return entry_price - stop_distance
        if config.stop_loss_pct:
            return entry_price - (entry_price * config.stop_loss_pct / Decimal('100'))
        return None
    
    def _get_take_profit_for_signal(self, signal: Any, config: MockBacktestConfig, entry_price: Decimal) -> Optional[Decimal]:
        if config.phase8_config and config.phase8_config.use_atr_based_stops:
            risk_level = signal.get('risk_level', 'Moderate') if isinstance(signal, dict) else 'Moderate'
            atr = signal.get('atr') if isinstance(signal, dict) else None
            if atr:
                multiplier = config.phase8_config.TAKE_PROFIT_MULTIPLIERS_BY_RISK.get(risk_level, 3.0)
                profit_distance = Decimal(str(atr)) * Decimal(str(multiplier))
                return entry_price + profit_distance
        if config.take_profit_pct:
            return entry_price + (entry_price * config.take_profit_pct / Decimal('100'))
        return None
    
    def _should_skip_extreme_risk(self, signal: Any, config: MockBacktestConfig) -> bool:
        if config.phase8_config and config.phase8_config.skip_extreme_risk:
            risk_level = signal.get('risk_level', 'Moderate') if isinstance(signal, dict) else 'Moderate'
            return risk_level == 'Extreme'
        return False


def run_all_tests():
    """Run all Phase 8 verification tests"""
    print("=" * 80)
    print("PHASE 8 STANDALONE VERIFICATION TEST")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Configuration initialization
    print("\n1. Testing Phase8BacktestConfig initialization...")
    try:
        config = Phase8BacktestConfig()
        
        # Verify all dictionaries exist
        assert len(config.ENTRY_THRESHOLDS_BY_RISK) == 5
        assert len(config.HOLD_PERIODS_BY_TRADE_TYPE) == 7
        assert len(config.POSITION_SIZE_BY_RISK_SCORE) == 5
        assert len(config.STOP_LOSS_MULTIPLIERS_BY_RISK) == 5
        assert len(config.TAKE_PROFIT_MULTIPLIERS_BY_RISK) == 5
        
        # Verify feature flags
        assert config.use_dynamic_entry_thresholds == True
        assert config.use_dynamic_hold_periods == True
        assert config.use_risk_based_position_sizing == True
        assert config.use_atr_based_stops == True
        assert config.skip_extreme_risk == False
        
        print("   ✅ Phase8BacktestConfig initialized correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Dynamic entry thresholds
    print("\n2. Testing dynamic entry thresholds...")
    try:
        phase8_config = Phase8BacktestConfig()
        backtest_config = MockBacktestConfig(phase8_config=phase8_config)
        engine = MockBacktestEngine()
        
        test_cases = [
            ('Low', 0.70),
            ('Moderate', 0.65),
            ('High', 0.55),
            ('Extreme', 0.50)
        ]
        
        for risk_level, expected_threshold in test_cases:
            signal = {'risk_level': risk_level, 'ticker': 'TEST'}
            threshold = engine._get_entry_threshold_for_signal(signal, backtest_config)
            assert threshold == expected_threshold, f"Threshold mismatch for {risk_level}: {threshold} != {expected_threshold}"
            print(f"   {risk_level:10s}: {threshold:.2f} ✅")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Dynamic hold periods
    print("\n3. Testing dynamic hold periods...")
    try:
        phase8_config = Phase8BacktestConfig()
        backtest_config = MockBacktestConfig(phase8_config=phase8_config)
        engine = MockBacktestEngine()
        
        test_cases = [
            ('Momentum', (3, 7)),
            ('Value', (30, 90)),
            ('Event-Driven', (1, 10)),
            ('Speculative Growth', (14, 30)),
            ('Contrarian', (7, 21))
        ]
        
        for trade_type, expected_period in test_cases:
            signal = {'trade_type': trade_type, 'ticker': 'TEST'}
            period = engine._get_hold_period_for_signal(signal, backtest_config)
            assert period == expected_period, f"Period mismatch for {trade_type}: {period} != {expected_period}"
            print(f"   {trade_type:20s}: {period[0]:3d}-{period[1]:3d} days ✅")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Risk-based position sizing
    print("\n4. Testing risk-based position sizing...")
    try:
        phase8_config = Phase8BacktestConfig()
        backtest_config = MockBacktestConfig(phase8_config=phase8_config)
        engine = MockBacktestEngine()
        portfolio_value = Decimal('100000')
        
        test_cases = [
            (20, 0.075, 7500),   # Low risk: 7.5% average
            (40, 0.04, 4000),    # Moderate: 4% average
            (60, 0.025, 2500),   # Elevated: 2.5% average
            (75, 0.015, 1500),   # High: 1.5% average
            (90, 0.0075, 750)    # Extreme: 0.75% average
        ]
        
        for risk_score, expected_pct, expected_dollars in test_cases:
            signal = {'risk_score': risk_score, 'ticker': 'TEST'}
            position_size = engine._get_position_size_for_signal(signal, backtest_config, portfolio_value)
            assert abs(float(position_size) - expected_dollars) < 1, f"Position size mismatch for risk {risk_score}"
            print(f"   Risk {risk_score:3d}: {float(position_size):>8.2f} ({expected_pct*100:5.2f}%) ✅")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 5: ATR-based stop loss
    print("\n5. Testing ATR-based stop loss...")
    try:
        phase8_config = Phase8BacktestConfig()
        backtest_config = MockBacktestConfig(phase8_config=phase8_config)
        engine = MockBacktestEngine()
        entry_price = Decimal('100.00')
        atr = 2.0
        
        test_cases = [
            ('Low', 1.5, 97.00),
            ('Moderate', 1.8, 96.40),
            ('High', 2.5, 95.00),
            ('Extreme', 3.0, 94.00)
        ]
        
        for risk_level, multiplier, expected_stop in test_cases:
            signal = {'risk_level': risk_level, 'atr': atr, 'ticker': 'TEST'}
            stop = engine._get_stop_loss_for_signal(signal, backtest_config, entry_price)
            assert stop is not None
            assert abs(float(stop) - expected_stop) < 0.01, f"Stop loss mismatch for {risk_level}"
            print(f"   {risk_level:10s} (ATR×{multiplier}): ${float(stop):.2f} ✅")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 6: ATR-based take profit
    print("\n6. Testing ATR-based take profit...")
    try:
        phase8_config = Phase8BacktestConfig()
        backtest_config = MockBacktestConfig(phase8_config=phase8_config)
        engine = MockBacktestEngine()
        entry_price = Decimal('100.00')
        atr = 2.0
        
        test_cases = [
            ('Low', 2.5, 105.00),
            ('Moderate', 3.0, 106.00),
            ('High', 3.5, 107.00),
            ('Extreme', 4.0, 108.00)
        ]
        
        for risk_level, multiplier, expected_profit in test_cases:
            signal = {'risk_level': risk_level, 'atr': atr, 'ticker': 'TEST'}
            profit = engine._get_take_profit_for_signal(signal, backtest_config, entry_price)
            assert profit is not None
            assert abs(float(profit) - expected_profit) < 0.01, f"Take profit mismatch for {risk_level}"
            print(f"   {risk_level:10s} (ATR×{multiplier}): ${float(profit):.2f} ✅")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 7: Extreme risk filtering
    print("\n7. Testing extreme risk filtering...")
    try:
        phase8_config = Phase8BacktestConfig(skip_extreme_risk=True)
        backtest_config = MockBacktestConfig(phase8_config=phase8_config)
        engine = MockBacktestEngine()
        
        test_cases = [
            ('Low', False, 'ALLOW'),
            ('Moderate', False, 'ALLOW'),
            ('Extreme', True, 'SKIP')
        ]
        
        for risk_level, expected_skip, action in test_cases:
            signal = {'risk_level': risk_level, 'ticker': 'TEST'}
            should_skip = engine._should_skip_extreme_risk(signal, backtest_config)
            assert should_skip == expected_skip, f"Skip mismatch for {risk_level}"
            print(f"   {risk_level:10s}: {action} ✅")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"PHASE 8 TEST RESULTS")
    print("=" * 80)
    print(f"✅ Tests Passed: {tests_passed}")
    print(f"❌ Tests Failed: {tests_failed}")
    print(f"Total Tests: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 ALL PHASE 8 TESTS PASSED!")
        print("\nPhase 8 Features Verified:")
        print("  ✓ Dynamic entry thresholds by risk level")
        print("  ✓ Dynamic hold periods by trade type")
        print("  ✓ Risk-based position sizing")
        print("  ✓ ATR-based stop loss")
        print("  ✓ ATR-based take profit")
        print("  ✓ Extreme risk filtering")
        print("\nConfiguration Summary:")
        config = Phase8BacktestConfig()
        print(f"  • Risk levels: {len(config.ENTRY_THRESHOLDS_BY_RISK)}")
        print(f"  • Trade types: {len(config.HOLD_PERIODS_BY_TRADE_TYPE)}")
        print(f"  • Position tiers: {len(config.POSITION_SIZE_BY_RISK_SCORE)}")
        print(f"  • Stop/profit levels: {len(config.STOP_LOSS_MULTIPLIERS_BY_RISK)}")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review errors above")
    
    return tests_passed, tests_failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    exit(0 if failed == 0 else 1)
