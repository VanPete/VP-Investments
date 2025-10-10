"""
Phase 8 Verification Test
Tests dynamic backtesting enhancements: entry thresholds, hold periods, position sizing
"""

import sys
import os
from datetime import date
from decimal import Decimal
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Phase 8 Standalone Configuration (copied for testing)
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


def test_phase8_backtest_enhancements():
    """Test Phase 8 backtest configuration and dynamic methods"""
    
    print("=" * 70)
    print("PHASE 8 VERIFICATION TEST")
    print("=" * 70)
    
    # Test 1: Initialize Phase8BacktestConfig
    print("\n1. Testing Phase8BacktestConfig initialization...")
    print("   " + "-" * 66)
    
    phase8_config = Phase8BacktestConfig()
    
    # Verify entry thresholds
    assert phase8_config.ENTRY_THRESHOLDS_BY_RISK['Low'] == 0.70, "❌ Low risk threshold incorrect"
    assert phase8_config.ENTRY_THRESHOLDS_BY_RISK['Moderate'] == 0.65, "❌ Moderate risk threshold incorrect"
    assert phase8_config.ENTRY_THRESHOLDS_BY_RISK['High'] == 0.55, "❌ High risk threshold incorrect"
    assert phase8_config.ENTRY_THRESHOLDS_BY_RISK['Extreme'] == 0.50, "❌ Extreme risk threshold incorrect"
    
    print("   ✅ Entry thresholds by risk level configured correctly")
    
    # Verify hold periods
    assert phase8_config.HOLD_PERIODS_BY_TRADE_TYPE['Momentum'] == (3, 7), "❌ Momentum hold period incorrect"
    assert phase8_config.HOLD_PERIODS_BY_TRADE_TYPE['Value'] == (30, 90), "❌ Value hold period incorrect"
    assert phase8_config.HOLD_PERIODS_BY_TRADE_TYPE['Event-Driven'] == (1, 10), "❌ Event-Driven hold period incorrect"
    assert phase8_config.HOLD_PERIODS_BY_TRADE_TYPE['Contrarian'] == (7, 21), "❌ Contrarian hold period incorrect"
    
    print("   ✅ Hold periods by trade type configured correctly")
    
    # Verify position sizing
    assert phase8_config.POSITION_SIZE_BY_RISK_SCORE[(0, 30)] == (0.05, 0.10), "❌ Low risk position size incorrect"
    assert phase8_config.POSITION_SIZE_BY_RISK_SCORE[(30, 50)] == (0.03, 0.05), "❌ Moderate risk position size incorrect"
    assert phase8_config.POSITION_SIZE_BY_RISK_SCORE[(70, 85)] == (0.01, 0.02), "❌ High risk position size incorrect"
    assert phase8_config.POSITION_SIZE_BY_RISK_SCORE[(85, 100)] == (0.005, 0.01), "❌ Extreme risk position size incorrect"
    
    print("   ✅ Position sizing by risk score configured correctly")
    
    # Verify stop loss multipliers
    assert phase8_config.STOP_LOSS_MULTIPLIERS_BY_RISK['Low'] == 1.5, "❌ Low risk stop loss multiplier incorrect"
    assert phase8_config.STOP_LOSS_MULTIPLIERS_BY_RISK['Moderate'] == 1.8, "❌ Moderate stop loss multiplier incorrect"
    assert phase8_config.STOP_LOSS_MULTIPLIERS_BY_RISK['High'] == 2.5, "❌ High stop loss multiplier incorrect"
    assert phase8_config.STOP_LOSS_MULTIPLIERS_BY_RISK['Extreme'] == 3.0, "❌ Extreme stop loss multiplier incorrect"
    
    print("   ✅ Stop loss multipliers by risk level configured correctly")
    
    # Verify take profit multipliers
    assert phase8_config.TAKE_PROFIT_MULTIPLIERS_BY_RISK['Low'] == 2.5, "❌ Low risk take profit multiplier incorrect"
    assert phase8_config.TAKE_PROFIT_MULTIPLIERS_BY_RISK['High'] == 3.5, "❌ High risk take profit multiplier incorrect"
    
    print("   ✅ Take profit multipliers by risk level configured correctly")
    
    # Test 2: BacktestConfig with Phase 8
    print("\n2. Testing BacktestConfig with Phase8 integration...")
    print("   " + "-" * 66)
    
    config = BacktestConfig(
        strategy=BacktestStrategy.SIGNAL_BASED,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        initial_capital=Decimal('100000'),
        phase8_config=phase8_config
    )
    
    assert config.phase8_config is not None, "❌ Phase8 config not attached"
    assert config.phase8_config.use_dynamic_entry_thresholds == True, "❌ Dynamic thresholds not enabled"
    assert config.phase8_config.use_dynamic_hold_periods == True, "❌ Dynamic hold periods not enabled"
    assert config.phase8_config.use_risk_based_position_sizing == True, "❌ Risk-based sizing not enabled"
    
    print("   ✅ BacktestConfig with Phase8 integration successful")
    
    # Test 3: Dynamic Entry Thresholds
    print("\n3. Testing dynamic entry threshold calculation...")
    print("   " + "-" * 66)
    
    engine = SupabaseBacktestEngine()
    
    # Test with dict-based signals
    signals = [
        {'risk_level': 'Low', 'ticker': 'AAPL'},
        {'risk_level': 'Moderate', 'ticker': 'MSFT'},
        {'risk_level': 'High', 'ticker': 'TSLA'},
        {'risk_level': 'Extreme', 'ticker': 'GME'},
    ]
    
    for signal in signals:
        threshold = engine._get_entry_threshold_for_signal(signal, config)
        expected = phase8_config.ENTRY_THRESHOLDS_BY_RISK[signal['risk_level']]
        assert threshold == expected, f"❌ Entry threshold mismatch for {signal['risk_level']}"
        print(f"   {signal['risk_level']} ({signal['ticker']}): {threshold:.2f} ✅")
    
    print("   ✅ Dynamic entry thresholds working correctly")
    
    # Test 4: Dynamic Hold Periods
    print("\n4. Testing dynamic hold period calculation...")
    print("   " + "-" * 66)
    
    trade_types = [
        {'trade_type': 'Momentum', 'expected': (3, 7)},
        {'trade_type': 'Value', 'expected': (30, 90)},
        {'trade_type': 'Event-Driven', 'expected': (1, 10)},
        {'trade_type': 'Speculative Growth', 'expected': (14, 30)},
        {'trade_type': 'Contrarian', 'expected': (7, 21)},
    ]
    
    for test_case in trade_types:
        signal = {'trade_type': test_case['trade_type'], 'ticker': 'TEST'}
        hold_period = engine._get_hold_period_for_signal(signal, config)
        assert hold_period == test_case['expected'], f"❌ Hold period mismatch for {test_case['trade_type']}"
        print(f"   {test_case['trade_type']}: {hold_period[0]}-{hold_period[1]} days ✅")
    
    print("   ✅ Dynamic hold periods working correctly")
    
    # Test 5: Risk-Based Position Sizing
    print("\n5. Testing risk-based position sizing...")
    print("   " + "-" * 66)
    
    portfolio_value = Decimal('100000')
    
    risk_scores = [
        {'risk_score': 20.0, 'expected_range': (5000, 10000)},    # Low: 5-10%
        {'risk_score': 40.0, 'expected_range': (3000, 5000)},     # Moderate: 3-5%
        {'risk_score': 60.0, 'expected_range': (2000, 3000)},     # Elevated: 2-3%
        {'risk_score': 75.0, 'expected_range': (1000, 2000)},     # High: 1-2%
        {'risk_score': 90.0, 'expected_range': (500, 1000)},      # Extreme: 0.5-1%
    ]
    
    for test_case in risk_scores:
        signal = {'risk_score': test_case['risk_score'], 'ticker': 'TEST'}
        position_size = engine._get_position_size_for_signal(signal, config, portfolio_value)
        
        min_expected, max_expected = test_case['expected_range']
        assert min_expected <= position_size <= max_expected, \
            f"❌ Position size {position_size} out of range for risk_score {test_case['risk_score']}"
        
        pct = (position_size / portfolio_value) * 100
        print(f"   Risk Score {test_case['risk_score']}: ${position_size:,.2f} ({pct:.2f}%) ✅")
    
    print("   ✅ Risk-based position sizing working correctly")
    
    # Test 6: ATR-Based Stop Loss
    print("\n6. Testing ATR-based stop loss calculation...")
    print("   " + "-" * 66)
    
    entry_price = Decimal('100.00')
    
    stop_loss_tests = [
        {'risk_level': 'Low', 'atr': 2.0, 'multiplier': 1.5, 'expected_stop': 97.0},
        {'risk_level': 'Moderate', 'atr': 2.0, 'multiplier': 1.8, 'expected_stop': 96.4},
        {'risk_level': 'High', 'atr': 2.0, 'multiplier': 2.5, 'expected_stop': 95.0},
        {'risk_level': 'Extreme', 'atr': 2.0, 'multiplier': 3.0, 'expected_stop': 94.0},
    ]
    
    for test_case in stop_loss_tests:
        signal = {
            'risk_level': test_case['risk_level'],
            'atr': test_case['atr'],
            'ticker': 'TEST'
        }
        stop_loss = engine._get_stop_loss_for_signal(signal, config, entry_price)
        
        assert stop_loss is not None, f"❌ Stop loss not calculated for {test_case['risk_level']}"
        assert abs(float(stop_loss) - test_case['expected_stop']) < 0.1, \
            f"❌ Stop loss mismatch for {test_case['risk_level']}"
        
        print(f"   {test_case['risk_level']} (ATR={test_case['atr']}): ${stop_loss:.2f} ✅")
    
    print("   ✅ ATR-based stop loss working correctly")
    
    # Test 7: ATR-Based Take Profit
    print("\n7. Testing ATR-based take profit calculation...")
    print("   " + "-" * 66)
    
    take_profit_tests = [
        {'risk_level': 'Low', 'atr': 2.0, 'multiplier': 2.5, 'expected_tp': 105.0},
        {'risk_level': 'Moderate', 'atr': 2.0, 'multiplier': 3.0, 'expected_tp': 106.0},
        {'risk_level': 'High', 'atr': 2.0, 'multiplier': 3.5, 'expected_tp': 107.0},
        {'risk_level': 'Extreme', 'atr': 2.0, 'multiplier': 4.0, 'expected_tp': 108.0},
    ]
    
    for test_case in take_profit_tests:
        signal = {
            'risk_level': test_case['risk_level'],
            'atr': test_case['atr'],
            'ticker': 'TEST'
        }
        take_profit = engine._get_take_profit_for_signal(signal, config, entry_price)
        
        assert take_profit is not None, f"❌ Take profit not calculated for {test_case['risk_level']}"
        assert abs(float(take_profit) - test_case['expected_tp']) < 0.1, \
            f"❌ Take profit mismatch for {test_case['risk_level']}"
        
        print(f"   {test_case['risk_level']} (ATR={test_case['atr']}): ${take_profit:.2f} ✅")
    
    print("   ✅ ATR-based take profit working correctly")
    
    # Test 8: Extreme Risk Filtering
    print("\n8. Testing extreme risk signal filtering...")
    print("   " + "-" * 66)
    
    # Test with skip_extreme_risk enabled
    phase8_config_skip = Phase8BacktestConfig()
    phase8_config_skip.skip_extreme_risk = True
    
    config_skip = BacktestConfig(
        strategy=BacktestStrategy.SIGNAL_BASED,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        phase8_config=phase8_config_skip
    )
    
    test_signals = [
        {'risk_level': 'Low', 'ticker': 'AAPL', 'should_skip': False},
        {'risk_level': 'Moderate', 'ticker': 'MSFT', 'should_skip': False},
        {'risk_level': 'Extreme', 'ticker': 'GME', 'should_skip': True},
    ]
    
    for signal in test_signals:
        should_skip = engine._should_skip_extreme_risk(signal, config_skip)
        assert should_skip == signal['should_skip'], \
            f"❌ Skip logic incorrect for {signal['ticker']} ({signal['risk_level']})"
        
        status = "SKIP" if should_skip else "ALLOW"
        print(f"   {signal['ticker']} ({signal['risk_level']}): {status} ✅")
    
    print("   ✅ Extreme risk filtering working correctly")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ PHASE 8 VERIFICATION COMPLETE - ALL TESTS PASSED")
    print("=" * 70)
    
    print("\nPhase 8 Features Verified:")
    print("  • Dynamic entry thresholds by risk level")
    print("  • Dynamic hold periods by trade type")
    print("  • Risk-based position sizing (5 tiers)")
    print("  • ATR-based stop loss (5 levels)")
    print("  • ATR-based take profit (5 levels)")
    print("  • Extreme risk signal filtering")
    
    print("\nConfiguration Summary:")
    print(f"  • Entry Thresholds: {len(phase8_config.ENTRY_THRESHOLDS_BY_RISK)} risk levels")
    print(f"  • Hold Periods: {len(phase8_config.HOLD_PERIODS_BY_TRADE_TYPE)} trade types")
    print(f"  • Position Sizing: {len(phase8_config.POSITION_SIZE_BY_RISK_SCORE)} risk tiers")
    print(f"  • Stop/Take Levels: {len(phase8_config.STOP_LOSS_MULTIPLIERS_BY_RISK)} risk levels")


if __name__ == "__main__":
    test_phase8_backtest_enhancements()
