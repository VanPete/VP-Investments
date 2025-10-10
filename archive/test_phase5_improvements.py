#!/usr/bin/env python3
"""
Quick Phase 5 Test Script

Tests Phase 5 improvements by processing a single ticker and checking
the population rate of all 11 Phase 5 columns.

Author: VP Investments
Date: October 10, 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.pipeline import VPInvestmentsPipeline
from dotenv import load_dotenv

load_dotenv()

def test_phase5_improvements():
    """Test Phase 5 with a known good ticker"""
    print("\n" + "="*80)
    print("PHASE 5 IMPROVEMENT TEST")
    print("="*80 + "\n")
    
    # Initialize pipeline
    pipeline = VPInvestmentsPipeline()
    
    # Test with AAPL (large-cap, should have all data)
    test_ticker = 'AAPL'
    
    print(f"Testing Phase 5 with ticker: {test_ticker}\n")
    
    # Create a mock signal
    mock_signal = {
        'ticker': test_ticker,
        'signal_score': 0.5,
        'risk_score': 50.0,
        'trade_type': 'Multi-Factor',
        'ai_commentary': 'Test signal',
        'created_at': '2025-10-10T12:00:00'
    }
    
    # Apply Phase 2-8 enhancements
    try:
        enhanced_signals = pipeline._apply_phase2_8_enhancements([mock_signal])
        
        if not enhanced_signals:
            print("❌ No enhanced signals returned")
            return False
        
        enhanced = enhanced_signals[0]
        
        # Check Phase 5 columns
        phase5_columns = [
            'atr',
            'atr_percent',
            'historical_volatility',
            'put_call_ratio',
            'open_interest',
            'operating_margin',
            'debt_to_equity',
            'current_ratio',
            'institutional_ownership',
            'insider_ownership',
            'short_interest'
        ]
        
        print("Phase 5 Column Population:\n")
        populated = 0
        
        for col in phase5_columns:
            value = enhanced.get(col)
            if value is not None:
                populated += 1
                print(f"  ✅ {col}: {value}")
            else:
                print(f"  ❌ {col}: NULL")
        
        population_rate = (populated / len(phase5_columns)) * 100
        
        print(f"\n{'='*80}")
        print(f"Phase 5 Population: {populated}/{len(phase5_columns)} ({population_rate:.1f}%)")
        
        if population_rate >= 60:
            print("✅ PASS - Phase 5 population meets target (60%+)")
            print(f"{'='*80}\n")
            return True
        else:
            print(f"⚠️  NEEDS IMPROVEMENT - Population below 60% target")
            print(f"{'='*80}\n")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during Phase 5 test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_phase5_improvements()
    sys.exit(0 if success else 1)
