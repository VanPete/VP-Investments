"""
VP Investments - Pipeline Architecture v3.2

This package contains the modular implementation of the 6-phase pipeline system:

Phase 1: FETCH - Fetch raw data (reddit→news→yfinance→market data)
Phase 2: CALCULATE - Compute 158 factors from raw data
Phase 3: NORMALIZE - Robust z-score normalization (cross-sectional)
Phase 4: SCORE & ASSEMBLE - Weighted scoring (factor→group→overall)
Phase 5: PERSIST - Save signals, factors, and performance baselines
Phase 6: PERFORMANCE TRACKING & ANALYTICS - Progressive interval tracking, risk metrics

Architecture Principles:
- NO API calls after Phase 1
- NO database reads during scoring (Phases 2-4)
- Clean separation of concerns
- Each phase has single responsibility
- Performance tracking uses hybrid approach (baselines + progressive updates)
"""

from .phase1_fetch import Phase1Fetcher
from .phase2_calculate import Phase2Calculator
from .phase3_normalize import Phase3Normalizer
from .phase4_score_assemble import Phase4ScoreAssembler
from .phase5_persist import add_phase5_methods_to_supabase_interface

__all__ = [
    'Phase1Fetcher',
    'Phase2Calculator',
    'Phase3Normalizer',
    'Phase4ScoreAssembler',
    'add_phase5_methods_to_supabase_interface',
]
