# Phase 6 Planning: Score Calculations Consolidation

**Status**: 🔄 IN PROGRESS  
**Target**: 759 lines to consolidate  
**Complexity**: HIGH - Multiple scoring methods with dependencies

## Overview

Phase 6 involves moving all scoring calculation methods from `pipeline.py` to `backend/core/signals.py` SignalScorer class.

## Methods to Move (9 total)

### 1. `_calculate_reddit_score()` - 18 lines (Line 965)
- Calculates Reddit-specific signal score
- Uses: mention_count, avg_sentiment, avg_score
- Dependencies: None
- **Status**: Ready to move

### 2. `_calculate_financial_score()` - 76 lines (Line 1221)
- Comprehensive financial metrics scoring
- Includes: P/E ratio, market cap, volume, growth metrics
- Dependencies: None
- **Status**: Ready to move

### 3. `_calculate_technical_score()` - 235 lines (Line 1297)
- Technical indicator scoring
- Includes: RSI, MACD, Bollinger Bands, momentum
- Dependencies: None  
- **Status**: Ready to move

### 4. `_calculate_fundamentals_score()` - 362 lines (Line 1532)
- Fundamental analysis scoring
- Includes: P/E, revenue growth, margins, ROE, debt ratios
- Dependencies: None
- **Status**: Ready to move

### 5. `_calculate_options_score()` - 18 lines (Line 1894)
- Options data scoring (put/call ratios)
- Dependencies: None
- **Status**: Ready to move

### 6. `_calculate_short_interest_score()` - 89 lines (Line 1912)
- Short interest analysis
- Includes: Short ratio, days to cover, squeeze potential
- Dependencies: None
- **Status**: Ready to move

### 7. `_calculate_news_score()` - 19 lines (Line 2001)
- News sentiment scoring
- Dependencies: None
- **Status**: Ready to move

### 8. `_calculate_score_components()` - 107 lines (Line 2594)
- Aggregates all component scores
- Dependencies: Calls methods 2-7 above
- **Status**: Move after dependencies

### 9. `_calculate_risk_penalty()` - 16 lines (Line 2701)
- Risk-based score adjustment
- Dependencies: None
- **Status**: Ready to move

## Total Lines Analysis

- **Direct methods**: ~940 lines
- **Overlap with existing SignalScorer**: ~181 lines already implemented
- **Net new lines to add**: ~759 lines

## Implementation Strategy

### Step 1: Add Helper Scoring Methods (573 lines)
Add methods 1-7, 9 to SignalScorer class in signals.py

### Step 2: Add Main Aggregator Method (107 lines)
Add method 8 (`_calculate_score_components`) which calls the helpers

### Step 3: Update calculate_signal_score() (79 lines savings)
Replace main scoring method in pipeline.py with delegate to SignalScorer

### Step 4: Update All Call Sites
Replace pipeline method calls with SignalScorer delegates

## Existing SignalScorer Analysis

The SignalScorer class already has:
- `_calculate_reddit_score()` - SIMILAR implementation
- `_calculate_news_score()` - SIMILAR implementation  
- `_calculate_financial_score()` - SIMILAR implementation
- `_calculate_technical_score()` - SIMILAR implementation

**Decision**: pipeline.py versions are MORE COMPREHENSIVE. Replace existing implementations with pipeline.py versions.

## Testing Strategy

1. Import verification (SignalScorer can instantiate)
2. Individual method testing (each scoring method)
3. Full signal generation test (AAPL & TSLA)
4. Verify scores match previous runs

## Risk Assessment

- **HIGH**: Many methods with complex logic
- **MEDIUM**: Some methods interdependent
- **MITIGATION**: Test after each major method addition

## Expected Outcome

- **pipeline.py**: 3,326 → ~2,567 lines (-759 lines)
- **signals.py**: ~1,161 → ~1,920 lines (+759 lines)
- **Progress**: 18.5% → 48.9% complete (462 → 1,221 lines)

---

**Next Action**: Execute Step 1 - Add helper scoring methods to SignalScorer class
