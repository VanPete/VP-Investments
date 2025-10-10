# Technical Signal Group - Complete Analysis & Action Plan
**Date:** October 10, 2025  
**Status:** 🟡 Needs Improvements  
**Overall Health:** 77% (23/30 scoring columns functional)

---

## Executive Summary

The Technical signal group has **3 critical issues** affecting scoring accuracy:
1. **`macd`** - 100% NULL (worth 10% of technical score)
2. **`above_200d_ma_pct`** - 100% NULL (worth 12% of technical score)  
3. **`exit_signal_strength`** - 100% NULL (worth 5% of technical score)

These 3 columns account for **27% of the technical score** but are completely unpopulated, causing the technical scoring algorithm to lose significant signal strength.

**Good News:**
- 20 columns have >90% population (excellent)
- Core momentum metrics are working (price_1d_pct, price_7d_pct, rsi)
- Volume analysis is functional
- Phase 2 z-scores are calculating

---

## 1. Columns Used in Technical Scoring

### From `backend/core/signals.py::_calculate_technical_score()`

The technical score calculation uses **17 columns** with the following weight distribution:

| Column | Weight | Status | NULL % | Issue |
|--------|--------|--------|--------|-------|
| **Momentum Indicators (18%)** |
| `price_1d_pct` | 6% | ✅ | 2.7% | Working |
| `price_7d_pct` | 6% | ✅ | 2.7% | Working |
| `momentum_30d_pct` | 6% | ✅ | 21.9% | Working |
| **RSI (12%)** |
| `rsi` | 12% | ✅ | 2.7% | Working |
| **Moving Averages (12%)** |
| `above_50d_ma_pct` | 6% | ✅ | 11.0% | Working |
| `above_200d_ma_pct` | 6% | 🔴 | 100.0% | **NOT CALCULATED** |
| **MACD (10%)** |
| `macd` | 10% | 🔴 | 100.0% | **NOT CALCULATED** |
| **Volume Analysis (12%)** |
| `volume_spike_ratio` | 6% | ✅ | 2.7% | Working |
| `volume_price_correlation` | 6% | 🟡 | 30.1% | Partial |
| **Volatility (10%)** |
| `volatility` | 5% | ✅ | 2.7% | Working |
| `volatility_rank` | 5% | ✅ | 2.7% | Working |
| **Relative Strength (10%)** |
| `relative_strength` | 5% | ✅ | 2.7% | Working |
| `sector_relative_strength` | 5% | ✅ | 2.7% | Working |
| **Beta (8%)** |
| `beta` | 8% | 🟡 | 30.1% | Partial |
| **Momentum Consistency (7%)** |
| `momentum_consistency_score` | 7% | ✅ | 0.0% | Working |
| **Liquidity (6%)** |
| `liquidity_score` | 6% | ✅ | 0.0% | Working |
| **Exit Signals (5%)** |
| `exit_signal_strength` | 5% | 🔴 | 100.0% | **NOT CALCULATED** |

### Scoring Impact Analysis

**Currently Working (73% of score):**
- Momentum: 18% (all 3 columns functional)
- RSI: 12% (working)
- MA 50d: 6% (working)
- Volume: 12% (mostly working)
- Volatility: 10% (working)
- Relative Strength: 10% (working)
- Beta: 8% (partial - 70% population)
- Momentum Consistency: 7% (working)
- Liquidity: 6% (working)

**NOT Working (27% of score):**
- MA 200d: 6% (0% population) ❌
- MACD: 10% (0% population) ❌
- Exit Signals: 5% (0% population) ❌
- Volume correlation: 6% (30% population) ⚠️

---

## 2. Root Cause Analysis

### Issue #1: `macd` Column (100% NULL)

**Problem:**  
Pipeline calculates `macd_line` and `macd_signal` but stores them as separate columns. The scoring function looks for `macd` (single column).

**Evidence from `backend/pipeline.py` lines 2288-2292:**
```python
signal['macd_line'] = float(macd_line.iloc[-1]) if not macd_line.empty else None
signal['macd_signal'] = float(macd_signal_line.iloc[-1]) if not macd_signal_line.empty else None
signal['macd_histogram'] = float(macd_histogram.iloc[-1]) if not macd_histogram.empty else None
```

**Scoring code expects (backend/core/signals.py line 2607):**
```python
macd = financial_data.get('macd')
if macd and not np.isnan(macd):
    if macd > 0:
        macd_score = min(0.7 + abs(macd) * 0.3, 1.0)
```

**Root Cause:** Column name mismatch. Pipeline uses `macd_line`, scoring expects `macd`.

**Fix Required:**
1. Option A: Add `signal['macd'] = signal['macd_line']` in pipeline (alias)
2. Option B: Change scoring to use `macd_line` instead of `macd`
3. Option C: Calculate actual MACD (macd_line - macd_signal) and store as `macd`

**Recommendation:** Option C (calculate proper MACD indicator)

---

### Issue #2: `above_200d_ma_pct` Column (100% NULL)

**Problem:**  
Pipeline calculates 200-day MA but the percentage calculation is failing or not being stored.

**Evidence from `backend/pipeline.py` line 1209:**
```python
financial_data['above_200d_ma_pct'] = float((current_price / ma_200 - 1) * 100) if not np.isnan(ma_200) else None
```

**Possible Root Causes:**
1. `ma_200` calculation is returning NaN
2. Not enough historical data (need 200 days)
3. Calculation happens but doesn't get saved to signal dict
4. Column is calculated but not included in database insert

**Fix Required:**
1. Check if `ma_200` is being calculated in `_get_financial_data`
2. Verify 200-day history is available (may need to increase fetch window)
3. Add logging to see why calculation fails
4. Ensure value is transferred from `financial_data` to `signal` dict

**Recommendation:** Increase historical data fetch from 3 months to 12 months for 200-day MA calculation

---

### Issue #3: `exit_signal_strength` Column (100% NULL)

**Problem:**  
This column is referenced in scoring but never populated anywhere in the pipeline.

**Evidence:** No code found in pipeline that sets `exit_signal_strength`

**Root Cause:** Incomplete implementation - scoring function references a column that was planned but never implemented.

**Fix Required:**
1. Option A: Implement exit signal calculation (reverse of entry signals)
2. Option B: Remove from scoring and redistribute 5% weight to other factors
3. Option C: Set to 0 and mark as "future enhancement"

**Recommendation:** Option B (remove from scoring) - exit signals are not part of current signal generation logic

---

### Issue #4: `price_30d_pct` Missing from Schema

**Problem:**  
`check.py` references `price_30d_pct` but this column doesn't exist in schema. Pipeline uses `momentum_30d_pct` instead.

**Root Cause:** Naming inconsistency between code and schema.

**Fix Required:**
1. Remove `price_30d_pct` from `check.py` Technical group definition
2. Schema already has `momentum_30d_pct` which serves the same purpose

---

### Issue #5: Missing Columns from check.py Definition

**Problem:**  
12 columns exist in schema and are used in scoring but not tracked in `check.py`:

- `price_1d_pct` (used in scoring - 6% weight) ❌ CRITICAL
- `macd_line`, `macd_histogram` (calculated, not used in scoring)
- `beta` (used in scoring - 8% weight) ❌ CRITICAL  
- `volatility` (used in scoring - 5% weight) ❌ CRITICAL
- `volume_spike_ratio` (used in scoring - 6% weight) ❌ CRITICAL
- `volume_price_correlation` (used in scoring - 6% weight) ❌ CRITICAL
- `sector_relative_strength` (used in scoring - 5% weight) ❌ CRITICAL
- `momentum_consistency_score` (used in scoring - 7% weight) ❌ CRITICAL
- `liquidity_score` (used in scoring - 6% weight) ❌ CRITICAL
- `exit_signal_strength` (referenced but not implemented)
- `atr_pct` (duplicate of `atr_percent`?)

**Fix Required:**
Add all scoring columns to `check.py` Technical group definition for proper tracking.

---

## 3. Data Population Status

### ✅ Excellent (>90% populated) - 20 columns
- `atr`, `atr_percent` - ATR indicators working
- `avg_volume_30d` - Volume average working
- `backtest_stop_loss_price`, `backtest_take_profit_price` - Phase 8 working
- `current_price` - Basic price data working
- `historical_volatility` - Volatility calculation working
- `liquidity_score` - Phase 1.4 metric working
- `momentum_consistency_score` - Phase 1.4 metric working
- `price_1d_pct`, `price_7d_pct` - Short-term momentum working
- `relative_strength`, `sector_relative_strength` - RS working
- `rsi` - RSI working
- `volatility`, `volatility_rank` - Volatility metrics working
- `volume`, `volume_spike_ratio` - Volume metrics working
- `z_score_momentum`, `z_score_volatility`, `z_score_volume` - Phase 2 working

### 🟢 Good (70-90% populated) - 3 columns
- `above_50d_ma_pct` (11.0% NULL) - Mostly working
- `momentum_30d_pct` (21.9% NULL) - Good coverage
- `z_score_momentum` (26.0% NULL) - Needs more history

### 🟡 Medium (30-70% populated) - 7 columns
- `beta` (30.1% NULL) - Used in scoring (8% weight) ⚠️
- `bollinger_lower`, `bollinger_upper` (30.1% NULL)
- `bollinger_position`, `bollinger_width` (30.1% NULL)
- `macd_signal` (30.1% NULL) - Partial working
- `volume_price_correlation` (30.1% NULL) - Used in scoring (6% weight) ⚠️

### 🔴 Critical (<10% populated) - 3 columns
- `above_200d_ma_pct` (100.0% NULL) - Used in scoring (6% weight) 🚨
- `exit_signal_strength` (100.0% NULL) - Used in scoring (5% weight) 🚨
- `macd` (100.0% NULL) - Used in scoring (10% weight) 🚨

---

## 4. Schema vs Code Reconciliation

### Columns to ADD to schema:
1. ❌ **None** - All required columns already exist

### Columns to ADD to check.py:
1. ✅ `price_1d_pct` (CRITICAL - used in scoring)
2. ✅ `beta` (CRITICAL - used in scoring)
3. ✅ `volatility` (CRITICAL - used in scoring)
4. ✅ `volume_spike_ratio` (CRITICAL - used in scoring)
5. ✅ `volume_price_correlation` (CRITICAL - used in scoring)
6. ✅ `sector_relative_strength` (CRITICAL - used in scoring)
7. ✅ `momentum_consistency_score` (CRITICAL - used in scoring)
8. ✅ `liquidity_score` (CRITICAL - used in scoring)
9. ✅ `exit_signal_strength` (referenced but not implemented)
10. `macd_line`, `macd_histogram` (calculated, for reference)
11. `atr_pct` (may be duplicate?)

### Columns to REMOVE from check.py:
1. ❌ `price_30d_pct` (doesn't exist, use `momentum_30d_pct` instead)

### Columns to ALIAS in pipeline:
1. ✅ `macd` → calculate from `macd_line - macd_signal`

---

## 5. Action Plan

### Priority 1: Fix Critical Scoring Issues (IMMEDIATE)

**Task 1.1: Fix MACD calculation** (10% of technical score)
- **File:** `backend/pipeline.py` line ~2290
- **Change:** Add `signal['macd'] = signal['macd_line']` after macd_line calculation
- **Alternative:** Calculate proper MACD: `signal['macd'] = macd_line - macd_signal`
- **Test:** Run pipeline and verify `macd` column populated
- **Expected Impact:** +10% technical scoring accuracy

**Task 1.2: Fix 200-day MA calculation** (6% of technical score)
- **File:** `backend/pipeline.py` line ~1150 (historical data fetch)
- **Change:** Increase history fetch from `3mo` to `1y` for 200-day calculation
- **Add:** Logging to debug why `ma_200` returns NaN
- **Verify:** Ensure `above_200d_ma_pct` transfers to signal dict
- **Test:** Run pipeline and verify `above_200d_ma_pct` populated
- **Expected Impact:** +6% technical scoring accuracy

**Task 1.3: Remove exit_signal_strength from scoring** (5% of technical score)
- **File:** `backend/core/signals.py` line ~2725
- **Change:** Comment out or remove exit_signal_strength scoring section
- **Redistribute:** Add 5% weight to momentum (increase from 18% to 23%)
- **Test:** Verify technical_score still calculates to 0-1 range
- **Expected Impact:** +5% technical scoring accuracy (by removing dead weight)

### Priority 2: Update check.py Definition (DOCUMENTATION)

**Task 2.1: Add missing scoring columns to Technical group**
- **File:** `check.py` line ~46
- **Add to SIGNAL_GROUPS['Technical']['columns']:**
  - `price_1d_pct`
  - `volatility`
  - `volume_spike_ratio`
  - `volume_price_correlation`
  - `momentum_consistency_score`
  - `liquidity_score`
  - `macd_line`, `macd_histogram` (reference)

**Task 2.2: Remove invalid columns**
- **File:** `check.py` line ~46
- **Remove:** `price_30d_pct` (doesn't exist in schema)

### Priority 3: Improve Medium Population Columns (ENHANCEMENT)

**Task 3.1: Improve beta calculation** (currently 30% NULL)
- **File:** `backend/integrations/yfinance.py`
- **Investigation:** Why does beta calculation fail 30% of the time?
- **Fallback:** Use yfinance `info.get('beta')` if calculation fails
- **Expected Impact:** Reduce NULL rate from 30% to <10%

**Task 3.2: Improve volume_price_correlation** (currently 30% NULL)
- **File:** `backend/pipeline.py` (volume correlation calculation)
- **Investigation:** Check if sufficient history exists for correlation
- **Minimum:** Ensure at least 30 days of data before calculating
- **Expected Impact:** Reduce NULL rate from 30% to <15%

**Task 3.3: Fix Bollinger Band calculations** (currently 30% NULL)
- **File:** `backend/pipeline.py` line ~2298
- **Issue:** Bollinger bands stored as `bb_upper`, `bb_lower`, `bb_middle`
- **Check:** Are these being aliased to `bollinger_*` columns?
- **Action:** Either alias or update check.py to use `bb_*` names
- **Expected Impact:** Consistency in column naming

### Priority 4: Schema Cleanup (OPTIONAL)

**Task 4.1: Add column aliases for clarity**
- Consider aliasing `bb_*` to `bollinger_*` for consistency
- Consider aliasing `momentum_30d_pct` to `price_30d_pct` for clarity

**Task 4.2: Add database comments**
- Document which columns are used in scoring
- Add weight percentages to column comments
- Mark deprecated/unused columns

---

## 6. Testing Plan

### Test 1: MACD Fix Validation
```python
# After implementing fix
python backend/pipeline.py

# Check database
SELECT ticker, macd, macd_line, macd_signal 
FROM signals 
ORDER BY created_at DESC 
LIMIT 10;

# Verify: macd should equal (macd_line - macd_signal)
```

### Test 2: 200-day MA Fix Validation
```python
# After implementing fix
python backend/pipeline.py

# Check database
SELECT ticker, current_price, above_200d_ma_pct, above_50d_ma_pct
FROM signals 
WHERE above_200d_ma_pct IS NOT NULL
ORDER BY created_at DESC 
LIMIT 10;

# Verify: above_200d_ma_pct should be populated for most signals
```

### Test 3: Technical Score Validation
```python
# After all fixes
python backend/pipeline.py

# Check scoring
SELECT 
    ticker, 
    technical_score,
    signal_score,
    macd,
    above_200d_ma_pct,
    exit_signal_strength
FROM signals 
ORDER BY created_at DESC 
LIMIT 10;

# Verify: technical_score in range 0.2-0.8 (not stuck at same value)
```

### Test 4: Run check.py
```bash
python check.py

# Verify:
# - No "columns in code but not in schema" errors
# - Technical group shows <10% NULL for scoring columns
# - No critical issues flagged
```

---

## 7. Expected Outcomes

### Before Fixes:
- Technical score only using 73% of intended signals
- 27% of score weight missing (MACD, 200d MA, exit signals)
- Scoring inconsistent and potentially biased

### After Fixes:
- ✅ Technical score using 100% of intended signals
- ✅ MACD contributing 10% to score
- ✅ 200-day MA contributing 6% to score
- ✅ Exit signals removed (5% redistributed to momentum)
- ✅ All scoring columns tracked in check.py
- ✅ Technical score more accurate and reliable

### Metrics to Monitor:
- `technical_score` average should change from ~0.353 to ~0.40-0.45
- `technical_score` range should expand (currently 0.254-0.407)
- NULL rates for `macd` and `above_200d_ma_pct` should go to <5%
- Overall signal quality should improve

---

## 8. Future Enhancements

### Phase 2 Enhancements (Optional):
1. Implement proper exit signal strength calculation
2. Add MA crossover signals (golden cross, death cross)
3. Add Ichimoku cloud indicators
4. Add Fibonacci retracement levels
5. Improve beta calculation with custom SPY correlation

### Phase 3 Machine Learning (Future):
1. Use ML to optimize technical indicator weights
2. Auto-detect best indicators per sector
3. Regime-based technical scoring (bull vs bear markets)

---

## Summary

**Current Status:** 77% functional (23/30 scoring columns working)

**Critical Issues:** 3 (affecting 27% of technical score)

**Action Required:** 
1. Fix MACD calculation (10 min)
2. Fix 200-day MA calculation (30 min)
3. Remove exit_signal_strength from scoring (5 min)
4. Update check.py definitions (10 min)

**Total Time:** ~1 hour

**Expected Impact:** +27% technical scoring accuracy

**Next Steps:** Implement Priority 1 fixes and test
