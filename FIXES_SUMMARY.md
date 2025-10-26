# Issue Resolution Summary

## Issues Fixed

### 1. Incomplete Backtest Column Population ✅

**Problem:** Only partial backtest data being saved (return_1d only, missing all other periods and SPY comparisons)

**Root Cause:** Old backtest script only populated return_1d, return_7d, return_30d, return_90d. Missing:
- `return_3d`, `return_10d`, `return_14d`  
- All SPY comparison returns (`spy_return_*`)
- `backtest_baseline_price` and `backtest_baseline_date`
- `backtest_status` and `backtest_last_update`

**Fix:** Created `backtest_complete.py` that populates ALL 19 backtest columns:
- Baseline data: `backtest_baseline_price`, `backtest_baseline_date`
- Returns: `return_1d/3d/7d/10d/14d/30d/90d`
- SPY Returns: `spy_return_1d/3d/7d/10d/14d/30d/90d`
- Status tracking: `backtest_status`, `backtest_last_update`

**File Changed:** `backtest_complete.py` (new)

---

### 2. Missing Runs in Frontend (Only Showing Subset) ✅

**Problem:** Frontend not showing all 48 runs from database

**Root Cause:** `useSupabaseSignals.ts` had two limiting filters:
```typescript
.eq('status', 'completed')  // Only completed runs
.limit(20);                 // Max 20 runs
```

**Fix:** Removed both filters to show ALL runs:
```typescript
.order('run_timestamp', { ascending: false });  // No filters
```

**File Changed:** `frontend/src/hooks/useSupabaseSignals.ts` (lines 42-49)

---

### 3. Incorrect Coverage Calculation (90.3% for All) ✅

**Problem:** All tickers showing 90.3% coverage regardless of actual data quality

**Root Cause:** Wrong MAX_FACTORS counts in coverage calculation:
```typescript
// WRONG - Phase 5 estimates
technical: 60,           // Actually 41
fundamental: 45,         // Correct
news_macro: 15,          // Actually 18
social_alternative: 10,  // Correct
risk_stability: 25,      // Actually 23
institutional_smart_money: 20, // Actually 21
```

This caused denominator to be 175 instead of actual 158, inflating percentages.

**Fix:** Updated to actual factor counts from `config/factor_to_group.yaml`:
```typescript
// CORRECT - From actual config
technical: 41,
fundamental: 45,
news_macro: 18,
social_alternative: 10,
risk_stability: 23,
institutional_smart_money: 21,
// Total: 158 factors
```

**File Changed:** `frontend/src/hooks/useSupabaseSignals.ts` (lines 84-91)

---

## How to Apply Fixes

### 1. Deploy Frontend Changes

```bash
cd frontend
npm run build
# Deploy to production (Vercel auto-deploys from GitHub)
```

### 2. Run Backtest for All Signals

```bash
# This will backfill ALL backtest columns for all signals
python backtest_complete.py
```

This will:
- Find ~1,687 signals missing complete backtest data
- Calculate returns for all 7 periods (1d, 3d, 7d, 10d, 14d, 30d, 90d)
- Calculate SPY benchmark returns for same periods
- Set status ('completed', 'partial', 'pending', or 'error')
- Update baseline price/date
- Rate-limit to avoid hammering yfinance (~5 signals/second)

**Expected Runtime:** ~5-10 minutes for 1,687 signals

---

## Verification

After running fixes, verify:

1. **All 48 runs showing:** Check runs dropdown in frontend
2. **Coverage varies:** AMD should show ~95%, tickers with missing data should show lower
3. **Backtest data complete:** Check database:
   ```sql
   SELECT 
     COUNT(*) as total,
     COUNT(backtest_baseline_price) as has_baseline,
     COUNT(return_1d) as has_1d,
     COUNT(return_7d) as has_7d,
     COUNT(spy_return_7d) as has_spy_7d,
     COUNT(CASE WHEN backtest_status = 'completed' THEN 1 END) as completed
   FROM signals;
   ```

---

## Files Changed

1. `frontend/src/hooks/useSupabaseSignals.ts`
   - Line 42-49: Removed run filters  
   - Line 84-91: Fixed MAX_FACTORS

2. `backtest_complete.py` (NEW)
   - Complete backtest implementation
   - Populates all 19 backtest columns

3. `check_backtest_schema.py` (NEW - temporary)
   - Schema verification tool

4. `check_backtest_coverage.py` (temporary)
   - Coverage check tool
