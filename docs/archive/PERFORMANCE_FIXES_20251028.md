# Performance Data Fixes - Oct 28, 2025

## Summary
Fixed critical bugs in performance tracking and added QQQ (Nasdaq) benchmark.

## Issues Fixed

### ✅ Issue #1: Performance Tab Showing "No Data"
**Problem**: Frontend wasn't fetching performance data even though it existed in database.

**Root Cause**: `useSupabaseSignals.ts` only queried `signals` table, didn't join with `performance` table.

**Fix**: Modified query to LEFT JOIN performance table and map all return fields.

**Files Changed**:
- `frontend/src/hooks/useSupabaseSignals.ts`

### ✅ Issue #2: SPY Returns Showing 0.0%
**Problem**: All SPY benchmark returns were 0.0% instead of actual market returns.

**Root Cause**: Phase 6 used forward-fill for BOTH baseline and target dates. When signal created on weekend (e.g., Oct 26 Saturday), it forward-filled baseline to Monday, making baseline = target, thus 0% return.

**Correct Logic**:
- **Baseline**: Use LAST available price before/on signal creation (backward fill)
- **Target**: Use NEXT available price on/after target date (forward fill)

**Example**:
```
Signal created: Oct 26 (Saturday) 
SPY data: Oct 24 (Fri) $677.25, Oct 27 (Mon) $685.24

OLD (broken):
  Baseline: Forward fill Oct 26 → Oct 27 = $685.24
  Target (1d): Oct 27 = $685.24  
  Return: 0.0% ❌

NEW (fixed):
  Baseline: Backward fill Oct 26 → Oct 24 = $677.25
  Target (1d): Forward fill Oct 27 → Oct 27 = $685.24
  Return: 1.18% ✅
```

**Files Changed**:
- `backend/phases/phase6_performance.py` - Updated `_get_price_at_date()` to support both directions

### ✅ Issue #8: Current Price Showing N/A
**Problem**: Dashboard showed "N/A" for current_price column.

**Root Cause**: Same as Issue #1 - performance join was missing.

**Fix**: Automatically fixed by adding performance table join.

### 🆕 Enhancement: QQQ (Nasdaq) Benchmark Added
**Why**: Many signals are tech-heavy, so comparing against Nasdaq (QQQ) provides better context than just S&P 500 (SPY).

**What's Added**:
- 7 QQQ return columns: `qqq_return_1d`, `qqq_return_3d`, `qqq_return_7d`, `qqq_return_10d`, `qqq_return_14d`, `qqq_return_30d`, `qqq_return_90d`
- 7 QQQ alpha columns (auto-calculated): `qqq_alpha_1d`, etc.
- Phase 6 now fetches and calculates QQQ returns alongside SPY

**Files Changed**:
- `migrations/013_add_qqq_benchmark.sql` - Database schema
- `backend/phases/phase6_performance.py` - QQQ calculation logic

## Migration Steps

### 1. Apply SQL Migration (Manual)
```sql
-- Copy/paste from migrations/013_add_qqq_benchmark.sql into Supabase SQL Editor
-- Adds qqq_return_* and qqq_alpha_* columns
```

### 2. Fix Existing Data
```bash
# Dry run first to preview changes
python scripts/fix_benchmark_returns.py --dry-run

# Apply fixes to all existing records
python scripts/fix_benchmark_returns.py

# Or limit to N records for testing
python scripts/fix_benchmark_returns.py --limit 50
```

This will:
- Recalculate all SPY returns with correct backward/forward fill logic
- Add QQQ returns for all existing signals
- Update `last_updated` timestamp

### 3. Deploy Frontend Changes
```bash
cd frontend
git add -A
git commit -m "Fix performance data display - join performance table"
git push
```

Vercel will auto-deploy.

### 4. Run Pipeline
```bash
python run_pipeline_and_push.py
```

New signals will automatically get correct SPY + QQQ returns from Phase 6.

## Verification

### Check Performance Tab
1. Go to https://vanpiq.com
2. Navigate to Performance tab
3. Should see:
   - 1d, 7d, 30d, 90d returns populated (not "No Data")
   - SPY returns showing ~1-2% (not 0.0%)
   - Win rates calculated
   - Top/worst performers listed

### Check Database
```sql
-- Should see corrected SPY and new QQQ values
SELECT 
  ticker,
  return_1d,
  spy_return_1d,
  qqq_return_1d,
  alpha_1d,
  qqq_alpha_1d
FROM performance p
JOIN signals s ON p.signal_id = s.id
WHERE return_1d IS NOT NULL
LIMIT 10;
```

## Impact

**Before**:
- Performance tab: "No Performance Data Available"
- SPY returns: 0.0% for all weekend signals
- No Nasdaq benchmark

**After**:
- Performance tab: Full backtest results with charts
- SPY returns: Accurate market benchmark (1-2% typical)
- QQQ returns: Tech-focused Nasdaq benchmark
- Alpha calculations: Properly show signal outperformance vs both benchmarks

## Files Modified

### Frontend
- `frontend/src/hooks/useSupabaseSignals.ts` - Added performance table join

### Backend
- `backend/phases/phase6_performance.py` - Fixed price lookup logic, added QQQ tracking

### Migrations
- `migrations/013_add_qqq_benchmark.sql` - New QQQ columns

### Scripts
- `scripts/fix_benchmark_returns.py` - Backfill correct SPY/QQQ data
- `scripts/apply_migration_013.py` - Helper to show migration SQL
- `scripts/check_spy_returns.py` - Diagnostics
- `scripts/verify_spy_return.py` - Verification
- `scripts/test_fixed_spy.py` - Unit test for fix
