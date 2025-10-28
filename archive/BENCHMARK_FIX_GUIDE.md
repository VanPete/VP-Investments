# Comprehensive Benchmark Data Fix Guide

**Date:** October 28, 2025  
**Issue:** All benchmark data (SPY, QQQ, Sector) is incorrect or missing  
**Root Causes:** 3 bugs discovered and fixed

---

## 🐛 Bugs Discovered

### Bug #1: SPY Returns Showing 0.0%
- **Location:** `backend/phases/phase6_performance.py` line 333-365
- **Cause:** Forward-fill used for baseline prices, causing weekend signals to have baseline=target
- **Example:** Oct 26 (Sat) signal → baseline forward-fills to Oct 27 (Mon) = target = 0% return
- **Fix:** Backward-fill for baseline (last price), forward-fill for target (next price)
- **Status:** ✅ FIXED

### Bug #2: Sector Data All NULL
- **Location:** `backend/phases/phase5_persist.py` line 1105
- **Cause:** `getattr(dict, 'sector', None)` - getattr doesn't work on Python dicts
- **Result:** ALL sector fields are NULL in database
- **Fix:** Changed to `dict.get('sector')` method
- **Status:** ✅ FIXED

### Bug #3: QQQ Benchmark Missing
- **Location:** Database schema + Phase 6
- **Cause:** No QQQ columns exist in performance table
- **Fix:** Migration 013 adds columns, Phase 6 calculates returns
- **Status:** ✅ CREATED, needs deployment

---

## 📋 Step-by-Step Fix Process

### Step 1: Apply QQQ Migration (CRITICAL - Do First)

```sql
-- Open Supabase SQL Editor
-- Copy and paste from migrations/013_add_qqq_benchmark.sql
-- This adds qqq_return_* and qqq_alpha_* columns

-- Verify columns exist:
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'performance' 
  AND column_name LIKE 'qqq_%';
```

**Expected result:** 14 columns (7 returns + 7 alphas)

---

### Step 2: Test Backfill Script (Dry Run)

```powershell
# Preview what will be fixed (no changes)
python scripts/comprehensive_benchmark_fix.py --dry-run --limit 5
```

**Expected output:**
```
[1/5] AAPL (intervals: [1])
  Current: SPY=0.0, QQQ=None, Sector=None
  Updated: SPY=1.1798, QQQ=1.7809, Sector=1.2345
  ℹ️  Would update (dry run)
```

---

### Step 3: Run Full Backfill

```powershell
# Apply fixes to ALL performance records
python scripts/comprehensive_benchmark_fix.py
```

**Expected results:**
- SPY Fixed: ~177 records (correct from 0.0% to actual returns)
- QQQ Added: ~177 records (add Nasdaq benchmark)
- Sector Added: ~177 records (populate sector ETF data)

---

### Step 4: Verify Database Changes

```sql
-- Check all benchmarks populated
SELECT 
    s.ticker,
    p.return_1d, 
    p.spy_return_1d, 
    p.qqq_return_1d, 
    p.sector_return_1d,
    p.alpha_1d, 
    p.qqq_alpha_1d, 
    p.sector_alpha_1d,
    p.sector, 
    p.sector_etf
FROM performance p
JOIN signals s ON p.signal_id = s.id
WHERE p.return_1d IS NOT NULL
ORDER BY p.baseline_date DESC
LIMIT 10;
```

**Expected:**
- ✅ spy_return_1d around 1.18% (NOT 0.0%)
- ✅ qqq_return_1d around 1.78% (NOT NULL)
- ✅ sector_return_1d populated (NOT NULL)
- ✅ alpha_1d, qqq_alpha_1d, sector_alpha_1d auto-calculated

---

### Step 5: Test Future Pipeline Runs

```powershell
# Run full pipeline to test Phase 5 + Phase 6
python run_pipeline_and_push.py
```

**Verify:**
1. **Phase 5:** New signals get `sector` and `sector_etf` populated
2. **Phase 6:** Calculates SPY, QQQ, sector returns with correct logic
3. **Phase 7:** Analytics use correct benchmark data

**Check new signals:**
```sql
-- Get most recent signal
SELECT 
    s.ticker,
    s.generated_at,
    p.spy_return_1d,
    p.qqq_return_1d, 
    p.sector,
    p.sector_etf,
    p.sector_return_1d
FROM signals s
JOIN performance p ON p.signal_id = s.id
ORDER BY s.generated_at DESC
LIMIT 1;
```

---

### Step 6: Verify All 7 Intervals Work

The fix applies to ALL intervals:
- 1d, 3d, 7d, 10d, 14d, 30d, 90d

**Check multi-interval signal:**
```sql
SELECT 
    s.ticker,
    p.intervals_completed,
    p.spy_return_1d, p.spy_return_7d, p.spy_return_30d,
    p.qqq_return_1d, p.qqq_return_7d, p.qqq_return_30d,
    p.sector_return_1d, p.sector_return_7d, p.sector_return_30d
FROM performance p
JOIN signals s ON p.signal_id = s.id
WHERE p.intervals_completed @> '[7]'::jsonb
ORDER BY p.baseline_date DESC
LIMIT 5;
```

**Expected:** All return columns properly populated for completed intervals

---

### Step 7: Test Frontend Display

1. Open https://vanpiq.com
2. Navigate to **Performance** tab
3. Verify:
   - ✅ Signal returns display
   - ✅ SPY returns NOT 0.0%
   - ✅ QQQ returns display
   - ✅ Alpha calculations show SPY vs QQQ comparison
   - ✅ Current price displays

---

## 🧪 Validation Queries

### Count Broken Records (Should be 0)

```sql
SELECT COUNT(*) as broken_records
FROM performance
WHERE return_1d IS NOT NULL 
  AND (
    spy_return_1d IS NULL 
    OR spy_return_1d = 0.0
    OR qqq_return_1d IS NULL
    OR sector_return_1d IS NULL
  );
```

### Check SPY 0.0% Bug Fixed

```sql
-- Before fix: Many records with SPY = 0.0
-- After fix: Should be 0 records
SELECT COUNT(*) 
FROM performance 
WHERE spy_return_1d = 0.0 
  AND return_1d IS NOT NULL;
```

### Verify QQQ Data Added

```sql
-- Should match count of records with return_1d
SELECT 
    COUNT(*) FILTER (WHERE return_1d IS NOT NULL) as total_returns,
    COUNT(*) FILTER (WHERE qqq_return_1d IS NOT NULL) as qqq_populated
FROM performance;
```

### Check Sector Coverage

```sql
-- Most tickers should have sector data
SELECT 
    sector,
    sector_etf,
    COUNT(*) as count
FROM performance
WHERE return_1d IS NOT NULL
GROUP BY sector, sector_etf
ORDER BY count DESC;
```

---

## 📊 Expected Changes Summary

| Metric | Before | After |
|--------|--------|-------|
| SPY returns = 0.0% | 177 records | 0 records |
| QQQ returns NULL | 177 records | 0 records |
| Sector data NULL | 177 records | ~10 records* |
| Alpha calculations | Incorrect | Correct |

*Some tickers may not have sector data (crypto, commodities, etc.)

---

## 🔄 Rollback Plan

If issues occur:

### Rollback Migration
```sql
-- Remove QQQ columns
ALTER TABLE performance 
  DROP COLUMN IF EXISTS qqq_return_1d,
  DROP COLUMN IF EXISTS qqq_return_3d,
  -- ... etc
```

### Revert Code Changes
```bash
git checkout HEAD~1 backend/phases/phase5_persist.py
git checkout HEAD~1 backend/phases/phase6_performance.py
git checkout HEAD~1 frontend/src/hooks/useSupabaseSignals.ts
```

---

## ⚠️ Important Notes

1. **Migration Order:** MUST apply migration 013 BEFORE running backfill script
2. **Dry Run First:** Always test with `--dry-run` flag
3. **Backup Data:** Consider exporting performance table first
4. **Monitor Logs:** Watch for errors during backfill
5. **Frontend Cache:** May need to clear browser cache to see changes

---

## 📞 Support

If you encounter issues:

1. Check `logs/vp_investments.log.1` for errors
2. Run validation queries above
3. Review Phase 6 execution for new signals
4. Compare old vs new data with dry-run output

---

## ✅ Success Criteria

- [ ] Migration 013 applied successfully
- [ ] Backfill script runs without errors
- [ ] No broken_records in validation query
- [ ] SPY returns show ~1.18% (not 0.0%)
- [ ] QQQ returns show ~1.78%
- [ ] Sector data populated
- [ ] Frontend Performance tab displays correctly
- [ ] New pipeline runs create correct benchmark data
- [ ] All 7 intervals work (1d through 90d)

---

**Status:** Ready to deploy  
**Estimated Time:** 10-15 minutes  
**Risk Level:** Low (dry-run tested, rollback available)
