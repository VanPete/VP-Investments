# Quick Start: Fix All Benchmark Data

**Time Required:** 15 minutes  
**Prerequisites:** Supabase access, Python environment set up

---

## ⚡ Quick Steps

### 1️⃣ Apply Database Migration

```sql
-- Open: https://supabase.com/dashboard/project/YOUR_PROJECT/sql/new
-- Copy/paste from: migrations/013_add_qqq_benchmark.sql
-- Click: Run
```

✅ **Verify:** Run this to check columns exist:
```sql
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'performance' AND column_name LIKE 'qqq_%';
```
Should return 14 rows.

---

### 2️⃣ Test Backfill Script

```powershell
cd "c:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments"
python scripts\comprehensive_benchmark_fix.py --dry-run --limit 5
```

✅ **Expected Output:**
```
[1/5] AAPL (intervals: [1])
  Current: SPY=0.0, QQQ=None, Sector=None
  Updated: SPY=1.1798, QQQ=1.7809, Sector=1.2345
  ℹ️  Would update (dry run)
```

---

### 3️⃣ Run Full Backfill

```powershell
python scripts\comprehensive_benchmark_fix.py
```

✅ **Expected:** ~177 records updated, no errors

---

### 4️⃣ Verify Database

```sql
-- Should show correct data
SELECT 
  s.ticker,
  p.spy_return_1d,
  p.qqq_return_1d,
  p.sector,
  p.sector_etf
FROM performance p
JOIN signals s ON p.signal_id = s.id
WHERE p.return_1d IS NOT NULL
LIMIT 5;
```

✅ **Expected:** 
- SPY ~1.18% (NOT 0.0)
- QQQ ~1.78% (NOT NULL)
- Sector populated

---

### 5️⃣ Test New Pipeline Run

```powershell
python run_pipeline_and_push.py
```

✅ **Check:** New signals should have all benchmark data populated

---

### 6️⃣ Verify Frontend

Open: https://vanpiq.com/performance

✅ **Check:**
- Performance tab shows data
- SPY returns NOT 0.0%
- Alpha calculations correct

---

## 🎯 Success Criteria

Run this validation query:

```sql
-- Should return 0 broken records
SELECT COUNT(*) as broken_records
FROM performance
WHERE return_1d IS NOT NULL 
  AND (
    spy_return_1d = 0.0 
    OR qqq_return_1d IS NULL
  );
```

**Result should be: 0**

---

## ❌ If Something Goes Wrong

### Rollback Migration
```sql
ALTER TABLE performance 
  DROP COLUMN qqq_return_1d CASCADE,
  DROP COLUMN qqq_return_3d CASCADE,
  DROP COLUMN qqq_return_7d CASCADE,
  DROP COLUMN qqq_return_10d CASCADE,
  DROP COLUMN qqq_return_14d CASCADE,
  DROP COLUMN qqq_return_30d CASCADE,
  DROP COLUMN qqq_return_90d CASCADE;
```

### Revert Code
```powershell
git checkout HEAD~1 backend/phases/phase5_persist.py
git checkout HEAD~1 backend/phases/phase6_performance.py
```

---

## 📊 What Gets Fixed

| Issue | Records Affected | Fix |
|-------|-----------------|-----|
| SPY = 0.0% | 177 | Correct backward/forward fill |
| QQQ = NULL | 177 | Add Nasdaq benchmark |
| Sector = NULL | 177 | Fix dict.get() extraction |

---

## 📁 Files Modified

**Code Changes:**
- ✅ `backend/phases/phase5_persist.py` (line 1105)
- ✅ `backend/phases/phase6_performance.py` (lines 256, 260-268, 333-365)
- ✅ `frontend/src/hooks/useSupabaseSignals.ts` (lines 54-82, 96-147)

**New Files:**
- ✅ `migrations/013_add_qqq_benchmark.sql`
- ✅ `scripts/comprehensive_benchmark_fix.py`
- ✅ `scripts/test_sector_extraction.py`
- ✅ `BENCHMARK_FIX_GUIDE.md`
- ✅ `docs/BENCHMARK_FIX_SUMMARY.md`

---

## 🔍 Detailed Guides

- **Full Guide:** See `BENCHMARK_FIX_GUIDE.md`
- **Technical Details:** See `docs/BENCHMARK_FIX_SUMMARY.md`
- **Migration SQL:** See `migrations/013_add_qqq_benchmark.sql`

---

**Ready to deploy!** ✅
