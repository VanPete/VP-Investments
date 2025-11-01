# Migration 017 Execution Guide
**Purpose:** Add market_cap and beta to signals table for VanPiQ Performance Tab  
**Date:** 2025-10-31  
**Status:** ✅ Backend code updated, ⏳ Migration pending execution

---

## Changes Summary

### ✅ Completed

**1. Migration Created**
- File: `migrations/017_add_mktcap_beta_to_signals.sql`
- Adds: `market_cap` (BIGINT), `beta` (REAL)
- Comments added for documentation

**2. Phase 5 Updated** (`backend/phases/phase5_persist.py`)
- Lines 1417-1433: Extract `market_cap` and `beta` from `ticker_raw_data.info`
- Lines 1463-1465: Add to `signal_record` dict
- Lines 660-707: Updated `insert_signals_batch()` to include 2 new columns (20→22 params)

**3. supabase.sql Reference Updated**
- Lines 195-196: Added `market_cap bigint` and `beta real` columns

---

## Migration 017 Execution

### Step 1: Review Migration

```powershell
# View migration SQL
cat migrations\017_add_mktcap_beta_to_signals.sql
```

**Expected content:**
```sql
ALTER TABLE signals 
ADD COLUMN IF NOT EXISTS market_cap BIGINT;

ALTER TABLE signals 
ADD COLUMN IF NOT EXISTS beta REAL;

COMMENT ON COLUMN signals.market_cap IS 'Market capitalization in USD (from YFinance info.marketCap)';
COMMENT ON COLUMN signals.beta IS 'Beta vs SPY - measures stock volatility relative to market (from YFinance info.beta)';
```

---

### Step 2: Execute in Supabase

1. Open Supabase Dashboard: https://supabase.com/dashboard
2. Navigate to: **Project → SQL Editor**
3. Click **+ New Query**
4. Copy/paste migration 017 content
5. Click **Run** (Ctrl+Enter)

---

### Step 3: Verify Migration

**In Supabase SQL Editor:**

```sql
-- Verify columns added
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'public'
  AND table_name = 'signals' 
  AND column_name IN ('market_cap', 'beta')
ORDER BY column_name;
```

**Expected output:**
```
column_name  | data_type | is_nullable | column_default
-------------+-----------+-------------+---------------
beta         | real      | YES         | NULL
market_cap   | bigint    | YES         | NULL
```

---

### Step 4: Test Pipeline Run

**Run pipeline to test data extraction:**

```powershell
# Activate environment (if needed)
# conda activate vp_investments

# Run pipeline
python run_pipeline_and_push.py
```

**Monitor logs for:**
- ✅ Phase 5 extraction of market_cap and beta
- ✅ Successful INSERT with 22 parameters
- ❌ Any SQL errors related to missing columns

---

### Step 5: Verify Data Populated

**Query most recent run:**

```sql
-- Get latest run
SELECT id, run_timestamp 
FROM signal_runs 
ORDER BY run_timestamp DESC 
LIMIT 1;

-- Check if market_cap and beta populated (replace with actual run_id)
SELECT 
    ticker,
    company_name,
    sector,
    market_cap,
    beta,
    overall_score
FROM signals 
WHERE run_id = '<LATEST_RUN_ID>'
ORDER BY overall_score DESC
LIMIT 10;
```

**Expected:**
- ✅ `market_cap` populated for most tickers (some may be NULL)
- ✅ `beta` populated for most tickers (some may be NULL)
- ⚠️ NULL values acceptable - not all tickers have this data in YFinance

---

## Troubleshooting

### Issue 1: Columns already exist

**Symptom:**
```
ERROR: column "market_cap" of relation "signals" already exists
```

**Solution:**
Migration uses `ADD COLUMN IF NOT EXISTS` - this should not occur. If it does:
```sql
-- Verify columns exist
\d signals;

-- If they exist with correct types, migration already applied
```

---

### Issue 2: Pipeline fails with parameter count mismatch

**Symptom:**
```
ERROR: INSERT has more target columns than expressions
```

**Solution:**
1. Check Phase 5 code updated (20→22 params)
2. Verify migration 017 executed in Supabase
3. Restart pipeline

---

### Issue 3: market_cap/beta always NULL

**Symptom:**
All rows have NULL for market_cap and beta

**Possible causes:**
1. YFinance info not available for tickers
2. Phase 1 cache missing data
3. Phase 5 extraction logic issue

**Debug:**
```python
# Add to phase5_persist.py after line 1425
if market_cap or beta:
    self.logger.info(f"  [{ticker}] market_cap={market_cap}, beta={beta}")
```

---

## Validation Checklist

- [ ] Migration 017 executed successfully in Supabase
- [ ] Columns `market_cap` and `beta` exist in signals table
- [ ] Pipeline runs without SQL errors
- [ ] At least 50% of tickers have non-NULL market_cap
- [ ] At least 50% of tickers have non-NULL beta
- [ ] Phase 6 Assessment document reviewed (next step)

---

## Next Steps

After migration 017 completes:

1. **Test Phase 7 run-based analytics** (10 min)
   - Run pipeline
   - Verify analytics table populated with run_id
   - Check for SQL errors

2. **Frontend Implementation** (12 hours)
   - Move Performance Tab files
   - Build 7-horizon grid with MktCap/Beta header
   - Implement SPY/QQQ toggle

3. **Phase 7 Analytics Functions** (1-2 days)
   - Add 4 computation functions
   - Test with real pipeline run

---

**Status:** Ready for execution  
**Risk:** Low (column-only addition, no data loss)  
**Rollback:** `ALTER TABLE signals DROP COLUMN market_cap, DROP COLUMN beta;`
