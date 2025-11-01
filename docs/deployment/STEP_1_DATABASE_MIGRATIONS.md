# Step 1: Database Schema Updates - Implementation Guide

**Date**: October 31, 2025  
**Phase**: Database Migrations  
**Status**: Ready to Execute  
**Risk Level**: Low (additive only, backwards compatible)

---

## 📋 **Overview**

Step 1 adds the database foundation for Performance + Analytics features through **two safe, additive migrations**:

1. **Migration 015**: Extend `analytics` table with 20 new columns
2. **Migration 016**: Create `benchmarks` cache table

**Key Insight**: QQQ benchmark data already exists in `performance` table! No changes needed there.

---

## ✅ **Pre-Migration Checklist**

Before running migrations, verify:

- [ ] Database backup created (recommended, though migrations are non-destructive)
- [ ] No active pipeline runs (to avoid conflicts)
- [ ] Supabase SQL Editor access confirmed
- [ ] Migration files reviewed and understood

---

## 🗄️ **Migration 015: Extend Analytics Table**

### **What It Does**

Adds 20 new columns to `public.analytics` table:

**Link to Runs** (1 column):
- `run_id` (uuid, FK to signal_runs) - Links analytics to specific pipeline run
- Creates unique constraint to enable UPSERT pattern

**Predictive Strength** (6 columns):
- `ic_series` (jsonb) - Rolling RankIC time series
- `ic_mean`, `ic_std` (numeric) - RankIC statistics
- `hit_rate_top_decile` (numeric) - Top 10% hit rate
- `profit_factor` (numeric) - Gross profit / gross loss ratio
- `win_loss_ratio` (numeric) - Avg win / avg loss size

**Global Performance** (8 columns):
- `cagr`, `volatility` (numeric) - Return and risk metrics
- `sortino_ratio`, `calmar_ratio` (numeric) - Risk-adjusted returns
- `alpha_vs_spy`, `beta_vs_spy` (numeric) - SPY benchmarking
- `alpha_vs_qqq`, `beta_vs_qqq` (numeric) - QQQ benchmarking

**Backtest Enhancement** (2 columns):
- `rolling_sharpe_30d` (jsonb) - 30-day rolling Sharpe time series
- `benchmark_correlations` (jsonb) - Correlations with SPY/QQQ

**Signal Correlations** (3 columns):
- `signal_correlations` (jsonb) - 158×158 correlation matrix (~12,403 pairs)
- `top_positive_pairs` (jsonb) - Top 20 positive correlations
- `top_negative_pairs` (jsonb) - Top 20 negative correlations

### **Data Types & Formats**

**JSONB Structures**:
```json
// ic_series
[{"date":"2025-10-31", "ic":0.045}, ...]

// rolling_sharpe_30d
[{"date":"2025-10-31", "sharpe":1.23}, ...]

// benchmark_correlations
{"SPY": 0.68, "QQQ": 0.57}

// signal_correlations
[{"i":"RSI_14", "j":"MACD", "r":0.42, "n":1284}, ...]

// top_positive_pairs / top_negative_pairs
[{"i":"RSI_14", "j":"momentum_30d_pct", "r":0.87}, ...]
```

**Numeric Types**:
- All fractions stored as decimals (0.1234 = 12.34%)
- All ratios stored as-is (Sharpe=1.5, not 150%)

### **Impact Assessment**

**Storage**:
- Current row size: ~2 KB
- New columns add: ~100 KB per row (mostly signal_correlations)
- With UPSERT pattern: Only 1 row per run_id (efficient!)

**Performance**:
- No impact on existing queries (columns added, not modified)
- New indexes on `run_id` for fast lookups
- Foreign key maintains referential integrity

**Backwards Compatibility**:
- ✅ All columns are nullable (existing rows unaffected)
- ✅ No column renames or drops
- ✅ No data type changes
- ✅ All existing queries continue working

### **Verification**

Migration includes automated verification that:
1. All 20 columns were added successfully
2. Constraints created (run_id unique, foreign key)
3. Indexes created successfully

Run the verification block at the end of the migration to confirm success.

---

## 🗄️ **Migration 016: Create Benchmarks Cache Table**

### **What It Does**

Creates new `public.benchmarks` table to cache historical SPY/QQQ data:

**Columns**:
- `id` (uuid, PK) - Auto-generated
- `symbol` (varchar) - ETF symbol (SPY, QQQ)
- `date` (date) - Trading date
- `open`, `high`, `low`, `close` (numeric) - OHLCV data
- `volume` (bigint) - Shares traded
- `daily_return` (numeric) - Computed daily return
- `source` (varchar) - Data source tracking
- `created_at`, `updated_at` (timestamptz) - Audit fields

**Constraints**:
- `benchmarks_symbol_date_unique` - One row per symbol per date
- `benchmarks_symbol_uppercase` - Symbol must be uppercase
- `benchmarks_close_positive` - Close price > 0
- `benchmarks_date_not_future` - Date <= current date

**Indexes**:
- `idx_benchmarks_symbol_date` - Fast range queries
- `idx_benchmarks_date` - Fast date-based queries
- `idx_benchmarks_symbol` - Fast symbol lookups

**Helper Function**:
- `get_benchmark_data(symbol, start_date, end_date)` - Query with cache status

### **Why This Table?**

**Problem**: Phase 6 fetches SPY/QQQ from yfinance on every run
- Hits API rate limits
- Inconsistent data (API could change historical values)
- Slow (~2-3 seconds per fetch)

**Solution**: Cache benchmark data locally
- Fetch once, use many times
- Consistent historical data
- Query in milliseconds

### **Usage Pattern**

```python
# In Phase 6 (Performance):
async def fetch_benchmark_data(symbol: str, start_date: date, end_date: date):
    """
    Fetch benchmark data with caching.
    
    Priority:
    1. Check benchmarks table for cached data
    2. Identify missing dates
    3. Fetch missing dates from yfinance
    4. Cache new data in benchmarks table
    5. Return complete dataset
    """
    # Step 1: Query cache
    cached = await db.query(
        "SELECT * FROM get_benchmark_data($1, $2, $3)",
        symbol, start_date, end_date
    )
    
    # Step 2: Find missing dates
    missing_dates = [row['date'] for row in cached if not row['is_cached']]
    
    # Step 3: Fetch missing from yfinance
    if missing_dates:
        data = yfinance.download(symbol, start=min(missing_dates), end=max(missing_dates))
        
        # Step 4: Cache in database
        await db.insert_benchmark_data(symbol, data)
    
    # Step 5: Return complete dataset
    return await db.query("SELECT * FROM benchmarks WHERE symbol=$1 AND date BETWEEN $2 AND $3", 
                          symbol, start_date, end_date)
```

### **Impact Assessment**

**Storage**:
- ~1 KB per row (symbol + date + OHLCV)
- 252 trading days/year × 2 symbols = 504 rows/year
- 5 years = ~2,500 rows = **2.5 MB** (negligible)

**Performance**:
- Indexed queries: <1ms vs yfinance: ~2000ms
- **2000x faster** than API calls!

**Maintenance**:
- Auto-populated during pipeline runs
- No manual maintenance needed
- Can be re-fetched if data changes

### **Verification**

Migration includes automated verification that:
1. Table created successfully
2. All 3 indexes created
3. All 4 constraints created
4. Helper function available

---

## 🚀 **Execution Instructions**

### **Step 1: Open Supabase SQL Editor**

1. Navigate to: https://supabase.com/dashboard/project/YOUR_PROJECT/sql
2. Create new query tab

### **Step 2: Run Migration 015**

1. Copy contents of `migrations/015_extend_analytics_for_performance_tab.sql`
2. Paste into SQL Editor
3. Click "Run" (bottom right)
4. Wait for "Migration 015 SUCCESS" message
5. Verify no errors in output

**Expected Output**:
```
NOTICE:  Migration 015 SUCCESS: All 20 columns added to analytics table
```

### **Step 3: Run Migration 016**

1. Copy contents of `migrations/016_create_benchmarks_cache_table.sql`
2. Paste into SQL Editor
3. Click "Run"
4. Wait for "Migration 016 SUCCESS" message
5. Verify no errors in output

**Expected Output**:
```
NOTICE:  Migration 016 SUCCESS: benchmarks table created with 3 indexes and 4 constraints
```

### **Step 4: Verify Schema Changes**

Run this query to confirm all changes:

```sql
-- Check analytics table new columns
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'analytics' 
  AND column_name IN (
    'run_id', 'ic_series', 'ic_mean', 'ic_std', 
    'hit_rate_top_decile', 'profit_factor', 'win_loss_ratio',
    'cagr', 'volatility', 'sortino_ratio', 'calmar_ratio',
    'alpha_vs_spy', 'beta_vs_spy', 'alpha_vs_qqq', 'beta_vs_qqq',
    'rolling_sharpe_30d', 'benchmark_correlations',
    'signal_correlations', 'top_positive_pairs', 'top_negative_pairs'
  )
ORDER BY column_name;

-- Check benchmarks table exists
SELECT table_name, table_type 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name = 'benchmarks';

-- Check indexes on benchmarks
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE schemaname = 'public' 
  AND tablename = 'benchmarks';
```

**Expected Results**:
- 20 columns returned from analytics query
- 1 table returned from benchmarks query
- 3+ indexes returned from indexes query

---

## 📝 **Update supabase.sql**

After successful migration, update `supabase.sql` with new schema:

### **Add to analytics table definition**:
```sql
-- Add these columns after existing columns:
run_id uuid,
ic_series jsonb,
ic_mean numeric,
ic_std numeric,
hit_rate_top_decile numeric,
profit_factor numeric,
win_loss_ratio numeric,
cagr numeric,
volatility numeric,
sortino_ratio numeric,
calmar_ratio numeric,
alpha_vs_spy numeric,
beta_vs_spy numeric,
alpha_vs_qqq numeric,
beta_vs_qqq numeric,
rolling_sharpe_30d jsonb,
benchmark_correlations jsonb,
signal_correlations jsonb,
top_positive_pairs jsonb,
top_negative_pairs jsonb,

-- Add these constraints:
CONSTRAINT analytics_run_id_unique UNIQUE (run_id),
CONSTRAINT analytics_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.signal_runs(id) ON DELETE CASCADE
```

### **Add new benchmarks table**:
```sql
CREATE TABLE public.benchmarks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol varchar(10) NOT NULL,
  date date NOT NULL,
  open numeric,
  high numeric,
  low numeric,
  close numeric NOT NULL,
  volume bigint,
  daily_return numeric,
  source varchar(50) DEFAULT 'yfinance',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  CONSTRAINT benchmarks_symbol_date_unique UNIQUE (symbol, date),
  CONSTRAINT benchmarks_symbol_uppercase CHECK (symbol = UPPER(symbol)),
  CONSTRAINT benchmarks_close_positive CHECK (close > 0),
  CONSTRAINT benchmarks_date_not_future CHECK (date <= CURRENT_DATE)
);
```

---

## ⚠️ **Troubleshooting**

### **Error: "relation 'signal_runs' does not exist"**
- **Cause**: Running migration in wrong database
- **Fix**: Verify you're in the correct Supabase project

### **Error: "column 'run_id' already exists"**
- **Cause**: Migration already run
- **Fix**: Safe to ignore, `IF NOT EXISTS` prevents duplicates

### **Warning: "Expected 3 indexes, found 2"**
- **Cause**: Index creation partially failed
- **Fix**: Manually create missing index from migration file

### **Error: "permission denied for table analytics"**
- **Cause**: Insufficient privileges
- **Fix**: Use Supabase admin credentials (SQL Editor has admin by default)

---

## 🎯 **Success Criteria**

After Step 1 completion, verify:

- [ ] ✅ Migration 015 success message displayed
- [ ] ✅ Migration 016 success message displayed
- [ ] ✅ 20 new columns in `analytics` table
- [ ] ✅ `benchmarks` table exists with 3 indexes
- [ ] ✅ `run_id` unique constraint on analytics
- [ ] ✅ Foreign key relationship established
- [ ] ✅ `supabase.sql` updated with new schema
- [ ] ✅ No errors in Supabase logs
- [ ] ✅ Existing pipeline still runs (backwards compatible)

---

## 📊 **What's Next: Step 2**

After database migrations complete, Step 2 will:

1. **Update Phase 6** (Performance):
   - Use `benchmarks` table caching
   - Verify QQQ data population (already exists!)
   
2. **Update Phase 4** (Score Assemble):
   - Standardize `factor_contributions` output format
   - Add `{alpha_pct, vol_pct}` per group

3. **Update Phase 7** (Analytics):
   - Implement UPSERT pattern for analytics
   - Add signal correlation computation
   - Add RankIC calculation
   - Add predictive strength metrics

**Estimated Time**: 3-4 days of development

---

## 🔄 **Rollback Procedure** (Emergency Only)

If issues arise, rollback using commented SQL at end of each migration:

```sql
-- Rollback 016 first (no dependencies)
DROP FUNCTION IF EXISTS get_benchmark_data(varchar, date, date);
DROP TABLE IF EXISTS public.benchmarks CASCADE;

-- Then rollback 015
ALTER TABLE public.analytics
  DROP CONSTRAINT IF EXISTS analytics_run_id_unique,
  DROP CONSTRAINT IF EXISTS analytics_run_id_fkey,
  DROP COLUMN IF EXISTS run_id,
  -- ... (see migration file for complete rollback)
```

**Note**: Rollback is non-destructive (no data loss) since columns are additive.

---

## 📞 **Support**

If you encounter issues:
1. Check Supabase logs: Project Settings → Database → Logs
2. Verify migration syntax in SQL Editor
3. Review error messages carefully
4. Check constraints aren't conflicting with existing data

---

**Ready to execute?** Run the migrations and let me know the results! 🚀
