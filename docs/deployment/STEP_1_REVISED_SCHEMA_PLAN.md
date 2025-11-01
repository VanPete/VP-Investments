# Step 1: Revised Database Schema Plan (Simplified)

**Date**: October 31, 2025  
**Status**: 🔄 REVISED - Streamlined Approach (Migration 016 Removed)  
**Focus**: Remove bloat, add only what spec requires, use existing cache

---

## 🎯 **Key Changes from Original Plan**

### **Problems Identified** ❌

1. **Analytics table has period-based bloat**: `period_start`, `period_end`, `period_type` create unnecessary duplication
   - Currently: Multiple rows per date range (daily, weekly, monthly, all_time)
   - Problem: Same data repeated 4x with different period types
   - Bloat: 56 horizon columns × 4 period types = **224 nearly-identical columns**

2. **Many columns are never populated**: Current schema has placeholders never written to
   - `win_rate_*` columns (7 horizons) - **not computed**
   - `sharpe_ratio_*` columns (7 horizons) - **not computed**
   - `max_drawdown_*` columns (7 horizons) - **not computed**
   - `avg_return_*`, `avg_alpha_*` columns (7 horizons each) - **partially used**

3. **Benchmark data already cached**: Phase 1 fetches SPY/QQQ/sector ETFs into memory
   - No separate DB table needed (migration 016 removed)
   - Performance table already stores SPY/QQQ returns (7 horizons)

4. **Performance tab data already exists**: QQQ columns already in `performance` table!

### **Solution** ✅

1. **Restructure analytics**: One row per `run_id` (not per period)
2. **Remove bloat**: Drop or ignore period-based duplication
3. **Add only spec requirements**: 20 new columns for Performance + Analytics features
4. **Use existing benchmark cache**: Phase 1 memory cache + performance table (no new DB table)

---

## 📊 **Current Analytics Table Analysis**

### **Bloat Columns** (56 columns, mostly empty)

```sql
-- Period-based keys (creates 4x duplication)
period_start, period_end, period_type

-- Win rates (7 horizons) - NOT POPULATED
win_rate_1d, win_rate_3d, win_rate_7d, win_rate_10d, 
win_rate_14d, win_rate_30d, win_rate_90d

-- Sharpe ratios (7 horizons) - NOT POPULATED  
sharpe_ratio_1d, sharpe_ratio_3d, sharpe_ratio_7d, sharpe_ratio_10d,
sharpe_ratio_14d, sharpe_ratio_30d, sharpe_ratio_90d

-- Max drawdowns (7 horizons) - NOT POPULATED
max_drawdown_1d, max_drawdown_3d, max_drawdown_7d, max_drawdown_10d,
max_drawdown_14d, max_drawdown_30d, max_drawdown_90d

-- Average returns (7 horizons) - PARTIALLY USED
avg_return_1d, avg_return_3d, avg_return_7d, avg_return_10d,
avg_return_14d, avg_return_30d, avg_return_90d

-- Average alphas (7 horizons) - PARTIALLY USED
avg_alpha_1d, avg_alpha_3d, avg_alpha_7d, avg_alpha_10d,
avg_alpha_14d, avg_alpha_30d, avg_alpha_90d

-- Sector stats (6 columns) - USED
top_sector, top_sector_avg_return, top_sector_count,
worst_sector, worst_sector_avg_return, worst_sector_count

-- Group scores (6 columns) - USED
avg_technical_score, avg_fundamental_score, avg_news_macro_score,
avg_social_alternative_score, avg_risk_stability_score, avg_institutional_score
```

### **Actually Used Columns** (14 columns)

```sql
-- Identifiers & metadata
id, created_at, updated_at
total_signals, signals_analyzed, performance_records_used
avg_overall_score

-- Sector analysis
sector_performance (jsonb)
top_sector, top_sector_avg_return, top_sector_count
worst_sector, worst_sector_avg_return, worst_sector_count

-- Group analysis  
avg_technical_score, avg_fundamental_score, avg_news_macro_score,
avg_social_alternative_score, avg_risk_stability_score, avg_institutional_score

-- Advanced metrics (JSONB)
score_bucket_performance (jsonb)
factor_correlations (jsonb)
factor_contributions (jsonb)
group_performance (jsonb)
backtest_cumulative_returns (jsonb)
top_factors (jsonb)
```

---

## 🔧 **Revised Analytics Table Schema**

### **Strategy**: Simplify + Add Spec Requirements

**Remove/Ignore**:
- ❌ Drop `period_type`, `period_start`, `period_end` (no UPSERT conflicts)
- ❌ Ignore 49 empty horizon columns (win_rate_*, sharpe_ratio_*, max_drawdown_*, avg_return_*, avg_alpha_*)

**Keep**:
- ✅ Core metadata (id, created_at, updated_at)
- ✅ Run tracking (total_signals, signals_analyzed, performance_records_used, avg_overall_score)
- ✅ Sector analysis (6 columns + sector_performance jsonb)
- ✅ Group scores (6 columns for avg scores)
- ✅ Advanced metrics (5 jsonb columns)

**Add (Per Spec)**:
- ✅ `run_id` (uuid, UNIQUE) - Link to signal_runs
- ✅ Predictive strength (6 columns): `ic_series`, `ic_mean`, `ic_std`, `hit_rate_top_decile`, `profit_factor`, `win_loss_ratio`
- ✅ Global performance (8 columns): `cagr`, `volatility`, `sortino_ratio`, `calmar_ratio`, `alpha_vs_spy`, `beta_vs_spy`, `alpha_vs_qqq`, `beta_vs_qqq`
- ✅ Backtest extras (2 columns): `rolling_sharpe_30d`, `benchmark_correlations`
- ✅ Signal correlations (3 columns): `signal_correlations`, `top_positive_pairs`, `top_negative_pairs`

---

## 📋 **Migration Plan**

### **Migration 015: Streamline Analytics Table**

**Goal**: Add spec-required columns, link to runs, ignore bloat

```sql
-- Add run_id for run-based analytics (replaces period_* approach)
ALTER TABLE public.analytics
  ADD COLUMN IF NOT EXISTS run_id uuid,
  ADD CONSTRAINT analytics_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.signal_runs(id) ON DELETE CASCADE,
  ADD CONSTRAINT analytics_run_id_unique UNIQUE (run_id);

-- Predictive Strength (6 columns)
ALTER TABLE public.analytics
  ADD COLUMN IF NOT EXISTS ic_series jsonb,
  ADD COLUMN IF NOT EXISTS ic_mean numeric,
  ADD COLUMN IF NOT EXISTS ic_std numeric,
  ADD COLUMN IF NOT EXISTS hit_rate_top_decile numeric,
  ADD COLUMN IF NOT EXISTS profit_factor numeric,
  ADD COLUMN IF NOT EXISTS win_loss_ratio numeric;

-- Global Performance Summary (8 columns)
ALTER TABLE public.analytics
  ADD COLUMN IF NOT EXISTS cagr numeric,
  ADD COLUMN IF NOT EXISTS volatility numeric,
  ADD COLUMN IF NOT EXISTS sortino_ratio numeric,
  ADD COLUMN IF NOT EXISTS calmar_ratio numeric,
  ADD COLUMN IF NOT EXISTS alpha_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS alpha_vs_qqq numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_qqq numeric;

-- Backtest Extras (2 columns)
ALTER TABLE public.analytics
  ADD COLUMN IF NOT EXISTS rolling_sharpe_30d jsonb,
  ADD COLUMN IF NOT EXISTS benchmark_correlations jsonb;

-- Signal Correlations (3 columns)
ALTER TABLE public.analytics
  ADD COLUMN IF NOT EXISTS signal_correlations jsonb,
  ADD COLUMN IF NOT EXISTS top_positive_pairs jsonb,
  ADD COLUMN IF NOT EXISTS top_negative_pairs jsonb;
```

**Total**: 1 + 6 + 8 + 2 + 3 = **20 new columns** (same as before, but simpler approach)

**Note**: We're NOT dropping old columns (backwards compatible), just adding new ones and ignoring bloat.

---

## 🎯 **Implementation Impact**

### **Analytics Table**

**Before** (current):
- 70 columns total
- 56 columns are bloat/unused
- Period-based duplication (4 rows per analysis)
- No run linkage

**After** (migration 015):
- 90 columns total (+20 new)
- 56 columns still exist (ignored, backwards compatible)
- 34 columns actively used
- Run-based (1 row per pipeline run)
- Linked to signal_runs via `run_id`

**Storage Impact**:
- Current: ~2 KB per row × 4 period types = 8 KB per analysis
- New: ~102 KB per row × 1 run = 102 KB per analysis
- **BUT**: Only 1 row per run (not 4), so net = 94 KB more per run
- Main growth: signal_correlations JSONB (~99 KB of the 102 KB)
- **Net Effect**: ~94 KB extra per run, but eliminates 6 KB of period duplication = +88 KB/run

**Note**: Benchmarks (SPY/QQQ/sector ETFs) use existing Phase 1 memory cache + performance table columns. No separate DB table needed.

---

## 🔄 **Phase 7 Analytics Updates Required**

### **Current Code** (phase7_analytics.py line 475)

```python
# WRONG: Uses period_type for UPSERT key
ON CONFLICT (period_type, period_start, period_end)
```

### **New Code** (required change)

```python
# CORRECT: Use run_id for UPSERT key
ON CONFLICT (run_id)
DO UPDATE SET
  # Update only changed columns
  ...
```

### **Required Changes**:

1. **Add `run_id` to INSERT statement**
   - Get run_id from pipeline context
   - Pass to _persist_analytics method

2. **Remove period_* logic**
   - No more `_calculate_period_ranges`
   - No more period_type selection

3. **Add new metrics computation**:
   - `compute_ic_series()` - Rolling RankIC
   - `compute_predictive_metrics()` - Hit rate, profit factor
   - `compute_global_performance()` - CAGR, Sortino, Calmar
   - `compute_signal_correlations()` - 158×158 matrix
   - `compute_benchmark_correlations()` - SPY/QQQ correlation

4. **Standardize factor_contributions format**:
   ```python
   {
     "technical": {"alpha_pct": 0.32, "vol_pct": 0.18},
     # ... other groups
   }
   ```

---

## 🚀 **Step 1 Execution Plan**

### **Phase A: Run Migrations** (10 minutes)

1. **Backup current analytics data** (optional but recommended)
   ```sql
   CREATE TABLE analytics_backup AS SELECT * FROM analytics;
   ```

2. **Run Migration 015** - Add columns to analytics
   - Open Supabase SQL Editor
   - Copy/paste migration SQL from `migrations/015_extend_analytics_for_performance_tab.sql`
   - Execute and verify success message
   - Verify 20 new columns added to analytics table

---

### **Phase B: Update Phase 7 Analytics** (Step 2, ~2 days)

1. **Simplify _persist_analytics method**:
   - Remove period_type/start/end logic
   - Add run_id parameter
   - Change UPSERT key to `ON CONFLICT (run_id)`
   - Remove 49 bloat column inserts

2. **Add new metric computations**:
   - `compute_ic_series()` using historical signals + returns
   - `compute_signal_correlations()` from factor_to_group.yaml
   - `compute_predictive_strength()` for hit rate, profit factor
   - `compute_global_performance()` for CAGR, Sortino, Calmar

3. **Update factor_contributions format**:
   - Ensure `{alpha_pct, vol_pct}` per group
   - Normalize to 0..1 fractions

---

### **Phase C: Verify Phase 6 Performance** (Step 2, ~30 min)

1. **Verify benchmark cache is working**:
   - Phase 1 already fetches SPY/QQQ/sector ETFs into memory ✅
   - Phase 6 already reuses cached data ✅
   - No code changes needed (already implemented in v3.3)

2. **Verify QQQ columns population**:
   - QQQ columns already exist in performance table schema ✅
   - Verify Phase 6 populates `qqq_return_*` and `qqq_alpha_*`
   - Run one pipeline and check performance table

---

## ✅ **Success Criteria**

After Step 1 migration:

- [ ] `analytics` table has 20 new columns
- [ ] `run_id` column added with UNIQUE constraint and foreign key
- [ ] Index on `run_id` created
- [ ] No errors in Supabase logs
- [ ] `supabase.sql` updated with new schema
- [ ] Existing pipeline still runs (backwards compatible)

---

## 📝 **Updated supabase.sql Schema**

Will update after migration 015 completes to reflect:
1. Analytics table with 20 new columns + run_id
2. Foreign key constraint to signal_runs
3. Index on run_id

---

## 🤔 **Next Steps**

**Today**: 
1. ✅ Review this plan
2. ✅ Approve simplified approach (no migration 016)
3. Run migration 015 in Supabase

**Tomorrow (Step 2)**:
1. Update Phase 7 analytics code
2. Add new metric computations
3. Test with one pipeline run

**Day 3-4 (Step 3)**:
1. Create `/api/analytics/global` endpoint
2. Create `/api/performance/:id/horizons` endpoint
3. Test API responses

---

**Questions?** Review the plan and let me know if this streamlined approach looks good!
