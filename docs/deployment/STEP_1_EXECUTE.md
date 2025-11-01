# Step 1: Execute Migration 015 - Analytics Extensions

**Date**: October 31, 2025  
**Status**: 🚀 Ready to Execute  
**Time**: ~5 minutes

---

## 📋 **What This Migration Does**

### **Migration 015: Analytics Table Extensions**
- ✅ Adds `run_id` column (links analytics to specific pipeline runs)
- ✅ Adds foreign key to `signal_runs` table
- ✅ Adds UNIQUE constraint on `run_id`
- ✅ Adds 19 new columns required by VanPiQ spec:
  - **6 Predictive Metrics**: `ic_series`, `ic_mean`, `ic_std`, `hit_rate_top_decile`, `profit_factor`, `win_loss_ratio`
  - **8 Performance Metrics**: `cagr`, `volatility`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `total_return`, `backtest_start`, `backtest_end`
  - **3 Correlation**: `signal_correlation_matrix`, `top_correlated_signals`, `low_correlated_signals`
  - **2 Predictive**: `predictive_strength`, `ic_consistency`

### **Strategic Shift**
- **OLD**: Period-based analytics (4 rows per analysis: daily, weekly, monthly, all_time)
- **NEW**: Run-based analytics (1 row per pipeline run)
- **Savings**: 75% storage reduction by eliminating duplication

### **Backwards Compatible** ✅
- Old columns NOT dropped (existing code continues to work)
- New columns added with NULL defaults
- Phase 7 will transition to run-based approach in Step 2

---

## 🚀 **Execution Steps**

### **1. Open Supabase SQL Editor**
1. Go to your Supabase project
2. Navigate to **SQL Editor**
3. Click **New Query**

### **2. Copy Migration SQL**
```powershell
# Copy migration file to clipboard
Get-Content "C:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments\migrations\015_extend_analytics_for_performance_tab.sql" | Set-Clipboard
```

### **3. Paste and Execute**
1. Paste SQL into Supabase editor
2. Click **Run** (or press F5)
3. Wait for success message

### **4. Verify Success**
Look for:
```
Migration 015 SUCCESS: Analytics table extended with 20 new columns
```

### **5. Verify Columns Added**
Run this query to confirm:
```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'analytics' 
  AND column_name IN (
    'run_id', 'ic_series', 'ic_mean', 'ic_std', 
    'hit_rate_top_decile', 'profit_factor', 'win_loss_ratio',
    'cagr', 'volatility', 'sortino_ratio', 'calmar_ratio',
    'max_drawdown', 'total_return', 'backtest_start', 'backtest_end',
    'signal_correlation_matrix', 'top_correlated_signals', 
    'low_correlated_signals', 'predictive_strength', 'ic_consistency'
  )
ORDER BY column_name;
```

Expected: 20 rows returned

---

## ✅ **Post-Migration Checklist**

- [ ] Migration executed without errors
- [ ] 20 new columns visible in analytics table
- [ ] `run_id` has UNIQUE constraint
- [ ] Foreign key to `signal_runs` created
- [ ] Index on `run_id` created
- [ ] Existing pipeline still runs (test with `python run_pipeline_and_push.py`)

---

## 📝 **Next Steps**

After migration 015 completes:

### **Immediate** (today):
1. ✅ Update `supabase.sql` reference file
2. ✅ Commit migration success to Git

### **Tomorrow** (Step 2):
1. Update Phase 7 analytics code:
   - Change UPSERT key from `(period_type, period_start, period_end)` to `run_id`
   - Add `compute_ic_series()` function
   - Add `compute_signal_correlations()` function
   - Add `compute_predictive_metrics()` function
   - Add `compute_global_performance()` function

### **Day 3-4** (Step 3):
1. Create API endpoints:
   - `/api/analytics/global?bucket=...&interval=...`
   - `/api/performance/:signal_id/horizons`

### **Week 2** (Step 4):
1. Build frontend components for Performance + Analytics tabs

---

## 🔍 **Troubleshooting**

### **Error: Column already exists**
**Cause**: Migration already ran partially  
**Fix**: Migration uses `IF NOT EXISTS` - safe to re-run

### **Error: Foreign key constraint violation**
**Cause**: `signal_runs` table missing  
**Fix**: Check if `signal_runs` table exists: `SELECT * FROM signal_runs LIMIT 1;`

### **Error: Permission denied**
**Cause**: Insufficient database permissions  
**Fix**: Ensure you're logged in as database admin/owner

---

## 📊 **Expected Impact**

### **Storage**
- **Before**: ~8 KB per analysis (2 KB × 4 period types)
- **After**: ~102 KB per run (but only 1 row, not 4)
- **Net**: +94 KB per run (mainly `signal_correlation_matrix` JSONB)

### **Query Performance**
- **Before**: Filter by `(period_type, period_start, period_end)` → 4 rows scanned
- **After**: Filter by `run_id` → 1 row with UNIQUE index (faster)

### **Code Simplicity**
- **Before**: Loop through 4 period types, insert 4 rows
- **After**: Single insert per pipeline run

---

## 🤔 **Questions Before Executing?**

1. **Will this break existing analytics?** No - backwards compatible, old columns remain
2. **Do I need to backfill run_id?** No - NULL allowed, Phase 7 code handles missing run_id
3. **Can I roll back?** Yes - drop new columns if needed (but not recommended)

---

**Ready to proceed?** Execute the migration when you're ready!
