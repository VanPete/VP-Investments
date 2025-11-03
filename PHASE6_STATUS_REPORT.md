# Phase 6 Performance Tracking - Status Report
**Date:** November 3, 2025  
**Total Performance Records:** 1,235

## ✅ Summary

**Phase 6 IS working**, but only for **older signals**. Recent signals (last 2 days) are not being updated.

## 📊 Current State

### Status Distribution
- **In Progress:** 144 records (14.4%) - ✅ Being updated
- **Pending:** 856 records (85.6%) - ❌ Stuck, not updating

### Data Population
- **Records with data:** 144/1,000 (14.4%)
- **Records eligible for 1d:** 847 (need updates)
- **Update rate:** Only 17% of eligible signals have data

### Age Distribution
- **Oldest signal:** 5 days
- **Newest signal:** 0 days (fresh)
- **Average age:** 3.3 days

### Eligibility Breakdown
| Interval | Eligible Signals | Have Data | Population % |
|----------|------------------|-----------|--------------|
| 1d       | 847              | 144       | 17%          |
| 3d       | 696              | ~100      | ~14%         |
| 7d       | 0                | 0         | N/A          |
| 30d      | 0                | 0         | N/A          |

## 🔍 Root Cause Analysis

### Issue: Recent Signals Not Updating

The 856 `pending` signals (85.6%) are **NOT being processed** by Phase 6, even though:
- 847 signals are ≥1 day old (eligible for 1d interval)
- 696 signals are ≥3 days old (eligible for 3d interval)

### Possible Causes

1. **Phase 6 Not Running Frequently Enough**
   - Phase 6 only runs during pipeline execution
   - If pipeline runs once per day, old signals get updated but recent ones stay pending
   - **Solution:** Run pipeline more frequently OR add scheduled Phase 6 job

2. **Batch Limit Too Small**
   - Current limit: 200 records per run (line 200 in phase6_performance.py)
   - With 856 pending signals, it takes 5 runs to catch up
   - **Solution:** Increase limit to 500-1000 OR run Phase 6 multiple times

3. **Query Order Issue**
   - Phase 6 queries `order('created_at', desc=False)` (oldest first)
   - This means it prioritizes OLD signals over RECENT signals
   - Recent signals (last 1-2 days) might not get processed if batch is full
   - **Solution:** Change order to prioritize signals by eligibility age

4. **Silent Failures**
   - yfinance API errors might be failing silently
   - Check logs for: "Failed to update performance record"
   - **Solution:** Review pipeline logs for error messages

## 🎯 Recommendations

### Immediate Actions

1. **Run Pipeline Again** - This will process next 200 pending signals
   ```bash
   python run_pipeline_and_push.py
   ```

2. **Check Pipeline Logs** - Look for Phase 6 errors:
   ```
   grep "Phase 6" logs/*.log
   grep "Failed to update" logs/*.log
   ```

### Short-term Fixes

3. **Increase Batch Size** (in `pipeline.py` line 316):
   ```python
   perf_stats = await p6_tracker.update_pending_performance(
       limit=500,  # Increased from 200
       benchmark_cache=benchmark_cache
   )
   ```

4. **Add Scheduled Phase 6 Job** - Run Phase 6 independently every hour:
   ```python
   # New script: scripts/run_phase6_only.py
   updater = PerformanceUpdater()
   await updater.update_pending_performance(limit=1000)
   ```

### Long-term Solutions

5. **Change Query Priority** - Update Phase 6 to process by eligibility:
   ```python
   # Sort by age (oldest baseline_date first)
   .order('baseline_date', desc=False)  # Instead of created_at
   ```

6. **Add Progress Monitoring** - Track Phase 6 performance:
   - Dashboard showing: pending vs in_progress vs completed
   - Alert when pending > 500 for more than 1 day

## 📈 Expected Behavior

**Ideal State:**
- ✓ All signals ≥1 day old should have 1d data (100% population)
- ✓ All signals ≥3 days old should have 1d+3d data (100% population)
- ✓ Status should transition: `pending` → `in_progress` → `completed`

**Current State:**
- ⚠️  Only 17% of eligible signals have 1d data
- ⚠️  856 signals stuck in `pending` (should be `in_progress`)
- ⚠️  Phase 6 is bottlenecked by batch size and run frequency

## 🔧 Next Steps

1. **Verify Issue Scope:**
   ```bash
   python scripts/verify_phase6_signals.py  # Already run
   python scripts/quick_phase6_check.py     # Already run
   ```

2. **Run Phase 6 Manually to Catch Up:**
   ```bash
   python scripts/archive/test_phase6_manual.py
   ```

3. **Increase Batch Size and Re-run Pipeline:**
   - Edit `backend/pipeline.py` line 316: `limit=500`
   - Run: `python run_pipeline_and_push.py`

4. **Monitor Results:**
   ```bash
   python scripts/quick_phase6_check.py  # Check if pending count decreased
   ```

---

**Status:** Phase 6 is functional but **underperforming**. Needs batch size increase or more frequent runs to keep up with signal generation rate.
