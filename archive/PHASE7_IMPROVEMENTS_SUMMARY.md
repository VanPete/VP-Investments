# Phase 7 Analytics Improvements - Summary

## Issues Fixed:

### 1. ✅ Benchmark Metrics (alpha_vs_spy, beta_vs_spy, alpha_vs_qqq, beta_vs_qqq) Now Interval-Specific
**Problem**: All benchmark metrics were 0 because `_calculate_benchmark_metrics` was hardcoded to use `return_7d`

**Solution**: Updated method signature to accept `interval` parameter:
```python
def _calculate_benchmark_metrics(self, performance_data: List[Dict], benchmark: str, interval: str = '7d')
```
Now uses interval-specific returns: `return_{interval}` and `{benchmark}_return_{interval}`

**Calls Updated**:
- Line 309-310: Now pass interval to benchmark calculations
- Line 383-384: Same for old _calculate_all_metrics method

### 2. ✅ Group Performance Analytics Added
**New Feature**: Added `_calculate_group_performance(performance_data, interval)` method

**What it does**:
- Analyzes performance by each score group (technical, fundamental, news_macro, social_alternative, risk_stability, institutional)
- Segments each group by score quintiles (top 20%, Q2, Q3, Q4, bottom 20%)
- Calculates for each quintile:
  - `count`: Number of signals
  - `avg_return`: Average return for that quintile
  - `win_rate`: Percentage of winning trades
  - `sharpe`: Sharpe ratio
  - `max_drawdown`: Maximum drawdown

**Example Output**:
```json
{
  "technical": {
    "top_20pct": {"count": 10, "avg_return": 2.5, "win_rate": 60.0, "sharpe": 1.2, "max_drawdown": 5.0},
    "q2": {...},
    "q3": {...},
    "q4": {...},
    "bottom_20pct": {...}
  },
  "fundamental": {...},
  ...
}
```

**Integration**:
- Line 288: Added call in `_calculate_interval_analytics()`
- Stored in `metrics['group_performance']`
- Will be persisted to analytics table as JSONB

### 3. ❓ avg_social_alternative_score Issue
**Status**: Needs verification

**Findings**:
- Database shows actual scores: Average is -0.0897 (negative, not 0)
- The `_safe_avg()` function rounds to 2 decimal places: -0.09
- Analytics table query showed 0, but this might be a display issue
- Need to run pipeline again to confirm it's now storing -0.09 correctly

**Code Location**: Line 659 - correctly calculating from signals

## Next Steps:

### Run Pipeline to Test
```powershell
python run_pipeline_and_push.py
```

### Verify Improvements
```powershell
python scripts/check_analytics_full.py
```

**Expected Results**:
1. ✅ Benchmark metrics (alpha/beta) should have non-zero values for 1d, 3d intervals
2. ✅ group_performance JSONB field should be populated
3. ✅ avg_social_alternative_score should show -0.09 (not 0)

## Additional Recommendations:

### 4. Store Group Performance in Separate Table (Future)
Currently stored as JSONB in analytics table. Consider:
- Create `analytics_group_performance` table
- Columns: `analytics_id`, `group_name`, `quintile`, `count`, `avg_return`, `win_rate`, `sharpe`, `max_drawdown`
- Benefit: Easier querying, better for frontend charting

### 5. Add Sector Performance by Interval
Currently `_analyze_sectors()` uses a default interval. Could enhance to:
- Calculate sector performance for each interval
- Store in `analytics_sector_performance` table or JSONB column

### 6. Add Factor-Level Drill-Down API
For the 158 individual factors:
- Create `/api/analytics/factors?interval=1d&group=technical`
- Return performance metrics for all factors in that group
- On-demand calculation (not in main pipeline)

### 7. Optimize "all_time" Calculation
Currently aggregates returns from all 7 intervals. Consider:
- Use a weighted average based on signal age
- Or calculate from actual signal creation date to now

### 8. Add Confidence Intervals
For metrics with low sample sizes (N < 30):
- Add `confidence_level` field
- Display warning in frontend: "Low sample size, results may not be statistically significant"

### 9. Frontend Enhancements
- Add interval selector dropdown
- Add group performance comparison charts
- Add benchmark comparison view (VP vs SPY vs QQQ)
- Add score bucket performance heatmap

## Files Modified:
- `backend/phases/phase7_analytics.py`:
  - Line 309-310: Updated benchmark metric calls with interval
  - Line 288: Added group performance calculation
  - Line 903: Added new `_calculate_group_performance()` method
  - Line 1114: Renamed old method to `_calculate_group_performance_old()` (deprecated)
  - Line 1467: Updated `_calculate_benchmark_metrics()` signature with interval parameter
