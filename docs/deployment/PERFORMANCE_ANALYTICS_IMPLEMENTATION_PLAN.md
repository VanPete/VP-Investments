# VanPiQ Performance + Analytics - Implementation Plan

**Version:** 1.0  
**Date:** October 31, 2025  
**Strategy:** Incremental, non-breaking, data-efficient  

---

## 📋 **Executive Summary**

Based on the comprehensive spec review and clarifications, here's the refined implementation strategy:

### **Key Decisions Made**

1. **Signal Count**: Dynamic based on `factor_to_group.yaml` (currently **158 factors**)
   - **Approach**: Precompute on each analytics run, cache in DB as JSONB
   - **Update trigger**: Runs automatically with each pipeline execution

2. **Timeline Consistency**: All metrics use **7 horizons**: `1d, 3d, 7d, 10d, 14d, 30d, 90d`
   - **Performance Tab**: Per-ticker horizon grid with returns + alpha
   - **Analytics Tab**: Rolling window calculations based on selected interval

3. **Benchmark Data**: SPY/QQQ fetched via yfinance and stored
   - **Storage**: New `benchmarks` table OR extend `performance` table
   - **Frequency**: Updated during Phase 6 (Performance) for each run

4. **Analytics Strategy**: **UPSERT (not append)** - single row per run_id
   - **Benefit**: Reduces data volume by ~90%, enables efficient historical queries
   - **Schema**: Add `run_id` as unique key to `analytics` table

5. **Factor Contributions**: Standardized `{alpha_pct, vol_pct}` per group (Phase 4 output)

---

## 🗄️ **Database Architecture**

### **Current State Analysis**

**Existing Tables:**
- ✅ `signal_runs` - Pipeline run metadata
- ✅ `signals` - Per-ticker signal data (one row per ticker per run)
- ✅ `performance` - Per-ticker horizon returns (1d/3d/7d/10d/14d/30d/90d)
- ✅ `analytics` - Global aggregates (needs extension)

**Current `performance` Table Columns:**
```
id, signal_id, baseline_price, baseline_date,
sector, sector_etf, market_cap,
return_1d, return_3d, return_7d, return_10d, return_14d, return_30d, return_90d,
alpha_1d, alpha_3d, alpha_7d, alpha_10d, alpha_14d, alpha_30d, alpha_90d,
spy_return_1d, spy_return_3d, spy_return_7d, spy_return_10d, spy_return_14d, spy_return_30d, spy_return_90d,
status, last_update, intervals_completed
```

**Current `analytics` Table Columns:**
```
id, run_id, run_timestamp, created_at,
sharpe_ratio, total_return, win_rate,
score_bucket_performance, factor_correlations, factor_contributions,
group_performance, backtest_cumulative_returns
```

---

## 🔧 **Phase 1: Database Extensions**

### **1.1 Add QQQ Benchmark Returns to `performance`**

**Rationale**: Currently only SPY returns are stored. Need QQQ for dual-benchmark comparison.

```sql
-- Migration 015: Add QQQ benchmark returns
ALTER TABLE public.performance
  ADD COLUMN IF NOT EXISTS qqq_return_1d numeric,
  ADD COLUMN IF NOT EXISTS qqq_return_3d numeric,
  ADD COLUMN IF NOT EXISTS qqq_return_7d numeric,
  ADD COLUMN IF NOT EXISTS qqq_return_10d numeric,
  ADD COLUMN IF NOT EXISTS qqq_return_14d numeric,
  ADD COLUMN IF NOT EXISTS qqq_return_30d numeric,
  ADD COLUMN IF NOT EXISTS qqq_return_90d numeric,
  
  ADD COLUMN IF NOT EXISTS alpha_qqq_1d numeric,
  ADD COLUMN IF NOT EXISTS alpha_qqq_3d numeric,
  ADD COLUMN IF NOT EXISTS alpha_qqq_7d numeric,
  ADD COLUMN IF NOT EXISTS alpha_qqq_10d numeric,
  ADD COLUMN IF NOT EXISTS alpha_qqq_14d numeric,
  ADD COLUMN IF NOT EXISTS alpha_qqq_30d numeric,
  ADD COLUMN IF NOT EXISTS alpha_qqq_90d numeric
;

COMMENT ON COLUMN public.performance.qqq_return_1d IS 'QQQ 1-day return for alpha calculation';
COMMENT ON COLUMN public.performance.alpha_qqq_1d IS 'Alpha vs QQQ: return_1d - qqq_return_1d';
```

**Impact**: 
- **Storage**: +14 columns × 8 bytes × 100 tickers/run × 90 days = **~1 MB** (negligible)
- **Query**: No impact (indexed on `signal_id`)
- **Computation**: yfinance fetch QQQ during Phase 6 (same pattern as SPY)

---

### **1.2 Extend `analytics` Table for New Metrics**

**Rationale**: Add columns from spec while keeping UPSERT strategy (one row per run_id).

```sql
-- Migration 016: Extend analytics table for Performance + Analytics features
ALTER TABLE public.analytics
  -- Make run_id unique to enable UPSERT pattern
  ADD CONSTRAINT IF NOT EXISTS analytics_run_id_unique UNIQUE (run_id),
  
  -- Predictive Strength
  ADD COLUMN IF NOT EXISTS ic_series jsonb,
  ADD COLUMN IF NOT EXISTS ic_mean numeric,
  ADD COLUMN IF NOT EXISTS ic_std numeric,
  ADD COLUMN IF NOT EXISTS hit_rate_top_decile numeric,
  ADD COLUMN IF NOT EXISTS profit_factor numeric,
  ADD COLUMN IF NOT EXISTS win_loss_ratio numeric,
  
  -- Global Performance Summary (already have sharpe_ratio)
  ADD COLUMN IF NOT EXISTS cagr numeric,
  ADD COLUMN IF NOT EXISTS volatility numeric,
  ADD COLUMN IF NOT EXISTS sortino_ratio numeric,
  ADD COLUMN IF NOT EXISTS calmar_ratio numeric,
  ADD COLUMN IF NOT EXISTS alpha_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS alpha_vs_qqq numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_qqq numeric,
  
  -- Backtest Extras
  ADD COLUMN IF NOT EXISTS rolling_sharpe_30d jsonb,
  ADD COLUMN IF NOT EXISTS benchmark_correlations jsonb,
  
  -- Signal-Level Correlations (158×158 matrix)
  ADD COLUMN IF NOT EXISTS signal_correlations jsonb,
  ADD COLUMN IF NOT EXISTS top_positive_pairs jsonb,
  ADD COLUMN IF NOT EXISTS top_negative_pairs jsonb
;

-- Add comments for documentation
COMMENT ON COLUMN public.analytics.ic_series IS '[{"date":"YYYY-MM-DD", "ic":<numeric>}, ...] - Rolling RankIC time series';
COMMENT ON COLUMN public.analytics.signal_correlations IS '[{"i":"RSI_14", "j":"MACD", "r":0.42, "n":1284}, ...] - Upper triangle of 158×158 correlation matrix';
COMMENT ON COLUMN public.analytics.top_positive_pairs IS '[{"i":"...", "j":"...", "r":...}, ...] - Top 20 positively correlated signal pairs';
COMMENT ON COLUMN public.analytics.top_negative_pairs IS '[{"i":"...", "j":"...", "r":...}, ...] - Top 20 negatively correlated signal pairs';
COMMENT ON COLUMN public.analytics.factor_contributions IS '{"technical":{"alpha_pct":0.32,"vol_pct":0.18}, ...} - Normalized group contributions';
```

**Impact**:
- **Storage per row**: ~100 KB (signal_correlations = 12,403 pairs × 8 bytes = 99 KB)
- **Total for 90 days**: 90 rows × 100 KB = **9 MB** (very manageable)
- **UPSERT benefit**: Without UPSERT, this would be 900 MB for same period!

---

### **1.3 Create `benchmarks` Table (Optional but Recommended)**

**Rationale**: Cache SPY/QQQ historical data to avoid repeated yfinance fetches.

```sql
-- Migration 017: Create benchmarks table for SPY/QQQ historical data
CREATE TABLE IF NOT EXISTS public.benchmarks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol varchar(10) NOT NULL,
  date date NOT NULL,
  open numeric,
  high numeric,
  low numeric,
  close numeric NOT NULL,
  volume bigint,
  created_at timestamptz DEFAULT now(),
  
  CONSTRAINT benchmarks_symbol_date_unique UNIQUE (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_symbol_date ON public.benchmarks (symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_benchmarks_date ON public.benchmarks (date DESC);

COMMENT ON TABLE public.benchmarks IS 'Historical OHLCV data for benchmark ETFs (SPY, QQQ)';
```

**Benefits**:
- **Avoid rate limits**: No repeated yfinance calls for same dates
- **Consistency**: All runs use same benchmark data
- **Performance**: Indexed queries much faster than API calls
- **Storage**: ~1 KB/day × 2 symbols × 365 days = **730 KB/year** (tiny)

**Usage Pattern**:
1. Phase 6 checks `benchmarks` table first
2. If data exists for required date range → use cached
3. If missing → fetch from yfinance, cache, then use

---

## 🔄 **Phase 2: Pipeline Integration**

### **2.1 Update Phase 6 (Performance) to Fetch QQQ**

**File**: `backend/phases/phase6_performance.py`

**Changes**:
1. Fetch both SPY and QQQ from yfinance (or `benchmarks` table)
2. Calculate QQQ returns for all horizons
3. Calculate alpha vs QQQ (`alpha_qqq_Xd = return_Xd - qqq_return_Xd`)
4. Store in `performance` table with new columns

**New Function**:
```python
async def fetch_benchmark_data(
    self, 
    symbol: str, 
    start_date: date, 
    end_date: date
) -> pd.DataFrame:
    """
    Fetch benchmark data with caching.
    
    Priority:
    1. Check benchmarks table first
    2. Fetch missing dates from yfinance
    3. Cache new data in benchmarks table
    4. Return complete DataFrame
    """
```

**Estimated LOC**: +150 lines  
**Risk**: Low (additive only, existing SPY logic remains unchanged)

---

### **2.2 Update Phase 4 (Score Assemble) Output Schema**

**File**: `backend/phases/phase4_score_assemble.py`

**Changes**:
Ensure `factor_contributions` output matches spec schema:
```python
{
  "technical": {"alpha_pct": 0.32, "vol_pct": 0.18},
  "fundamental": {"alpha_pct": 0.21, "vol_pct": 0.22},
  "news_macro": {"alpha_pct": 0.14, "vol_pct": 0.12},
  "social_alternative": {"alpha_pct": 0.09, "vol_pct": 0.15},
  "risk_stability": {"alpha_pct": 0.16, "vol_pct": 0.20},
  "institutional_smart_money": {"alpha_pct": 0.08, "vol_pct": 0.13}
}
```

**Calculation**:
- `alpha_pct`: Group's contribution to overall score variance (normalized to 0..1)
- `vol_pct`: Group's contribution to score volatility (normalized to 0..1)

**Estimated LOC**: +50 lines (refactor existing contribution calculation)  
**Risk**: Medium (changes existing logic, requires validation)

---

### **2.3 Create Phase 7 (Analytics) UPSERT Logic**

**File**: `backend/phases/phase7_analytics.py`

**Changes**:
1. **Replace INSERT with UPSERT** pattern:
   ```python
   result = self.db.client.table('analytics').upsert(
       analytics_data,
       on_conflict='run_id'  # Update if run_id exists
   ).execute()
   ```

2. **Add Signal Correlation Computation**:
   ```python
   async def compute_signal_correlations(
       self, 
       run_id: str
   ) -> Dict[str, Any]:
       """
       Compute 158×158 signal correlation matrix.
       
       Returns:
       {
         "signal_correlations": [{"i":"RSI_14", "j":"MACD", "r":0.42, "n":1284}, ...],
         "top_positive_pairs": [{"i":"...", "j":"...", "r":...}, ...],
         "top_negative_pairs": [{"i":"...", "j":"...", "r":...}, ...]
       }
       """
   ```

3. **Dynamic Factor Count** from `factor_to_group.yaml`:
   ```python
   def get_all_factors(self) -> List[str]:
       """Load factor list from factor_to_group.yaml"""
       with open('config/factor_to_group.yaml') as f:
           config = yaml.safe_load(f)
       
       factors = []
       for group in ['technical', 'fundamental', 'news_macro', 
                     'social_alternative', 'risk_stability', 
                     'institutional_smart_money']:
           factors.extend(config[group].keys())
       
       return factors  # Currently 158, auto-updates when YAML changes
   ```

**Estimated LOC**: +300 lines  
**Risk**: High (complex correlations, performance-sensitive)

---

## 🖥️ **Phase 3: Backend API**

### **3.1 Create `/api/analytics/global` Endpoint**

**File**: `backend/api/analytics.py` (new)

**Endpoint**:
```python
@router.get("/analytics/global")
async def get_global_analytics(
    bucket: str = "all",  # all, top10, top25, bottom10, etc.
    interval: str = "7d",  # 1d, 3d, 7d, 10d, 14d, 30d, 90d, custom
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    run_id: Optional[str] = None  # If provided, fetch specific run
) -> GlobalAnalyticsResponse:
    """
    Fetch global analytics with score bucket and interval filtering.
    
    Returns all data needed for Analytics Tab in single request:
    - perf_summary: CAGR, Sharpe, Sortino, etc.
    - predictive: RankIC series, hit rate, profit factor
    - buckets: Performance by score bucket
    - correlations: Signal/group correlations with top pairs
    - factor_contributions: Alpha/vol by group
    - backtest: Cumulative returns, rolling Sharpe, benchmark corrs
    """
```

**Response Schema**:
```typescript
interface GlobalAnalyticsResponse {
  run_id: string;
  run_timestamp: string;
  bucket: string;
  interval: string;
  
  perf_summary: {
    cagr: number;
    volatility: number;
    sharpe: number;
    sortino: number;
    calmar: number;
    alpha_vs_spy: number;
    beta_vs_spy: number;
    alpha_vs_qqq: number;
    beta_vs_qqq: number;
  };
  
  predictive: {
    ic_series: Array<{date: string; ic: number}>;
    ic_mean: number;
    ic_std: number;
    hit_rate_top_decile: number;
    profit_factor: number;
    win_loss_ratio: number;
  };
  
  buckets: {
    [bucket_name: string]: {
      avg_return: number;
      win_rate: number;
      count: number;
    };
  };
  
  correlations: {
    signal_correlations: Array<{i: string; j: string; r: number; n: number}>;
    factor_correlations: any;  // Existing group-level
    top_positive_pairs: Array<{i: string; j: string; r: number}>;
    top_negative_pairs: Array<{i: string; j: string; r: number}>;
  };
  
  factor_contributions: {
    [group: string]: {
      alpha_pct: number;
      vol_pct: number;
    };
  };
  
  backtest: {
    cumulative_returns: Array<{date: string; vp: number; spy: number; qqq: number}>;
    rolling_sharpe_30d: Array<{date: string; sharpe: number}>;
    benchmark_correlations: {
      SPY: number;
      QQQ: number;
    };
  };
}
```

**Estimated LOC**: +200 lines  
**Risk**: Low (read-only, well-defined schema)

---

### **3.2 Extend `/api/performance` Endpoints**

**File**: `backend/api/performance.py` (existing)

**New Endpoint**:
```python
@router.get("/performance/{signal_id}/horizons")
async def get_performance_horizons(
    signal_id: str,
    benchmark: str = "spy"  # spy or qqq
) -> PerformanceHorizonsResponse:
    """
    Get all horizon data for Performance Tab grid.
    
    Returns:
    - Horizons: 1d, 3d, 7d, 10d, 14d, 30d, 90d
    - For each: VP return, benchmark return, alpha, status
    """
```

**Response Schema**:
```typescript
interface PerformanceHorizonsResponse {
  ticker: string;
  sector: string;
  market_cap: number;
  beta: number;
  baseline_date: string;
  last_updated: string;
  
  horizons: Array<{
    period: string;  // "1d", "3d", etc.
    vp_return: number | null;
    spy_return: number | null;
    qqq_return: number | null;
    alpha_vs_spy: number | null;
    alpha_vs_qqq: number | null;
    status: "completed" | "pending";
  }>;
  
  countdown: {
    next_horizon: string;
    time_remaining: number;  // seconds
    is_active: boolean;
  };
}
```

**Estimated LOC**: +100 lines  
**Risk**: Low (query existing `performance` table)

---

## 🎨 **Phase 4: Frontend Components**

### **4.1 Performance Tab** 

**Files to Create**:
- `frontend/src/performance/PerformanceTab.tsx` (move from dashboard)
- `frontend/src/performance/PerformanceCountdown.tsx` (move from dashboard)
- `frontend/src/performance/HorizonGrid.tsx` (new)
- `frontend/src/performance/AlphaSparkline.tsx` (new)
- `frontend/src/performance/HorizonQualitySummary.tsx` (new)
- `frontend/src/performance/index.ts` (barrel export)

**Layout Structure**:
```tsx
<PerformanceTab>
  <Header />  {/* Ticker • Sector • MktCap • β • Baseline • Last Updated */}
  
  <Row className="main-content">
    <HorizonGrid horizons={data.horizons} />
    <PerformanceCountdown countdown={data.countdown} />
  </Row>
  
  <Row className="insights">
    <AlphaSparkline 
      horizons={data.horizons} 
      benchmark={selectedBenchmark}
      onBenchmarkToggle={setBenchmark}
    />
    <HorizonQualitySummary 
      horizons={data.horizons}
      benchmark={selectedBenchmark}
    />
  </Row>
  
  <Row className="optional">
    <TopSignalContributors contributors={data.top_contributors} />
    <DataStaleness staleness={data.staleness} />
  </Row>
</PerformanceTab>
```

**Key Features**:
- SPY/QQQ toggle (affects sparkline + summary only)
- Monospace numbers, right-aligned
- Green/red color coding for returns + alpha
- "Pending" status for missing data (no errors)
- Countdown auto-hides after 90d

**Estimated LOC**: ~800 lines total  
**Risk**: Low (read-only, no complex state)

---

### **4.2 Analytics Tab**

**Files to Create**:
- `frontend/src/analytics/AnalyticsTab.tsx` (new)
- `frontend/src/analytics/GlobalControls.tsx` (new)
- `frontend/src/analytics/PerformanceSummary.tsx` (new)
- `frontend/src/analytics/PredictiveStrength.tsx` (new)
- `frontend/src/analytics/ScoreBucketPerformance.tsx` (new)
- `frontend/src/analytics/CorrelationHeatmap.tsx` (new - complex!)
- `frontend/src/analytics/FactorContributions.tsx` (new)
- `frontend/src/analytics/BacktestChart.tsx` (new)
- `frontend/src/analytics/index.ts` (barrel export)

**Layout Structure**:
```tsx
<AnalyticsTab>
  <GlobalControls 
    bucket={bucket}
    interval={interval}
    onBucketChange={setBucket}
    onIntervalChange={setInterval}
  />
  
  <PerformanceSummary data={analytics.perf_summary} />
  
  <PredictiveStrength data={analytics.predictive} />
  
  <ScoreBucketPerformance 
    data={analytics.buckets}
    bucket={bucket}
  />
  
  <CorrelationHeatmap 
    mode={heatmapMode}  // "signals" or "groups"
    data={analytics.correlations}
    onModeToggle={setHeatmapMode}
    onGroupFilter={handleGroupFilter}
  />
  
  <FactorContributions 
    data={analytics.factor_contributions}
    onGroupClick={handleGroupFilter}
  />
  
  <BacktestChart 
    data={analytics.backtest}
    logScale={logScale}
    onLogScaleToggle={setLogScale}
  />
</AnalyticsTab>
```

**Most Complex Component**: `CorrelationHeatmap.tsx`
- 158×158 matrix = 24,806 cells
- Multi-select search/filter
- Threshold slider
- Clustered ordering
- Hover tooltips
- Click → rolling correlation modal
- Export PNG/CSV

**Implementation Strategy**:
1. Use **canvas rendering** (not DOM) for 158×158 grid (performance)
2. **Virtualization** if needed (only render visible cells)
3. **Web Worker** for clustering algorithm (doesn't block UI)
4. **Debounced search** (300ms delay)
5. **Memoization** for expensive calculations

**Estimated LOC**: ~2000 lines total (heatmap = 800 lines)  
**Risk**: High (complex visualization, performance-sensitive)

---

## 📦 **Phase 5: Testing & Validation**

### **5.1 Unit Tests**

**Backend**:
- Phase 6 benchmark fetching logic
- Phase 4 contribution calculation
- Phase 7 correlation computation
- API endpoint response schemas

**Frontend**:
- Performance Tab horizon grid rendering
- Analytics Tab global controls state
- Heatmap search/filter logic
- Alpha calculations

**Target Coverage**: 80%+

---

### **5.2 Integration Tests**

**End-to-End Scenarios**:
1. Run pipeline → verify QQQ data stored in `performance`
2. Run analytics → verify UPSERT (single row per run_id)
3. Fetch `/analytics/global` → verify all sections populated
4. Fetch `/performance/:id/horizons` → verify 7 horizons returned
5. Toggle benchmark → verify alpha sparkline updates
6. Filter heatmap → verify correct subset displayed

---

### **5.3 Performance Tests**

**Targets**:
- `/analytics/global` response < 500ms (with 158×158 correlations)
- Heatmap initial render < 1 second
- Heatmap search/filter < 100ms
- Analytics job (Phase 7) < 60 seconds for 100 tickers

**Tools**:
- Backend: `pytest-benchmark`
- Frontend: Chrome DevTools Performance tab
- Database: `EXPLAIN ANALYZE` for query plans

---

## 📊 **Data Volume Projections**

### **Current State** (before changes)
- `signals`: ~100 tickers/run × 50 columns × 8 bytes = **40 KB/run**
- `performance`: ~100 tickers/run × 40 columns × 8 bytes = **32 KB/run**
- `analytics`: ~1 row/run × 10 columns × 1 KB = **1 KB/run**
- **Total per run**: ~73 KB
- **90 days**: 73 KB × 90 = **6.6 MB**

### **After Phase 1 Changes**
- `signals`: No change = **40 KB/run**
- `performance`: +14 columns = **43 KB/run** (+10%)
- `analytics`: +15 columns (including 158×158 matrix) = **101 KB/run** (+10,000%!)
- **Total per run**: ~184 KB (+152%)
- **90 days**: 184 KB × 90 = **16.6 MB** (+151%)

### **With UPSERT Strategy**
- Same as above BUT only 1 analytics row per run_id (not cumulative)
- **90 days with UPSERT**: 43 KB × 90 + 101 KB × 90 = **13 MB** (vs 16.6 MB)
- **Savings**: Small now, but huge over time (prevents linear growth)

### **6 Months Projection**
- **Without UPSERT**: 184 KB × 180 runs = **33 MB**
- **With UPSERT**: 43 KB × 180 + 101 KB × 1 = **7.8 MB** (76% savings!)

### **1 Year Projection**
- **Without UPSERT**: 184 KB × 365 = **67 MB**
- **With UPSERT**: 43 KB × 365 + 101 KB × 1 = **15.8 MB** (76% savings!)

**Recommendation**: ✅ **Use UPSERT strategy** for `analytics` table

---

## ⚠️ **Risks & Mitigations**

### **Risk 1: 158×158 Correlation Computation Performance**
- **Impact**: Phase 7 analytics job could take >5 minutes
- **Mitigation**: 
  - Use NumPy vectorized operations
  - Parallelize with `multiprocessing` (one process per group)
  - Precompute only upper triangle (symmetric matrix)
  - Store as sparse format (skip |r| < 0.1)

### **Risk 2: Heatmap Rendering Performance**
- **Impact**: 24,806 cells could freeze browser
- **Mitigation**:
  - Canvas rendering (not DOM)
  - Virtualization (only render visible cells)
  - Lazy load on scroll
  - Debounced interactions

### **Risk 3: yfinance Rate Limits**
- **Impact**: Fetching SPY + QQQ daily could hit limits
- **Mitigation**:
  - Use `benchmarks` table caching
  - Batch fetch (90 days at once, not daily)
  - Exponential backoff on errors
  - Fallback to cached data if fetch fails

### **Risk 4: UPSERT Breaking Historical Analytics**
- **Impact**: Overwriting analytics rows loses history
- **Mitigation**:
  - **Don't use UPSERT initially** - append for first 90 days
  - Add `version` column to track schema changes
  - Create `analytics_history` table for long-term trends
  - Or: keep UPSERT but add `updated_at` to track refreshes

### **Risk 5: Dynamic Factor Count Breaking Frontend**
- **Impact**: Adding factors to YAML breaks heatmap sizing
- **Mitigation**:
  - Frontend reads factor count from API response
  - Heatmap dimensions = `data.signal_correlations.length`
  - No hardcoded "158" anywhere

---

## 🗓️ **Implementation Timeline**

### **Week 1: Database + Pipeline**
- **Day 1-2**: Run migrations (015, 016, 017)
- **Day 3-4**: Update Phase 6 for QQQ fetching
- **Day 5-7**: Update Phase 4 contribution schema + Phase 7 UPSERT logic

**Deliverable**: Pipeline runs successfully, new columns populated

---

### **Week 2: Backend API**
- **Day 1-3**: Create `/analytics/global` endpoint with stub data
- **Day 4-5**: Create `/performance/:id/horizons` endpoint
- **Day 6-7**: Integration tests + performance profiling

**Deliverable**: API endpoints return correct schemas, <500ms response

---

### **Week 3: Frontend - Performance Tab**
- **Day 1-2**: File restructure (move to `/performance/`)
- **Day 3-4**: Build `HorizonGrid` and `PerformanceCountdown`
- **Day 5-6**: Build `AlphaSparkline` and `HorizonQualitySummary`
- **Day 7**: Integration + testing

**Deliverable**: Performance Tab fully functional with SPY/QQQ toggle

---

### **Week 4: Frontend - Analytics Tab (Part 1)**
- **Day 1-2**: Build `GlobalControls` and `PerformanceSummary`
- **Day 3-4**: Build `PredictiveStrength` and `ScoreBucketPerformance`
- **Day 5-7**: Build `FactorContributions` and `BacktestChart`

**Deliverable**: Analytics Tab sections 1-4 functional (no heatmap yet)

---

### **Week 5: Frontend - Analytics Tab (Part 2 - Heatmap)**
- **Day 1-3**: Build basic heatmap (canvas rendering)
- **Day 4-5**: Add search/filter/threshold controls
- **Day 6-7**: Add clustering + export features

**Deliverable**: Heatmap fully functional (groups mode)

---

### **Week 6: Frontend - Analytics Tab (Part 3 - Signal Heatmap)**
- **Day 1-3**: Extend heatmap to 158×158 signal mode
- **Day 4-5**: Optimize performance (virtualization, Web Worker)
- **Day 6-7**: Add rolling correlation modal

**Deliverable**: Signal heatmap fully functional with all features

---

### **Week 7: Polish & Testing**
- **Day 1-2**: Cross-browser testing (Chrome, Firefox, Safari)
- **Day 3-4**: Mobile responsive adjustments
- **Day 5-6**: A11y audit + keyboard navigation
- **Day 7**: Performance profiling + optimization

**Deliverable**: Production-ready, all features tested

---

### **Week 8: Documentation & Deployment**
- **Day 1-2**: Update README, API docs, frontend docs
- **Day 3-4**: Create user guide (screenshots + GIFs)
- **Day 5**: Deploy to staging
- **Day 6**: User acceptance testing
- **Day 7**: Deploy to production

**Deliverable**: Live in production, documented

---

## ✅ **Success Criteria**

### **Database**
- ✅ All migrations run without errors
- ✅ QQQ data populated for all performance rows
- ✅ Analytics table uses UPSERT (one row per run_id)
- ✅ 158×158 correlation matrix stored as JSONB
- ✅ Factor contributions match `{alpha_pct, vol_pct}` schema

### **Backend**
- ✅ `/analytics/global` returns complete payload
- ✅ `/performance/:id/horizons` returns 7 horizons
- ✅ API response time < 500ms (95th percentile)
- ✅ Phase 7 analytics job completes in < 60 seconds

### **Frontend - Performance Tab**
- ✅ Horizon grid shows all 7 horizons with returns + alpha
- ✅ SPY/QQQ toggle updates sparkline + summary
- ✅ Countdown accurate and auto-hides after 90d
- ✅ Missing data shows "Pending" (not errors)
- ✅ All numbers monospace, right-aligned, color-coded

### **Frontend - Analytics Tab**
- ✅ Global controls (bucket + interval) persist to URL
- ✅ All sections update when controls change
- ✅ Heatmap supports both signals (158×158) and groups modes
- ✅ Heatmap search/filter/threshold work smoothly
- ✅ Factor contributions clickable → filters heatmap
- ✅ Export features work (PNG, CSV)

### **Performance**
- ✅ Analytics Tab initial load < 2 seconds
- ✅ Heatmap render < 1 second
- ✅ Heatmap interactions < 100ms
- ✅ No browser freezes or lag

### **Quality**
- ✅ Unit test coverage > 80%
- ✅ All integration tests pass
- ✅ No console errors or warnings
- ✅ Accessibility score > 90 (Lighthouse)

---

## 🤔 **Open Questions for User**

1. **Analytics UPSERT Strategy**:
   - ✅ **Recommended**: UPSERT (one row per run_id, overwrite on each run)
   - ❌ **Alternative**: Append (new row each run, keeps history)
   - **Your preference?** UPSERT saves 76% storage but loses historical snapshots

2. **Benchmark Caching**:
   - ✅ **Recommended**: Create `benchmarks` table, cache SPY/QQQ
   - ❌ **Alternative**: Fetch from yfinance every run (simpler but slower)
   - **Your preference?** I recommend caching for performance + consistency

3. **Correlation Sparsity**:
   - Store all 12,403 pairs OR only |r| > threshold (e.g., 0.1)?
   - **Trade-off**: All pairs = 99 KB, sparse = ~20 KB but less flexible
   - **Your preference?** I recommend all pairs (flexibility > 79 KB savings)

4. **Heatmap Clustering Algorithm**:
   - Hierarchical clustering (slow, accurate) vs K-means (fast, approximate)
   - **Your preference?** I recommend hierarchical (run once, cache result)

5. **Phase Prioritization**:
   - Start with Performance Tab (simpler) or Analytics Tab (more valuable)?
   - **Your preference?** I recommend Performance Tab first (validate foundation)

---

## 📝 **Next Steps**

**Immediate Actions** (today):
1. ✅ Review this plan
2. ✅ Answer open questions above
3. ✅ Approve migration SQL
4. ⏸️ Wait for confirmation before starting

**Week 1 Kickoff** (after approval):
1. Run migrations 015, 016, 017
2. Create feature branch: `feature/performance-analytics`
3. Update Phase 6 for QQQ benchmarks
4. Test pipeline with new columns

---

**Ready to proceed?** Let me know your thoughts on:
1. UPSERT vs Append for analytics
2. Benchmark caching table
3. Phase prioritization (Performance vs Analytics first)
4. Any concerns or questions

Then we'll create the first migration and get started! 🚀
