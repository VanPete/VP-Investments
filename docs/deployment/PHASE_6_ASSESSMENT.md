# Phase 6 Performance Assessment for VanPiQ

**Purpose:** Determine what Phase 6 changes are needed for the VanPiQ Performance Tab  
**Date:** $(Get-Date)  
**Status:** ✅ **QQQ already supported!** Minimal backend work needed.

---

## Executive Summary

### Finding: **Phase 6 Already Supports QQQ** ✅

**Code Evidence:**
- **Lines 356-357** in `phase6_performance.py`: Computes `qqq_return_pct` for all 7 horizons
- **Line 244**: Returns dict includes `qqq_return_Xd` for all intervals
- **Lines 359-360**: Comments confirm `qqq_alpha_Xd` auto-calculated by database GENERATED columns
- **Database Schema** (`supabase.sql` lines 141-154): All 7 horizons have `qqq_return_*` and `qqq_alpha_*` columns

**Verdict:** Phase 6 requires **no QQQ additions**. Work is **95% frontend**, **5% backend verification**.

---

## 1. Current Phase 6 Implementation

### 1.1 Performance Table Schema (Existing)
```sql
-- All 7 horizons already have:
qqq_return_1d real,
qqq_return_3d real,
qqq_return_7d real,
qqq_return_10d real,
qqq_return_14d real,
qqq_return_30d real,
qqq_return_90d real,

-- GENERATED columns auto-calculate alpha:
qqq_alpha_1d real DEFAULT ((return_1d) - qqq_return_1d),
qqq_alpha_3d real DEFAULT ((return_3d) - qqq_return_3d),
qqq_alpha_7d real DEFAULT ((return_7d) - qqq_return_7d),
qqq_alpha_10d real DEFAULT ((return_10d) - qqq_return_10d),
qqq_alpha_14d real DEFAULT ((return_14d) - qqq_return_14d),
qqq_alpha_30d real DEFAULT ((return_30d) - qqq_return_30d),
qqq_alpha_90d real DEFAULT ((return_90d) - qqq_return_90d),
```

**Status:** ✅ **Complete** - No schema changes needed

### 1.2 Phase 6 Code (`phase6_performance.py`)

**Key Code Sections:**

```python
# Lines 356-357: QQQ return calculation
for interval in self.intervals:
    qqq_return_pct = ((target_qqq - baseline_qqq) / baseline_qqq) * 100
    perf_metrics[f"qqq_return_{interval}d"] = qqq_return_pct
    
# Lines 359-360: Alpha auto-calculated
# qqq_alpha_Xd columns are GENERATED in DB (return_Xd - qqq_return_Xd)
```

**Status:** ✅ **Complete** - QQQ returns computed for all 7 intervals

### 1.3 Benchmark Cache (Phase 1 Integration)

**From Phase 1:**
- SPY, QQQ, and 11 sector ETFs fetched into `benchmark_cache` dict
- Passed to Phase 6 constructor
- Phase 6 uses cached prices (no extra API calls)

**Status:** ✅ **Complete** - Architecture already optimized

---

## 2. VanPiQ Performance Tab Requirements

### 2.1 Data Required by Frontend

**From VanPiQ Spec (`VanPiQ_Performance+Analytics_Complete_vFinal.md`):**

| Data Point | Status | Source |
|------------|--------|--------|
| **7-Horizon Grid** | | |
| - VP Return (1d/3d/7d/10d/14d/30d/90d) | ✅ Exists | `return_Xd` columns |
| - SPY Return (all horizons) | ✅ Exists | `spy_return_Xd` columns |
| - QQQ Return (all horizons) | ✅ Exists | `qqq_return_Xd` columns |
| - Alpha vs SPY (all horizons) | ✅ Exists | `alpha_Xd` GENERATED columns |
| - Alpha vs QQQ (all horizons) | ✅ Exists | `qqq_alpha_Xd` GENERATED columns |
| - Status (Completed/Pending) | ⚠️ Frontend | Compute from `created_at` + interval |
| **Countdown Card** | | |
| - Next horizon unlock time | ⚠️ Frontend | `created_at` + next incomplete interval |
| - Auto-hide after 90D | ⚠️ Frontend | Check if 90d elapsed |
| **Alpha Sparkline** | | |
| - Cumulative alpha series (SPY/QQQ toggle) | ✅ Data exists | `alpha_Xd` or `qqq_alpha_Xd` arrays |
| - X-axis = completed horizons | ⚠️ Frontend | Filter by current date vs horizon |
| **Horizon Quality Summary** | | |
| - Earliest completed | ⚠️ Frontend | Find first elapsed horizon |
| - Most recent | ⚠️ Frontend | Find last elapsed horizon |
| - Best/Worst horizon | ✅ Data exists | Compare `alpha_Xd` values |
| - "Beating SPY/QQQ: X/Y" | ✅ Data exists | Count positive `alpha_Xd` |
| **Header Metadata** | | |
| - Ticker | ✅ Exists | `signals.ticker` |
| - Sector | ✅ Exists | `signals.sector` |
| - MktCap | ❌ **MISSING** | **Not in signals table** |
| - β (Beta) | ❌ **MISSING** | **Not in signals table** |
| - Baseline Date | ✅ Exists | `signals.created_at` |
| - Last Updated | ✅ Exists | `signals.updated_at` |

**Key Findings:**
1. ✅ All performance data exists in database
2. ⚠️ Countdown/status logic = frontend computation only
3. ❌ **MktCap and Beta missing** from `signals` table

---

## 3. Gap Analysis

### 3.1 Backend Gaps

#### **Gap 1: MktCap and Beta Missing from Signals Table** ❌

**Problem:** VanPiQ Performance Tab header requires:
- Market Cap (formatted: `$12.3B`)
- Beta vs SPY (formatted: `β 1.23`)

**Current State:**
- `signals` table has: `ticker`, `sector`, `current_price`, `company_name`
- Missing: `market_cap`, `beta`

**Options:**

**Option A: Add to Phase 1 (Recommended)** ✅
- Phase 1 already fetches fundamentals via Polygon/AlphaVantage
- Add `market_cap` and `beta` to Phase 1 enrichment
- Store in `signals` table (2 new columns)
- Minimal latency (already in same API calls)

**Option B: Compute in Phase 6** ⚠️
- Phase 6 computes beta from price history (SPY regression)
- Market cap = `current_price * shares_outstanding` (requires new data fetch)
- Slower, duplicates work Phase 1 should do

**Option C: Skip for MVP** ❌
- Show "N/A" for MktCap/Beta
- Not ideal, spec explicitly calls for these

**Recommendation:** **Option A** - Add to Phase 1 enrichment (~30 min work)

---

#### **Gap 2: Countdown Logic** ⚠️ (Frontend Only)

**Requirement:** 
- Show time remaining until next horizon unlocks
- Auto-hide after 90D complete
- Display format: "X days Y hours to 14D" or "Completed"

**Status:** 
- All data exists (`created_at` timestamp in `signals` table)
- Pure frontend computation: `current_time - created_at` vs horizon thresholds
- No backend work needed

---

#### **Gap 3: Horizon Status (Completed/Pending)** ⚠️ (Frontend Only)

**Requirement:**
- Show "Completed" if horizon elapsed
- Show "Pending" if not yet elapsed

**Status:**
- Computation: `created_at + interval <= current_time`
- Frontend logic only
- No backend work needed

---

### 3.2 Frontend Gaps (95% of work)

**Major Work Items:**

1. **Move Files** (~15 min)
   - `dashboard/PerformanceTab.tsx` → `performance/PerformanceTab.tsx`
   - `dashboard/PerformanceCountdown.tsx` → `performance/PerformanceCountdown.tsx`
   - Create barrel `performance/index.ts`

2. **Rebuild Layout** (~4 hours)
   - **Row A:** 7-horizon grid (7 rows × 6 columns) + Countdown card
   - **Row B:** Alpha sparkline (SPY/QQQ toggle) + Horizon quality summary
   - **Row C:** Top signal contributors (if available) + Data staleness

3. **Implement SPY/QQQ Toggle** (~1 hour)
   - State: `benchmark: 'SPY' | 'QQQ'`
   - Affects: Alpha sparkline, "Beating X/Y" count
   - Use `alpha_Xd` (SPY) or `qqq_alpha_Xd` (QQQ) from data

4. **Alpha Sparkline Component** (~2 hours)
   - Input: Array of completed horizon alphas
   - Cumulative sum: `[α₁, α₁+α₃, α₁+α₃+α₇, ...]`
   - X-axis labels: `["1D", "3D", "7D", ...]` (only completed)
   - Recharts LineChart with gradient fill

5. **Countdown Timer Component** (~2 hours)
   - Calculate next incomplete horizon
   - Real-time countdown (useEffect + setInterval)
   - Auto-hide if all horizons complete
   - Format: "X days Y hours to 14D"

6. **Horizon Quality Summary** (~1 hour)
   - Parse completed horizons
   - Find earliest, latest, best (max alpha), worst (min alpha)
   - Count beating benchmark: `sum(alpha_Xd > 0)`
   - Display: "Beating SPY: 5/7"

7. **Header Metadata** (~30 min)
   - Display: Ticker • Sector • MktCap • β • Baseline • Last Updated
   - Format: `text-sm`, compact, `gap-2`
   - Show "N/A" if MktCap/Beta missing (until Phase 1 update)

8. **API Endpoint** (~1 hour)
   - `GET /api/performance/:signal_id/horizons`
   - Returns all 7-horizon data + metadata
   - Used by Performance Tab for single-signal view

**Total Frontend Estimate:** ~12 hours

---

## 4. Recommended Action Plan

### Phase 1: Backend Validation & Enrichment (~1 hour)

**Step 1: Test Phase 6 QQQ Data** (30 min)
```bash
# Run one pipeline execution
python run_pipeline_and_push.py

# Query Supabase to verify QQQ data populated
SELECT 
  ticker, 
  qqq_return_1d, qqq_return_7d, qqq_return_30d,
  qqq_alpha_1d, qqq_alpha_7d, qqq_alpha_30d
FROM performance 
WHERE signal_id = (SELECT id FROM signals ORDER BY created_at DESC LIMIT 1)
LIMIT 5;
```

**Expected:**
- ✅ `qqq_return_*` columns populated with decimal values
- ✅ `qqq_alpha_*` columns auto-calculated (GENERATED)
- ❌ If NULL, investigate Phase 6 benchmark_cache access

---

**Step 2: Add MktCap & Beta to Phase 1** (30 min)

**File:** `backend/phases/phase1_data_fetch.py`

**Changes:**
1. Add to signal enrichment section:
```python
# Fetch market cap and beta from Polygon/AlphaVantage
market_cap = fundamentals.get('marketCap', None)
beta = fundamentals.get('beta', None)

signal_data['market_cap'] = market_cap
signal_data['beta'] = beta
```

2. **Migration 017:** Add columns to `signals` table
```sql
-- migrations/017_add_mktcap_beta_to_signals.sql
ALTER TABLE signals 
ADD COLUMN market_cap BIGINT,
ADD COLUMN beta REAL;

COMMENT ON COLUMN signals.market_cap IS 'Market capitalization in USD';
COMMENT ON COLUMN signals.beta IS 'Beta vs SPY (volatility vs market)';
```

3. Update `pipeline.py` to pass market_cap/beta to Phase 6 (if needed)

**Test:**
```bash
# Run pipeline after changes
python run_pipeline_and_push.py

# Verify new columns populated
SELECT ticker, market_cap, beta, sector 
FROM signals 
ORDER BY created_at DESC 
LIMIT 10;
```

---

### Phase 2: Frontend Implementation (~12 hours)

**Blocked by:** None (can start immediately, show "N/A" for MktCap/Beta until Phase 1 done)

**Priority Order:**
1. Move files + create barrel (15 min)
2. Build 7-horizon grid layout (2 hours)
3. Implement SPY/QQQ toggle (1 hour)
4. Alpha sparkline component (2 hours)
5. Countdown timer (2 hours)
6. Horizon quality summary (1 hour)
7. Header metadata (30 min)
8. API endpoint (1 hour)
9. Polish + testing (2 hours)

---

### Phase 3: Integration Testing (~30 min)

**Test Cases:**
1. ✅ Performance tab shows all 7 horizons with SPY/QQQ data
2. ✅ Countdown shows correct time to next horizon
3. ✅ Alpha sparkline toggles between SPY/QQQ correctly
4. ✅ "Beating SPY: X/Y" count matches positive alpha_Xd
5. ✅ Missing horizons show "Pending" (not error)
6. ✅ MktCap/Beta display correctly (after Phase 1 update)
7. ✅ Countdown auto-hides after 90D complete

---

## 5. Implementation Checklist

### Backend Tasks

- [ ] **Test Phase 6 QQQ Data** (30 min)
  - Run pipeline
  - Query `qqq_return_*` and `qqq_alpha_*` columns
  - Verify GENERATED columns auto-populate
  
- [ ] **Add MktCap & Beta to Phase 1** (30 min)
  - Update `phase1_data_fetch.py` to fetch market_cap/beta
  - Create migration 017 (2 new columns)
  - Execute migration in Supabase
  - Test with pipeline run
  
- [ ] **Update Supabase.sql** (5 min)
  - Add migration 017 schema to reference file

### Frontend Tasks

- [ ] **Move Files** (15 min)
  - `dashboard/PerformanceTab.tsx` → `performance/PerformanceTab.tsx`
  - `dashboard/PerformanceCountdown.tsx` → `performance/PerformanceCountdown.tsx`
  - Create `performance/index.ts` barrel
  - Update routes/imports

- [ ] **7-Horizon Grid Component** (2 hours)
  - Table with 7 rows (1D/3D/7D/10D/14D/30D/90D)
  - 6 columns: VP Return, SPY Return, QQQ Return, Alpha SPY, Alpha QQQ, Status
  - Color coding: green (positive), red (negative), neutral (pending)
  - Right-aligned, monospace numbers

- [ ] **SPY/QQQ Toggle** (1 hour)
  - State: `const [benchmark, setBenchmark] = useState<'SPY' | 'QQQ'>('SPY')`
  - Affects: Alpha sparkline, "Beating X/Y" count
  - Use `alpha_Xd` or `qqq_alpha_Xd` based on toggle

- [ ] **Alpha Sparkline** (2 hours)
  - Compute cumulative alpha array
  - Filter to completed horizons only
  - Recharts LineChart with gradient fill
  - X-axis: ["1D", "3D", "7D", ...] (completed only)
  - Y-axis: Cumulative alpha (%)

- [ ] **Countdown Timer** (2 hours)
  - Calculate next incomplete horizon
  - Real-time countdown (useEffect + setInterval)
  - Format: "X days Y hours to 14D"
  - Auto-hide if all horizons complete

- [ ] **Horizon Quality Summary** (1 hour)
  - Earliest/latest completed horizon
  - Best/worst horizon (max/min alpha)
  - "Beating SPY: X/Y" count

- [ ] **Header Metadata** (30 min)
  - Display: Ticker • Sector • MktCap • β • Baseline • Last Updated
  - Format MktCap: `$12.3B`
  - Format Beta: `β 1.23`
  - Show "N/A" if missing (until Phase 1 update)

- [ ] **API Endpoint** (1 hour)
  - `GET /api/performance/:signal_id/horizons`
  - Return: all 7-horizon data + metadata + sector/mktcap/beta
  - Used by Performance Tab

- [ ] **Testing & Polish** (2 hours)
  - Test all 7 test cases
  - Accessibility (keyboard nav)
  - Responsive layout
  - Error states (missing data)

---

## 6. Decision Required

**Question for User:**

Should we add **MktCap and Beta** to Phase 1 enrichment now, or proceed with frontend showing "N/A" for MVP?

**Option A (Recommended):** Add to Phase 1 (~30 min)
- ✅ Complete VanPiQ spec implementation
- ✅ Data available for all future uses
- ✅ Only 30 min work (Phase 1 already fetches fundamentals)

**Option B:** Skip for MVP, show "N/A"
- ⚠️ Incomplete spec implementation
- ⚠️ Need to revisit later
- ⚠️ Frontend shows placeholder data

---

## 7. Summary

### What Phase 6 Already Has ✅
- All 7 QQQ horizons (return + alpha)
- Benchmark cache integration (SPY/QQQ/sectors)
- GENERATED columns for auto-calculated alpha
- Efficient architecture (no redundant API calls)

### What's Needed ⚠️
1. **Backend (5%)**: Add MktCap/Beta to Phase 1 + Test QQQ data (~1 hour)
2. **Frontend (95%)**: Rebuild Performance Tab per VanPiQ spec (~12 hours)

### Next Steps
1. **User Decision:** Add MktCap/Beta to Phase 1 now? (Option A recommended)
2. **Test Phase 6:** Run pipeline, verify QQQ data populates correctly
3. **Frontend Work:** Rebuild Performance Tab (can start immediately)

**Estimated Total Time:** ~13 hours (1 hour backend + 12 hours frontend)

---

## 8. Files to Update

### Backend
- `backend/phases/phase1_data_fetch.py` - Add market_cap/beta enrichment
- `migrations/017_add_mktcap_beta_to_signals.sql` - New migration (if Option A)
- `supabase.sql` - Add migration 017 schema (if Option A)

### Frontend
- Move: `dashboard/PerformanceTab.tsx` → `performance/PerformanceTab.tsx`
- Move: `dashboard/PerformanceCountdown.tsx` → `performance/PerformanceCountdown.tsx`
- Create: `performance/index.ts` (barrel export)
- Create: `performance/AlphaSparkline.tsx` (new component)
- Create: `performance/HorizonQuality.tsx` (new component)
- Update: Routes/imports to use new paths

### API
- Create: `backend/api/routes/performance.py` - `/api/performance/:signal_id/horizons`

---

**Status:** Ready for user decision on MktCap/Beta approach. Phase 6 assessment complete.
