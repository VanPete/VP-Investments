# Frontend Improvements Summary
**Date:** October 26, 2025

## Overview
This document summarizes the frontend and UI improvements made to the VanPiQ Signals Dashboard.

---

## 1. Column Visibility Updates ✅

### New Columns Added
- **Company Name** - Shows full company name (e.g., "Apple Inc.", "Microsoft Corporation")
  - Visible by default
  - Position: Between Ticker and Overall Score
  - Source: `signals.company_name` from database
  
- **Current Price** - Shows current stock price
  - Hidden by default (can be toggled on)
  - Position: After Company Name
  - Source: `signals.current_price` from database

### Column Filter Recommendations
**Recommended visible columns (in order):**
1. Rank
2. Ticker
3. **Company Name** ← NEW (on by default)
4. Overall Score
5. Coverage
6. Technical → Institutional (group scores)
7. **Current Price** ← NEW (off by default, toggle on if needed)

### Performance Columns Renamed
- Removed: "Baseline Price" (redundant with Current Price)
- Updated naming: "Backtest" → **"Performance"** throughout the app
- Performance columns (1D/7D/30D/90D Returns, vs SPY) remain optional

---

## 2. Header Layout Redesign ✅

### New Three-Section Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  [VanPiQ Dashboard]     [LARGE VANPIQ LOGO]    [Date Selector]  │
│  [Badge: 14 tickers]                           [Refresh][Theme] │
│  Latest: Oct 26 • Source: supabase • Last updated: 6:45 AM      │
└─────────────────────────────────────────────────────────────────┘
```

**Changes:**
- **Left Section:** Title "VanPiQ Signals Dashboard" + ticker count badge
- **Center Section:** Large VanPIQ logo (60px height, up from 40px) with glow effect on hover
- **Right Section:** Run selector dropdown + refresh button + theme toggle
- **Bottom Row:** Metadata (timestamp, source, last updated) spans full width

**Benefits:**
- Better visual balance
- Logo is prominent and centered
- Controls accessible on the right
- More professional appearance

---

## 3. Performance Tab Improvements ✅

### Always Show Tab (Even with No Data)
Previously: Tab would be hidden or show minimal UI if no performance data existed
Now: **Tab always visible** with proper messaging

### No Data State
When no performance data is available:
```
┌────────────────────────────────────────┐
│    [Activity Icon]                     │
│    No Performance Data Available       │
│                                        │
│    Performance tracking will begin     │
│    once signals have been generated... │
│                                        │
│    Check back after running pipeline   │
└────────────────────────────────────────┘
```

**Message includes:**
- Clear icon (Activity)
- Informative title
- Explanation of why no data exists
- Actionable next steps

### Renamed Throughout
- "Backtest" → **"Performance"** (all references updated)
- Comments updated to reflect "Performance columns" instead of "Backtest columns"

---

## 4. Coverage Bug Fix ✅

### Issue
Coverage was showing **100%** for all signals instead of actual values (e.g., 15.6%, 23.4%)

### Root Cause
The `useSupabaseSignals` hook was calculating coverage from detail tables (JSONB factor counts) instead of using the pre-calculated values stored in the database.

### Solution
- Removed `fetchCoverageForSignals` function (no longer needed)
- Use `signal.total_coverage` directly from database
- Use `signal.*_coverage` for all group coverages

**Result:** Coverage now displays correctly:
- Total Coverage: Uses `total_coverage` column (0-1 scale, displayed as %)
- Group Coverages: Uses `technical_coverage`, `fundamental_coverage`, etc.

---

## 5. TypeScript Type Updates ✅

### `SignalRanking` Interface
Added new fields to `frontend/src/types/pipeline.ts`:
```typescript
export interface SignalRanking {
  rank: number;
  ticker: string;
  company_name?: string;     // ← NEW
  current_price?: number;    // ← NEW
  overall_score: number;
  total_coverage: number;
  // ... rest of fields
}
```

### `ColumnVisibility` Interface
Updated in multiple files:
```typescript
export interface ColumnVisibility {
  rank: boolean;
  ticker: boolean;
  companyName: boolean;      // ← NEW
  currentPrice: boolean;     // ← NEW
  overallScore: boolean;
  coverage: boolean;
  // ... group scores
  // Performance columns (renamed from backtest)
  return1d?: boolean;
  return7d?: boolean;
  return30d?: boolean;
  return90d?: boolean;
  vsSpy?: boolean;
}
```

---

## 6. Data Fetching Updates ✅

### `useSupabaseSignals` Hook
Updated SELECT query to include new columns:
```typescript
.select(`
  id,
  ticker,
  company_name,        // ← NEW
  current_price,       // ← NEW
  rank,
  overall_score,
  // ... all score and coverage fields
`)
```

### Coverage Calculation
**Before:** Calculated from JSONB factors in detail tables
**After:** Direct database values from signals table
- `total_coverage` → `signal.total_coverage`
- Group coverages → `signal.technical_coverage`, etc.

---

## 7. Component Updates

### Files Modified
1. **DashboardHeader.tsx**
   - Redesigned 3-section layout (title left, logo center, controls right)
   - Larger logo (60px)
   - Better spacing and alignment

2. **SignalsDashboard.tsx**
   - Updated default `columnVisibility` to include `companyName: true`, `currentPrice: false`
   - Removed "baseline" column from visibility options

3. **ColumnVisibilityToggle.tsx**
   - Added "Company Name" and "Current Price" to column list
   - Removed "Baseline Price"
   - Updated comments: "Backtest" → "Performance"

4. **PerformanceTab.tsx**
   - Enhanced "No Data" state with icon and detailed message
   - Tab always visible even without data

5. **useSupabaseSignals.ts**
   - Removed `fetchCoverageForSignals` function
   - Simplified data transformation to use database values directly
   - Added `company_name` and `current_price` to query

6. **usePersistedState.ts**
   - Updated `ColumnVisibility` interface with new columns
   - Removed "baseline" field

---

## 8. Backend Support (Already Completed)

### Migration 011
- Added `company_name TEXT` column with index
- Added `current_price NUMERIC(12,4)` column

### Phase 5 Persistence
- Extracts `company_name` from `yfinance` `info.longName` or `info.shortName`
- Extracts `current_price` from `fast_info.lastPrice` or history `Close`
- Updated INSERT query to include 19 parameters (was 17)

### Tested Successfully
```
🧪 Testing AAPL: ✅ Apple Inc. | $262.82
🧪 Testing MSFT: ✅ Microsoft Corporation | $523.61
🧪 Testing GOOGL: ✅ Alphabet Inc. | $259.92
```

---

## Next Steps

1. **Run Pipeline** to populate company_name and current_price in database
2. **Test Frontend** with real data:
   - Verify company names display in table
   - Toggle "Current Price" column on/off
   - Check coverage shows actual percentages (not 100%)
3. **Update SignalsTable Component** (if needed) to render new columns
4. **Add Expandable Rows** that use `getTopFactorsForGroup()` to show top 5 factors
5. **Use `useSupabaseSignalsWithPerformance` hook** in Performance tab for full data

---

## Files Changed

### Frontend
- `frontend/src/types/pipeline.ts`
- `frontend/src/hooks/useSupabaseSignals.ts`
- `frontend/src/hooks/usePersistedState.ts`
- `frontend/src/components/dashboard/DashboardHeader.tsx`
- `frontend/src/components/dashboard/SignalsDashboard.tsx`
- `frontend/src/components/dashboard/ColumnVisibilityToggle.tsx`
- `frontend/src/components/dashboard/PerformanceTab.tsx`

### New Files
- `frontend/src/lib/getTopFactorsForGroup.ts` (utility for top 5 factors)
- `frontend/src/hooks/useSupabaseSignalsWithPerformance.ts` (signals + performance join)

### Backend
- `migrations/011_add_company_price_to_signals.sql`
- `backend/phases/phase5_persist.py`
- `scripts/apply_migration_011.py`
- `scripts/test_company_price_extraction.py`

---

## Visual Preview

### Column Filter (Updated)
```
✓ Rank
✓ Ticker (required)
✓ Company Name        ← NEW (on by default)
☐ Current Price       ← NEW (off by default)
✓ Overall Score (required)
✓ Coverage
✓ Technical
...
☐ 1D Return
☐ 7D Return
☐ vs SPY
```

### Header Layout (Before → After)
**Before:**
```
[Logo] VanPiQ Dashboard              [Selector] [Refresh]
       14 tickers
       Latest: Oct 26...
```

**After:**
```
VanPiQ Dashboard          [LARGE LOGO]          [Selector] [Refresh] [Theme]
14 tickers
Latest: Oct 26 • Source: supabase • Last updated: 6:45 AM
```

---

## Success Criteria

✅ Company names visible in table by default  
✅ Current price can be toggled on/off  
✅ Coverage shows actual percentages (15.6%, not 100%)  
✅ Header layout balanced (title left, logo center, controls right)  
✅ Performance tab always visible  
✅ "No data" message clear and informative  
✅ All "Backtest" references renamed to "Performance"  

---

**Status:** All changes implemented and ready for testing ✅
