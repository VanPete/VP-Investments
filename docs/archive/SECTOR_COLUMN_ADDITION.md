# Sector Column Addition Summary

**Date:** October 28, 2025  
**Feature:** Add Sector Column to Signals Table & Dashboard  
**Version:** v3.3

---

## 📋 Overview

Moved `sector` from `performance` table to `signals` table for better data normalization. Sector is a fundamental property of the ticker, not performance-specific data.

### Benefits

1. **Data Normalization** - Sector is intrinsic to ticker, belongs in signals
2. **Frontend Efficiency** - Dashboard displays sector without performance JOIN
3. **Query Performance** - Faster queries, no unnecessary joins
4. **Immediate Availability** - Sector available when signal created

---

## 🔧 Changes Made

### Database Migration

**File:** `migrations/014_add_sector_to_signals.sql`

```sql
-- Add sector column to signals table
ALTER TABLE signals ADD COLUMN IF NOT EXISTS sector TEXT;

-- Create indexes for sector queries
CREATE INDEX idx_signals_sector ON signals(sector);
CREATE INDEX idx_signals_sector_score ON signals(sector, overall_score DESC);

-- Backfill from performance table
UPDATE signals s
SET sector = p.sector
FROM performance p
WHERE s.id = p.signal_id AND p.sector IS NOT NULL;
```

### Backend Changes

**File:** `backend/phases/phase5_persist.py`

**Lines 353-388** - Extract sector when building signal record:
```python
# Extract company name, current price, and sector
sector = None  # v3.3: Extract sector for signals table

raw_data = ticker_data.get('raw_data')
if raw_data:
    info = raw_data.get('info', {})
    company_name = info.get('longName') or info.get('shortName')
    sector = info.get('sector')  # v3.3: Extract sector
    
signal_record = {
    'ticker': ticker,
    # ... other fields ...
    'sector': sector  # v3.3: Add sector to signals table
}
```

**Lines 625-706** - Update INSERT to include sector:
```python
INSERT INTO signals (
    run_id, ticker, rank, overall_score,
    # ... other fields ...
    company_name,
    current_price,
    sector  -- v3.3: Include sector column
) VALUES (...)
```

### Frontend Changes

**Type Definitions:**

1. `frontend/src/types/pipeline.ts` - Added sector to SignalRanking:
```typescript
export interface SignalRanking {
  rank: number;
  ticker: string;
  company_name?: string;
  sector?: string;  // v3.3: Industry sector
  current_price?: number;
  // ... rest of fields
}
```

2. `frontend/src/hooks/usePersistedState.ts` - Added to ColumnVisibility:
```typescript
export interface ColumnVisibility {
  rank: boolean;
  ticker: boolean;
  companyName: boolean;
  sector?: boolean;  // v3.3: Sector column
  currentPrice: boolean;
  // ... rest of columns
}
```

**Component Updates:**

3. `frontend/src/components/dashboard/SignalsTable.tsx`:
   - Added sector header column (line ~302)
   - Added sector cell rendering (line ~539)
   - Sortable by sector

4. `frontend/src/components/dashboard/ColumnVisibilityToggle.tsx`:
   - Added sector to column list
   - Toggleable visibility

5. `frontend/src/components/dashboard/SignalsDashboard.tsx`:
   - Enabled sector by default: `sector: true`

6. `frontend/src/hooks/useSupabaseSignals.ts`:
   - Added sector to SELECT query
   - Maps sector to SignalRanking

---

## 📊 Example Output

### Dashboard Table

| Rank | Ticker | Company Name | **Sector** | Price | Score |
|------|--------|--------------|------------|-------|-------|
| 1 | AAPL | Apple Inc. | **Technology** | $177.25 | 88.5 |
| 2 | GOOGL | Alphabet Inc. | **Communication Services** | $139.40 | 85.2 |
| 3 | JPM | JPMorgan Chase & Co. | **Financial Services** | $158.30 | 82.1 |

---

## 🧪 Testing

### Database Query
```sql
SELECT 
  ticker,
  company_name,
  sector,
  overall_score
FROM signals
WHERE sector IS NOT NULL
ORDER BY overall_score DESC
LIMIT 10;
```

### Expected Result
All new signals should have sector populated from yfinance data.

### Frontend Verification
1. Open https://vanpiq.com
2. Check "Columns" dropdown - Sector should be listed
3. Verify sector column displays between "Company Name" and "Current Price"
4. Confirm sector values display (e.g., "Technology", "Healthcare")

---

## 📝 Deployment Steps

### 1. Apply Migration
```sql
-- Run in Supabase SQL Editor
-- Copy from migrations/014_add_sector_to_signals.sql
```

### 2. Verify Column Exists
```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'signals' AND column_name = 'sector';
```

### 3. Check Backfill Results
```sql
-- Should show sector data copied from performance table
SELECT COUNT(*) FROM signals WHERE sector IS NOT NULL;
```

### 4. Run Pipeline
```powershell
python run_pipeline_and_push.py
```

New signals should have sector populated directly in signals table.

### 5. Verify Frontend
- Deploy frontend changes (auto via Vercel)
- Check dashboard shows sector column
- Toggle visibility works

---

## ✅ Success Criteria

- [ ] Migration 014 applied successfully
- [ ] Sector column exists in signals table
- [ ] Existing signals backfilled from performance table
- [ ] New pipeline run populates sector in signals
- [ ] Frontend displays sector column
- [ ] Column visibility toggle includes sector
- [ ] Sector sorting works
- [ ] Performance table still has sector (for benchmark tracking)

---

## 🔄 Data Flow (v3.3)

### Before (v3.2)
```
Phase 5 → signals table (no sector)
Phase 5 → performance table (with sector, sector_etf)
Frontend → JOIN signals + performance → get sector
```

### After (v3.3)
```
Phase 5 → signals table (with sector)
Phase 5 → performance table (with sector, sector_etf for benchmarking)
Frontend → SELECT from signals → sector available immediately
```

---

## 📚 Related Files

**Database:**
- `migrations/014_add_sector_to_signals.sql` - Schema change

**Backend:**
- `backend/phases/phase5_persist.py` - Extract & persist sector

**Frontend:**
- `frontend/src/types/pipeline.ts` - Type definitions
- `frontend/src/hooks/usePersistedState.ts` - Column visibility type
- `frontend/src/hooks/useSupabaseSignals.ts` - Fetch sector from DB
- `frontend/src/components/dashboard/SignalsTable.tsx` - Display column
- `frontend/src/components/dashboard/ColumnVisibilityToggle.tsx` - Toggle control
- `frontend/src/components/dashboard/SignalsDashboard.tsx` - Default visibility

---

## 💡 Future Enhancements

1. **Sector Filtering** - Add dropdown to filter signals by sector
2. **Sector Analytics** - Group performance by sector
3. **Sector Comparison** - Compare ticker vs sector average
4. **Sector ETF Display** - Show matched sector ETF in table
5. **Color Coding** - Visual sector grouping with colors

---

**Status:** ✅ Ready for deployment  
**Testing:** Local verification needed  
**Risk:** Low (additive change, backward compatible)
