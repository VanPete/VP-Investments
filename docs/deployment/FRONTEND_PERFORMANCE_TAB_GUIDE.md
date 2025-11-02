# Frontend Performance Tab - Development Guide

**Backend Status:** ✅ **READY** (Phase 6 production verified)  
**Estimated Time:** 12 hours  
**Priority:** HIGH

---

## 📊 Backend Data Structure

### API Endpoint (To Be Created)
```typescript
GET /api/performance/:signal_id/horizons

Response:
{
  signal_id: string;
  ticker: string;
  baseline_date: string;
  baseline_price: number;
  
  // Header data
  market_cap: number;
  beta: number;
  sector: string;
  
  // 7-horizon grid
  horizons: [
    {
      interval: "1d",
      days: 1,
      status: "complete" | "pending" | "in_progress",
      
      // Returns
      ticker_return: number | null,
      spy_return: number | null,
      qqq_return: number | null,
      sector_return: number | null,
      
      // Alpha (auto-calculated)
      alpha_vs_spy: number | null,
      alpha_vs_qqq: number | null,
      alpha_vs_sector: number | null,
      
      // Countdown
      eligible_at: string, // ISO timestamp when interval becomes eligible
      hours_remaining: number | null
    },
    // ... repeat for 3d, 7d, 10d, 14d, 30d, 90d
  ]
}
```

### Database Query (For Backend Implementation)
```sql
SELECT 
  p.signal_id,
  s.ticker,
  p.baseline_date,
  p.baseline_price,
  p.intervals_completed,
  
  -- Signal metadata
  s.market_cap,
  s.beta,
  p.sector,
  
  -- 1d interval
  p.return_1d, p.spy_return_1d, p.qqq_return_1d, p.sector_return_1d,
  p.alpha_1d, p.qqq_alpha_1d, p.sector_alpha_1d,
  
  -- 3d interval
  p.return_3d, p.spy_return_3d, p.qqq_return_3d, p.sector_return_3d,
  p.alpha_3d, p.qqq_alpha_3d, p.sector_alpha_3d,
  
  -- ... (7d, 10d, 14d, 30d, 90d)
  
FROM performance p
INNER JOIN signals s ON p.signal_id = s.id
WHERE p.signal_id = $1
```

---

## 🎨 UI Component Structure

### 1. Performance Tab Container
```tsx
<PerformanceTab>
  <SignalHeader 
    ticker={ticker}
    marketCap={marketCap}
    beta={beta}
    sector={sector}
  />
  
  <BenchmarkToggle 
    selected={benchmark} // "SPY" | "QQQ"
    onChange={setBenchmark}
  />
  
  <HorizonGrid 
    horizons={horizons}
    benchmark={benchmark}
  />
  
  <HorizonQualitySummary 
    completedCount={completedCount}
    pendingCount={pendingCount}
  />
</PerformanceTab>
```

### 2. Horizon Grid (7 columns)
```tsx
<HorizonGrid>
  {horizons.map(horizon => (
    <HorizonCard key={horizon.interval}>
      <IntervalLabel>{horizon.interval}</IntervalLabel>
      
      {horizon.status === "complete" ? (
        <>
          <ReturnDisplay 
            ticker={horizon.ticker_return}
            benchmark={horizon[`${benchmark.toLowerCase()}_return`]}
          />
          
          <AlphaSparkline 
            alpha={horizon[`alpha_vs_${benchmark.toLowerCase()}`]}
          />
          
          <AlphaValue 
            value={horizon[`alpha_vs_${benchmark.toLowerCase()}`]}
            positive={horizon[`alpha_vs_${benchmark.toLowerCase()}`] > 0}
          />
        </>
      ) : (
        <>
          <PendingBadge />
          <CountdownTimer 
            hoursRemaining={horizon.hours_remaining}
          />
        </>
      )}
    </HorizonCard>
  ))}
</HorizonGrid>
```

---

## 🔧 Implementation Checklist

### Phase 1: Backend API Endpoint (2-3 hours)
- [ ] Create `/api/performance/:signal_id/horizons` endpoint
- [ ] Query performance + signals tables
- [ ] Transform data into 7-horizon array
- [ ] Calculate `hours_remaining` for pending intervals
- [ ] Add error handling for missing signal_id
- [ ] Test with existing signals

### Phase 2: Frontend Data Layer (1-2 hours)
- [ ] Create `usePerformanceData(signalId)` hook
- [ ] Implement API call with error handling
- [ ] Add loading state
- [ ] Add caching (react-query or SWR)
- [ ] Handle NULL values gracefully

### Phase 3: UI Components (6-8 hours)

#### SignalHeader Component (30 min)
- [ ] Display ticker, market cap, beta, sector
- [ ] Format market cap (e.g., "$1.2B")
- [ ] Style with Tailwind/shadcn

#### BenchmarkToggle Component (30 min)
- [ ] Toggle button: SPY / QQQ
- [ ] Persist selection in localStorage
- [ ] Update grid when changed

#### HorizonGrid Component (3-4 hours)
- [ ] 7-column responsive grid
- [ ] Interval labels (1d, 3d, 7d, 10d, 14d, 30d, 90d)
- [ ] Return display with color coding:
  - Green for positive returns
  - Red for negative returns
  - Gray for NULL/pending
- [ ] Handle NULL values (show "Pending")

#### AlphaSparkline Component (1-2 hours)
- [ ] Mini chart showing alpha trend
- [ ] Use recharts or victory-native
- [ ] Show only for completed intervals
- [ ] Color based on positive/negative

#### CountdownTimer Component (1 hour)
- [ ] Display "Available in X hours"
- [ ] Update every minute
- [ ] Show as badge

#### HorizonQualitySummary Component (30 min)
- [ ] Summary: "3/7 intervals complete"
- [ ] Progress bar or badge
- [ ] Show next eligible interval

### Phase 4: Testing & Polish (1-2 hours)
- [ ] Test with fresh signals (all NULL)
- [ ] Test with partial data (some intervals complete)
- [ ] Test with complete signals (all 7 intervals)
- [ ] Responsive design (mobile, tablet, desktop)
- [ ] Loading states
- [ ] Error states (signal not found, API error)

---

## 🎯 Key Business Logic

### Countdown Calculation
```typescript
function calculateHoursRemaining(baselineDate: string, intervalDays: number): number | null {
  const baseline = new Date(baselineDate);
  const now = new Date();
  const eligibleAt = new Date(baseline.getTime() + intervalDays * 24 * 60 * 60 * 1000);
  
  if (now >= eligibleAt) {
    return null; // Already eligible
  }
  
  const msRemaining = eligibleAt.getTime() - now.getTime();
  const hoursRemaining = Math.ceil(msRemaining / (60 * 60 * 1000));
  
  return hoursRemaining;
}
```

### Status Determination
```typescript
function getHorizonStatus(
  intervalsCompleted: number[],
  intervalDays: number,
  hoursRemaining: number | null
): "complete" | "pending" | "in_progress" {
  if (intervalsCompleted.includes(intervalDays)) {
    return "complete";
  }
  
  if (hoursRemaining === null) {
    return "in_progress"; // Eligible but not yet calculated
  }
  
  return "pending";
}
```

### Alpha Color Coding
```typescript
function getAlphaColor(alpha: number | null): string {
  if (alpha === null) return "text-gray-400";
  if (alpha > 0) return "text-green-600";
  if (alpha < 0) return "text-red-600";
  return "text-gray-600";
}
```

---

## 📝 Sample Data for Testing

### Fresh Signal (< 1 day old)
```json
{
  "ticker": "AAPL",
  "baseline_date": "2025-11-01T09:53:36",
  "intervals_completed": [],
  "horizons": [
    {
      "interval": "1d",
      "status": "pending",
      "ticker_return": null,
      "spy_return": null,
      "alpha_vs_spy": null,
      "hours_remaining": 15
    },
    // All other intervals: status="pending", hours_remaining calculated
  ]
}
```

### Partial Signal (3 days old)
```json
{
  "ticker": "MSFT",
  "baseline_date": "2025-10-29T14:20:00",
  "intervals_completed": [1, 3],
  "horizons": [
    {
      "interval": "1d",
      "status": "complete",
      "ticker_return": 2.5,
      "spy_return": 0.8,
      "alpha_vs_spy": 1.7,
      "hours_remaining": null
    },
    {
      "interval": "3d",
      "status": "complete",
      "ticker_return": 5.2,
      "spy_return": 1.5,
      "alpha_vs_spy": 3.7,
      "hours_remaining": null
    },
    {
      "interval": "7d",
      "status": "pending",
      "ticker_return": null,
      "spy_return": null,
      "alpha_vs_spy": null,
      "hours_remaining": 96
    },
    // 10d, 14d, 30d, 90d: all pending
  ]
}
```

### Complete Signal (90+ days old)
```json
{
  "ticker": "GOOGL",
  "baseline_date": "2025-08-01T10:00:00",
  "intervals_completed": [1, 3, 7, 10, 14, 30, 90],
  "horizons": [
    // All 7 intervals: status="complete", data populated
  ]
}
```

---

## 🚀 Getting Started

### Step 1: Create API Endpoint
```bash
# Create file: backend/api/routes/performance.py
touch backend/api/routes/performance.py
```

```python
from fastapi import APIRouter, HTTPException
from backend.storage.database import get_supabase_database

router = APIRouter(prefix="/api/performance", tags=["performance"])

@router.get("/{signal_id}/horizons")
async def get_performance_horizons(signal_id: str):
    """Get 7-horizon performance grid for a signal."""
    db = await get_supabase_database()
    
    # Query performance data
    result = db.client.table('performance').select('''
        *,
        signals!inner(ticker, market_cap, beta)
    ''').eq('signal_id', signal_id).execute()
    
    if not result.data:
        raise HTTPException(404, "Performance data not found")
    
    # Transform into 7-horizon structure
    # (implementation here)
    
    return {"signal_id": signal_id, "horizons": horizons}
```

### Step 2: Create Frontend Hook
```bash
# Create file: frontend/src/hooks/usePerformanceData.ts
touch frontend/src/hooks/usePerformanceData.ts
```

```typescript
import { useQuery } from '@tanstack/react-query';

export function usePerformanceData(signalId: string) {
  return useQuery({
    queryKey: ['performance', signalId],
    queryFn: async () => {
      const res = await fetch(`/api/performance/${signalId}/horizons`);
      if (!res.ok) throw new Error('Failed to fetch performance data');
      return res.json();
    },
    staleTime: 60000, // 1 minute
  });
}
```

### Step 3: Build UI Component
```bash
# Create file: frontend/src/components/PerformanceTab.tsx
touch frontend/src/components/PerformanceTab.tsx
```

```typescript
import { usePerformanceData } from '@/hooks/usePerformanceData';

export function PerformanceTab({ signalId }: { signalId: string }) {
  const { data, isLoading, error } = usePerformanceData(signalId);
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage />;
  
  return (
    <div className="space-y-6">
      <SignalHeader {...data} />
      <BenchmarkToggle />
      <HorizonGrid horizons={data.horizons} />
      <HorizonQualitySummary horizons={data.horizons} />
    </div>
  );
}
```

---

## 📚 Resources

### Backend
- Performance table schema: `backend/storage/supabase.sql`
- Phase 6 implementation: `backend/phases/phase6_performance.py`
- Phase 6 verification: `docs/deployment/PHASE_6_PRODUCTION_VERIFICATION.md`

### Frontend
- shadcn/ui components: https://ui.shadcn.com/
- Recharts for sparklines: https://recharts.org/
- React Query: https://tanstack.com/query/latest

### Design Reference
- See: `docs/deployment/PHASE_6_ASSESSMENT.md` section 6
- Figma: (if available)

---

## ✅ Success Criteria

- [ ] API endpoint returns correct 7-horizon data
- [ ] UI displays all 7 intervals in grid format
- [ ] SPY/QQQ toggle switches alpha calculations
- [ ] Pending intervals show countdown timer
- [ ] Complete intervals show alpha sparkline
- [ ] NULL values handled gracefully
- [ ] Responsive on mobile/tablet/desktop
- [ ] Loading and error states work correctly

---

**Ready to build!** 🚀 Backend is fully functional, all data is available. Focus on creating a clean, intuitive UI that helps users understand signal performance across different time horizons.
