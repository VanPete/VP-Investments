# Migration 021 Verification & Admin Backend Access

## ✅ Status: Backend Server Running

**Backend API Server**: http://127.0.0.1:8000

## 📋 Admin Endpoints Available

### 1. API Documentation (Interactive Swagger UI)
**URL**: http://127.0.0.1:8000/docs

This provides an interactive interface to test all admin endpoints.

### 2. Admin Endpoints

#### List All Pipeline Runs
```
GET http://127.0.0.1:8000/api/admin/runs/list
```

#### Get Specific Run Details
```
GET http://127.0.0.1:8000/api/admin/runs/{run_id}
```

#### Delete Single Run (with preview)
```
POST http://127.0.0.1:8000/api/admin/runs/delete
Body: {
  "run_id": "your-run-id-here",
  "confirm": false  // Set to true to actually delete
}
```

#### Bulk Delete Multiple Runs
```
POST http://127.0.0.1:8000/api/admin/runs/bulk-delete
Body: {
  "run_ids": ["run-id-1", "run-id-2", "run-id-3"],
  "confirm": false  // Set to true to actually delete
}
```

## 🔍 Verifying CASCADE DELETE Migration

### Step 1: Check Constraints in Supabase

1. Go to: https://supabase.com/dashboard
2. Select your project
3. Click **SQL Editor** in the left menu
4. Run this query:

```sql
SELECT
    tc.table_name,
    tc.constraint_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    rc.delete_rule
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
JOIN information_schema.referential_constraints AS rc
    ON rc.constraint_name = tc.constraint_name
    AND rc.constraint_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_schema = 'public'
AND tc.table_name IN (
    'analytics',
    'signals', 
    'performance',
    'signals_technical',
    'signals_fundamental',
    'signals_news_macro',
    'signals_social_alternative',
    'signals_risk_stability',
    'signals_institutional_smart_money'
)
ORDER BY tc.table_name, tc.constraint_name;
```

### Expected Results

All 9 constraints should show `delete_rule = 'CASCADE'`:

| Table | Constraint | Expected Result |
|-------|-----------|----------------|
| analytics | analytics_run_id_fkey | CASCADE |
| signals | fk_signals_run | CASCADE |
| performance | performance_signal_id_fkey | CASCADE |
| signals_technical | signals_technical_signal_id_fkey | CASCADE |
| signals_fundamental | signals_fundamental_signal_id_fkey | CASCADE |
| signals_news_macro | signals_news_macro_signal_id_fkey | CASCADE |
| signals_social_alternative | signals_social_alternative_signal_id_fkey | CASCADE |
| signals_risk_stability | signals_risk_stability_signal_id_fkey | CASCADE |
| signals_institutional_smart_money | signals_institutional_smart_money_signal_id_fkey | CASCADE |

### Step 2: Test CASCADE Deletion (Optional)

Find old test runs from 11/1 or 11/2:

```sql
-- 1. List recent runs
SELECT id, run_timestamp, total_tickers 
FROM signal_runs 
ORDER BY run_timestamp DESC 
LIMIT 20;

-- 2. Preview what will be deleted (replace YOUR_RUN_ID)
SELECT
  (SELECT COUNT(*) FROM signals WHERE run_id = 'YOUR_RUN_ID') as signals_count,
  (SELECT COUNT(*) FROM analytics WHERE run_id = 'YOUR_RUN_ID') as analytics_count,
  (SELECT COUNT(*) FROM performance p 
   JOIN signals s ON p.signal_id = s.id 
   WHERE s.run_id = 'YOUR_RUN_ID') as performance_count;

-- 3. Delete the run (replace YOUR_RUN_ID)
DELETE FROM signal_runs WHERE id = 'YOUR_RUN_ID';

-- 4. Verify cascade worked (should all return 0)
SELECT COUNT(*) FROM signals WHERE run_id = 'YOUR_RUN_ID';
SELECT COUNT(*) FROM analytics WHERE run_id = 'YOUR_RUN_ID';
```

## 🎯 Using the Admin API for Deletion

### Via Swagger UI (Recommended)

1. Go to: http://127.0.0.1:8000/docs
2. Find the **admin** section
3. Click on **GET /api/admin/runs/list** to see all runs
4. Click **Try it out** → **Execute**
5. Copy a `run_id` from the response
6. Use **POST /api/admin/runs/delete** to delete:
   - First with `confirm: false` to preview
   - Then with `confirm: true` to actually delete

### Via Python Script

```python
import requests

# List runs
response = requests.get("http://127.0.0.1:8000/api/admin/runs/list")
runs = response.json()
print(f"Found {runs['total_count']} runs")

# Preview deletion
run_id = runs['runs'][0]['run_id']  # First run
response = requests.post(
    "http://127.0.0.1:8000/api/admin/runs/delete",
    json={"run_id": run_id, "confirm": False}
)
preview = response.json()
print(f"Would delete: {preview['deleted_counts']}")

# Actually delete (uncomment to execute)
# response = requests.post(
#     "http://127.0.0.1:8000/api/admin/runs/delete",
#     json={"run_id": run_id, "confirm": True}
# )
# result = response.json()
# print(f"Deleted: {result['message']}")
```

## 📝 Next Steps

1. ✅ **Backend server is running** at http://127.0.0.1:8000
2. ⏳ **Verify migration** - Run the SQL query above in Supabase SQL Editor
3. ⏳ **Delete test runs** - Use admin API to clean up 11/1 and 11/2 runs
4. ⏳ **Run pipeline** - Test that everything works after deletion
5. ⏳ **Update frontend** - Add interval selector for Phase 7 analytics

## 🔗 Quick Links

- **API Docs**: http://127.0.0.1:8000/docs
- **OpenAPI JSON**: http://127.0.0.1:8000/openapi.json
- **Frontend**: http://localhost:3000
- **Supabase Dashboard**: https://supabase.com/dashboard
