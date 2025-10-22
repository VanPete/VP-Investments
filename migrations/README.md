# Phase 5 Migration - Manual Execution Guide

## Issue
The Supabase Python SDK doesn't support raw SQL execution. We need to execute the migration SQL directly via the Supabase dashboard.

## Solution: Execute Migration via Supabase SQL Editor

### Step 1: Open Supabase SQL Editor
Go to: https://supabase.com/dashboard/project/rdkxwoqevjicupmefbem/sql/new

### Step 2: Copy and Paste the Migration SQL
Open the file: `migrations/001_phase5_core_schema.sql`
Copy the entire content and paste it into the SQL Editor

### Step 3: Run the Migration
Click "RUN" button in the SQL Editor

### Step 4: Verify Tables Created
After running, you should see 8 new tables in your database:
- signals
- signal_runs
- signals_technical
- signals_fundamental
- signals_news_macro
- signals_social_alternative
- signals_risk_stability
- signals_institutional_smart_money

## Alternative: Install psycopg2 for Direct PostgreSQL Access

If you want to run migrations from Python, install psycopg2:

```bash
pip install psycopg2-binary
```

Then we can modify the SupabaseInterface to use psycopg2 instead of asyncpg for migrations.

## Recommended Approach
**Use the Supabase SQL Editor** - it's the simplest and most reliable way to run migrations.
