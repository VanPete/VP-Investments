"""Check all analytics rows and runs"""
import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect(
        "postgresql://postgres.rdkxwoqevjicupmefbem:1qaz1QAZ2wsx2WSX@aws-1-us-east-2.pooler.supabase.com:6543/postgres",
        statement_cache_size=0
    )
    
    try:
        print("=" * 80)
        print("SIGNALS RUNS")
        print("=" * 80)
        
        # Check all signal runs
        runs = await conn.fetch("""
            SELECT id, run_timestamp, total_tickers, created_at
            FROM signal_runs
            ORDER BY run_timestamp DESC
            LIMIT 10;
        """)
        
        print(f"\nTotal signal runs: {len(runs)}")
        for i, run in enumerate(runs, 1):
            print(f"{i}. {run['id'][:8]}... - {run['run_timestamp']} ({run['total_tickers']} tickers)")
        
        print("\n" + "=" * 80)
        print("ANALYTICS ROWS")
        print("=" * 80)
        
        # Check all analytics rows
        analytics = await conn.fetch("""
            SELECT run_id, total_signals, created_at
            FROM analytics
            ORDER BY created_at DESC;
        """)
        
        print(f"\nTotal analytics rows: {len(analytics)}")
        for i, row in enumerate(analytics, 1):
            print(f"{i}. {row['run_id'][:8] if row['run_id'] else 'NULL'}... - {row['created_at']} ({row['total_signals']} signals)")
        
        print("\n" + "=" * 80)
        print("MISMATCH CHECK")
        print("=" * 80)
        
        if len(runs) > len(analytics):
            print(f"\n⚠️  We have {len(runs)} signal runs but only {len(analytics)} analytics rows!")
            print(f"Missing analytics for {len(runs) - len(analytics)} runs.")
            
            # Find runs without analytics
            analytics_run_ids = set(str(a['run_id']) for a in analytics if a['run_id'])
            runs_run_ids = set(str(r['id']) for r in runs)
            missing = runs_run_ids - analytics_run_ids
            
            if missing:
                print(f"\nRuns missing analytics:")
                for run_id in list(missing)[:5]:
                    run = next((r for r in runs if str(r['id']) == run_id), None)
                    if run:
                        print(f"  - {run_id[:8]}... ({run['run_timestamp']})")
        else:
            print(f"\n✅ All runs have analytics! ({len(analytics)} rows)")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
