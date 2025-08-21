import sqlite3
from datetime import datetime, timezone

from config.config import DB_PATH
from utils.db import ensure_schema, upsert_run
from processors.weekly_rollup import compute_rollup


def test_weekly_rollup_minimal():
    ensure_schema()
    run_id = datetime.now(timezone.utc).isoformat()
    upsert_run(run_id, started_at=run_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO signals_norm (run_id,ticker,score,rank,trade_type,risk_level,reddit_score,news_score,financial_score,run_datetime) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (run_id, "TEST", 1.0, 1, None, None, None, None, None, run_id))
        conn.execute("INSERT INTO labels (run_id,ticker,window,fwd_return,beat_spy,ready_at) VALUES (?,?,?,?,?,?)",
                     (run_id, "TEST", "3D", 2.0, 1, run_id))
    df = compute_rollup(days=7, windows=["3D"], ks=[1, 5])
    assert not df.empty
    assert set(["run_id", "window", "ic", "p@1", "p@5"]).issubset(df.columns)
