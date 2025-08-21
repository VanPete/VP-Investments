from datetime import datetime, timedelta, timezone
import sqlite3

from utils.db import ensure_schema, upsert_run, upsert_feature
from config.config import DB_PATH


def test_feature_snapshot_asof_within_run_window():
    ensure_schema()
    start = datetime.now(timezone.utc)
    end = start + timedelta(minutes=1)
    run_id = start.isoformat()
    upsert_run(run_id, started_at=run_id)
    as_of = datetime.now(timezone.utc).isoformat()
    upsert_feature(run_id, "TEST", "Reddit Sentiment", 1.23, as_of)
    # Set ended_at after insert
    upsert_run(run_id, ended_at=end.isoformat())
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT as_of FROM features WHERE run_id=? AND ticker=? AND key=?", (run_id, "TEST", "Reddit Sentiment")).fetchone()
        assert row is not None
        as_of_db = datetime.fromisoformat(row[0].replace("Z", "+00:00")) if row[0] else None
        assert as_of_db is not None
        # as_of should be >= started_at and <= ended_at
        assert as_of_db >= start.replace(tzinfo=timezone.utc)
        assert as_of_db <= end.replace(tzinfo=timezone.utc)
