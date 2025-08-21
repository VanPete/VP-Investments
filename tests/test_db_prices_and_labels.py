import os
import sqlite3
from datetime import datetime

from utils.db import ensure_schema, upsert_prices, upsert_label
from config.config import DB_PATH


def setup_module(module):
    # Ensure schema exists in the shared DB
    ensure_schema()


def test_price_upsert_dedupe_overwrite(tmp_path):
    # Insert the same (ticker, date) twice and ensure second write overwrites values
    rows1 = [{
        "ticker": "TEST",
        "date": "2024-01-02",
        "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
        "adj_close": 1.4, "volume": 1000.0,
    }]
    rows2 = [{
        "ticker": "TEST",
        "date": "2024-01-02",
        "open": 10.0, "high": 20.0, "low": 5.0, "close": 15.0,
        "adj_close": 14.0, "volume": 2000.0,
    }]
    upsert_prices(rows1)
    upsert_prices(rows2)
    with sqlite3.connect(DB_PATH) as conn:
        r = conn.execute("SELECT open, high, low, close, adj_close, volume FROM prices WHERE ticker='TEST' AND date='2024-01-02'").fetchone()
        assert r == (10.0, 20.0, 5.0, 15.0, 14.0, 2000.0)


def test_labels_upsert_window_correctness():
    # Insert a simple label for a known run/ticker/window and ensure upsert works idempotently
    run_id = "run-test-1"
    upsert_label(run_id, "TEST", "3D", 5.0, 1, datetime.utcnow().isoformat())
    upsert_label(run_id, "TEST", "3D", 6.0, 0, datetime.utcnow().isoformat())
    with sqlite3.connect(DB_PATH) as conn:
        r = conn.execute("SELECT fwd_return, beat_spy FROM labels WHERE run_id=? AND ticker=? AND window=?", (run_id, "TEST", "3D")).fetchone()
        assert r == (6.0, 0)
