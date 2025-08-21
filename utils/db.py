"""
SQLite DB utilities: schema initialization, indexes, and simple UPSERT helpers.

Tables ensured (IF NOT EXISTS):
 - runs(run_id PK, started_at, ended_at, config_json, code_version, notes)
 - prices(ticker, date, open, high, low, close, adj_close, volume, PK(ticker,date))
 - features(run_id, ticker, key, value, as_of, PK(run_id,ticker,key))
 - labels(run_id, ticker, window, fwd_return, beat_spy, ready_at, PK(run_id,ticker,window))
 - signals_norm(run_id, ticker, score, rank, trade_type, risk_level, reddit_score, news_score, financial_score, run_datetime, PK(run_id,ticker))
 - experiments(exp_id PK, run_id, profile, params_json, code_version, started_at, ended_at, notes)
 - metrics(id PK, run_id, name, value, context_json, created_at)

Also attempts to add a helpful index on the legacy wide 'signals' table: ("Run ID", "Ticker").
This will succeed only if that table/columns exist.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, Mapping, Any

from config.config import DB_PATH
try:
    from config.config import DB_CONTRACTS_VALIDATE as _DB_VALIDATE
except Exception:
    _DB_VALIDATE = False
try:
    from utils.contracts import FeatureRow as _FeatureRow, LabelRow as _LabelRow, PriceRow as _PriceRow, SignalNormRow as _SignalNormRow, MetricRow as _MetricRow, ExperimentRow as _ExperimentRow, RunRow as _RunRow
except Exception:
    _FeatureRow = _LabelRow = _PriceRow = _SignalNormRow = _MetricRow = _ExperimentRow = _RunRow = None


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def connect() -> sqlite3.Connection:
    """Return a SQLite connection with pragmatic defaults for better reliability."""
    _ensure_parent_dir(DB_PATH)
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)  # autocommit
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA mmap_size = 30000000000;")  # best-effort; ignored if not supported
    except Exception:
        pass
    return conn


@contextmanager
def get_conn():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema() -> None:
    """Create core tables and indexes if they don't exist."""
    with get_conn() as conn:
        cur = conn.cursor()
        # runs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                config_json TEXT,
                code_version TEXT,
                notes TEXT
            );
            """
        )
        # prices (PK ensures UPSERT target)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                adj_close REAL, volume REAL,
                PRIMARY KEY (ticker, date)
            );
            """
        )
        # features (long form)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                key    TEXT NOT NULL,
                value  REAL,
                as_of  TEXT,
                PRIMARY KEY (run_id, ticker, key)
            );
            """
        )
        # labels
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                window TEXT NOT NULL,
                fwd_return REAL,
                beat_spy   INTEGER,
                ready_at   TEXT,
                PRIMARY KEY (run_id, ticker, window)
            );
            """
        )
        # normalized signals (compact reference separate from legacy wide 'signals')
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signals_norm (
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                score REAL,
                rank INTEGER,
                trade_type TEXT,
                risk_level TEXT,
                reddit_score REAL,
                news_score REAL,
                financial_score REAL,
                run_datetime TEXT,
                PRIMARY KEY (run_id, ticker)
            );
            """
        )
        # experiments
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                exp_id TEXT PRIMARY KEY,
                run_id TEXT,
                profile TEXT,
                params_json TEXT,
                code_version TEXT,
                started_at TEXT,
                ended_at TEXT,
                notes TEXT
            );
            """
        )
        # metrics (id as INTEGER PK for simplicity)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                name TEXT,
                value REAL,
                context_json TEXT,
                created_at TEXT
            );
            """
        )

        # Helpful indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_features_run_ticker ON features(run_id, ticker);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_labels_run_ticker ON labels(run_id, ticker);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_snorm_run_ticker ON signals_norm(run_id, ticker);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_snorm_ticker ON signals_norm(ticker);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_name ON metrics(run_id, name);")

        # Attempt to index legacy wide 'signals' table if columns exist
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_runid_ticker ON signals(\"Run ID\", \"Ticker\");"
            )
        except Exception:
            pass


# === UPSERT helpers ===

def upsert_prices(rows: Iterable[Mapping[str, Any]]) -> int:
    """UPSERT price rows.

    Required keys per row: ticker, date, open, high, low, close, adj_close, volume
    Returns the number of rows processed.
    """
    sql = (
        "INSERT INTO prices (ticker,date,open,high,low,close,adj_close,volume) "
        "VALUES (:ticker,:date,:open,:high,:low,:close,:adj_close,:volume) "
        "ON CONFLICT(ticker,date) DO UPDATE SET "
        "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, "
        "adj_close=excluded.adj_close, volume=excluded.volume"
    )
    rows = list(rows)
    if _DB_VALIDATE and _PriceRow is not None:
        rows = [dict(_PriceRow(**r).dict()) for r in rows]
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(sql, rows)
    return len(rows)


def upsert_run(run_id: str, started_at: str | None = None, ended_at: str | None = None,
               config_json: str | None = None, code_version: str | None = None,
               notes: str | None = None) -> None:
    sql = (
        "INSERT INTO runs (run_id, started_at, ended_at, config_json, code_version, notes) "
        "VALUES (:run_id,:started_at,:ended_at,:config_json,:code_version,:notes) "
        "ON CONFLICT(run_id) DO UPDATE SET "
        "started_at=COALESCE(excluded.started_at, started_at), "
        "ended_at=COALESCE(excluded.ended_at, ended_at), "
        "config_json=COALESCE(excluded.config_json, config_json), "
        "code_version=COALESCE(excluded.code_version, code_version), "
        "notes=COALESCE(excluded.notes, notes)"
    )
    if _DB_VALIDATE and _RunRow is not None:
        _ = _RunRow(run_id=run_id, started_at=started_at, ended_at=ended_at, config_json=config_json, code_version=code_version, notes=notes)
    with get_conn() as conn:
        conn.execute(sql, {
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "config_json": config_json,
            "code_version": code_version,
            "notes": notes,
        })


def upsert_label(run_id: str, ticker: str, window: str, fwd_return: float | None,
                 beat_spy: int | None, ready_at: str | None) -> None:
    sql = (
        "INSERT INTO labels (run_id,ticker,window,fwd_return,beat_spy,ready_at) "
        "VALUES (:run_id,:ticker,:window,:fwd_return,:beat_spy,:ready_at) "
        "ON CONFLICT(run_id,ticker,window) DO UPDATE SET "
        "fwd_return=excluded.fwd_return, beat_spy=excluded.beat_spy, ready_at=excluded.ready_at"
    )
    if _DB_VALIDATE and _LabelRow is not None:
        _ = _LabelRow(run_id=run_id, ticker=ticker, window=window, fwd_return=fwd_return, beat_spy=beat_spy, ready_at=ready_at)
    with get_conn() as conn:
        conn.execute(sql, {
            "run_id": run_id,
            "ticker": ticker,
            "window": window,
            "fwd_return": fwd_return,
            "beat_spy": beat_spy,
            "ready_at": ready_at,
        })


def upsert_feature(run_id: str, ticker: str, key: str, value: float | None, as_of: str | None) -> None:
    sql = (
        "INSERT INTO features (run_id,ticker,key,value,as_of) "
        "VALUES (:run_id,:ticker,:key,:value,:as_of) "
        "ON CONFLICT(run_id,ticker,key) DO UPDATE SET value=excluded.value, as_of=excluded.as_of"
    )
    if _DB_VALIDATE and _FeatureRow is not None:
        _ = _FeatureRow(run_id=run_id, ticker=ticker, key=key, value=value, as_of=as_of)
    with get_conn() as conn:
        conn.execute(sql, {"run_id": run_id, "ticker": ticker, "key": key, "value": value, "as_of": as_of})


def insert_metric(run_id: str, name: str, value: float | None, context_json: str | None, created_at: str) -> None:
    if _DB_VALIDATE and _MetricRow is not None:
        _ = _MetricRow(run_id=run_id, name=name, value=value, context_json=context_json, created_at=created_at)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO metrics (run_id,name,value,context_json,created_at) VALUES (?,?,?,?,?)",
            (run_id, name, value, context_json, created_at)
        )


def insert_experiment(exp_id: str, run_id: str | None, profile: str | None,
                      params_json: str | None, code_version: str | None,
                      started_at: str | None, ended_at: str | None, notes: str | None) -> None:
    if _DB_VALIDATE and _ExperimentRow is not None:
        _ = _ExperimentRow(exp_id=exp_id, run_id=run_id, profile=profile, params_json=params_json, code_version=code_version, started_at=started_at, ended_at=ended_at, notes=notes)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO experiments (exp_id,run_id,profile,params_json,code_version,started_at,ended_at,notes)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(exp_id) DO UPDATE SET
              run_id=excluded.run_id,
              profile=excluded.profile,
              params_json=excluded.params_json,
              code_version=excluded.code_version,
              started_at=excluded.started_at,
              ended_at=excluded.ended_at,
              notes=excluded.notes
            """,
            (exp_id, run_id, profile, params_json, code_version, started_at, ended_at, notes)
        )


def upsert_signal_norm(run_id: str, ticker: str, score: float | None, rank: int | None,
                       trade_type: str | None, risk_level: str | None,
                       reddit_score: float | None, news_score: float | None,
                       financial_score: float | None, run_datetime: str | None) -> None:
    sql = (
        "INSERT INTO signals_norm (run_id,ticker,score,rank,trade_type,risk_level,reddit_score,news_score,financial_score,run_datetime) "
        "VALUES (:run_id,:ticker,:score,:rank,:trade_type,:risk_level,:reddit_score,:news_score,:financial_score,:run_datetime) "
        "ON CONFLICT(run_id,ticker) DO UPDATE SET "
        "score=excluded.score, rank=excluded.rank, trade_type=excluded.trade_type, risk_level=excluded.risk_level, "
        "reddit_score=excluded.reddit_score, news_score=excluded.news_score, financial_score=excluded.financial_score, run_datetime=excluded.run_datetime"
    )
    if _DB_VALIDATE and _SignalNormRow is not None:
        _ = _SignalNormRow(run_id=run_id, ticker=ticker, score=score, rank=rank, trade_type=trade_type, risk_level=risk_level, reddit_score=reddit_score, news_score=news_score, financial_score=financial_score, run_datetime=run_datetime)
    with get_conn() as conn:
        conn.execute(sql, {
            "run_id": run_id,
            "ticker": ticker,
            "score": score,
            "rank": rank,
            "trade_type": trade_type,
            "risk_level": risk_level,
            "reddit_score": reddit_score,
            "news_score": news_score,
            "financial_score": financial_score,
            "run_datetime": run_datetime,
        })
