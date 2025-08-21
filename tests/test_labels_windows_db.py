from datetime import datetime, timezone
import sqlite3
import pandas as pd

from config.config import DB_PATH, RETURN_WINDOWS
from utils.db import ensure_schema


def test_labels_windows_correctness(monkeypatch):
    ensure_schema()
    # Prepare one signals row
    run_id = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        # Reset table to isolate the test row
        conn.execute("DROP TABLE IF EXISTS signals")
        conn.execute("CREATE TABLE IF NOT EXISTS signals (dummy TEXT)")
        # Ensure required legacy columns are present; add if missing
        cols = {r[1] for r in conn.execute("PRAGMA table_info(signals)")}
        for col in ["Run ID", "Ticker", "Run Datetime"]:
            if col not in cols:
                conn.execute(f'ALTER TABLE signals ADD COLUMN "{col}"')
        conn.execute("INSERT INTO signals (\"Run ID\", \"Ticker\", \"Run Datetime\") VALUES (?,?,?)", (run_id, "TEST", run_id))

    # Synthetic price series: base 100, increments to simulate returns
    idx = pd.date_range(start=pd.Timestamp(datetime.now().date()), periods=30, freq="B")
    series = pd.Series([100 + i for i in range(len(idx))], index=idx)

    from processors import backtest as bt
    # Ensure our test ticker is allowed by company list filter
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'company_names.csv')
    try:
        import pandas as _pd
        df_comp = _pd.read_csv(csv_path)
        if 'Ticker' not in df_comp.columns:
            df_comp['Ticker'] = []
        if not (df_comp['Ticker'].astype(str).str.upper() == 'TEST').any():
            df_comp = _pd.concat([df_comp, _pd.DataFrame([{'Ticker': 'TEST'}])], ignore_index=True)
            df_comp.to_csv(csv_path, index=False)
    except Exception:
        # Best effort; if this fails the function will drop TEST, so make sure fallback path ignores the filter
        pass

    def fake_fetch_price_series(ticker, start, end):
        return series

    monkeypatch.setattr(bt, "fetch_price_series", fake_fetch_price_series)
    # Force fallback path to use fetch_price_series by making bulk download fail
    def _raise(*args, **kwargs):
        raise Exception("force fallback")
    monkeypatch.setattr(bt.yf, "download", _raise)

    # Run only-labels path (process all; our inserted row will be included)
    bt.enrich_future_returns_in_db(force_all=True, only_labels=True)

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT window, fwd_return FROM labels WHERE run_id=? AND ticker=?", (run_id, "TEST")).fetchall()
    got = {w: v for (w, v) in rows}

    # Expected returns using the same alignment as implementation (start next trading day at searchsorted)
    run_dt = datetime.fromisoformat(run_id.replace("Z", "+00:00"))
    start_time = (run_dt + pd.Timedelta(days=1)).replace(tzinfo=None)
    base_idx = series.index.searchsorted(pd.Timestamp(start_time), side='left')
    assert base_idx < len(series)
    base = float(series.iloc[base_idx])
    for w in RETURN_WINDOWS:
        tgt = base_idx + w
        if tgt < len(series):
            expect = round((float(series.iloc[tgt]) - base) / base * 100, 2)
            assert got.get(f"{w}D") == expect
