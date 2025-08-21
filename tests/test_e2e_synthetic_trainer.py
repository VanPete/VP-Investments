import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config.config import DB_PATH
from utils.db import ensure_schema, upsert_run, upsert_feature


def test_e2e_enrich_and_train(monkeypatch):
    ensure_schema()
    # Fresh minimal signals table
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS signals")
        conn.execute("CREATE TABLE signals (\"Run ID\" TEXT, \"Ticker\" TEXT, \"Run Datetime\" TEXT)")
    # One run with two tickers
    run_id = datetime.now(timezone.utc).isoformat()
    start = datetime.now(timezone.utc) - timedelta(days=15)
    end = start + timedelta(minutes=1)
    upsert_run(run_id, started_at=start.isoformat(), ended_at=end.isoformat())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO signals (\"Run ID\",\"Ticker\",\"Run Datetime\") VALUES (?,?,?)", (run_id, "AAA", (start - timedelta(days=10)).isoformat()))
        conn.execute("INSERT INTO signals (\"Run ID\",\"Ticker\",\"Run Datetime\") VALUES (?,?,?)", (run_id, "BBB", (start - timedelta(days=9)).isoformat()))

    # Deterministic price series for both tickers + SPY
    idx = pd.date_range(start=start.date(), periods=40, freq="B")
    series = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    from processors import backtest as bt

    def fake_fetch_price_series(ticker, s, e):
        return series

    monkeypatch.setattr(bt, "fetch_price_series", fake_fetch_price_series)
    # Force fallback path for stability in CI
    monkeypatch.setattr(bt.yf, "download", lambda *a, **k: (_ for _ in ()).throw(Exception("force fallback")))

    # Ensure tickers pass the valid universe filter
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'company_names.csv')
    try:
        import pandas as _pd
        df_comp = _pd.read_csv(csv_path)
        if 'Ticker' not in df_comp.columns:
            df_comp['Ticker'] = []
        need = {"AAA", "BBB"}
        have = set(df_comp['Ticker'].astype(str).str.upper())
        add = list(need - have)
        if add:
            df_comp = _pd.concat([df_comp, _pd.DataFrame([{ 'Ticker': t } for t in add])], ignore_index=True)
            df_comp.to_csv(csv_path, index=False)
    except Exception:
        pass

    # Label only, writes to labels table for run_id
    bt.enrich_future_returns_in_db(force_all=True, only_labels=True)

    # Minimal features for training
    for t in ("AAA", "BBB"):
        upsert_feature(run_id, t, "Reddit Sentiment", 0.1, start.isoformat())
        upsert_feature(run_id, t, "News Sentiment", 0.2, start.isoformat())
        upsert_feature(run_id, t, "Momentum 30D %", 1.5, start.isoformat())

    # Load and train using helpers (no CLI)
    from processors.scoring_trainer import load_backtest_data, extract_features_targets
    df = load_backtest_data(DB_PATH, "3D Return")
    assert not df.empty
    X, y, feats = extract_features_targets(df, "3D Return")
    assert not X.empty and not y.empty and len(feats) > 0
    # Build a no-CV pipeline for tiny deterministic dataset
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    preproc = ColumnTransformer(transformers=[("num", numeric_transformer, slice(0, None))])
    pipe = Pipeline(steps=[("preprocessor", preproc), ("model", Ridge(alpha=1.0))])
    pipe.fit(X.values, y.values)
    model = pipe.named_steps["model"]
    assert hasattr(model, "coef_")
