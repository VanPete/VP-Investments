# Schema Overview

Normalized SQLite schema stored at `outputs/backtest.db`.

## Tables

- runs
  - run_id TEXT PRIMARY KEY
  - started_at TEXT
  - ended_at TEXT
  - notes TEXT

- prices
  - ticker TEXT
  - date TEXT
  - open REAL
  - high REAL
  - low REAL
  - close REAL
  - adj_close REAL
  - volume REAL
  - PRIMARY KEY (ticker, date)

- features
  - run_id TEXT
  - ticker TEXT
  - key TEXT
  - value REAL
  - as_of TEXT
  - PRIMARY KEY (run_id, ticker, key)

- labels
  - run_id TEXT
  - ticker TEXT
  - window TEXT
  - fwd_return REAL
  - beat_spy REAL
  - ready_at TEXT
  - PRIMARY KEY (run_id, ticker, window)

- signals_norm
  - run_id TEXT
  - ticker TEXT
  - score REAL
  - rank INTEGER
  - trade_type TEXT
  - risk_level TEXT
  - subscores_json TEXT
  - PRIMARY KEY (run_id, ticker)

- experiments
  - id TEXT PRIMARY KEY
  - run_id TEXT
  - profile TEXT
  - params_json TEXT
  - code_version TEXT
  - started_at TEXT
  - ended_at TEXT
  - notes TEXT

- metrics
  - id TEXT PRIMARY KEY
  - run_id TEXT
  - name TEXT
  - value REAL
  - context_json TEXT
  - created_at TEXT

## Indexes (recommended)

- prices(date)
- features(as_of)
- labels(window)
- metrics(name, created_at)

## Notes

- Timestamps in UTC ISO 8601.
- Use WAL mode for better concurrency.
- Avoid wide tables in favor of normalized joins when training or reporting.
