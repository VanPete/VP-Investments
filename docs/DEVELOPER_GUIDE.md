# Developer Guide

This guide helps you extend and operate the VP Investments pipeline locally. It complements README.md and focuses on data contracts, DB schema, and common workflows.

## Data Contracts (normalized)

Tables in `outputs/backtest.db`:

- runs(run_id PK, started_at, ended_at, notes)
- prices(ticker, date, open, high, low, close, adj_close, volume, PRIMARY KEY(ticker, date))
- features(run_id, ticker, key, value, as_of, PRIMARY KEY(run_id, ticker, key))
- labels(run_id, ticker, window, fwd_return, beat_spy, ready_at, PRIMARY KEY(run_id, ticker, window))
- signals_norm(run_id, ticker, score, rank, trade_type, risk_level, subscores_json, PRIMARY KEY(run_id, ticker))
- experiments(id PK, run_id, profile, params_json, code_version, started_at, ended_at, notes)
- metrics(id PK, run_id, name, value, context_json, created_at)

Notes:

- Use UTC timestamps in ISO 8601 in DB.
- Keep features atomic (one key/value per row). Prefer numeric values.
- Labels window strings are like "1D", "3D", "7D", "10D".

## Common Workflows

- End-to-end run once:
  - `python run_all.py --once --stream --timeout 1800`
- Schedule:
  - set env `SCHED_ENABLED=1` then `python run_all.py`
- Weekly rollup:
  - `python processors/weekly_rollup.py --days 7 --windows 3D,10D --ks 10,20`
- Train weights:
  - `python processors/scoring_trainer.py --target "3D Return" --model ridge`

## Adding a New Feature

1) Implement the computation where signals are created, producing a numeric value per (run_id, ticker).
2) Upsert via `utils.db.upsert_feature(run_id, ticker, key, value, as_of)`.
3) Add the feature name to `config/config.py` under `FEATURE_TOGGLES = {"Your Feature": True, ...}`.
4) Ensure it appears in analytics by re-running `run_all.py` or backtest/trainer.

## Adding a New Signal Field (signals_norm)

- Extend the object you persist for each ticker with new subscores or metadata. Persist through `utils.db.upsert_signal_norm(run_id, ticker, score, rank, trade_type, risk_level, subscores_json)`.
- Keep subscores in a compact JSON field.

## Labels-only Backfill

- To only compute labels (forward returns) for recent runs without re-generating charts:
  - `python processors/backtest.py --only-labels --only-pending`

## Error Budgets & Observability

- Per-run component reliability is aggregated to `outputs/(run)/References/Fetch_Reliability.csv` and `Error_Budgets.csv`.
- A 30-day trend is exported to `outputs/tables/Error_Budget_Trend.csv` and plotted.
- Configure target with `ERROR_BUDGET_SUCCESS_TARGET`.

### Interpreting the Feature Standardization Audit

- The audit computes per-run cross-sectional mean (cs_mean) and std (cs_std) for each feature and averages across runs.
- Ideal standardized features have avg_cs_mean ≈ 0 and avg_cs_std ≈ 1; deviations are reported as center_bias and scale_bias.
- If scale_bias is large, consider adding/adjusting scaling in your feature creation, or standardize within the trainer pipeline.

## Minimal Code Contract Snippets

- Feature upsert (Python):
  - `from utils.db import upsert_feature`
  - `upsert_feature(run_id, ticker, "My Feature", value, as_of_ts)`
- Metric logging:
  - `from utils.db import insert_metric`
  - `insert_metric(run_id, "fetch_success_reddit", 0.97, None, ts)`

## Troubleshooting

- Missing tables: ensure `utils.db.ensure_schema()` ran (run_all calls it on start).
- DB locked: SQLite uses WAL; avoid long-lived transactions; retry with small backoff.
- YFinance gaps: inspect `outputs/tables/backtest_missing_prices_*.csv`.
- Tiny datasets: disable CV in trainer (`--rolling-cv` off) or lower splits.

