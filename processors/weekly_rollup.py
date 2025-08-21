"""
Compute a simple weekly rollup of predictive performance using the normalized DB tables.

For the last N days of runs, joins signals_norm with labels per run and computes:
- Spearman IC (score vs fwd_return) per window
- Precision@K for K in a provided list (fraction with fwd_return>0)

Exports a CSV to outputs/tables and writes summary metrics to the metrics table (run_id=None).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List

import pandas as pd

# Ensure project root imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
import sys
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.config import DB_PATH
from utils.db import insert_metric


def _spearman_ic(score: pd.Series, target: pd.Series) -> float:
    # Compute Spearman via rank-then-Pearson to avoid SciPy dependency
    if score.empty or target.empty:
        return float('nan')
    s = score.rank(pct=True)
    t = target.rank(pct=True)
    return float(s.corr(t, method='pearson'))


def _precision_at_k(score: pd.Series, target: pd.Series, k: int) -> float:
    if len(score) == 0 or k <= 0:
        return float('nan')
    order = score.sort_values(ascending=False)
    top = order.head(min(k, len(order))).index
    hits = (target.loc[top] > 0).sum()
    return float(hits) / float(min(k, len(order)))


def compute_rollup(days: int, windows: List[str], ks: List[int]) -> pd.DataFrame:
    since = (datetime.utcnow() - timedelta(days=max(1, days))).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        runs = pd.read_sql_query("SELECT run_id, started_at FROM runs WHERE started_at >= ? ORDER BY started_at", conn, params=(since,))
        if runs.empty:
            return pd.DataFrame()
        rows = []
        for _, r in runs.iterrows():
            run_id = str(r["run_id"]) if pd.notna(r["run_id"]) else None
            if not run_id:
                continue
            s = pd.read_sql_query("SELECT ticker, score FROM signals_norm WHERE run_id = ?", conn, params=(run_id,))
            if s.empty:
                continue
            for w in windows:
                l = pd.read_sql_query(
                    "SELECT ticker, fwd_return FROM labels WHERE run_id = ? AND window = ?",
                    conn,
                    params=(run_id, w),
                )
                if l.empty:
                    continue
                df = s.merge(l, on="ticker", how="inner").dropna(subset=["score", "fwd_return"])
                if df.empty:
                    continue
                ic = _spearman_ic(pd.to_numeric(df["score"], errors='coerce'), pd.to_numeric(df["fwd_return"], errors='coerce'))
                row = {
                    "run_id": run_id,
                    "started_at": r.get("started_at"),
                    "window": w,
                    "ic": ic,
                }
                for k in ks:
                    row[f"p@{k}"] = _precision_at_k(df["score"], df["fwd_return"], k)
                rows.append(row)
        return pd.DataFrame(rows)


def export_rollup(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    out_dir = os.path.join(_ROOT, "outputs", "tables")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"weekly_rollup_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def persist_summary_metrics(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    # Global averages per window across the computed period
    try:
        ts = datetime.utcnow().isoformat()
        for w, g in df.groupby("window"):
            avg_ic = float(pd.to_numeric(g["ic"], errors='coerce').mean())
            insert_metric(None, f"weekly_ic_{w}", avg_ic, None, ts)  # run_id is nullable
            for col in [c for c in g.columns if c.startswith("p@")]:
                avg_pk = float(pd.to_numeric(g[col], errors='coerce').mean())
                insert_metric(None, f"weekly_{col}_{w}", avg_pk, None, ts)
    except Exception:
        pass


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Weekly rollup of IC and precision@K from normalized DB")
    p.add_argument("--days", type=int, default=7, help="Lookback window in days")
    p.add_argument("--windows", type=str, default="3D,10D", help="Comma-separated label windows (e.g., 3D,10D)")
    p.add_argument("--ks", type=str, default="10,20", help="Comma-separated K values for precision@K")
    args = p.parse_args(argv)

    windows = [w.strip() for w in args.windows.split(',') if w.strip()]
    ks = [int(k.strip()) for k in args.ks.split(',') if k.strip()]

    df = compute_rollup(args.days, windows, ks)
    path = export_rollup(df)
    if path:
        print(f"[WEEKLY] Rollup exported to {path} ({len(df)} rows)")
    persist_summary_metrics(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
