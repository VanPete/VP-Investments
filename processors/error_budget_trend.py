"""
Export an error-budget trend CSV from metrics.

Reads metrics with names like 'fetch_success_<component>' in the last N days,
groups by UTC date and component, and writes a tidy CSV for dashboards.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# Ensure project root imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
import sys
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.config import DB_PATH


def compute_error_budget_trend(days: int = 30) -> pd.DataFrame:
    since = (datetime.utcnow() - timedelta(days=max(1, days))).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT date(created_at) AS day, name, value
            FROM metrics
            WHERE name LIKE 'fetch_success_%' AND created_at >= ?
            ORDER BY created_at
            """,
            conn,
            params=(since,),
        )
    if df.empty:
        return df
    # Extract component suffix from name
    df["component"] = df["name"].astype(str).str.replace("fetch_success_", "", regex=False)
    df = df[["day", "component", "value"]].rename(columns={"value": "success_rate"})
    return df


def export_error_budget_trend(days: int = 30) -> Optional[str]:
    df = compute_error_budget_trend(days)
    if df is None or df.empty:
        return None
    out_dir = os.path.join(_ROOT, "outputs", "tables")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Error_Budget_Trend.csv")
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    p = export_error_budget_trend(30)
    if p:
        print(f"[OBS] Error budget trend exported: {p}")
    else:
        print("[OBS] No error-budget metrics found to export.")
