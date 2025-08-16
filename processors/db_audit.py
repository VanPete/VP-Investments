import argparse
import os
import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd

# Ensure project root is on sys.path for 'config' imports when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH, RETURN_WINDOWS


def _load_signals(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT rowid, * FROM signals", conn)
    if df.empty:
        return df
    df["Run Datetime"] = pd.to_datetime(df["Run Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Run Datetime"])  # ensure valid timestamps
    if tickers:
        tickset = {t.upper().replace("$", "").strip() for t in tickers}
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
        df = df[df["Ticker"].isin(tickset)]
    return df


def _eligible_mask(df: pd.DataFrame, buffer_days: int = 2) -> pd.Series:
    max_w = max(RETURN_WINDOWS)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_w + buffer_days)
    return df["Run Datetime"] <= cutoff


def audit(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "eligible_rows": 0,
            "complete": 0,
            "partial": 0,
            "missing": 0,
            "mismatch_phase": 0,
            "by_ticker": pd.DataFrame(),
            "issues": pd.DataFrame(),
        }

    ret_cols = [f"{d}D Return" for d in RETURN_WINDOWS]
    for c in ret_cols:
        if c not in df.columns:
            df[c] = pd.NA

    eligible = df[_eligible_mask(df)].copy()
    if eligible.empty:
        return {
            "eligible_rows": 0,
            "complete": 0,
            "partial": 0,
            "missing": 0,
            "mismatch_phase": 0,
            "by_ticker": pd.DataFrame(),
            "issues": pd.DataFrame(),
        }

    eligible["ret_present"] = eligible[ret_cols].notna().sum(axis=1)
    eligible["status"] = eligible["ret_present"].apply(lambda k: "complete" if k == len(ret_cols) else ("partial" if k > 0 else "missing"))

    # Backtest Phase mismatches
    phase = eligible.get("Backtest Phase").fillna("")
    eligible["phase_mismatch"] = (
        ((eligible["status"] == "complete") & (phase != "Complete")) |
        ((eligible["status"] == "partial") & (phase == "Pending"))
    )

    summary = {
        "eligible_rows": int(len(eligible)),
        "complete": int((eligible["status"] == "complete").sum()),
        "partial": int((eligible["status"] == "partial").sum()),
        "missing": int((eligible["status"] == "missing").sum()),
        "mismatch_phase": int(eligible["phase_mismatch"].sum()),
    }

    # By-ticker aggregation
    by_ticker = (
        eligible.groupby("Ticker")["status"]
        .value_counts()
        .unstack(fill_value=0)
        .sort_values(by=["missing", "partial"], ascending=False)
        .reset_index()
    )

    issues = eligible[eligible["status"] != "complete"][
        ["rowid", "Ticker", "Run Datetime", "Backtest Phase", *ret_cols, "status"]
    ].sort_values("Run Datetime")

    return {
        **summary,
        "by_ticker": by_ticker,
        "issues": issues,
    }


def export_reports(report: dict, out_dir: str = os.path.join("outputs", "tables")) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(report.get("by_ticker"), pd.DataFrame) and not report["by_ticker"].empty:
        report["by_ticker"].to_csv(os.path.join(out_dir, f"db_audit_by_ticker_{ts}.csv"), index=False)
    if isinstance(report.get("issues"), pd.DataFrame) and not report["issues"].empty:
        report["issues"].to_csv(os.path.join(out_dir, f"db_audit_issues_{ts}.csv"), index=False)


def mark_unavailable(issues_df: pd.DataFrame, reason: str = "No price data") -> int:
    if issues_df.empty:
        return 0
    mask = issues_df["status"] == "missing"
    target_rows = issues_df.loc[mask, ["rowid"]]
    if target_rows.empty:
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        # Ensure columns exist
        try:
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
        except Exception:
            existing_cols = set()
        for col in ["Backtest Phase", "Backtest Notes"]:
            if col not in existing_cols:
                try:
                    conn.execute(f'ALTER TABLE signals ADD COLUMN "{col}"')
                except Exception:
                    pass
        # Update rows
        for rid in target_rows["rowid"].tolist():
            try:
                conn.execute(
                    'UPDATE signals SET "Backtest Phase" = ?, "Backtest Notes" = ? WHERE rowid = ?',
                    ("Unavailable", reason, int(rid))
                )
            except Exception:
                continue
        return int(len(target_rows))


def main():
    p = argparse.ArgumentParser(description="Audit backtest completeness and optionally mark unavailable rows.")
    p.add_argument("--tickers", type=str, default="", help="Comma-separated ticker filter")
    p.add_argument("--export", action="store_true", help="Export CSV reports to outputs/tables")
    p.add_argument("--mark-unavailable", action="store_true", help="Mark eligible missing rows as Unavailable")
    p.add_argument("--reason", type=str, default="No price data", help="Reason to store in Backtest Notes when marking")
    args = p.parse_args()

    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()] if args.tickers else None
    df = _load_signals(tickers)
    report = audit(df)

    print("=== DB AUDIT SUMMARY ===")
    print(f"Eligible rows: {report['eligible_rows']}")
    print(f"Complete: {report['complete']}")
    print(f"Partial: {report['partial']}")
    print(f"Missing: {report['missing']}")
    print(f"Backtest Phase mismatches: {report['mismatch_phase']}")

    if args.export:
        export_reports(report)
        print("[INFO] Reports exported to outputs/tables")

    if args.mark_unavailable and isinstance(report.get("issues"), pd.DataFrame):
        updated = mark_unavailable(report["issues"], reason=args.reason)
        print(f"[INFO] Marked {updated} rows as Unavailable")


if __name__ == "__main__":
    main()
