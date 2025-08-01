import sys
import os
import json
import sqlite3
import hashlib
import argparse
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone, date

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from charts import generate_charts_and_tables
from config.config import DB_PATH, RETURN_WINDOWS, FUTURE_PROOF_COLS
from config.labels import FINAL_COLUMN_ORDER

import numpy as np
import pandas as pd
import yfinance as yf
import pytz


def hash_config(config_dict: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()


def append_backtest_to_db(df: pd.DataFrame, feature_toggles: Optional[Dict[str, bool]] = None) -> None:
    run_id = datetime.now(timezone.utc).isoformat()
    df["Run ID"] = run_id
    all_columns = FINAL_COLUMN_ORDER + FUTURE_PROOF_COLS + ["Run ID", "Signal Type"]
    for col in all_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[all_columns]

    with sqlite3.connect(DB_PATH) as conn:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
        for col in set(df.columns) - existing_cols:
            conn.execute(f'ALTER TABLE signals ADD COLUMN "{col}"')
        df.to_sql("signals", conn, if_exists="append", index=False)
        print(f"[INFO] Appended {len(df)} rows to 'signals' table.")

        if feature_toggles:
            config_meta = {
                "Run ID": run_id,
                "timestamp": run_id,
                "feature_toggles": json.dumps(feature_toggles),
                "hash": hash_config(feature_toggles),
                "notes": ""
            }
            os.makedirs("outputs/config_logs", exist_ok=True)
            path = os.path.join("outputs/config_logs", f"{run_id}.json")
            with open(path, "w") as f:
                json.dump(feature_toggles, f, indent=2)
            pd.DataFrame([config_meta]).to_sql("run_metadata", conn, if_exists="append", index=False)
            print(f"[INFO] Config log saved: {path}")


def enrich_with_backtest_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Run Datetime"] = pd.to_datetime(df["Run Datetime"], utc=True, errors="coerce")
    df.sort_values("Run Datetime", inplace=True)

    for d in RETURN_WINDOWS:
        df[f"{d}D Return"] = pd.to_numeric(df.get(f"{d}D Return"), errors="coerce")

    df["Signal Δ 3D"] = df["3D Return"] - df.get("Avg 3D Return", 0)
    df["Signal Δ 10D"] = df["3D Return"] - df.get("Avg 10D Return", 0)
    df["Outperformed"] = df["Signal Δ 3D"] > 0
    df["Cumulative Return"] = (
        df.groupby("Ticker")["3D Return"]
          .transform(lambda x: x.fillna(0).ewm(halflife=5, min_periods=1).mean().cumsum())
    )
    df["Alpha 3D"] = df["3D Return"] - df.get("SPY 3D Return", 0)
    df["Alpha 10D"] = df["10D Return"] - df.get("SPY 10D Return", 0)
    df["Fully Realized"] = (datetime.now(timezone.utc) - df["Run Datetime"]).dt.days >= max(RETURN_WINDOWS)
    df["Top Factor Weight"] = df["Top Factors"].apply(lambda x: 1.0 if isinstance(x, str) else np.nan)
    df["Normalized Rank"] = df["Rank"].rank(pct=True)
    df["Anomaly Flag"] = df.apply(
        lambda row: "Extreme Return" if any(abs(row.get(f"{d}D Return", 0)) > 100 for d in RETURN_WINDOWS) else None,
        axis=1
    )

    rolling = (
        df.set_index("Run Datetime")
          .groupby("Ticker")["Ticker"]
          .rolling("30D").count()
          .rename("Signal Count 30D")
          .reset_index()
    )
    return df.merge(rolling, on=["Ticker", "Run Datetime"], how="left")


def fetch_price_series(ticker: str, start: date, end: date) -> Optional[pd.Series]:
    try:
        return yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                           progress=False)["Close"].dropna()
    except Exception:
        return None


def compute_returns(row: pd.Series) -> Optional[Dict[str, Any]]:
    ticker = row["Ticker"].replace("$", "")
    run_dt = pd.to_datetime(row["Run Datetime"], utc=True).tz_convert(None)
    start = run_dt + timedelta(days=1)
    end = run_dt + timedelta(days=max(RETURN_WINDOWS) + 5)

    if (date.today() - run_dt.date()).days < 1 or start.date() >= date.today():
        return None

    prices = fetch_price_series(ticker, start, end)
    spy = fetch_price_series("SPY", start, end)

    if prices is None or prices.empty:
        return "empty"

    base = prices.iloc[0]
    realized, updates = [], {}
    max_ret, peak, drawdown = -float("inf"), base, 0.0

    for w in RETURN_WINDOWS:
        if len(prices) > w:
            val = round((prices.iloc[w] - base) / base * 100, 2)
            updates[f"{w}D Return"] = val
            realized.append(str(w))

    for val in prices:
        pct = (val - base) / base * 100
        max_ret = max(max_ret, pct)
        peak = max(peak, val)
        drawdown = min(drawdown, (val - peak) / peak * 100)

    updates.update({
        "Max Return %": round(max_ret, 2),
        "Drawdown %": round(drawdown, 2),
        "Signal Duration": len(prices),
        "Forward Volatility": round(prices.pct_change().std() * 100, 2),
        "Forward Sharpe Ratio": round(prices.pct_change().mean() / prices.pct_change().std(), 2)
        if prices.pct_change().std() > 0 else None,
        "Realized Returns": ", ".join(realized),
        "Backtest Phase": "Complete" if len(realized) == len(RETURN_WINDOWS) else "Partial" if realized else "Pending",
        "Backtest Timestamp": datetime.now(timezone.utc).isoformat()
    })

    if spy is not None and not spy.empty:
        spy_base = spy.iloc[0]
        if len(spy) > 3:
            updates["SPY 3D Return"] = round((spy.iloc[3] - spy_base) / spy_base * 100, 2)
            if "3D Return" in updates:
                updates["Beat SPY 3D"] = updates["3D Return"] > updates["SPY 3D Return"]
        if len(spy) > 10:
            updates["SPY 10D Return"] = round((spy.iloc[10] - spy_base) / spy_base * 100, 2)
            if "10D Return" in updates:
                updates["Beat SPY 10D"] = updates["10D Return"] > updates["SPY 10D Return"]

    return updates


def enrich_future_returns_in_db(force_all: bool = True) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT rowid, * FROM signals", conn)
        df["Run Datetime"] = pd.to_datetime(df["Run Datetime"], utc=True, errors="coerce")
        eastern = pytz.timezone("US/Eastern")
        updated, skipped, skipped_tickers = 0, [], []

        for _, row in df.iterrows():
            run_dt = row["Run Datetime"].tz_convert(eastern)
            if not force_all and datetime.now(eastern) < run_dt.replace(hour=16) + timedelta(days=1):
                skipped.append(f"{row['Ticker']} ({run_dt:%Y-%m-%d %H:%M})")
                continue

            result = compute_returns(row)
            if result is None:
                continue
            elif result == "empty":
                skipped_tickers.append(row["Ticker"])
                continue
            elif isinstance(result, dict) and "error" in result:
                print(f"[ERROR] {row['Ticker']} row {row['rowid']}: {result['error']}")
                continue

            sets = ", ".join([f'"{k}" = ?' for k in result])
            values = list(result.values())
            conn.execute(f'UPDATE signals SET {sets} WHERE rowid = ?', values + [row["rowid"]])
            updated += 1
            print(f"[UPDATE] {row['Ticker']} at row {row['rowid']}")

        print(f"[INFO] Updated rows: {updated}")
        if skipped:
            print(f"[INFO] Skipped (too recent): {', '.join(skipped[:10])}...")
        if skipped_tickers:
            print(f"[SKIPPED] No data for: {', '.join(set(skipped_tickers))[:100]}...")

        print("[INFO] Generating charts and tables...")
        df_latest = pd.read_sql_query("SELECT * FROM signals", conn)
        df_latest["Run Datetime"] = pd.to_datetime(df_latest["Run Datetime"], utc=True, errors="coerce")
        df_latest = df_latest.sort_values("Run Datetime").dropna(subset=["Run Datetime"])
        generate_charts_and_tables(df_latest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true", help="Force backtest all rows")
    args = parser.parse_args()
    enrich_future_returns_in_db(force_all=args.force_all)


if __name__ == "__main__":
    main()
