import sys
import os
import json
import sqlite3
import hashlib
import argparse
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta, timezone, date

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from charts import generate_charts_and_tables
from config.config import DB_PATH, RETURN_WINDOWS, FUTURE_PROOF_COLS, SLIPPAGE_BPS, FEES_BPS
from utils.db import upsert_label, upsert_prices, insert_metric
from config.labels import FINAL_COLUMN_ORDER

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import csv
import warnings

# Suppress noisy NumPy warnings when computing stats over empty sets during early runs
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


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
    # Use the correct window for 10D delta
    df["Signal Δ 10D"] = df["10D Return"] - df.get("Avg 10D Return", 0)
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
    """Legacy per-ticker fetch. Kept for compatibility; prefer batched path below."""
    try:
        data = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
        return data["Close"].dropna()
    except Exception:
        return None

def _upsert_price_series_to_db(ticker: str, series: Optional[pd.Series]) -> int:
    """Convert a Close series to normalized rows and upsert into prices table."""
    try:
        if series is None or series.empty:
            return 0
        rows = []
        for dt, close in series.dropna().items():
            try:
                # We only have Close here; set others to null
                rows.append({
                    "ticker": ticker,
                    "date": (dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else pd.Timestamp(dt)).strftime("%Y-%m-%d"),
                    "open": None, "high": None, "low": None, "close": float(close),
                    "adj_close": None, "volume": None,
                })
            except Exception:
                continue
        return upsert_prices(rows)
    except Exception:
        return 0


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
            # Net of simple transaction costs (in/out once):
            try:
                cost_bps = float(SLIPPAGE_BPS) + float(FEES_BPS)
            except Exception:
                cost_bps = 0.0
            net = val - (cost_bps / 100.0)  # convert bps to percent
            updates[f"{w}D Return (net)"] = round(net, 2)
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


def enrich_future_returns_in_db(force_all: bool = True,
                                since_days: int = 30,
                                tickers: Optional[List[str]] = None,
                                limit: int = 0,
                                only_pending: bool = False,
                                only_labels: bool = False) -> None:
    """Compute forward returns for signals and persist into DB.

    Optimizations:
    - Filters by date/tickers and pending status.
    - Prefetches price series for all tickers to minimize API calls.
    """
    with sqlite3.connect(DB_PATH) as conn:
        base_df = pd.read_sql_query("SELECT rowid, * FROM signals", conn)
        base_df["Run Datetime"] = pd.to_datetime(base_df["Run Datetime"], utc=True, errors="coerce")
        base_df = base_df.dropna(subset=["Run Datetime"])  # ensure valid timestamps

        # Apply filters
        if tickers:
            tick_set = {t.replace("$", "").upper() for t in tickers}
            base_df["Ticker"] = base_df["Ticker"].astype(str).str.replace("$", "", regex=False).str.upper()
            base_df = base_df[base_df["Ticker"].isin(tick_set)]

        if not force_all:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, since_days))
            if "Backtest Phase" in base_df.columns:
                mask_recent = base_df["Run Datetime"] >= cutoff
                mask_incomplete = base_df["Backtest Phase"].fillna("") != "Complete"
                base_df = base_df[mask_recent | mask_incomplete]
            else:
                base_df = base_df[base_df["Run Datetime"] >= cutoff]

        if only_pending and "Backtest Phase" in base_df.columns:
            base_df = base_df[base_df["Backtest Phase"].fillna("") != "Complete"]

        # Sort and optionally limit
        base_df = base_df.sort_values("Run Datetime")
        if limit and limit > 0:
            base_df = base_df.tail(limit)

        if base_df.empty:
            print("[INFO] No rows to backtest after filtering.")
            return

        eastern = pytz.timezone("US/Eastern")

        # Restrict to known tickers from company list to avoid bad symbols in historical DB
        try:
            companies = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'company_names.csv'))
            valid_set = set(companies['Ticker'].astype(str).str.upper().str.strip().str.replace('$','', regex=False))
        except Exception:
            valid_set = set()

        # Normalize Ticker for downstream use and optional filtering
        base_df["Ticker"] = base_df["Ticker"].astype(str).str.replace("$", "", regex=False).str.upper()
        all_symbols = {str(t).replace("$", "").upper() for t in base_df["Ticker"].unique()}
        if valid_set:
            drop_syms = sorted(all_symbols - valid_set)
            if drop_syms:
                print(f"[FILTER] Dropping {len(drop_syms)} unknown symbols (e.g., {drop_syms[:10]})")
            # Filter base_df rows to valid tickers only
            base_df = base_df[base_df["Ticker"].isin(valid_set)]
            use_syms = sorted(set(base_df["Ticker"].unique()) & valid_set)
        else:
            use_syms = sorted(all_symbols)

    # Prefetch price data for all tickers and SPY in one go
    # Always include SPY for benchmarks
    uniq_tickers = sorted(set(use_syms) | {"SPY"})
    min_run = base_df["Run Datetime"].min().tz_convert(None).date() if hasattr(base_df["Run Datetime"], 'dt') else date.today()
    # Broaden the window to avoid weekend/holiday gaps and yfinance's exclusive end bound
    # Start a week BEFORE the earliest run to ensure baseline alignment exists
    start_global = max(date(1990, 1, 1), min_run - timedelta(days=7))
    today = date.today()
    # yfinance "end" is exclusive; include today by adding one day
    end_global = today + timedelta(days=1)

    print(f"[FETCH] Downloading price history for {len(uniq_tickers)} tickers from {start_global} to {end_global}…")
    price_map: Dict[str, pd.Series] = {}
    try:
            data = yf.download(uniq_tickers, start=start_global.strftime("%Y-%m-%d"), end=end_global.strftime("%Y-%m-%d"),
                               progress=False, group_by='ticker', auto_adjust=False, threads=True)
            if isinstance(data.columns, pd.MultiIndex):
                for t in uniq_tickers:
                    try:
                        if t in data.columns.get_level_values(0):
                            frame = data[t].copy()
                            # Upsert OHLCV into normalized prices table
                            try:
                                rows = []
                                for idx, r in frame.iterrows():
                                    rows.append({
                                        "ticker": t,
                                        "date": (idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else pd.Timestamp(idx)).strftime("%Y-%m-%d"),
                                        "open": float(r.get("Open")) if pd.notna(r.get("Open")) else None,
                                        "high": float(r.get("High")) if pd.notna(r.get("High")) else None,
                                        "low": float(r.get("Low")) if pd.notna(r.get("Low")) else None,
                                        "close": float(r.get("Close")) if pd.notna(r.get("Close")) else None,
                                        "adj_close": float(r.get("Adj Close")) if pd.notna(r.get("Adj Close")) else None,
                                        "volume": float(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
                                    })
                                if rows:
                                    upsert_prices(rows)
                            except Exception:
                                pass
                            s = frame["Close"].dropna()
                            if not s.empty:
                                price_map[t] = s
                    except Exception:
                        continue
            else:
                # Single ticker path
                try:
                    # Attempt normalized upsert for single-ticker download
                    if {'Open','High','Low','Close','Adj Close','Volume'}.issubset(set(data.columns)):
                        rows = []
                        for idx, r in data.iterrows():
                            rows.append({
                                "ticker": uniq_tickers[0],
                                "date": (idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else pd.Timestamp(idx)).strftime("%Y-%m-%d"),
                                "open": float(r.get("Open")) if pd.notna(r.get("Open")) else None,
                                "high": float(r.get("High")) if pd.notna(r.get("High")) else None,
                                "low": float(r.get("Low")) if pd.notna(r.get("Low")) else None,
                                "close": float(r.get("Close")) if pd.notna(r.get("Close")) else None,
                                "adj_close": float(r.get("Adj Close")) if pd.notna(r.get("Adj Close")) else None,
                                "volume": float(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
                            })
                        if rows:
                            upsert_prices(rows)
                except Exception:
                    pass
                if 'Close' in data:
                    s = data['Close'].dropna()
                    if not s.empty:
                        price_map[uniq_tickers[0]] = s
        except Exception as e:
            print(f"[WARN] Bulk download failed: {e}. Falling back to per-ticker fetch.")
            for t in uniq_tickers:
                s = fetch_price_series(t, start_global, end_global)
                if s is not None and not s.empty:
                    price_map[t] = s

        # Fetch SPY once (from bulk map if available; else fallback to per-ticker)
        s_spy = price_map.get("SPY")
        if s_spy is None or s_spy.empty:
            s_spy = fetch_price_series("SPY", start_global, end_global)
        spy_series = s_spy if s_spy is not None else pd.Series(dtype=float)

        updated, skipped, skipped_tickers = 0, [], []
        missing_detail: Dict[str, str] = {}
        # Cache existing column names to avoid repeated PRAGMA calls
        try:
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
        except Exception:
            existing_cols = set()

        for _, row in base_df.iterrows():
            run_dt_utc = row["Run Datetime"]
            try:
                run_dt_et = run_dt_utc.tz_convert(eastern)
            except Exception:
                run_dt_et = run_dt_utc

            # Skip if too recent (next day after 4pm ET) and not forcing
            if not force_all and datetime.now(eastern) < run_dt_et.replace(hour=16) + timedelta(days=1):
                skipped.append(f"{row['Ticker']} ({run_dt_et:%Y-%m-%d %H:%M})")
                continue

            tkr = str(row["Ticker"]).replace("$", "").upper()
            s = price_map.get(tkr)
            if s is None or s.empty:
                skipped_tickers.append(tkr)
                missing_detail[tkr] = missing_detail.get(tkr, "no series after bulk/per-ticker fetch")
                continue

            # Align to first trading day after signal
            run_naive = run_dt_utc.tz_convert(None)
            start_time = (run_naive + timedelta(days=1)).replace(tzinfo=None)
            idx = s.index.searchsorted(pd.Timestamp(start_time), side='left')
            if idx >= len(s):
                skipped_tickers.append(tkr)
                missing_detail[tkr] = "no trading days after start"
                continue

            base_price = float(s.iloc[idx])
            realized, updates = [], {}

            # Compute windowed returns using trading-day offsets
            for w in RETURN_WINDOWS:
                tgt_idx = idx + w
                if tgt_idx < len(s):
                    val = round((float(s.iloc[tgt_idx]) - base_price) / base_price * 100, 2)
                    updates[f"{w}D Return"] = val
                    try:
                        cost_bps = float(SLIPPAGE_BPS) + float(FEES_BPS)
                    except Exception:
                        cost_bps = 0.0
                    updates[f"{w}D Return (net)"] = round(val - (cost_bps / 100.0), 2)
                    realized.append(str(w))

            # Path metrics
            path = s.iloc[idx:]
            if not path.empty:
                max_ret = ((path.max() - base_price) / base_price) * 100
                # Drawdown from subsequent peaks
                roll_max = path.cummax()
                dd = ((path - roll_max) / roll_max * 100).min()
                updates.update({
                    "Max Return %": round(float(max_ret), 2),
                    "Drawdown %": round(float(dd), 2),
                    "Signal Duration": int(len(path)),
                })

                pct = path.pct_change().dropna()
                if not pct.empty and pct.std() > 0:
                    updates["Forward Volatility"] = round(float(pct.std() * 100), 2)
                    updates["Forward Sharpe Ratio"] = round(float(pct.mean() / pct.std()), 2)

            # Benchmark vs SPY for all return windows
            if spy_series is not None and not spy_series.empty:
                spy_idx = spy_series.index.searchsorted(pd.Timestamp(start_time), side='left')
                if spy_idx < len(spy_series):
                    spy_base_val = spy_series.iloc[spy_idx]
                    spy_base = float(spy_base_val.item() if hasattr(spy_base_val, 'item') else spy_base_val)
                    for w in RETURN_WINDOWS:
                        tgt = spy_idx + w
                        if tgt < len(spy_series):
                            v = spy_series.iloc[tgt]
                            v = float(v.item() if hasattr(v, 'item') else v)
                            updates[f"SPY {w}D Return"] = round((v - spy_base) / spy_base * 100, 2)
                            if f"{w}D Return" in updates:
                                updates[f"Beat SPY {w}D"] = updates[f"{w}D Return"] > updates[f"SPY {w}D Return"]

            updates.update({
                "Realized Returns": ", ".join(realized),
                "Backtest Phase": "Complete" if len(realized) == len(RETURN_WINDOWS) else "Partial" if realized else "Pending",
                "Backtest Timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Persist labels into normalized table when Run ID present
            try:
                run_id_val = str(row.get("Run ID") or "").strip()
                if run_id_val:
                    t_tkr = str(row.get("Ticker", "")).replace("$", "").upper()
                    for w in RETURN_WINDOWS:
                        win_key = f"{w}D Return"
                        if win_key in updates:
                            fwd = float(updates[win_key])
                            beat = None
                            beat_key = f"Beat SPY {w}D"
                            if beat_key in updates:
                                try:
                                    beat = 1 if bool(updates[beat_key]) else 0
                                except Exception:
                                    beat = None
                            upsert_label(run_id_val, t_tkr, f"{w}D", fwd, beat, datetime.now(timezone.utc).isoformat())
            except Exception:
                pass

            # Update wide signals table unless in label-only mode
            if not only_labels:
                # Ensure all columns exist before updating
                missing_cols = set(updates.keys()) - existing_cols
                for col in missing_cols:
                    try:
                        conn.execute(f'ALTER TABLE signals ADD COLUMN "{col}"')
                        existing_cols.add(col)
                    except Exception:
                        # Ignore if column was added concurrently
                        pass

                sets = ", ".join([f'"{k}" = ?' for k in updates])
                values = list(updates.values())
                conn.execute(f'UPDATE signals SET {sets} WHERE rowid = ?', values + [row["rowid"]])
                updated += 1
            if updated % 50 == 0:
                print(f"[UPDATE] Processed {updated} rows…")

        print(f"[INFO] Updated rows: {updated}")
        if skipped:
            print(f"[INFO] Skipped (too recent): {', '.join(skipped[:10])}...")
        if skipped_tickers:
            print(f"[SKIPPED] No data for: {', '.join(sorted(set(skipped_tickers)))[:100]}...")

        # Emit a diagnostics CSV for missing tickers
        if missing_detail:
            diag_path = os.path.join('outputs', 'tables', f"backtest_missing_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.makedirs(os.path.dirname(diag_path), exist_ok=True)
            with open(diag_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["Ticker", "Reason"])
                for t, r in sorted(missing_detail.items()):
                    w.writerow([t, r])
            print(f"[INFO] Missing price diagnostics saved: {diag_path}")

        if not only_labels:
            print("[INFO] Generating charts and tables…")
            df_latest = pd.read_sql_query("SELECT * FROM signals", conn)
            df_latest["Run Datetime"] = pd.to_datetime(df_latest["Run Datetime"], utc=True, errors="coerce")
            df_latest = df_latest.sort_values("Run Datetime").dropna(subset=["Run Datetime"]) 
            try:
                generate_charts_and_tables(df_latest)
            except Exception as e:
                # Non-fatal: emit a warning and continue
                print(f"[WARN] Chart/table generation failed: {e}")

        # Insert a simple run metric for this backtest pass
        try:
            run_ids = [r for r in df_latest.get("Run ID", pd.Series(dtype=str)).dropna().unique()]
            if run_ids:
                rid = str(run_ids[-1])
                insert_metric(rid, "backtest_updated_rows", float(updated), None, datetime.now(timezone.utc).isoformat())
        except Exception:
            pass


def net_returns_from_series(series: List[float], windows: List[int]) -> Dict[str, float]:
    """Compute gross and net returns (%) from a simple price series for given windows.

    Returns dict with keys like '3D Return' and '3D Return (net)'.
    """
    if not series or len(series) < 2:
        return {}
    try:
        cost_bps = float(SLIPPAGE_BPS) + float(FEES_BPS)
    except Exception:
        cost_bps = 0.0
    base = float(series[0])
    out: Dict[str, float] = {}
    for w in windows:
        if w < len(series):
            v = float(series[w])
            gross = (v - base) / base * 100.0
            out[f"{w}D Return"] = round(gross, 2)
            out[f"{w}D Return (net)"] = round(gross - (cost_bps / 100.0), 2)
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true", help="Force backtest all rows")
    parser.add_argument("--since-days", type=int, default=30, help="Process signals from the last N days (ignored if --force-all)")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers to filter (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of rows to process (0 = no limit)")
    parser.add_argument("--only-pending", action="store_true", help="Only rows without complete backtest")
    parser.add_argument("--only-labels", action="store_true", help="Skip wide table updates/plots; only persist labels to normalized DB")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()] if args.tickers else None
    enrich_future_returns_in_db(
        force_all=args.force_all,
        since_days=args.since_days,
        tickers=tickers,
        limit=args.limit,
        only_pending=args.only_pending,
        only_labels=args.only_labels,
    )


if __name__ == "__main__":
    main()
