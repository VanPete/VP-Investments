"""
Backfill historical prices into the normalized SQLite DB (utils.db.prices).

Usage (PowerShell):
    python -m tools.price_backfill --tickers AAPL,MSFT,SPY --start 2023-01-01 --end 2024-12-31

Notes:
- Uses yfinance bulk download when possible, with per-ticker fallback + retries.
- Writes to DB via utils.db.upsert_prices.
- Emits a diagnostics CSV for missing business days per ticker in the requested range.
- Idempotent: safe to rerun for overlapping dates.
"""
from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime
from typing import List, Dict, Tuple, Set

import pandas as pd
import time

# Ensure project root on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.db import upsert_prices
from config.config import YF_BATCH_PAUSE_SEC
from datetime import date as _date


def _bday_range_set(start: str, end: str) -> Set[str]:
    # Inclusive business-day range using pandas bdate_range
    rng = pd.bdate_range(start=start, end=end)
    return {d.date().isoformat() for d in rng}


def _chunk(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]


def fetch_and_upsert(tickers: List[str], start: str, end: str) -> Tuple[int, List[Dict[str, str]]]:
    import yfinance as yf  # type: ignore

    total = 0
    diagnostics: List[Dict[str, str]] = []
    expected_days = _bday_range_set(start, end)
    have_days: Dict[str, Set[str]] = {t: set() for t in tickers}
    # Try bulk download with simple retries
    attempts = 0
    last_err = None
    while attempts < 3:
        try:
            df = yf.download(tickers=tickers, start=start, end=end, group_by='ticker', auto_adjust=False, threads=True)
            last_err = None
            break
        except Exception as e:
            last_err = e
            attempts += 1
            time.sleep(1.5 * attempts)
            df = None  # type: ignore
    if last_err and df is None:  # type: ignore
        # Fallback per ticker with small delay between
        for t in tickers:
            try:
                sub = yf.download(tickers=t, start=start, end=end, group_by='ticker', auto_adjust=False, threads=False)
                sub = sub.dropna(how='all')
                rows: List[Dict] = []
                for idx, row in sub.iterrows():
                    d_iso = pd.Timestamp(idx).date().isoformat()
                    rows.append({
                        'ticker': t,
                        'date': d_iso,
                        'open': float(row.get('Open', float('nan'))),
                        'high': float(row.get('High', float('nan'))),
                        'low': float(row.get('Low', float('nan'))),
                        'close': float(row.get('Close', float('nan'))),
                        'adj_close': float(row.get('Adj Close', row.get('Adj Close', float('nan')))),
                        'volume': float(row.get('Volume', float('nan'))),
                    })
                    have_days[t].add(d_iso)
                total += upsert_prices(rows)
            except Exception:
                continue
            time.sleep(YF_BATCH_PAUSE_SEC)
        # Diagnostics for fallback-only path
        for t in tickers:
            miss = sorted(expected_days - have_days.get(t, set()))
            if miss:
                diagnostics.append({'ticker': t, 'missing_count': str(len(miss)), 'missing_dates': ",".join(miss[:50])})
        return total, diagnostics

    # If we have df from bulk path, process it
    if df is None:
        return total, diagnostics
    # If single ticker, yfinance returns a flat columns MultiIndex with OHLCV
    if isinstance(df.columns, pd.MultiIndex):
        for t in (tickers if len(tickers) > 1 else [tickers[0]]):
            try:
                sub = df[t].dropna(how='all')
            except Exception:
                # Some tickers missing
                continue
            rows: List[Dict] = []
            for idx, row in sub.iterrows():
                d_iso = pd.Timestamp(idx).date().isoformat()
                rows.append({
                    'ticker': t,
                    'date': d_iso,
                    'open': float(row.get('Open', float('nan'))),
                    'high': float(row.get('High', float('nan'))),
                    'low': float(row.get('Low', float('nan'))),
                    'close': float(row.get('Close', float('nan'))),
                    'adj_close': float(row.get('Adj Close', row.get('Adj Close', float('nan')))),
                    'volume': float(row.get('Volume', float('nan'))),
                })
                have_days[t].add(d_iso)
            total += upsert_prices(rows)
    else:
        # Single ticker flat frame
        t = tickers[0]
        sub = df.dropna(how='all')
        rows: List[Dict] = []
        for idx, row in sub.iterrows():
            d_iso = pd.Timestamp(idx).date().isoformat()
            rows.append({
                'ticker': t,
                'date': d_iso,
                'open': float(row.get('Open', float('nan'))),
                'high': float(row.get('High', float('nan'))),
                'low': float(row.get('Low', float('nan'))),
                'close': float(row.get('Close', float('nan'))),
                'adj_close': float(row.get('Adj Close', row.get('Adj Close', float('nan')))),
                'volume': float(row.get('Volume', float('nan'))),
            })
            have_days[t].add(d_iso)
        total += upsert_prices(rows)
    # Build diagnostics
    for t in tickers:
        miss = sorted(expected_days - have_days.get(t, set()))
        if miss:
            diagnostics.append({'ticker': t, 'missing_count': str(len(miss)), 'missing_dates': ",".join(miss[:50])})
    return total, diagnostics


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Backfill prices into SQLite DB')
    p.add_argument('--tickers', required=True, help='Comma-separated list')
    p.add_argument('--start', required=True, help='YYYY-MM-DD')
    p.add_argument('--end', required=True, help='YYYY-MM-DD')
    p.add_argument('--batch', type=int, default=20, help='Tickers per batch')
    p.add_argument('--retries', type=int, default=2, help='Additional per-ticker retries for sparse/missing data')
    args = p.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    if not tickers:
        print('No tickers provided')
        return 2

    total = 0
    all_diags: List[Dict[str, str]] = []
    # Diagnostics CSV path
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    diag_path = os.path.join(_ROOT, 'outputs', 'tables', f'price_backfill_gaps_{ts}.csv')
    os.makedirs(os.path.dirname(diag_path), exist_ok=True)
    for group in _chunk(tickers, max(1, args.batch)):
        up, diags = fetch_and_upsert(group, args.start, args.end)
        total += up
        all_diags.extend(diags)
        # Retry queue for tickers with large gaps
        retry_syms = [d['ticker'] for d in diags if int(d.get('missing_count', '0')) > 5]
        for attempt in range(max(0, args.retries)):
            if not retry_syms:
                break
            time.sleep(1.0 + attempt)
            up2, diags2 = fetch_and_upsert(retry_syms, args.start, args.end)
            total += up2
            all_diags = [d for d in all_diags if d['ticker'] not in set(retry_syms)] + diags2
            # Narrow retry list to those still with sizeable gaps
            retry_syms = [d['ticker'] for d in diags2 if int(d.get('missing_count', '0')) > 5]

    # Write diagnostics CSV if any
    if all_diags:
        pd.DataFrame(all_diags).to_csv(diag_path, index=False)
        print(f'[DIAG] Missing-day diagnostics saved to {diag_path} ({len(all_diags)} tickers with gaps)')
    print(f'Upserted {total} price rows for {len(tickers)} tickers from {args.start} to {args.end}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
