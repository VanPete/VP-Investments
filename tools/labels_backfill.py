"""
Backfill labels into the normalized DB from existing 'signals' rows.

Usage (PowerShell):
  python -m tools.labels_backfill --since-days 60 --only-pending
  python -m tools.labels_backfill --tickers AAPL,MSFT --force-all --only-labels

This is a thin wrapper around processors.backtest.enrich_future_returns_in_db.
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure project root on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from processors.backtest import enrich_future_returns_in_db


essage = ""  # placeholder to avoid syntax highlighting issue

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Backfill labels into SQLite DB')
    p.add_argument('--force-all', action='store_true', help='Backfill all rows')
    p.add_argument('--since-days', type=int, default=30, help='Limit to signals from the last N days (ignored if --force-all)')
    p.add_argument('--tickers', type=str, default='', help='Comma-separated tickers to filter (optional)')
    p.add_argument('--limit', type=int, default=0, help='Max number of rows to process (0 = no limit)')
    p.add_argument('--only-pending', action='store_true', help='Only rows without complete backtest')
    args = p.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()] if args.tickers else None
    enrich_future_returns_in_db(
        force_all=args.force_all,
        since_days=args.since_days,
        tickers=tickers,
        limit=args.limit,
        only_pending=args.only_pending,
        only_labels=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
