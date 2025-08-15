import argparse
import os
from datetime import date, timedelta
from typing import List

import pandas as pd
import yfinance as yf

# Load known tickers
DEF_COMPANY_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'company_names.csv')


def download_symbol(sym: str, start: date, end: date):
    try:
        df = yf.download(sym, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False, auto_adjust=False)
        return df
    except Exception as e:
        return e


def main():
    p = argparse.ArgumentParser(description='Diagnose yfinance price availability for symbols')
    p.add_argument('--symbols', type=str, default='', help='Comma-separated symbols; if empty, sample from issues CSV or company list')
    p.add_argument('--days', type=int, default=30, help='Lookback days')
    p.add_argument('--sample', type=int, default=50, help='Sample size when symbols omitted')
    args = p.parse_args()

    today = date.today()
    start = today - timedelta(days=max(2, args.days))

    syms: List[str] = []
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        # Try to seed from the latest backtest_missing_prices_*.csv
        tables_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'tables')
        if os.path.isdir(tables_dir):
            cand = sorted([f for f in os.listdir(tables_dir) if f.startswith('backtest_missing_prices_')])
            if cand:
                latest = os.path.join(tables_dir, cand[-1])
                try:
                    df = pd.read_csv(latest)
                    syms = df['Ticker'].astype(str).str.upper().tolist()[:args.sample]
                except Exception:
                    pass
        if not syms and os.path.exists(DEF_COMPANY_CSV):
            try:
                df = pd.read_csv(DEF_COMPANY_CSV)
                syms = df['Ticker'].astype(str).str.upper().drop_duplicates().sample(min(args.sample, len(df)), random_state=42).tolist()
            except Exception:
                pass

    if not syms:
        print('No symbols to test.')
        return

    rows = []
    for s in syms:
        res = download_symbol(s, start, today)
        if isinstance(res, Exception):
            rows.append({'Ticker': s, 'Status': 'error', 'Detail': str(res)})
        elif res is None or res.empty:
            rows.append({'Ticker': s, 'Status': 'empty', 'Detail': 'no rows returned'})
        else:
            # Basic stats
            rows.append({'Ticker': s, 'Status': 'ok', 'Detail': f'{len(res)} rows'})

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'tables')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'yf_diagnose_{today.strftime('%Y%m%d')}.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
