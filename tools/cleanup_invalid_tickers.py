import os
import sqlite3
import pandas as pd

from config.config import DB_PATH

def load_valid_set() -> set[str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'data', 'company_names.csv')
    df = pd.read_csv(path)
    return set(df['Ticker'].astype(str).str.upper().str.replace('$', '', regex=False))

def main():
    vs = load_valid_set()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            cur = conn.cursor()
            cur.execute('SELECT DISTINCT "Ticker" FROM signals')
            all_syms = {str(r[0]).replace('$','').upper() for r in cur.fetchall() if r and r[0]}
            bad = sorted(all_syms - vs)
            if not bad:
                print('[CLEANUP] No invalid tickers found.')
                return
            qmarks = ','.join('?' for _ in bad)
            sql = f'DELETE FROM signals WHERE REPLACE(UPPER("Ticker"),"$","") IN ({qmarks})'
            cur.execute(sql, bad)
            print(f'[CLEANUP] Deleted {cur.rowcount} rows for {len(bad)} invalid tickers.')
            conn.commit()
        except Exception as e:
            print(f'[CLEANUP] Failed: {e}')

if __name__ == '__main__':
    main()
