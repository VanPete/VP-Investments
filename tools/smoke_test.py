import os
import sys
import pandas as pd
from datetime import datetime

# Ensure project root is on sys.path when executing from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from processors.reports import export_excel_report, export_csv_report, export_min_signals, export_full_json

def main():
    run_dir = os.path.join('outputs', datetime.now().strftime('(%d %B %y, %H_%M_%S)'))
    os.makedirs(run_dir, exist_ok=True)
    df = pd.DataFrame([
        {"Ticker": "$TEST", "Company": "Test Co", "Sector": "Tech", "Trade Type": "Momentum",
         "Weighted Score": 1.23, "Risk Level": "Moderate", "Risk Tags": "Low Liquidity",
         "Reddit Score": 0.2, "Financial Score": 0.9, "News Score": 0.13,
         "Rank": 1, "Run Datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ])
    export_excel_report(df, run_dir, metadata={"pipeline": "smoke"})
    export_csv_report(df, run_dir)
    export_min_signals(df, run_dir)
    export_full_json(df, run_dir)
    print("SMOKE_OK")

if __name__ == '__main__':
    main()
