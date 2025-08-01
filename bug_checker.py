# bug_checker.py

import os
import sys
import pandas as pd
import numpy as np

KEY_ZSCORE_COLUMNS = [
    "Market Cap", "Avg Daily Value Traded",
    "Reddit Sentiment", "Mentions", "Upvotes", "Post Recency"
]

KEY_SCORE_COLUMNS = [
    "Reddit Sentiment", "News Sentiment", "Price 1D %", "Price 7D %",
    "Volume", "Market Cap", "EPS Growth", "ROE",
    "Insider Buys 30D", "Insider Buy Volume", "ETF Flow Spike Ratio",
    "Earnings Gap %", "Trend Spike", "Google Interest"
]

FINANCE_FETCHER_COLUMNS = [
    "Current Price", "Price 1D %", "Price 7D %", "Price 30D %",
    "Volume", "Avg Daily Value Traded", "Market Cap",
    "Volatility", "Above 50-Day MA %", "Above 200-Day MA %",
    "RSI", "MACD Histogram", "Bollinger Width",
    "EPS Growth", "ROE", "Debt/Equity", "FCF Margin", "P/E Ratio",
    "Shares Short", "Short Ratio", "Short Percent Float", "Short Percent Outstanding",
    "Put/Call OI Ratio", "Put/Call Volume Ratio", "Options Skew", "IV Spike %",
    "Insider Buys 30D", "Insider Buy Volume", "ETF Flow Spike Ratio", "Sector Inflows"
]

def analyze_file(path):
    print(f"\n=== Checking file: {path} ===")
    ext = os.path.splitext(path)[1].lower()
    print(f"📄 Detected file type: {ext[1:].upper()}")

    try:
        df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    if df.shape[0] < 5:
        print(f"⚠️ Warning: Only {df.shape[0]} rows detected. File may be incomplete.")

    null_report = df.isna().mean().sort_values(ascending=False)

    # Print columns with >20% nulls
    high_nulls = null_report[null_report > 0.2]
    if not high_nulls.empty:
        print("\n⚠️ Columns with >20% missing values:")
        for col, ratio in high_nulls.items():
            print(f" - {col}: {ratio:.1%}")
    else:
        print("\n✅ No columns with >20% missing values.")

    print("\n🔍 Z-Score column checks:")
    for col in KEY_ZSCORE_COLUMNS:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
        elif df[col].isna().all():
            print(f"⚠️ {col} exists but is fully NaN.")
        elif df[col].apply(lambda x: str(x).strip()).eq("").all():
            print(f"⚠️ {col} exists but all entries are blank.")
        else:
            print(f"✅ {col} is present and has data.")

    print("\n📊 Score-driving column checks:")
    for col in KEY_SCORE_COLUMNS:
        if col not in df.columns:
            print(f"❌ Missing: {col}")
        elif df[col].isna().all():
            print(f"⚠️ Present but fully NaN: {col}")
        elif (df[col] == 0).all():
            print(f"⚠️ Present but all zero: {col}")
        else:
            print(f"✅ {col} OK.")

    print("\n🧮 Finance fetcher column checks:")
    for col in FINANCE_FETCHER_COLUMNS:
        if col not in df.columns:
            print(f"❌ Missing: {col}")
        elif df[col].isna().all():
            print(f"⚠️ {col} exists but is fully NaN.")
        else:
            print(f"✅ {col} OK.")

    print("\n📋 Summary:")
    print(f" - Total columns: {len(df.columns)}")
    print(f" - Total rows: {len(df)}")
    print(f" - High-null columns: {len(high_nulls)}")
    print("✅ Bug check complete.\n")

if __name__ == "__main__":
    target_file = None
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Auto-locate most recent file in ./outputs/
        candidate_files = []
        for root, _, files in os.walk("outputs"):
            for name in files:
                if name in {"Final Analysis.csv", "Signal_Report.xlsx"}:
                    candidate_files.append(os.path.join(root, name))

        if not candidate_files:
            print("❌ No Final Analysis or Excel reports found in ./outputs/")
            sys.exit(1)
        target_file = max(candidate_files, key=os.path.getmtime)

    analyze_file(target_file)
