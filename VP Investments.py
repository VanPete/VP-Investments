import os
import logging
import warnings
import sqlite3
import unicodedata
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd
import numpy as np

from tqdm import tqdm
from fetchers.reddit_scraper import fetch_reddit_data
from fetchers.finance_fetcher import fetch_yf_data, load_company_data
from fetchers.news_fetcher import fetch_news_for_tickers, summarize_news_sentiment
from fetchers.google_trends_fetcher import fetch_google_trends
from processors.backtest import FUTURE_PROOF_COLS
from processors.signal_scoring import SignalScorer, calculate_post_recency, detect_new_signals
from processors.reports import export_excel_report, generate_html_dashboard
from config.config import (
    MIN_MENTIONS, MIN_UPVOTES, SUBREDDIT_WEIGHTS, FEATURE_TOGGLES, DB_PATH
)
from config.labels import FINAL_COLUMN_ORDER
from utils.logger import setup_logging

setup_logging()
warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays on .*fillna.* is deprecated.*")

def initialize_run_directory() -> Tuple[str, str]:
    now = datetime.now()
    run_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    run_dir = os.path.join("outputs", now.strftime("(%d %B %y, %H_%M_%S)"))
    os.makedirs(run_dir, exist_ok=True)
    return run_datetime, run_dir

def enrich_reddit_data(reddit_df: pd.DataFrame) -> pd.DataFrame:
    reddit_df["created"] = pd.to_datetime(reddit_df["created"], utc=True)
    reddit_df["Post Recency"] = calculate_post_recency(reddit_df)
    reddit_df["Reddit Summary"] = reddit_df["Reddit Summary"].fillna("[No summary available]")
    reddit_df["Weighted Mention"] = reddit_df["Subreddit"].map(SUBREDDIT_WEIGHTS).fillna(1.0)
    return reddit_df

def aggregate_reddit_signals(reddit_df: pd.DataFrame) -> pd.DataFrame:
    mention_weights = reddit_df.groupby("Ticker")["Weighted Mention"].sum()
    agg = reddit_df.groupby("Ticker").agg(
        sentiment=("Reddit Sentiment", "mean"),
        Upvotes=("Upvotes", "mean"),
        Post_Recency=("Post Recency", "max")
    ).rename(columns={"sentiment": "Reddit Sentiment", "Post_Recency": "Post Recency"})
    agg["Mentions"] = agg.index.map(mention_weights)
    return agg

def merge_with_optional_sources(df: pd.DataFrame, tickers: set) -> pd.DataFrame:
    if FEATURE_TOGGLES.get("Enable News Fetch") and (
        FEATURE_TOGGLES.get("News Sentiment") or FEATURE_TOGGLES.get("News Mentions")
    ):
        news = summarize_news_sentiment(fetch_news_for_tickers(sorted(tickers)))
        df_news = pd.DataFrame.from_dict(news, orient='index').reset_index().rename(columns={"index": "Ticker"})
        df = df.merge(df_news, on="Ticker", how="left")

    if FEATURE_TOGGLES.get("Google Trends"):
        trends = fetch_google_trends(sorted(tickers), use_cache=False)
        if isinstance(trends, pd.DataFrame) and "Ticker" in trends.columns:
            trends["Ticker"] = trends["Ticker"].str.upper()
            merge_cols = [col for col in ["Ticker", "Trend Spike", "Google Interest", "Source"] if col in trends.columns]
            df = df.merge(trends[merge_cols], on="Ticker", how="left")
            if "Source" not in df.columns:
                df["Source"] = "n/a"
    return df

def apply_ranking_and_scoring(df: pd.DataFrame, reddit_df: pd.DataFrame, run_datetime: str, summary_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(summary_lookup, on="Ticker", how="left")
    scorer = SignalScorer(profile_name="default")
    scorer.fit_normalization(df)
    scores = df.apply(lambda row: scorer.score_row(row, debug=True), axis=1)

    df["Weighted Score"] = scores.map(lambda x: x.weighted_score)
    df["Trade Type"] = scores.map(lambda x: x.trade_type)
    df["Signal Type"] = scores.map(lambda x: x.signal_type)
    df["Top Factors"] = scores.map(lambda x: ', '.join(x.top_features or []))
    df["Highest Contributor"] = scores.map(lambda x: x.highest_feature or "")
    df["Lowest Contributor"] = scores.map(lambda x: x.lowest_feature or "")
    df["Secondary Flags"] = scores.map(lambda x: x.secondary_flags or "")
    df["Risk Level"] = scores.map(lambda x: x.risk_level or "")
    df["Risk Tags"] = scores.map(lambda x: x.risk_tags or "")
    df["Reddit Score"] = scores.map(lambda x: x.reddit_score)
    df["Financial Score"] = scores.map(lambda x: x.financial_score)
    df["News Score"] = scores.map(lambda x: x.news_score)
    df["Score (0–100)"] = (df["Weighted Score"] * 100).round(2)
    df["Rank"] = df["Weighted Score"].rank(ascending=False, method="min").astype("Int64")
    df["Run Datetime"] = run_datetime

    # Add z-score normalization for selected metrics
    for col in ["Reddit Sentiment", "Price 7D %", "Volume Spike Ratio"]:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            z_col = col + " Z"
            df[z_col] = ((df[col] - mean) / std).round(2)

    # Flag low liquidity
    df["Low Liquidity Flag"] = df["Avg Daily Value Traded"].apply(
        lambda x: "Yes" if pd.notna(x) and x < 1_000_000 else "No"
    )

    return df


def handle_emerging_and_thread_tags(df: pd.DataFrame, reddit_df: pd.DataFrame) -> pd.DataFrame:
    seen_tickers = set()
    hist_path = os.path.join("outputs", "historical_scores.csv")
    if os.path.exists(hist_path):
        try:
            seen_tickers = set(pd.read_csv(hist_path)["Ticker"].unique())
        except Exception as e:
            logging.warning(f"Could not load historical_scores.csv: {e}")

    df["Emerging"] = df["Ticker"].apply(lambda x: "Emerging" if x not in seen_tickers else "")

    if FEATURE_TOGGLES.get("Thread Detection") and "Thread Tag" in reddit_df.columns:
        threads = reddit_df.groupby("Ticker")["Thread Tag"].agg(
            lambda tags: tags.mode()[0] if not tags.mode().empty else "Single"
        ).reset_index()
        df = df.merge(threads, on="Ticker", how="left")
        df["Emerging/Thread"] = df.apply(
            lambda row: f"{row['Emerging']} + {row['Thread Tag']}" if row["Emerging"] and row["Thread Tag"] == "Series"
            else row["Emerging"] or row["Thread Tag"], axis=1
        )
        df.drop(columns=["Emerging", "Thread Tag"], inplace=True)
    else:
        df.rename(columns={"Emerging": "Emerging/Thread"}, inplace=True)

    return df

def write_to_db(df: pd.DataFrame):
    try:
        all_cols = set(FINAL_COLUMN_ORDER + FUTURE_PROOF_COLS + ["Signal Type"])
        for col in all_cols:
            if col not in df.columns:
                df[col] = ""

        df.columns = [unicodedata.normalize("NFKC", col) for col in df.columns]

        with sqlite3.connect(DB_PATH) as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'").fetchone():
                existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
                missing_cols = set(df.columns) - existing_cols
                for col in missing_cols:
                    conn.execute(f'ALTER TABLE signals ADD COLUMN "{col}"')

            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii') if isinstance(x, str) else x)

            df.to_sql("signals", conn, if_exists="append", index=False)
            print("[INFO] Saved signals to database.")
    except Exception as e:
        logging.warning(f"[ERROR] Failed to save to signals table: {e}")

def clean_and_validate_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Signal Type.1" in df.columns:
        df.drop(columns=["Signal Type.1"], inplace=True)
    if "Source" not in df.columns:
        df["Source"] = "Unknown"
    df["Source"] = df["Source"].fillna("Unknown")

    for col in ["Twitter Mentions", "Next Earnings Date", "Google Interest", "Squeeze Signal"]:
        if col in df.columns and df[col].isna().all():
            logging.warning(f"[WARNING] Column '{col}' is fully null.")

    critical = ["Current Price", "Reddit Sentiment", "Weighted Score"]
    before = len(df)
    df = df.dropna(subset=critical)
    after = len(df)
    if after < before:
        logging.warning(f"[WARNING] Dropped {before - after} rows missing: {critical}")
    return df

# (all imports unchanged)

def process_data() -> Tuple[Optional[pd.DataFrame], str]:
    run_datetime, run_dir = initialize_run_directory()
    reddit_df = fetch_reddit_data(enable_scrape=True)
    if reddit_df.empty:
        logging.warning("No Reddit posts after fetch.")
        return None, run_dir

    reddit_df = enrich_reddit_data(reddit_df)
    company_df = load_company_data()
    valid_tickers = set(company_df["Ticker"].str.upper().str.strip().str.replace("$", "", regex=False))

    reddit_df["Ticker"] = reddit_df["Ticker"].str.upper().str.strip().str.replace("$", "", regex=False)
    reddit_df = reddit_df[reddit_df["Ticker"].str.len() <= 5]
    reddit_df = reddit_df[reddit_df["Ticker"].isin(valid_tickers)]
    if reddit_df.empty:
        logging.warning("No Reddit posts left after ticker filtering.")
        return None, run_dir

    summary_lookup = reddit_df.groupby("Ticker")["Reddit Summary"].first().reset_index()
    reddit_agg = aggregate_reddit_signals(reddit_df)
    reddit_filtered = reddit_agg.query(f"Mentions >= {MIN_MENTIONS} and Upvotes > {MIN_UPVOTES}")
    detect_new_signals(reddit_filtered.reset_index(), run_dir)

    tickers = set(reddit_agg.index)
    feature_matrix = reddit_agg.reset_index()
    feature_matrix = merge_with_optional_sources(feature_matrix, tickers)

    # === PATCHED SECTION START ===
    yf_data = fetch_yf_data(sorted(tickers))
    logging.info(f"[DEBUG] fetch_yf_data received {len(tickers)} tickers.")
    logging.info(f"[DEBUG] fetch_yf_data returned {len(yf_data)} rows.")
    if isinstance(yf_data, pd.DataFrame):
        missing_yf = set(tickers) - set(yf_data["Ticker"].unique())
        if missing_yf:
            logging.warning(f"[WARNING] Tickers missing from YF data: {sorted(missing_yf)}")
        feature_matrix = feature_matrix.merge(yf_data, on="Ticker", how="left")

        for col in ["Market Cap", "Price 1D %", "EPS Growth", "Avg Daily Value Traded"]:
            if col in feature_matrix.columns:
                null_rate = feature_matrix[col].isna().mean()
                logging.info(f"[INFO] Null rate for {col}: {null_rate:.1%}")
            else:
                logging.warning(f"[WARNING] Column {col} missing after YF merge.")
    else:
        logging.error("[ERROR] fetch_yf_data did not return a DataFrame.")
    # === PATCHED SECTION END ===

    for col in ["Volatility", "Momentum 30D %", "Avg Daily Value Traded", "Retail Holding %", "Float % Held by Institutions", "Earnings Gap %"]:
        if col in feature_matrix.columns:
            rank_col = col.replace(" %", "").replace(" ", "") + "Rank"
            feature_matrix[rank_col] = feature_matrix[col].rank(pct=True)

    feature_matrix = apply_ranking_and_scoring(feature_matrix, reddit_df, run_datetime, summary_lookup)
    feature_matrix = handle_emerging_and_thread_tags(feature_matrix, reddit_df)
    feature_matrix["Signal Type"] = "Reddit + Financial"

    for col in FINAL_COLUMN_ORDER:
        if col not in feature_matrix.columns:
            feature_matrix[col] = ""

    final_df = feature_matrix[FINAL_COLUMN_ORDER].sort_values(by="Weighted Score", ascending=False).reset_index(drop=True)
    final_df = clean_and_validate_df(final_df)
    final_df.to_csv(os.path.join(run_dir, "Final Analysis.csv"), index=False)
    write_to_db(final_df)
    return final_df, run_dir

if __name__ == "__main__":
    result = process_data()
    if result:
        combined, run_dir = result
        if combined is not None and not combined.empty:
            version = "v1.0.0"  # Adjust version tag as needed
            export_excel_report(
                combined,
                run_dir,
                metadata={
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "version": version,
                    "toggles": FEATURE_TOGGLES
                }
            )
            generate_html_dashboard(combined)
        else:
            print("[WARNING] No records passed final filter.")
    else:
        print("[ERROR] process_data() failed.")
