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
from fetchers.universe_fetcher import build_trending_universe
from processors.backtest import FUTURE_PROOF_COLS
from processors.signal_scoring import SignalScorer, calculate_post_recency, detect_new_signals
from processors.reports import export_excel_report, export_csv_report, generate_html_dashboard, export_picks, write_run_readme
# ChatGPT Integration
from processors.chatgpt_integrator import enhance_dataframe_with_chatgpt, generate_executive_summary
from config.config import (
    MIN_MENTIONS, MIN_UPVOTES, SUBREDDIT_WEIGHTS, FEATURE_TOGGLES, DB_PATH, OUTPUTS_DIR_FORMAT,
    LIQUIDITY_WARNING_ADV, CURRENT_SIGNAL_PROFILE, OUTPUTS_DIR
)
from config.labels import FINAL_COLUMN_ORDER
from utils.logger import setup_logging
from config.config import SLACK_WEBHOOK_URL

setup_logging()
warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays on .*fillna.* is deprecated.*")

# Ensure DB schema exists even if no signals are generated in this run
def _ensure_db_tables() -> None:
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            # Minimal schema; columns will be extended dynamically as needed elsewhere
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    "Ticker" TEXT,
                    "Run Datetime" TEXT,
                    "Weighted Score" REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    "Run ID" TEXT,
                    timestamp TEXT,
                    feature_toggles TEXT,
                    hash TEXT,
                    notes TEXT
                )
                """
            )
    except Exception:
        # Non-fatal; downstream code handles missing tables conservatively
        pass

def initialize_run_directory() -> Tuple[str, str]:
    now = datetime.now()
    run_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    run_dir = os.path.join(OUTPUTS_DIR, now.strftime(OUTPUTS_DIR_FORMAT))
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
            merge_cols = [col for col in ["Ticker", "Trend Spike", "Google Interest", "AI Trends Commentary", "Source"] if col in trends.columns]
            df = df.merge(trends[merge_cols], on="Ticker", how="left")
            if "Source" not in df.columns:
                df["Source"] = "n/a"
    else:
        # Ensure columns exist even when trends disabled
        if "Trend Spike" not in df.columns:
            df["Trend Spike"] = 0.0
        if "Google Interest" not in df.columns:
            df["Google Interest"] = 0.0
        if "AI Trends Commentary" not in df.columns:
            df["AI Trends Commentary"] = ""
    return df

def apply_ranking_and_scoring(df: pd.DataFrame, reddit_df: pd.DataFrame, run_datetime: str, summary_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(summary_lookup, on="Ticker", how="left")
    # Prefer AI-enhanced Reddit summary when present
    if "AI Reddit Summary" in df.columns:
        df["Reddit Summary"] = df.apply(lambda r: r.get("AI Reddit Summary") if isinstance(r.get("AI Reddit Summary"), str) and r.get("AI Reddit Summary").strip() else r.get("Reddit Summary"), axis=1)
    scorer = SignalScorer(profile_name=CURRENT_SIGNAL_PROFILE)
    scorer.fit_normalization(df)
    scores = df.apply(lambda row: scorer.score_row(row, debug=True), axis=1)
    # Compact summary of missing feature values across all rows
    try:
        scorer.log_missing_summary()
    except Exception:
        pass

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

    # Additional Z-Score fields needed in report
    z_map = {
        "Market Cap": "Z-Score: Market Cap",
        "Avg Daily Value Traded": "Z-Score: Avg Daily Value",
        "Mentions": "Z-Score: Reddit Activity"
    }
    for src, dest in z_map.items():
        if src in df.columns:
            m, s = df[src].mean(), df[src].std()
            df[dest] = ((df[src] - m) / s).round(2) if pd.notna(s) and s != 0 else 0.0
        else:
            df[dest] = 0.0

    # Standardized Liquidity Flags
    df["Liquidity Flags"] = df["Avg Daily Value Traded"].apply(
        lambda x: "Low Liquidity" if pd.notna(x) and x < LIQUIDITY_WARNING_ADV else ""
    )

    # Ensure Squeeze Signal exists if not set by scorer
    if "Squeeze Signal" not in df.columns or df["Squeeze Signal"].isna().all():
        def _squeeze_rule(r):
            spf = pd.to_numeric(r.get("Short Percent Float", 0), errors="coerce")
            sr = pd.to_numeric(r.get("Short Ratio", 0), errors="coerce")
            spf_ok = (pd.notna(spf) and float(spf) > 20)
            sr_ok = (pd.notna(sr) and float(sr) > 3)
            return "Yes" if (spf_ok and sr_ok) else "No"
        df["Squeeze Signal"] = df.apply(_squeeze_rule, axis=1)

    # Sentiment Spike (relative to median Reddit Sentiment)
    if "Reddit Sentiment" in df.columns:
        med = df["Reddit Sentiment"].median()
        df["Sentiment Spike"] = (df["Reddit Sentiment"] - med).round(3)
    else:
        df["Sentiment Spike"] = 0.0

    # Twitter Mentions default to 0 when missing
    if "Twitter Mentions" not in df.columns:
        df["Twitter Mentions"] = 0.0

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
    _ensure_db_tables()
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
    # Expand universe with external trending sources (FMP/Stocktwits/Yahoo)
    try:
        extra = build_trending_universe()
        if extra:
            tickers |= {t for t in extra if t in valid_tickers}
            logging.info(f"[UNIVERSE] Added {len(extra)} trending tickers; total universe: {len(tickers)}")
    except Exception as e:
        logging.warning(f"[UNIVERSE] Failed to build trending universe: {e}")
    feature_matrix = reddit_agg.reset_index()
    logging.info(f"[PIPELINE] Merging optional sources (news/trends) for {len(tickers)} tickers…")
    feature_matrix = merge_with_optional_sources(feature_matrix, tickers)
    logging.info("[PIPELINE] Optional sources merge complete.")

    # === PATCHED SECTION START ===
    yf_data = fetch_yf_data(sorted(tickers))
    logging.info(f"[YF] Requested: {len(tickers)} tickers; Received: {len(yf_data)} rows")
    if isinstance(yf_data, pd.DataFrame):
        missing_yf = set(tickers) - set(yf_data["Ticker"].unique())
        if missing_yf:
            some = sorted(list(missing_yf))[:10]
            more = f" …(+{len(missing_yf)-10})" if len(missing_yf) > 10 else ""
            logging.info(f"[YF] Missing from YF data ({len(missing_yf)}): {some}{more}")
        feature_matrix = feature_matrix.merge(yf_data, on="Ticker", how="left")

        for col in ["Market Cap", "Price 1D %", "EPS Growth", "Avg Daily Value Traded"]:
            if col in feature_matrix.columns:
                null_rate = feature_matrix[col].isna().mean()
                logging.info(f"[YF] Null rate for {col}: {null_rate:.1%}")
            else:
                logging.warning(f"[WARNING] Column {col} missing after YF merge.")
    else:
        logging.error("[ERROR] fetch_yf_data did not return a DataFrame.")
    # === PATCHED SECTION END ===

    for col in ["Volatility", "Momentum 30D %", "Avg Daily Value Traded", "Retail Holding %", "Float % Held by Institutions", "Earnings Gap %"]:
        if col in feature_matrix.columns:
            rank_col = col.replace(" %", "").replace(" ", "") + "Rank"
            feature_matrix[rank_col] = feature_matrix[col].rank(pct=True)

    # Standard rank names expected by frontend/report
    if "Momentum 30D %" in feature_matrix.columns:
        feature_matrix["Momentum Rank"] = feature_matrix["Momentum 30D %"].rank(pct=True)
    if "Avg Daily Value Traded" in feature_matrix.columns:
        feature_matrix["Liquidity Rank"] = feature_matrix["Avg Daily Value Traded"].rank(pct=True)

    feature_matrix = apply_ranking_and_scoring(feature_matrix, reddit_df, run_datetime, summary_lookup)
    feature_matrix = handle_emerging_and_thread_tags(feature_matrix, reddit_df)
    feature_matrix["Signal Type"] = "Reddit + Financial"

    for col in FINAL_COLUMN_ORDER:
        if col not in feature_matrix.columns:
            feature_matrix[col] = ""

    final_df = feature_matrix[FINAL_COLUMN_ORDER].sort_values(by="Weighted Score", ascending=False).reset_index(drop=True)
    final_df = clean_and_validate_df(final_df)
    # Normalize percent-like columns to fractional scale in Final Analysis (configurable)
    from config.config import PERCENT_NORMALIZE as _PCTN
    if _PCTN:
        try:
            from config.labels import COLUMN_FORMAT_HINTS as _COLFMT
            pct_cols = [c for c, hint in _COLFMT.items() if hint == "percent" and c in final_df.columns]
            for pc in pct_cols:
                s = pd.to_numeric(final_df[pc], errors='coerce')
                nonnull = s.dropna()
                if nonnull.empty:
                    continue
                frac_gt = (nonnull.abs() > 1.5).mean()
                if pd.notna(frac_gt) and frac_gt >= 0.6:
                    final_df[pc] = s / 100.0
        except Exception:
            pass
    
    # ChatGPT Enhancement
    if os.getenv("OPENAI_API_KEY"):
        logging.info("Enhancing signals with ChatGPT analysis...")
        final_df = enhance_dataframe_with_chatgpt(final_df)
        executive_summary = generate_executive_summary(final_df)
        logging.info(f"AI Executive Summary: {executive_summary.get('portfolio_insights', 'N/A')}")
    
    # Save outputs (CSV + Parquet if available)
    out_csv = os.path.join(run_dir, "Final Analysis.csv")
    try:
        from utils.parquet_writer import write_both
        pq_path = write_both(final_df, out_csv)
        if pq_path:
            logging.info(f"Saved Parquet alongside CSV: {pq_path}")
    except Exception:
        final_df.to_csv(out_csv, index=False)
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
            # Also write a plain CSV signal report for quick inspection
            export_csv_report(combined, run_dir)
            # Picks and run README
            export_picks(combined, run_dir)
            write_run_readme(run_dir)
            generate_html_dashboard(combined)
            # Optional Slack alert
            if SLACK_WEBHOOK_URL:
                try:
                    import json, requests
                    top = combined[['Ticker', 'Score (0–100)']].head(5).to_dict(orient='records')
                    text = f"VP Investments run complete. Top 5: {top}"
                    requests.post(SLACK_WEBHOOK_URL, data=json.dumps({"text": text}), headers={"Content-Type": "application/json"}, timeout=10)
                except Exception:
                    pass
        else:
            print("[WARNING] No records passed final filter.")
    else:
        print("[ERROR] process_data() failed.")
