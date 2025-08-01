import glob
import logging
from datetime import datetime, timezone
from typing import Optional, NamedTuple, Dict, List, Tuple

import pandas as pd

from config.config import (
    SIGNAL_WEIGHT_PROFILES, CURRENT_SIGNAL_PROFILE, THRESHOLDS,
    RECENCY_HALFLIFE_HOURS, GROUP_LABELS, EMERGING_SCORE_BOOST,
    REDDIT_FINANCIAL_WEIGHT_RATIO, FEATURE_TOGGLES, FEATURE_NORMALIZATION
)

# === Cap definitions ===
SOFT_CAPS = {
    "Reddit Sentiment": 2.0, "Post Recency": 1.0, "Mentions": 20.0, "Upvotes": 50.0,
    "News Mentions": 10.0, "News Sentiment": 2.0, "Volatility Rank": 1.0,
    "Retail Ownership Rank": 1.0, "Institutional Ownership Rank": 1.0, "Liquidity Rank": 1.0,
    "Momentum Rank": 1.0, "MACD Histogram": 2.0, "Bollinger Width": 1.5,
    "Volatility": 0.1, "Beta vs SPY": 3.0, "Earnings Gap %": 20.0,
    "Retail Holding %": 0.9, "Avg Daily Value Traded": 2e9, "Sector Inflows": 5e9
}

# === Trade type weights ===
TRADE_TYPE_PROFILE = {
    "Swing": {"Sentiment_Reddit": 0.4, "Price_Action": 0.4, "Technical_Oscillator": 0.2},
    "Momentum": {"Momentum_Trend": 0.5, "Price_Action": 0.4, "Volume": 0.1},
    "Growth": {"Profitability": 0.6, "EPS Growth": 0.3, "Sentiment_News": 0.1},
    "Value": {"Valuation": 0.6, "Balance_Sheet": 0.3, "Ownership": 0.1},
    "Speculative": {"Sentiment_Reddit": 0.5, "Sentiment_News": 0.3, "Technical_Oscillator": 0.2}
}


class SignalScore(NamedTuple):
    weighted_score: float
    trade_type: str
    top_features: Optional[List[str]] = None
    highest_feature: Optional[str] = None
    lowest_feature: Optional[str] = None
    secondary_flags: Optional[str] = None
    risk_level: Optional[str] = None
    risk_tags: Optional[str] = None
    signal_type: Optional[str] = None
    reddit_score: Optional[float] = None
    financial_score: Optional[float] = None
    news_score: Optional[float] = None


def safe_float(val: Optional[float], default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def post_recency_score(created_time, now=None) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    if created_time is None:
        return 0.0
    age_hours = (now - created_time).total_seconds() / 3600
    return round(2 ** (-age_hours / RECENCY_HALFLIFE_HOURS), 4)


def calculate_post_recency(df: pd.DataFrame) -> pd.Series:
    def parse(val):
        if pd.isna(val): return None
        try: return pd.to_datetime(val, utc=True)
        except: return None
    times = df["created"] if "created" in df.columns else pd.Series([None] * len(df))
    return times.apply(parse).apply(post_recency_score)


class SignalScorer:
    def __init__(self, profile_name: str = CURRENT_SIGNAL_PROFILE):
        base = SIGNAL_WEIGHT_PROFILES[profile_name]
        enabled = {k for k in base if FEATURE_TOGGLES.get(k, True)}
        reddit = {k for k in enabled if "Reddit" in GROUP_LABELS.get(k, "")}
        financial = enabled - reddit

        r_total = sum(base[k] for k in reddit) or 1e-5
        f_total = sum(base[k] for k in financial) or 1e-5
        self.weights = {
            **{k: REDDIT_FINANCIAL_WEIGHT_RATIO * base[k] / r_total for k in reddit},
            **{k: (1 - REDDIT_FINANCIAL_WEIGHT_RATIO) * base[k] / f_total for k in financial}
        }
        self.normalization_means = {}
        self.normalization_stds = {}

    def fit_normalization(self, df: pd.DataFrame) -> None:
        if not FEATURE_NORMALIZATION:
            return
        for k in self.weights:
            if k in df.columns:
                vals = pd.to_numeric(df[k], errors="coerce")
                self.normalization_means[k] = vals.mean()
                self.normalization_stds[k] = vals.std(ddof=0)

        # Compute and store z-scores
        Z_SCORE_FIELDS = ["Reddit Sentiment", "Price 7D %", "Volume Spike Ratio", "Market Cap", "Avg Daily Value Traded"]
        for z_col in Z_SCORE_FIELDS:
            if z_col in df.columns:
                mean = df[z_col].mean()
                std = df[z_col].std(ddof=0)
                if std == 0 or pd.isna(std):
                    logging.warning(f"⚠️ Skipped Z-score for {z_col} — std=0 or NaN")
                    continue
                df[z_col + " Z-Score"] = df[z_col].apply(lambda x: (x - mean) / std if pd.notna(x) else 0.0)
            else:
                logging.warning(f"⚠️ Skipped Z-score for {z_col} — column missing")

        # Liquidity warning
        df["Liquidity Warning"] = df["Avg Daily Value Traded"].apply(
            lambda x: "Low Liquidity" if safe_float(x) < 1e6 else ""
        )

    def normalize(self, key: str, val: float) -> float:
        if not FEATURE_NORMALIZATION:
            return val
        mean = self.normalization_means.get(key)
        std = self.normalization_stds.get(key)
        return (val - mean) / std if std else val

    def evaluate_risk(self, row: pd.Series) -> Tuple[str, str]:
        tags = []
        if safe_float(row.get("Volatility")) > 0.06: tags.append("High Volatility")
        if safe_float(row.get("Beta vs SPY")) > 1.5: tags.append("High Beta")
        if safe_float(row.get("Avg Daily Value Traded")) < 5e6: tags.append("Low Liquidity")
        if safe_float(row.get("Retail Holding %")) > 0.5: tags.append("Retail Driven")
        if safe_float(row.get("Earnings Gap %")) > 10: tags.append("Event Sensitive")
        if safe_float(row.get("Short Percent Float")) > 20 and safe_float(row.get("Short Ratio")) > 3:
            tags.append("Short Squeeze Risk")
        score = len(tags)
        return ("Low", "Stable Metrics") if score == 0 else ("Moderate", ", ".join(tags)) if score <= 2 else ("High", ", ".join(tags))

    def infer_signal_type(self, row: pd.Series) -> str:
        if row.get("Reddit Sentiment", 0) > 0.5 and row.get("Mentions", 0) >= 5:
            return "Reddit Surge"
        if row.get("News Sentiment", 0) > 0.3 and row.get("News Mentions", 0) >= 3:
            return "News Momentum"
        if row.get("Earnings Gap %", 0) > 5:
            return "Earnings Reaction"
        if row.get("Retail Holding %", 0) > 15 and row.get("Volatility", 0) > 0.05:
            return "Retail Speculative"
        if row.get("Momentum 30D %", 0) > 10:
            return "Technical Momentum"
        return "Multi-Factor"

    def score_row(self, row: pd.Series, debug: bool = False) -> SignalScore:
        w_score, f_weights, flags = 0.0, {}, []

        for key in self.weights:
            if not FEATURE_TOGGLES.get(key, True):
                continue
            val = 1.0 if key in ["ETF Flow Signal", "Insider Signal"] and str(row.get(key)).strip().lower() == "yes" else safe_float(row.get(key))
            if pd.isna(val):
                logging.warning(f"Missing value for '{key}' on ticker '{row.get('Ticker')}'")
                val = 0.0
            val = self.normalize(key, val)
            val = min(val, SOFT_CAPS.get(key, val))
            weight = 0.0

            if key == "RSI" and THRESHOLDS["RSI_LOW"] <= val <= THRESHOLDS["RSI_HIGH"]:
                weight = val * self.weights.get(key, 0.0)
            elif key == "P/E Ratio" and THRESHOLDS["PE_LOW"] <= val <= THRESHOLDS["PE_HIGH"]:
                weight = self.weights.get(key, 0.0)
            elif val >= THRESHOLDS.get(key, 0):
                weight = val * self.weights.get(key, 0.0)

            f_weights[key] = weight
            w_score += weight

        if row.get("Emerging") == "Emerging":
            w_score *= EMERGING_SCORE_BOOST

        if safe_float(row.get("Short Percent Float")) > 20 and safe_float(row.get("Short Ratio")) > 3:
            row["Squeeze Signal"] = "Yes"
            flags.append("Squeeze Watch")
            s_weight = self.weights.get("Squeeze Signal", 0.25)
            f_weights["Squeeze Signal"] = s_weight
            w_score += s_weight
        else:
            row["Squeeze Signal"] = "No"

        if safe_float(row.get("MACD Histogram")) > 0.5 and safe_float(row.get("Above 50-Day MA %")) > 0 and safe_float(row.get("RSI")) > 60:
            flags.append("Breakout Candidate")
        if safe_float(row.get("RSI")) < 30 and safe_float(row.get("Price 1D %")) > 0:
            flags.append("Oversold Rebound")
        if safe_float(row.get("Momentum 30D %")) > 10:
            flags.append("Momentum Spike")
        if safe_float(row.get("Volatility")) > 0.04:
            flags.append("High Volatility")
        if safe_float(row.get("Beta vs SPY")) > 1.3:
            flags.append("High Beta")

        sorted_feats = sorted(f_weights.items(), key=lambda i: i[1], reverse=True)
        top_feats = [k for k, v in sorted_feats if v > 0][:3]
        high_feat = sorted_feats[0][0] if sorted_feats else ""
        low_feat = sorted(f_weights.items(), key=lambda i: i[1])[0][0] if f_weights else ""

        group_scores = {}
        for k, v in f_weights.items():
            g = GROUP_LABELS.get(k)
            if FEATURE_TOGGLES.get(k, True) and g:
                group_scores[g] = group_scores.get(g, 0) + v
        g_total = sum(group_scores.values()) or 1e-5
        norm_group = {k: v / g_total for k, v in group_scores.items()}

        best_type, best_val = "Balanced", 0.0
        for t_type, t_map in TRADE_TYPE_PROFILE.items():
            score = sum(t_map.get(k, 0.0) * norm_group.get(k, 0.0) for k in t_map)
            if score > best_val:
                best_type, best_val = t_type, score

        risk_lvl, risk_tags = self.evaluate_risk(row)
        signal_type = self.infer_signal_type(row)

        reddit_score = sum(v for k, v in f_weights.items() if "Reddit" in GROUP_LABELS.get(k, ""))
        financial_score = sum(v for k, v in f_weights.items() if "Financial" in GROUP_LABELS.get(k, ""))
        news_score = sum(v for k, v in f_weights.items() if "News" in GROUP_LABELS.get(k, ""))

        return SignalScore(
            round(w_score, 2), best_type, top_feats if debug else None, high_feat, low_feat,
            ", ".join(flags) if flags else None, risk_lvl, risk_tags, signal_type,
            round(reddit_score, 4), round(financial_score, 4), round(news_score, 4)
        )


def detect_new_signals(current_df: pd.DataFrame, run_dir: str) -> None:
    try:
        prev = sorted(glob.glob("outputs/*/Filtered Reddit Signals.csv"))
        if len(prev) >= 2:
            df_prev = pd.read_csv(prev[-2])
            if "Ticker" in df_prev.columns:
                new = current_df[~current_df["Ticker"].isin(df_prev["Ticker"])]
                print("\nNew Signals:", new["Ticker"].tolist())
            else:
                logging.warning(f"Previous CSV missing 'Ticker' column: {prev[-2]}")
    except Exception as e:
        logging.warning(f"Comparison error: {e}")
