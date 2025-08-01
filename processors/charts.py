import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.labels import FINAL_COLUMN_ORDER
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

# === Directory Setup ===
PLOT_DIR = "outputs/plots"
TABLE_DIR = "outputs/tables"
SIGNAL_DIR = "outputs/top_signals"
BREAKOUT_DIR = "outputs/breakouts"
DASHBOARD_DIR = "outputs/dashboard"
for d in [PLOT_DIR, TABLE_DIR, SIGNAL_DIR, BREAKOUT_DIR, DASHBOARD_DIR]:
    os.makedirs(d, exist_ok=True)

# === Utility ===
def save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[PLOT] Saved: {path}")

def save_table(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"[TABLE] Saved: {path}")

# === Plots ===
def plot_risk_return_scatter(df: pd.DataFrame):
    if {"Max Return %", "Drawdown %", "Trade Type"}.issubset(df.columns):
        plot_df = df.dropna(subset=["Max Return %", "Drawdown %"])
        if not plot_df.empty:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x="Drawdown %", y="Max Return %", hue="Trade Type", alpha=0.7)
            plt.axvline(0, color="gray", linestyle="--")
            plt.title("Max Return vs Drawdown by Trade Type")
            save_plot(os.path.join(PLOT_DIR, "risk_return_scatter.png"))

# PATCHED chart logic — top of file unchanged

def plot_bucket_returns(df: pd.DataFrame):
    if {"1D Return", "Score (0–100)"}.issubset(df.columns):
        df = df.dropna(subset=["1D Return", "Score (0–100)"])
        if df["Score (0–100)"].between(0, 100).all():
            df["score_bucket"] = pd.cut(df["Score (0–100)"], bins=[0, 60, 70, 80, 90, 100])
            grouped = df.groupby("score_bucket", observed=False)["1D Return"].mean().reset_index()
            grouped.plot(x="score_bucket", y="1D Return", kind="bar", figsize=(8, 5))
            plt.title("Avg 1D Return by Score Bucket")
            plt.ylabel("Return (%)")
            plt.xticks(rotation=0)
            save_plot(os.path.join(PLOT_DIR, "bucket_returns.png"))

def plot_feature_correlation(df: pd.DataFrame):
    cols = list(set(FINAL_COLUMN_ORDER + [
        "1D Return", "3D Return", "7D Return", "10D Return",
        "Reddit Score", "Financial Score", "News Score",
        "SPY 3D Return", "SPY 10D Return",
        "Avg 3D Return", "Avg 10D Return", "Signal Δ 3D", "Signal Δ 10D",
        "Cumulative Return", "Max Return %", "Drawdown %",
        "Forward Volatility", "Forward Sharpe Ratio"
    ]))
    cols = [c for c in cols if c in df.columns]
    numeric_df = df[cols].select_dtypes(include="number").dropna()
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        if not corr.isnull().values.all():
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
            plt.title("Feature Correlation Heatmap")
            save_plot(os.path.join(PLOT_DIR, "feature_correlation_heatmap.png"))

def plot_outperformance_heatmap(df: pd.DataFrame):
    if {"Signal Δ 3D", "Sector", "Run Datetime"}.issubset(df.columns):
        df = df[df["Run Datetime"].notnull()].copy()
        df["Weekday"] = df["Run Datetime"].dt.day_name()
        pivot = df.pivot_table(values="Signal Δ 3D", index="Sector", columns="Weekday", aggfunc="mean")
        if not pivot.empty and not pivot.isnull().all().all():
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
            plt.title("Signal Δ 3D by Weekday and Sector")
            save_plot(os.path.join(PLOT_DIR, "outperformance_heatmap.png"))

def plot_signal_duration_hist(df: pd.DataFrame):
    if "Signal Duration" in df.columns:
        durations = df["Signal Duration"].dropna()
        if not durations.empty:
            plt.figure(figsize=(7, 4))
            plt.hist(durations, bins=20, edgecolor="black")
            plt.title("Signal Duration Histogram")
            plt.xlabel("Days Held")
            save_plot(os.path.join(PLOT_DIR, "signal_duration_hist.png"))

def plot_score_return_trend(df: pd.DataFrame):
    if {"Run Datetime", "Score (0–100)", "3D Return"}.issubset(df.columns):
        df = df.dropna(subset=["Run Datetime", "Score (0–100)", "3D Return"])
        df = df.sort_values("Run Datetime").copy()
        df["Date"] = df["Run Datetime"].dt.date
        daily = df.groupby("Date")[["Score (0–100)", "3D Return"]].mean()
        if not daily.empty:
            daily.rolling(7).mean().plot(figsize=(10, 5), title="7D Rolling Avg: Score vs 3D Return")
            save_plot(os.path.join(PLOT_DIR, "score_return_trend.png"))

def plot_signaltype_vs_weekday(df: pd.DataFrame):
    if {"Signal Type", "Run Datetime", "3D Return"}.issubset(df.columns):
        df = df.dropna(subset=["Signal Type", "Run Datetime", "3D Return"]).copy()
        df["Weekday"] = df["Run Datetime"].dt.day_name()
        pivot = df.pivot_table(index="Signal Type", columns="Weekday", values="3D Return", aggfunc="mean")
        if not pivot.empty:
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title("3D Return by Signal Type and Weekday")
            save_plot(os.path.join(PLOT_DIR, "signaltype_vs_weekday.png"))

# === Tables ===
def summarize_factor_returns(df: pd.DataFrame):
    if {"Top Factors", "3D Return"}.issubset(df.columns):
        records = []
        for _, row in df[["Top Factors", "3D Return"]].dropna().iterrows():
            for f in str(row["Top Factors"]).split(","):
                f = f.strip()
                if f:
                    records.append({"Factor": f, "3D Return": row["3D Return"]})
        if records:
            factor_df = pd.DataFrame(records)
            summary = factor_df.groupby("Factor")["3D Return"].mean().reset_index()
            summary = summary.sort_values("3D Return", ascending=False)
            save_table(summary, os.path.join(TABLE_DIR, "factor_return_summary.csv"))

def export_top_signals(df: pd.DataFrame):
    if {"Signal Type", "3D Return"}.issubset(df.columns):
        for sig_type, group in df.groupby("Signal Type"):
            top = group.sort_values("3D Return", ascending=False).head(10)
            save_table(top, os.path.join(SIGNAL_DIR, f"{sig_type}_top.csv"))

def export_breakout_csvs(df: pd.DataFrame):
    if "Run Datetime" in df.columns:
        df = df.copy()
        df["Date"] = df["Run Datetime"].dt.date
        for sector, g in df.groupby("Sector", dropna=True):
            save_table(g, os.path.join(BREAKOUT_DIR, f"sector_{sector}.csv"))
        for sig, g in df.groupby("Signal Type", dropna=True):
            save_table(g, os.path.join(BREAKOUT_DIR, f"type_{sig}.csv"))
        for d, g in df.groupby("Date", dropna=True):
            save_table(g, os.path.join(BREAKOUT_DIR, f"date_{d}.csv"))

def export_feature_correlations(df: pd.DataFrame):
    targets = ["3D Return", "Score (0–100)"]
    features = [col for col in df.columns if col not in targets and df[col].dtype != 'O']
    df_corr = df[features + targets].copy().dropna(subset=targets)
    corr_result = {target: df_corr[features].corrwith(df_corr[target]).dropna() for target in targets}
    output_df = pd.DataFrame(corr_result).sort_values(by="3D Return", ascending=False)
    save_table(output_df.reset_index().rename(columns={"index": "Feature"}), os.path.join(TABLE_DIR, "feature_correlations.csv"))

def export_metadata_summary(df: pd.DataFrame):
    stats = {
        "Total Signals": len(df),
        "Most Recent Date": str(df["Run Datetime"].max().date()) if "Run Datetime" in df.columns else "N/A",
        "Tickers": df["Ticker"].nunique() if "Ticker" in df.columns else 0,
        "Signal Types": df["Signal Type"].nunique() if "Signal Type" in df.columns else 0,
    }
    path = os.path.join(TABLE_DIR, "metadata_summary.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Metadata summary saved: {path}")

def clean_output_dirs():
    for folder in [PLOT_DIR, TABLE_DIR, SIGNAL_DIR, BREAKOUT_DIR]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    os.remove(path)

def generate_charts_and_tables(df: pd.DataFrame):
    clean_output_dirs()
    plot_risk_return_scatter(df)
    plot_bucket_returns(df)
    plot_feature_correlation(df)
    plot_outperformance_heatmap(df)
    plot_signal_duration_hist(df)
    plot_score_return_trend(df)
    plot_signaltype_vs_weekday(df)
    summarize_factor_returns(df)
    export_top_signals(df)
    export_breakout_csvs(df)
    export_feature_correlations(df)
    export_metadata_summary(df)
    generate_html_dashboard(df)

def generate_html_dashboard(df: pd.DataFrame):
    plot_files = sorted(Path(PLOT_DIR).glob("*.png"))

    metadata_path = os.path.join(TABLE_DIR, "metadata_summary.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    tables = {}
    for name in ["factor_return_summary", "feature_correlations"]:
        csv_path = os.path.join(TABLE_DIR, f"{name}.csv")
        if os.path.exists(csv_path):
            tables[name.replace("_", " ").title()] = pd.read_csv(csv_path)

    top_signals = df.sort_values("Score (0–100)", ascending=False).head(10).copy()
    display_cols = ["Rank", "Ticker", "Company", "Sector", "Trade Type",
                    "Score (0–100)", "Reddit Sentiment", "News Sentiment", "3D Return"]
    top_signals = top_signals[[c for c in display_cols if c in top_signals.columns]]

    env = Environment(loader=FileSystemLoader(searchpath="."), autoescape=select_autoescape())
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VP Investments Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-size: 0.95rem; }
            img { max-width: 100%; height: auto; margin-bottom: 20px; }
            .table-sm th, .table-sm td { padding: 0.3rem; }
        </style>
    </head>
    <body class="bg-light">
        <div class="container py-4">
            <h1 class="mb-4">VP Investments Dashboard</h1>

            <h3>📊 Top Signals</h3>
            {{ top_signals|safe }}

            <h3 class="mt-5">🧠 Metadata</h3>
            <ul>
                {% for key, value in metadata.items() %}
                <li><strong>{{ key }}</strong>: {{ value }}</li>
                {% endfor %}
            </ul>

            {% for label, table in tables.items() %}
                <h3 class="mt-5">{{ label }}</h3>
                {{ table.to_html(classes="table table-striped table-sm", index=False) | safe }}
            {% endfor %}

            <h3 class="mt-5">📈 Charts</h3>
            {% for img in plot_files %}
                <div class="mb-4">
                    <img src="../plots/{{ img.name }}" alt="{{ img.name }}">
                </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    template = env.from_string(template_str)
    html = template.render(
        top_signals=top_signals.to_html(classes="table table-bordered table-sm", index=False),
        metadata=metadata,
        tables=tables,
        plot_files=plot_files
    )

    output_path = os.path.join(DASHBOARD_DIR, "dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DASHBOARD] Saved: {output_path}")