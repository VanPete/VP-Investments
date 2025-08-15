import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
from config.labels import FINAL_COLUMN_ORDER
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from config.config import OUTPUTS_DIR

# === Directory Setup ===
PLOT_DIR = os.path.join(OUTPUTS_DIR, "plots")
TABLE_DIR = os.path.join(OUTPUTS_DIR, "tables")
SIGNAL_DIR = os.path.join(OUTPUTS_DIR, "top_signals")
BREAKOUT_DIR = os.path.join(OUTPUTS_DIR, "breakouts")
DASHBOARD_DIR = os.path.join(OUTPUTS_DIR, "dashboard")
for d in [PLOT_DIR, TABLE_DIR, SIGNAL_DIR, BREAKOUT_DIR, DASHBOARD_DIR]:
    os.makedirs(d, exist_ok=True)

# === Utility ===
def save_plot(path: str, dpi: int = 150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")

def save_table(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"[TABLE] Saved: {path}")

# === Plots ===
def _percent_formatter():
    return FuncFormatter(lambda x, _: f"{x:.0f}%")


def _make_percent_formatter(scale: float = 1.0, decimals: int = 0):
    return FuncFormatter(lambda x, _: f"{(x*scale):.{decimals}f}%")


def _guess_percent_scale(values: pd.Series) -> float:
    """Heuristic: if values look like decimals (|max| <= ~1.5), scale by 100 for labels."""
    try:
        vmax = float(np.nanmax(np.abs(values.values)))
        return 100.0 if vmax <= 1.5 else 1.0
    except Exception:
        return 1.0


def plot_risk_return_scatter(df: pd.DataFrame):
    """Risk/return scatter with outlier clipping, density, and facets.

    - Clips x/y to the 1st–99th percentile to avoid squashed axes.
    - Uses small, semi-transparent markers (or hexbin for very large N).
    - Adds 0 lines and a simple trend (median per drawdown bin).
    - Saves combined plot and per-trade-type facets.
    """
    required = {"Max Return %", "Drawdown %"}
    if not required.issubset(df.columns):
        return
    plot_df = df.dropna(subset=list(required)).copy()
    # Coerce to numeric to avoid string types from DB joins
    for c in ["Drawdown %", "Max Return %"]:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df = plot_df.dropna(subset=["Drawdown %", "Max Return %"]).copy()
    if plot_df.empty:
        return

    # Clip outliers to improve readability
    xq = plot_df["Drawdown %"].quantile([0.01, 0.99]).values
    yq = plot_df["Max Return %"].quantile([0.01, 0.99]).values
    plot_df = plot_df[(plot_df["Drawdown %"].between(xq[0], xq[1])) & (plot_df["Max Return %"].between(yq[0], yq[1]))]
    if plot_df.empty:
        return

    # Combined view
    plt.figure(figsize=(10, 7))
    N = len(plot_df)
    if N > 1500:
        # Dense: hexbin for readability
        hb = plt.hexbin(plot_df["Drawdown %"], plot_df["Max Return %"], gridsize=40, cmap="viridis", mincnt=3)
        cb = plt.colorbar(hb)
        cb.set_label("Density")
        # Overlay a light scatter sample
        sample = plot_df.sample(min(800, N), random_state=42)
        hue = "Trade Type" if "Trade Type" in sample.columns else None
        sns.scatterplot(data=sample, x="Drawdown %", y="Max Return %", hue=hue, s=18, alpha=0.3, legend=False)
    else:
        hue = "Trade Type" if "Trade Type" in plot_df.columns else None
        sns.scatterplot(data=plot_df, x="Drawdown %", y="Max Return %", hue=hue, s=20, alpha=0.35, edgecolor=None)

    # Baselines and formatting
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Max Return vs Drawdown (clipped 1–99th percentile)")
    plt.xlabel("Drawdown %")
    plt.ylabel("Max Return %")
    x_scale = _guess_percent_scale(plot_df["Drawdown %"])
    y_scale = _guess_percent_scale(plot_df["Max Return %"])
    plt.gca().xaxis.set_major_formatter(_make_percent_formatter(scale=x_scale))
    plt.gca().yaxis.set_major_formatter(_make_percent_formatter(scale=y_scale))
    plt.grid(True, which="major", color="#eaeaea")
    # Place legend outside if present
    leg = plt.gca().get_legend()
    if leg is not None:
        plt.legend(title=leg.get_title().get_text() if leg.get_title() else None, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_plot(os.path.join(PLOT_DIR, "risk_return_scatter.png"), dpi=180)

    # Trend: median Max Return per Drawdown bin
    try:
        bins = pd.interval_range(start=plot_df["Drawdown %"].min(), end=plot_df["Drawdown %"].max(), periods=25)
        binned = plot_df.assign(_bin=pd.cut(plot_df["Drawdown %"], bins=bins, include_lowest=True))
        trend = binned.groupby("_bin", observed=False)["Max Return %"].median().reset_index()
        trend["Drawdown Mid %"] = trend["_bin"].apply(lambda iv: (iv.left + iv.right) / 2)
        plt.figure(figsize=(10, 4))
        plt.plot(trend["Drawdown Mid %"], trend["Max Return %"], color="#2a9d8f", linewidth=2)
        plt.title("Median Max Return by Drawdown")
        plt.xlabel("Drawdown %")
        plt.ylabel("Median Max Return %")
        x_scale = _guess_percent_scale(plot_df["Drawdown %"])
        y_scale = _guess_percent_scale(plot_df["Max Return %"])
        plt.gca().xaxis.set_major_formatter(_make_percent_formatter(scale=x_scale))
        plt.gca().yaxis.set_major_formatter(_make_percent_formatter(scale=y_scale))
        plt.grid(True, color="#eaeaea")
        save_plot(os.path.join(PLOT_DIR, "risk_return_trend.png"), dpi=180)
    except Exception:
        pass

    # Facets per Trade Type (small multiples)
    if "Trade Type" in plot_df.columns:
        g = sns.FacetGrid(plot_df, col="Trade Type", col_wrap=3, height=3.3, sharex=True, sharey=True)
        g.map_dataframe(sns.scatterplot, x="Drawdown %", y="Max Return %", s=16, alpha=0.35)
        for ax in g.axes.flatten():
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.xaxis.set_major_formatter(_make_percent_formatter(scale=_guess_percent_scale(plot_df["Drawdown %"])))
            ax.yaxis.set_major_formatter(_make_percent_formatter(scale=_guess_percent_scale(plot_df["Max Return %"])))
        g.fig.suptitle("Max Return vs Drawdown by Trade Type (clipped)", y=1.02)
        save_plot(os.path.join(PLOT_DIR, "risk_return_scatter_facets.png"), dpi=180)

# PATCHED chart logic — top of file unchanged

def plot_bucket_returns(df: pd.DataFrame):
    if {"1D Return", "Score (0–100)"}.issubset(df.columns):
        d = df.dropna(subset=["1D Return", "Score (0–100)"]).copy()
        if not d.empty and d["Score (0–100)"].between(0, 100).all():
            bins = [0, 50, 60, 70, 80, 90, 100]
            d["score_bucket"] = pd.cut(d["Score (0–100)"], bins=bins, right=True)
            grouped = d.groupby("score_bucket", observed=False)["1D Return"].agg(["mean", "count", "std"]).reset_index()
            grouped["se"] = grouped["std"].div(grouped["count"].clip(lower=1) ** 0.5)
            plt.figure(figsize=(9, 5))
            ax = sns.barplot(data=grouped, x="score_bucket", y="mean", color="#6baed6", edgecolor="#3b82f6")
            ax.errorbar(x=range(len(grouped)), y=grouped["mean"], yerr=grouped["se"], fmt="none", ecolor="#1f2937", capsize=3)
            plt.title("Avg 1D Return by Score Bucket (± SE)")
            plt.ylabel("Avg 1D Return %")
            plt.xlabel("Score (0–100) Buckets")
            y_scale = _guess_percent_scale(grouped["mean"])
            plt.gca().yaxis.set_major_formatter(_make_percent_formatter(scale=y_scale))
            for i, v in enumerate(grouped["mean"]):
                plt.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
            save_plot(os.path.join(PLOT_DIR, "bucket_returns.png"), dpi=160)

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
    # Drop constant columns to avoid divide-by-zero in correlation
    nunique = numeric_df.nunique()
    numeric_df = numeric_df.loc[:, nunique[nunique > 1].index]
    if numeric_df.shape[1] >= 2:
        import numpy as _np
        with _np.errstate(invalid='ignore', divide='ignore'):
            corr = numeric_df.corr()
        if not corr.isnull().values.all():
            # Keep top-N by absolute correlation to avoid unreadable grids
            abs_corr = corr.abs()
            # Pick top 25 features by max abs correlation with any target-like column if present
            if abs_corr.shape[0] > 30:
                top_idx = abs_corr.max(axis=1).sort_values(ascending=False).head(30).index
                corr = corr.loc[top_idx, top_idx]
            plt.figure(figsize=(min(0.6 * corr.shape[1] + 4, 18), min(0.6 * corr.shape[0] + 4, 18)))
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle mask (exclude diagonal)
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.3, cbar_kws={"shrink": 0.8})
            plt.title("Feature Correlation Heatmap")
            save_plot(os.path.join(PLOT_DIR, "feature_correlation_heatmap.png"))

def plot_outperformance_heatmap(df: pd.DataFrame):
    if {"Signal Δ 3D", "Sector", "Run Datetime"}.issubset(df.columns):
        df = df[df["Run Datetime"].notnull()].copy()
        df["Weekday"] = df["Run Datetime"].dt.day_name()
        pivot = df.pivot_table(values="Signal Δ 3D", index="Sector", columns="Weekday", aggfunc="mean")
        if not pivot.empty and not pivot.isnull().all().all():
            # Reorder weekdays and re-validate
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            pivot = pivot[[c for c in order if c in pivot.columns]]
            if pivot.shape[0] > 0 and pivot.shape[1] > 0 and not pivot.isna().all().all():
                vals = pivot.to_numpy(dtype=float)
                if np.isfinite(vals).any():
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.3, cbar_kws={"shrink": 0.8})
                    plt.title("Signal Δ 3D by Weekday and Sector")
                    save_plot(os.path.join(PLOT_DIR, "outperformance_heatmap.png"))

def plot_signal_duration_hist(df: pd.DataFrame):
    """Histogram of signal duration with robust numeric coercion."""
    if "Signal Duration" not in df.columns:
        return
    durations = pd.to_numeric(df["Signal Duration"], errors="coerce").dropna()
    if durations.empty:
        return
    # Freedman–Diaconis bin width
    try:
        iqr = float(durations.quantile(0.75) - durations.quantile(0.25))
        data_range = float(durations.max() - durations.min())
        if iqr <= 0 or data_range <= 0:
            bins = min(max(int(len(durations) ** 0.5), 10), 50)
        else:
            bin_w = 2 * iqr / max(len(durations) ** (1/3), 1)
            if bin_w <= 0:
                bins = min(max(int(len(durations) ** 0.5), 10), 50)
            else:
                bins = max(int(data_range / bin_w), 10)
    except Exception:
        bins = 20
    plt.figure(figsize=(8, 4.5))
    plt.hist(durations, bins=bins, edgecolor="black", color="#60a5fa")
    med = float(durations.median())
    plt.axvline(med, color="#ef4444", linestyle="--", label=f"Median {med:.1f}d")
    plt.title("Signal Duration Histogram")
    plt.xlabel("Days Held")
    plt.legend()
    save_plot(os.path.join(PLOT_DIR, "signal_duration_hist.png"))

def plot_score_return_trend(df: pd.DataFrame):
    if {"Run Datetime", "Score (0–100)", "3D Return"}.issubset(df.columns):
        df = df.dropna(subset=["Run Datetime", "Score (0–100)", "3D Return"])
        df = df.sort_values("Run Datetime").copy()
        df["Date"] = df["Run Datetime"].dt.date
        daily = df.groupby("Date")[["Score (0–100)", "3D Return"]].mean()
        if not daily.empty:
            roll = daily.rolling(7).mean()
            plt.figure(figsize=(10, 5))
            ax = roll["Score (0–100)"].plot(color="#0ea5e9", label="Score (7D avg)")
            ax2 = ax.twinx()
            roll["3D Return"].plot(ax=ax2, color="#22c55e", label="3D Return (7D avg)")
            ax.set_ylabel("Score (0–100)")
            ax2.set_ylabel("3D Return %")
            y_scale = _guess_percent_scale(roll["3D Return"]) if not roll["3D Return"].dropna().empty else 1.0
            ax2.yaxis.set_major_formatter(_make_percent_formatter(scale=y_scale))
            ax.grid(True, color="#eaeaea")
            ax.set_title("7D Rolling Avg: Score vs 3D Return")
            # Build a combined legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper left")
            save_plot(os.path.join(PLOT_DIR, "score_return_trend.png"))

def plot_signaltype_vs_weekday(df: pd.DataFrame):
    if {"Signal Type", "Run Datetime", "3D Return"}.issubset(df.columns):
        df = df.dropna(subset=["Signal Type", "Run Datetime", "3D Return"]).copy()
        df["Weekday"] = df["Run Datetime"].dt.day_name()
        pivot = df.pivot_table(index="Signal Type", columns="Weekday", values="3D Return", aggfunc="mean")
        # Guard: seaborn heatmap fails if all values are NaN (zero-size reduction)
        if not pivot.empty and not pivot.isna().all().all():
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            pivot = pivot[[c for c in order if c in pivot.columns]]
            if pivot.shape[0] > 0 and pivot.shape[1] > 0 and not pivot.isna().all().all():
                vals = pivot.to_numpy(dtype=float)
                if np.isfinite(vals).any():
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.3)
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
    # Drop constant feature columns to avoid invalid operations
    nunique = df_corr[features].nunique()
    features = [f for f in features if nunique.get(f, 0) > 1]
    import numpy as _np
    with _np.errstate(invalid='ignore', divide='ignore'):
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

def export_score_quantile_performance(df: pd.DataFrame):
    """Export performance by score quantile to help assess calibration.

    Emits CSV with mean forward returns and beat-SPY rates per score decile.
    """
    required = {"Score (0–100)", "1D Return", "3D Return", "10D Return"}
    cols_present = required.intersection(df.columns)
    if "Score (0–100)" not in cols_present:
        return
    d = df.dropna(subset=["Score (0–100)"]).copy()
    if d.empty:
        return
    d["score_decile"] = pd.qcut(d["Score (0–100)"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    metrics = {}
    for w in [1, 3, 10]:
        col = f"{w}D Return"
        if col in d.columns:
            metrics[f"avg_{w}d"] = (col, "mean")
            metrics[f"med_{w}d"] = (col, "median")
            # Beat SPY if benchmark present
            spy_col = f"SPY {w}D Return"
            if spy_col in d.columns:
                d[f"beat_spy_{w}d"] = (d[col] > d[spy_col]).astype(float)
                metrics[f"beat_spy_{w}d"] = (f"beat_spy_{w}d", "mean")
    if not metrics:
        return
    agg = d.groupby("score_decile", observed=False).agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in metrics.items()}, count=("Score (0–100)", "size")).reset_index()
    # Coerce to numeric before rounding in case of mixed dtypes
    for k in list(metrics.keys()):
        agg[k] = pd.to_numeric(agg[k], errors="coerce")
        if k.startswith("avg_") or k.startswith("med_"):
            agg[k] = agg[k].round(2)
        if k.startswith("beat_spy_"):
            agg[k] = (agg[k] * 100).round(1)
    save_table(agg, os.path.join(TABLE_DIR, "score_quantile_performance.csv"))

def plot_score_quantile_performance():
    """Plot score decile vs average forward returns if CSV exists."""
    path = os.path.join(TABLE_DIR, "score_quantile_performance.csv")
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    # Melt averages for plotting
    avg_cols = [c for c in df.columns if c.startswith("avg_")]
    if not avg_cols:
        return
    m = df.melt(id_vars=["score_decile"], value_vars=avg_cols, var_name="window", value_name="avg_return")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=m, x="score_decile", y="avg_return", hue="window", palette="Blues")
    plt.title("Average Forward Return by Score Decile")
    plt.ylabel("Avg Return %")
    plt.xlabel("Score Decile (low→high)")
    plt.gca().yaxis.set_major_formatter(_make_percent_formatter(scale=1.0))
    save_plot(os.path.join(PLOT_DIR, "score_decile_performance.png"), dpi=160)

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
    export_score_quantile_performance(df)
    plot_score_quantile_performance()
    generate_html_dashboard(df)

def generate_html_dashboard(df: pd.DataFrame):
    """Generate charts dashboard with proper meta tags and external CSS.

    Removes inline <style> and style="..." attributes for accessibility tooling.
    """
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

    # Ensure dashboard directory and write CSS
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    css_path = os.path.join(DASHBOARD_DIR, "styles.css")
    css = (
        "body{font-size:0.95rem;}" \
        "img{max-width:100%;height:auto;margin-bottom:20px;}" \
        ".table-sm th,.table-sm td{padding:0.3rem;}"
    )
    try:
        with open(css_path, "w", encoding="utf-8") as cf:
            cf.write(css)
    except Exception:
        pass

    env = Environment(loader=FileSystemLoader(searchpath="."), autoescape=select_autoescape())
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>VP Investments Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="styles.css" rel="stylesheet">
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

    # Strip inline style attributes introduced by pandas
    import re as _re
    html = _re.sub(r"\sstyle=\"[^\"]*\"", "", html)

    output_path = os.path.join(DASHBOARD_DIR, "dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DASHBOARD] Saved: {output_path}")