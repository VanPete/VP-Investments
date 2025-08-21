import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import sqlite3
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
from config.labels import FINAL_COLUMN_ORDER
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from config.config import OUTPUTS_DIR, SLIPPAGE_BPS, FEES_BPS
from config.config import TURNOVER_TOP_K
from config.config import ERROR_BUDGET_SUCCESS_TARGET
from config.config import DB_PATH

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
    # Use Weighted Score deciles to assess calibration
    if {"1D Return", "Weighted Score"}.issubset(df.columns):
        d = df.dropna(subset=["1D Return", "Weighted Score"]).copy()
        if not d.empty:
            try:
                d["score_decile"] = pd.qcut(d["Weighted Score"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
            except Exception:
                # Fallback to rank-based buckets
                ranks = d["Weighted Score"].rank(method="average", pct=True)
                d["score_decile"] = (ranks * 10).clip(1, 10).round().astype(int).map(lambda x: f"D{int(x)}")
            grouped = d.groupby("score_decile", observed=False)["1D Return"].agg(["mean", "count", "std"]).reset_index()
            grouped["se"] = grouped["std"].div(grouped["count"].clip(lower=1) ** 0.5)
            plt.figure(figsize=(9, 5))
            ax = sns.barplot(data=grouped, x="score_decile", y="mean", color="#6baed6", edgecolor="#3b82f6")
            ax.errorbar(x=range(len(grouped)), y=grouped["mean"], yerr=grouped["se"], fmt="none", ecolor="#1f2937", capsize=3)
            plt.title("Avg 1D Return by Weighted Score Decile (± SE)")
            plt.ylabel("Avg 1D Return %")
            plt.xlabel("Weighted Score Decile (low→high)")
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
    if {"Run Datetime", "Weighted Score", "3D Return"}.issubset(df.columns):
        df = df.dropna(subset=["Run Datetime", "Weighted Score", "3D Return"])
        df = df.sort_values("Run Datetime").copy()
        df["Date"] = df["Run Datetime"].dt.date
        daily = df.groupby("Date")[ ["Weighted Score", "3D Return"] ].mean()
        if not daily.empty:
            roll = daily.rolling(7).mean()
            plt.figure(figsize=(10, 5))
            ax = roll["Weighted Score"].plot(color="#0ea5e9", label="Weighted Score (7D avg)")
            ax2 = ax.twinx()
            roll["3D Return"].plot(ax=ax2, color="#22c55e", label="3D Return (7D avg)")
            ax.set_ylabel("Weighted Score")
            ax2.set_ylabel("3D Return %")
            y_scale = _guess_percent_scale(roll["3D Return"]) if not roll["3D Return"].dropna().empty else 1.0
            ax2.yaxis.set_major_formatter(_make_percent_formatter(scale=y_scale))
            ax.grid(True, color="#eaeaea")
            ax.set_title("7D Rolling Avg: Weighted Score vs 3D Return")
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
    # Build target list dynamically based on availability
    candidate_targets = ["3D Return", "Weighted Score"]
    targets = [t for t in candidate_targets if t in df.columns]
    if not targets:
        # Nothing to correlate against yet
        return
    # Numeric features excluding targets and obvious non-numerics
    features = [col for col in df.columns if col not in targets and df[col].dtype != 'O']
    if not features:
        return
    # Restrict frame to available cols and rows that have target values
    use_cols = [c for c in (features + targets) if c in df.columns]
    df_corr = df[use_cols].copy()
    df_corr = df_corr.dropna(subset=targets)
    if df_corr.empty:
        return
    # Drop constant feature columns to avoid invalid operations
    nunique = df_corr[features].nunique()
    features = [f for f in features if nunique.get(f, 0) > 1]
    if not features:
        return
    import numpy as _np
    with _np.errstate(invalid='ignore', divide='ignore'):
        corr_result = {target: df_corr[features].corrwith(df_corr[target]).dropna() for target in targets}
    output_df = pd.DataFrame(corr_result)
    # Sort by the first available target for stability
    sort_col = targets[0]
    if sort_col in output_df.columns:
        output_df = output_df.sort_values(by=sort_col, ascending=False)
    save_table(output_df.reset_index().rename(columns={"index": "Feature"}), os.path.join(TABLE_DIR, "feature_correlations.csv"))

def export_model_comparison() -> None:
    """Export a compact model comparison table from experiments+metrics.

    Columns: exp_id, profile, target, model, started_at, ended_at, test_r2, test_mae, best_fold_r2, best_fold_mae
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            exps = pd.read_sql("SELECT id as exp_id, profile, params_json, started_at, ended_at FROM experiments", conn)
            if exps.empty:
                return
            mets = pd.read_sql("SELECT exp_id, name, value FROM metrics", conn)
    except Exception:
        return
    if mets.empty:
        return
    # Parse params for target/model
    def _parse_params(s: str) -> tuple[str | None, str | None]:
        try:
            obj = json.loads(s) if isinstance(s, str) and s else {}
            return obj.get("target"), obj.get("model")
        except Exception:
            return None, None
    exps[["target", "model"]] = exps.apply(lambda r: pd.Series(_parse_params(r.get("params_json"))), axis=1)
    # Aggregate metrics
    pivot = mets.pivot_table(index=["exp_id"], columns="name", values="value", aggfunc="last")
    pivot = pivot.reset_index()
    # Derive best fold metrics if present
    fold_cols_r2 = [c for c in pivot.columns if isinstance(c, str) and c.endswith("_test_r2")]
    fold_cols_mae = [c for c in pivot.columns if isinstance(c, str) and c.endswith("_test_mae")]
    def _best(series_like: pd.Series) -> float | None:
        try:
            s = pd.to_numeric(series_like, errors="coerce")
            return float(s.max()) if s.notna().any() else None
        except Exception:
            return None
    pivot["best_fold_r2"] = _best(pivot[fold_cols_r2]) if fold_cols_r2 else None
    pivot["best_fold_mae"] = _best(pivot[fold_cols_mae]) if fold_cols_mae else None
    # Keep key columns
    keep = ["exp_id", "test_r2", "test_mae", "best_fold_r2", "best_fold_mae"]
    for k in list(keep):
        if k not in pivot.columns:
            pivot[k] = None
    out = exps.merge(pivot[keep], on="exp_id", how="left")
    # Order and save
    cols = ["exp_id", "profile", "target", "model", "started_at", "ended_at", "test_r2", "test_mae", "best_fold_r2", "best_fold_mae"]
    out = out[cols]
    save_table(out, os.path.join(TABLE_DIR, "Model_Comparison.csv"))

def export_feature_standardization_audit() -> None:
    """Export cross-sectional standardization audit for features.

    Computes per-feature overall stats and per-run cross-sectional mean/std, then
    aggregates across runs to assess centering and scaling.
    Output: outputs/tables/feature_standardization_audit.csv
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            f = pd.read_sql("SELECT run_id, ticker, key, value FROM features", conn)
    except Exception:
        return
    if f.empty:
        return
    # Coerce numeric and drop NaNs
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    f = f.dropna(subset=["value"]) 
    if f.empty:
        return
    # Overall stats per key
    overall = f.groupby("key").agg(
        count=("value", "size"),
        runs_covered=("run_id", pd.Series.nunique),
        mean_all=("value", "mean"),
        std_all=("value", "std"),
        min_all=("value", "min"),
        max_all=("value", "max"),
    )
    # Per-run cross-sectional mean/std
    cs = f.groupby(["key", "run_id"]).agg(cs_mean=("value", "mean"), cs_std=("value", "std")).reset_index()
    # Aggregate across runs
    agg = cs.groupby("key").agg(
        runs_with_values=("run_id", pd.Series.nunique),
        avg_cs_mean=("cs_mean", "mean"),
        med_cs_mean=("cs_mean", "median"),
        avg_cs_std=("cs_std", "mean"),
        med_cs_std=("cs_std", "median"),
        std_of_cs_std=("cs_std", "std")
    )
    out = overall.join(agg, how="left")
    # Centering and scaling proximity (informational)
    out["center_bias"] = out["avg_cs_mean"].abs()
    out["scale_bias"] = (out["avg_cs_std"] - 1.0).abs()
    out = out.reset_index().rename(columns={"key": "feature"})
    save_table(out, os.path.join(TABLE_DIR, "feature_standardization_audit.csv"))

def export_feature_drift_summary(last_runs: int = 20) -> None:
    """Export a simple drift summary for features across recent runs.

    Uses cross-sectional means per run and reports volatility of cs_mean.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            f = pd.read_sql("SELECT run_id, ticker, key, value FROM features", conn)
            runs = pd.read_sql("SELECT run_id, started_at FROM runs", conn)
    except Exception:
        return
    if f.empty or runs.empty:
        return
    runs = runs.dropna(subset=["started_at"]).copy()
    runs["started_at"] = pd.to_datetime(runs["started_at"], errors="coerce")
    runs = runs.dropna(subset=["started_at"]).sort_values("started_at").tail(last_runs)
    f = f.merge(runs[["run_id"]], on="run_id", how="inner")
    if f.empty:
        return
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    f = f.dropna(subset=["value"]) 
    cs = f.groupby(["key", "run_id"]).agg(cs_mean=("value", "mean")).reset_index()
    # Drift = std of cs_mean across runs; also max abs deviation from overall mean
    overall = f.groupby("key")["value"].mean().rename("overall_mean")
    g = cs.groupby("key")["cs_mean"].agg(["std", "mean"]).rename(columns={"std": "cs_mean_std", "mean": "cs_mean_mean"})
    out = g.join(overall, how="left")
    out["max_abs_dev"] = (out["cs_mean_mean"] - out["overall_mean"]).abs()
    out = out.reset_index().rename(columns={"key": "feature"})
    save_table(out, os.path.join(TABLE_DIR, "feature_drift_summary.csv"))

def export_feature_importance_corr(return_target: str = "3D Return") -> None:
    """Quick correlation-based importance between features and target return.

    Joins normalized features with labels for the requested window.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            f = pd.read_sql("SELECT run_id, ticker, key, value FROM features", conn)
            l = pd.read_sql("SELECT run_id, ticker, window, fwd_return FROM labels", conn)
    except Exception:
        return
    if f.empty or l.empty:
        return
    win = return_target.replace(" Return", "")
    l = l[l["window"].astype(str) == win].copy()
    if l.empty:
        return
    piv = f.pivot_table(index=["run_id", "ticker"], columns="key", values="value", aggfunc="first").reset_index()
    df = piv.merge(l[["run_id", "ticker", "fwd_return"]], on=["run_id", "ticker"], how="inner").dropna()
    if df.shape[0] < 3:
        return
    X = df.drop(columns=["run_id", "ticker", "fwd_return"]).select_dtypes(include="number")
    y = df["fwd_return"].astype(float)
    if X.empty:
        return
    corr = X.apply(lambda col: pd.Series({"corr": pd.to_numeric(col, errors="coerce").corr(y)}))
    corr = corr.reset_index().rename(columns={"index": "feature"})
    save_table(corr, os.path.join(TABLE_DIR, "feature_importance_corr.csv"))

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
    required = {"Weighted Score", "1D Return", "3D Return", "10D Return"}
    cols_present = required.intersection(df.columns)
    if "Weighted Score" not in cols_present:
        return
    d = df.dropna(subset=["Weighted Score"]).copy()
    if d.empty:
        return
    d["score_decile"] = pd.qcut(d["Weighted Score"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
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
    agg = d.groupby("score_decile", observed=False).agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in metrics.items()}, count=("Weighted Score", "size")).reset_index()
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

def export_score_quantile_performance_net(df: pd.DataFrame):
    """Export performance by score decile using net forward returns if present."""
    if "Weighted Score" not in df.columns:
        return
    d = df.dropna(subset=["Weighted Score"]).copy()
    if d.empty:
        return
    d["score_decile"] = pd.qcut(d["Weighted Score"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    metrics = {}
    for w in [1, 3, 10]:
        col = f"{w}D Return (net)"
        if col in d.columns:
            metrics[f"avg_{w}d_net"] = (col, "mean")
            metrics[f"med_{w}d_net"] = (col, "median")
    if not metrics:
        return
    agg = d.groupby("score_decile", observed=False).agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in metrics.items()}, count=("Weighted Score", "size")).reset_index()
    for k in list(metrics.keys()):
        agg[k] = pd.to_numeric(agg[k], errors="coerce").round(2)
    save_table(agg, os.path.join(TABLE_DIR, "score_quantile_performance_net.csv"))

def plot_score_quantile_performance_net():
    path = os.path.join(TABLE_DIR, "score_quantile_performance_net.csv")
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    avg_cols = [c for c in df.columns if c.startswith("avg_")]
    if not avg_cols:
        return
    m = df.melt(id_vars=["score_decile"], value_vars=avg_cols, var_name="window", value_name="avg_return")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=m, x="score_decile", y="avg_return", hue="window", palette="Greens")
    plt.title("Average Net Forward Return by Score Decile")
    plt.ylabel("Avg Return %")
    plt.xlabel("Score Decile (low→high)")
    plt.gca().yaxis.set_major_formatter(_make_percent_formatter(scale=1.0))
    save_plot(os.path.join(PLOT_DIR, "score_decile_performance_net.png"), dpi=160)

def export_backtest_cost_summary(df: pd.DataFrame):
    rows = []
    for w in [1, 3, 7, 10]:
        g = f"{w}D Return"
        n = f"{w}D Return (net)"
        if g in df.columns and n in df.columns:
            s_g = pd.to_numeric(df[g], errors="coerce")
            s_n = pd.to_numeric(df[n], errors="coerce")
            m = s_g.notna() & s_n.notna()
            if m.any():
                gg = float(s_g[m].mean())
                nn = float(s_n[m].mean())
                rows.append({
                    "window": f"{w}D",
                    "rows": int(m.sum()),
                    "avg_gross_%": round(gg, 3),
                    "avg_net_%": round(nn, 3),
                    "avg_cost_%": round(gg - nn, 3)
                })
    if rows:
        save_table(pd.DataFrame(rows), os.path.join(TABLE_DIR, "backtest_cost_summary.csv"))

def export_daily_turnover(df: pd.DataFrame, top_k: int = TURNOVER_TOP_K):
    """Compute daily turnover of top-K signals by Weighted Score.

    Turnover_t = 1 - |TopK_t ∩ TopK_{t-1}| / K
    """
    if {"Run Datetime", "Ticker", "Weighted Score"}.issubset(df.columns) is False:
        return
    d = df.dropna(subset=["Run Datetime", "Ticker", "Weighted Score"]).copy()
    if d.empty:
        return
    d["Date"] = pd.to_datetime(d["Run Datetime"], errors="coerce").dt.date
    if d["Date"].isna().all():
        return
    daily = []
    prev_set = None
    for day, g in d.groupby("Date", sort=True):
        g = g.sort_values("Weighted Score", ascending=False)
        top = list(g["Ticker"].astype(str).head(top_k))
        cur_set = set(top)
        if prev_set is None:
            overlap = 0
            turnover = None
        else:
            overlap = len(cur_set & prev_set)
            turnover = round(1.0 - (overlap / float(top_k)), 4)
        daily.append({
            "date": str(day),
            "k": int(top_k),
            "overlap": int(overlap),
            "turnover": turnover
        })
        prev_set = cur_set
    if daily:
        df_out = pd.DataFrame(daily)
        save_table(df_out, os.path.join(TABLE_DIR, "daily_turnover.csv"))

def export_turnover_summary(df: pd.DataFrame, top_k: int = TURNOVER_TOP_K):
    """Summarize turnover and implied cost drag from slippage+fees.

    Uses average daily turnover and cost in percent derived from (SLIPPAGE_BPS+FEES_BPS)/100.
    """
    path = os.path.join(TABLE_DIR, "daily_turnover.csv")
    if not os.path.exists(path):
        export_daily_turnover(df, top_k=top_k)
    if not os.path.exists(path):
        return
    try:
        tdf = pd.read_csv(path)
    except Exception:
        return
    t = pd.to_numeric(tdf["turnover"], errors="coerce").dropna()
    if t.empty:
        return
    avg_daily_turn = float(t.mean())
    med_daily_turn = float(t.median())
    # Cost per rebalance (%): (bps_in + bps_out) / 100; we model a single round-trip
    try:
        cost_pct = (float(SLIPPAGE_BPS) + float(FEES_BPS)) / 100.0
    except Exception:
        cost_pct = 0.0
    # Implied expected daily cost from turnover of fraction of portfolio
    exp_daily_cost_pct = avg_daily_turn * cost_pct
    # Rough month = 21 trading days, year = 252
    exp_monthly_cost_pct = exp_daily_cost_pct * 21
    exp_annual_cost_pct = exp_daily_cost_pct * 252
    summary = pd.DataFrame([
        {
            "k": int(top_k),
            "avg_daily_turnover": round(avg_daily_turn, 4),
            "med_daily_turnover": round(med_daily_turn, 4),
            "cost_per_roundtrip_pct": round(cost_pct, 4),
            "exp_daily_cost_pct": round(exp_daily_cost_pct, 4),
            "exp_monthly_cost_pct": round(exp_monthly_cost_pct, 2),
            "exp_annual_cost_pct": round(exp_annual_cost_pct, 2),
        }
    ])
    save_table(summary, os.path.join(TABLE_DIR, "turnover_summary.csv"))

def plot_daily_turnover():
    path = os.path.join(TABLE_DIR, "daily_turnover.csv")
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    if df.empty or "turnover" not in df.columns:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(df["date"]), df["turnover"].astype(float), color="#fb7185")
    plt.title("Daily Turnover of Top-K Signals")
    plt.ylabel("Turnover (fraction)")
    plt.xlabel("Date")
    plt.grid(True, color="#eaeaea")
    save_plot(os.path.join(PLOT_DIR, "daily_turnover.png"))

def export_turnover_by_group(df: pd.DataFrame, group_col: str, top_k: int = TURNOVER_TOP_K, out_name: str = "turnover_by_group.csv"):
    """Compute daily top-K turnover within each group (e.g., Sector or Trade Type)."""
    required = {"Run Datetime", "Ticker", "Weighted Score", group_col}
    if not required.issubset(df.columns):
        return
    d = df.dropna(subset=["Run Datetime", "Ticker", "Weighted Score", group_col]).copy()
    if d.empty:
        return
    d["Date"] = pd.to_datetime(d["Run Datetime"], errors="coerce").dt.date
    if d["Date"].isna().all():
        return
    rows = []
    # Precompute previous day's top per group
    d_sorted = d.sort_values([group_col, "Date", "Weighted Score"], ascending=[True, True, False])
    for (gval, day), g in d_sorted.groupby([group_col, "Date" ], sort=True):
        top_today = set(g["Ticker"].astype(str).head(top_k))
        prev_mask = (d_sorted[group_col] == gval) & (d_sorted["Date"] < day)
        prev = d_sorted.loc[prev_mask]
        if prev.empty:
            overlap, turnover = 0, None
        else:
            prev_day = prev["Date"].max()
            prev_top = set(prev[prev["Date"] == prev_day]["Ticker"].astype(str).head(top_k))
            overlap = len(top_today & prev_top)
            turnover = round(1.0 - (overlap / float(top_k)), 4)
        rows.append({
            "group": gval,
            "date": str(day),
            "k": int(top_k),
            "overlap": int(overlap),
            "turnover": turnover,
        })
    if rows:
        save_table(pd.DataFrame(rows), os.path.join(TABLE_DIR, out_name))

def plot_turnover_by_group(out_name: str, plot_name: str) -> None:
    path = os.path.join(TABLE_DIR, out_name)
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    if df.empty or "turnover" not in df.columns:
        return
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        return
    plt.figure(figsize=(11, 6))
    med = df.groupby(["group", "date"])['turnover'].median().reset_index()
    for gval, g in med.groupby("group"):
        plt.plot(g["date"], g["turnover"], label=str(gval))
    plt.title("Daily Turnover by Group")
    plt.ylabel("Turnover (fraction)")
    plt.xlabel("Date")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.grid(True, color="#eaeaea")
    save_plot(os.path.join(PLOT_DIR, plot_name))

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
    # Observability: error budget trend
    try:
        plot_error_budget_trend()
    except Exception:
        pass
    summarize_factor_returns(df)
    export_top_signals(df)
    export_breakout_csvs(df)
    export_feature_correlations(df)
    # DB-driven exports
    export_model_comparison()
    export_feature_standardization_audit()
    export_feature_drift_summary()
    export_feature_importance_corr()
    export_metadata_summary(df)
    export_score_quantile_performance(df)
    plot_score_quantile_performance()
    export_score_quantile_performance_net(df)
    plot_score_quantile_performance_net()
    export_backtest_cost_summary(df)
    export_daily_turnover(df, top_k=TURNOVER_TOP_K)
    export_turnover_summary(df, top_k=TURNOVER_TOP_K)
    plot_daily_turnover()
    # By Sector and Trade Type
    if "Sector" in df.columns:
        export_turnover_by_group(df, "Sector", top_k=TURNOVER_TOP_K, out_name="turnover_by_sector.csv")
        plot_turnover_by_group("turnover_by_sector.csv", "turnover_by_sector.png")
    if "Trade Type" in df.columns:
        export_turnover_by_group(df, "Trade Type", top_k=TURNOVER_TOP_K, out_name="turnover_by_tradetype.csv")
        plot_turnover_by_group("turnover_by_tradetype.csv", "turnover_by_tradetype.png")
    generate_html_dashboard(df)
    # Snapshot selected tables into current run's References folder if RUN_DIR is set
    try:
        run_dir = os.environ.get("RUN_DIR")
        if run_dir:
            refs = os.path.join(run_dir, "References")
            os.makedirs(refs, exist_ok=True)
            for name in [
                "score_quantile_performance.csv",
                "score_quantile_performance_net.csv",
                "backtest_cost_summary.csv",
                "turnover_summary.csv",
                "turnover_by_sector.csv",
                "turnover_by_tradetype.csv",
                "daily_turnover.csv",
                "Error_Budget_Trend.csv",
                "Model_Comparison.csv",
                "feature_standardization_audit.csv",
                "feature_drift_summary.csv",
                "feature_importance_corr.csv",
                "metadata_summary.json",
            ]:
                src = os.path.join(TABLE_DIR, name)
                if os.path.exists(src):
                    import shutil
                    shutil.copy2(src, os.path.join(refs, name))
    except Exception:
        pass

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
    for name in [
        "factor_return_summary",
        "feature_correlations",
    "Model_Comparison",
    "feature_standardization_audit",
    "feature_drift_summary",
    "feature_importance_corr",
        "score_quantile_performance",
        "score_quantile_performance_net",
        "backtest_cost_summary",
        "turnover_summary",
        "turnover_by_sector",
        "turnover_by_tradetype",
        "Error_Budget_Trend",
    ]:
        csv_path = os.path.join(TABLE_DIR, f"{name}.csv")
        if os.path.exists(csv_path):
            label = name.replace("_", " ").title()
            # Slightly nicer label for the error budget trend
            if name == "Error_Budget_Trend":
                label = "Error Budget Trend"
            tables[label] = pd.read_csv(csv_path)

    top_signals = df.sort_values("Weighted Score", ascending=False).head(10).copy()
    display_cols = ["Rank", "Ticker", "Company", "Sector", "Trade Type",
                    "Weighted Score", "Reddit Sentiment", "News Sentiment", "3D Return"]
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


def plot_error_budget_trend(days: int = 30):
    """Plot component success-rate trend over the last N days and export CSV if needed."""
    # Try computing directly; fall back to CSV if present
    try:
        from processors.error_budget_trend import compute_error_budget_trend, export_error_budget_trend
    except Exception:
        return
    try:
        df = compute_error_budget_trend(days)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        # Try read from tables
        csv_path = os.path.join(TABLE_DIR, "Error_Budget_Trend.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = pd.DataFrame()
    else:
        # Save a fresh CSV for dashboard consumption
        try:
            export_error_budget_trend(days)
        except Exception:
            pass
    if df is None or df.empty:
        return
    # Pivot for plotting lines per component
    try:
        df_plot = df.copy()
        df_plot["day"] = pd.to_datetime(df_plot["day"], errors="coerce")
        df_plot = df_plot.dropna(subset=["day"])  # ensure valid dates
        wide = df_plot.pivot_table(index="day", columns="component", values="success_rate", aggfunc="mean")
        if wide.empty:
            return
        plt.figure(figsize=(10, 5))
        for col in wide.columns:
            plt.plot(wide.index, wide[col] * 100.0, marker="o", linewidth=1.6, markersize=3, label=str(col))
        # Target line
        try:
            target = float(ERROR_BUDGET_SUCCESS_TARGET) * 100.0
            plt.axhline(target, color="#ef4444", linestyle="--", linewidth=1.2, label=f"Target {target:.0f}%")
        except Exception:
            pass
        plt.title("Fetch Success Rate Trend (last 30 days)")
        plt.ylabel("Success Rate %")
        plt.xlabel("Day (UTC)")
        plt.grid(True, color="#eaeaea")
        plt.legend(fontsize=8, ncol=2, loc="lower right")
        save_plot(os.path.join(PLOT_DIR, "error_budget_trend.png"), dpi=170)
    except Exception:
        pass