import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import xlsxwriter
from typing import Optional
import numpy as np
import sqlite3

from config.labels import FINAL_COLUMN_ORDER, COLUMN_FORMAT_HINTS
from utils.db import insert_metric
from datetime import datetime, timezone
from config.config import PERCENT_NORMALIZE, LIQUIDITY_WARNING_ADV
from config.config import OUTPUTS_DIR
from config.report_legend import LEGEND

INVERSE_COLUMNS = {"Volatility", "Debt/Equity", "P/E Ratio"}

TRADE_TYPE_COLORS = {
    "Swing": "#FFD700", "Momentum": "#FFA07A", "Growth": "#98FB98",
    "Value": "#ADD8E6", "Speculative": "#DDA0DD"
}

HEADER_STYLE = {
    'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center',
    'font_size': 11, 'fg_color': '#C6E2FF', 'border': 2, 'bottom': 6
}

ZEBRA_COLOR = "#F2F2F2"
ZERO_FONT_COLOR = "#B0B0B0"
SUBDUED_TEXT_COLOR = "#666666"
YAHOO_FINANCE_URL = "https://finance.yahoo.com/quote/{ticker}"

COLUMN_FORMAT_HINTS.update({
    # Z-score variants (support both naming styles)
    "Reddit Sentiment Z-Score": "float",
    "Price 7D % Z-Score": "float",
    "Volume Spike Ratio Z-Score": "float",
    "Z-Score: Market Cap": "float",
    "Z-Score: Avg Daily Value": "float",
    "Z-Score: Reddit Activity": "float",
    # Text flags
    "Liquidity Warning": "text",
    # Percent-like columns (rendered as 0.00%)
    "Price 1D %": "percent",
    "Price 7D %": "percent",
    "Above 50-Day MA %": "percent",
    "Above 200-Day MA %": "percent",
    "Momentum 30D %": "percent",
    "Retail Holding %": "percent",
    "Short Percent Float": "percent",
    "Short Percent Outstanding": "percent",
    "EPS Growth": "percent",
    "ROE": "percent",
    "FCF Margin": "percent",
    "IV Spike %": "percent",
    # Large numeric columns (render as human-readable K/M/B/T)
    "Market Cap": "human",
    "Volume": "human",
    "Avg Daily Value Traded": "human",
    "Shares Short": "human",
    # Keep as currency to reflect dollars
    "Insider Buy Volume": "currency",
    "Sector Inflows": "currency",
    # Currency
    "Current Price": "currency"
})

def human_format(num):
    if pd.isna(num): return ""
    try: num = float(num)
    except (ValueError, TypeError): return ""
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000
    return f"{num:.2f}P"

def autosize_and_center(ws, df_like, workbook):
    center_fmt = workbook.add_format({'align': 'center'})
    for idx, col in enumerate(df_like.columns):
        width = max(df_like[col].astype(str).map(len).max(), len(str(col))) + 2
        ws.set_column(idx, idx, max(min(width, 50), 12), center_fmt)


def compute_backtest_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple backtest diagnostics from current DataFrame if future return columns exist.

    Returns a table with one row per available return window containing:
    - Count: rows with both Weighted Score and the window's return present
    - Pearson IC / Spearman IC between Weighted Score and the return
    - Top-Decile Avg Return vs Rest Avg Return and Excess (Top - Rest)
    - Hit Rate in Top Decile (share of names with positive return)
    """
    rows = []
    if "Weighted Score" not in df.columns:
        return pd.DataFrame(columns=[
            "Window", "Count", "Pearson IC", "Spearman IC", "Top-Decile Avg %", "Rest Avg %", "Excess %", "Top-Decile Hit %"
        ])

    score = pd.to_numeric(df["Weighted Score"], errors="coerce")
    ret_cols = [c for c in ("3D Return", "10D Return") if c in df.columns]
    for col in ret_cols:
        r = pd.to_numeric(df[col], errors="coerce")
        mask = score.notna() & r.notna()
        if not mask.any():
            continue
        s = score[mask]
        y = r[mask]
        n = int(mask.sum())
        pearson = float(round(s.corr(y, method="pearson"), 4)) if n > 2 else None
        spearman = float(round(s.corr(y, method="spearman"), 4)) if n > 2 else None
        # Top decile split by score
        try:
            ranks = s.rank(pct=True)
            top_mask = ranks >= 0.9
        except Exception:
            # Fallback: quantile threshold
            thr = s.quantile(0.9)
            top_mask = s >= thr
        top_mean = float(round(y[top_mask].mean(), 2)) if top_mask.any() else None
        rest_mean = float(round(y[~top_mask].mean(), 2)) if (~top_mask).any() else None
        excess = float(round((top_mean - rest_mean), 2)) if top_mean is not None and rest_mean is not None else None
        hit = float(round((y[top_mask] > 0).mean() * 100.0, 2)) if top_mask.any() else None
        rows.append({
            "Window": col,
            "Count": n,
            "Pearson IC": pearson,
            "Spearman IC": spearman,
            "Top-Decile Avg %": top_mean,
            "Rest Avg %": rest_mean,
            "Excess %": excess,
            "Top-Decile Hit %": hit,
        })
    return pd.DataFrame(rows, columns=[
        "Window", "Count", "Pearson IC", "Spearman IC", "Top-Decile Avg %", "Rest Avg %", "Excess %", "Top-Decile Hit %"
    ])


def compute_and_save_daily_ic_pk(df: pd.DataFrame, run_dir: str, ks: list[int] | None = None) -> None:
    """Compute daily Spearman IC and precision@K per available return window and persist CSVs.

    - Groups by date from 'Run Datetime'.
    - For each day, computes Spearman IC between Weighted Score and return (3D/10D if present).
    - precision@K: fraction of top-K by score with positive return for that window.
    Writes per-run tables to run_dir/tables/ and also updates global outputs/tables/ files.
    """
    try:
        if "Run Datetime" not in df.columns or "Weighted Score" not in df.columns:
            return
        d = df.copy()
        d["Run Datetime"] = pd.to_datetime(d["Run Datetime"], errors='coerce')
        d = d.dropna(subset=["Run Datetime"])  # ensure valid timestamps
        d["date"] = d["Run Datetime"].dt.date
        score = pd.to_numeric(d["Weighted Score"], errors="coerce")
        d = d.assign(_score=score)
        ret_windows = [c for c in ("3D Return", "10D Return") if c in d.columns]
        if not ret_windows:
            return
        if ks is None:
            ks = [10, 20]
        rows = []
        for win in ret_windows:
            rr = pd.to_numeric(d[win], errors='coerce')
            tmp = d.assign(_ret=rr).dropna(subset=["_score", "_ret"])  # keep rows with both
            if tmp.empty:
                continue
            for day, grp in tmp.groupby("date"):
                if len(grp) < 3:
                    continue
                try:
                    ic_s = float(grp["_score"].corr(grp["_ret"], method="spearman"))
                except Exception:
                    ic_s = None
                # precision@K on that day
                pk_vals = {}
                try:
                    gsorted = grp.sort_values("_score", ascending=False)
                    for k in ks:
                        topk = gsorted.head(k)
                        if not topk.empty:
                            pk = float((topk["_ret"] > 0).mean() * 100.0)
                        else:
                            pk = None
                        pk_vals[f"p@{k}"] = pk
                except Exception:
                    for k in ks:
                        pk_vals[f"p@{k}"] = None
                row = {"Date": str(day), "Window": win, "Spearman IC": ic_s}
                row.update(pk_vals)
                rows.append(row)
        if not rows:
            return
        out = pd.DataFrame(rows)
        # Write per-run table
        try:
            run_tables = os.path.join(run_dir, "tables")
            os.makedirs(run_tables, exist_ok=True)
            out_path = os.path.join(run_tables, "daily_ic_pk.csv")
            out.to_csv(out_path, index=False)
        except Exception:
            pass
        # Write global table
        try:
            global_tables = os.path.join(OUTPUTS_DIR, "tables")
            os.makedirs(global_tables, exist_ok=True)
            out_path2 = os.path.join(global_tables, "daily_ic_pk.csv")
            # Append to global with de-dup on (Date, Window) by keeping latest
            if os.path.exists(out_path2):
                prev = pd.read_csv(out_path2)
                all_df = pd.concat([prev, out], ignore_index=True)
                all_df = all_df.drop_duplicates(subset=["Date", "Window"], keep="last")
            else:
                all_df = out
            all_df.to_csv(out_path2, index=False)
        except Exception:
            pass
    except Exception as e:
        logging.warning(f"[DAILY_METRICS] Failed: {e}")

def export_excel_report(df: pd.DataFrame, output_dir: str, metadata: Optional[dict] = None) -> Optional[str]:
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "Signal_Report.xlsx")
        logging.info("Preparing to export Excel report...")

        # Work on a copy to avoid mutating the pipeline DataFrame
        out = df.copy(deep=True)

    # Remove legacy normalized score; rely on Weighted Score directly

        # Build a combined Risk field early so it's available for ordering/formatting
        try:
            def _build_risk(row):
                lvl = str(row.get("Risk Level") or "").strip()
                tags = str(row.get("Risk Tags") or "").strip()
                ai = str(row.get("Risk Assessment") or "").strip()
                parts = []
                if lvl:
                    parts.append(lvl)
                if tags:
                    parts.append(tags)
                if ai:
                    low_ai = ai.lower()
                    if lvl and (low_ai == f"{lvl.lower()} risk" or low_ai == lvl.lower()):
                        pass
                    else:
                        parts.append(ai)
                return " — ".join([p for p in parts if p])
            out["Risk"] = out.apply(_build_risk, axis=1)
        except Exception:
            # If anything goes wrong, at least ensure the column exists
            if "Risk" not in out.columns:
                out["Risk"] = ""

        # Sort for presentation and compute a stable unique Rank
        sort_prim = [c for c in ["Weighted Score", "Reddit Score", "News Score", "Financial Score"] if c in out.columns]
        sort_keys = sort_prim + (["Ticker"] if "Ticker" in out.columns else [])
        if sort_keys:
            ascending = [False] * len(sort_prim) + ([True] if ("Ticker" in out.columns) else [])
            out = out.sort_values(by=sort_keys, ascending=ascending, kind="mergesort").reset_index(drop=True)
            out["Rank"] = range(1, len(out) + 1)

        out = out.loc[:, ~out.columns.duplicated()]
        for col in FINAL_COLUMN_ORDER:
            if col not in out.columns:
                out[col] = ""
        # Reorder to put preferred columns first but KEEP all other columns afterward
        ordered = [c for c in FINAL_COLUMN_ORDER if c in out.columns]
        others = [c for c in out.columns if c not in ordered]
        out = out[ordered + others]

        with pd.ExcelWriter(file_path, engine="xlsxwriter",
            engine_kwargs={"options": {"nan_inf_to_errors": True}}) as writer:
            workbook = writer.book
            ws = workbook.add_worksheet("Signals")
            writer.sheets["Signals"] = ws

            header_fmt = workbook.add_format(HEADER_STYLE)
            subdued_fmt = workbook.add_format({'font_color': SUBDUED_TEXT_COLOR, 'align': 'center'})
            faded_fmt = workbook.add_format({'font_color': ZERO_FONT_COLOR, 'align': 'center'})
            emerging_fmt = workbook.add_format({'bg_color': '#FFEB9C', 'bold': True, 'align': 'center'})

            formats = {
                'text': workbook.add_format({'num_format': '@', 'align': 'left'}),
                'text_zebra': workbook.add_format({'num_format': '@', 'bg_color': ZEBRA_COLOR, 'align': 'left'}),
                'float': workbook.add_format({'num_format': '0.00', 'align': 'center'}),
                'float_zebra': workbook.add_format({'num_format': '0.00', 'bg_color': ZEBRA_COLOR, 'align': 'center'}),
                'percent': workbook.add_format({'num_format': '0.00%', 'align': 'center'}),
                'percent_zebra': workbook.add_format({'num_format': '0.00%', 'bg_color': ZEBRA_COLOR, 'align': 'center'}),
                'integer': workbook.add_format({'num_format': '#,##0', 'align': 'center'}),
                'integer_zebra': workbook.add_format({'num_format': '#,##0', 'bg_color': ZEBRA_COLOR, 'align': 'center'}),
                # Use text for human-readable big numbers to avoid complex Excel formats
                'human': workbook.add_format({'num_format': '@', 'align': 'center'}),
                'human_zebra': workbook.add_format({'num_format': '@', 'bg_color': ZEBRA_COLOR, 'align': 'center'}),
                'currency': workbook.add_format({'num_format': '$#,##0.00', 'align': 'center'}),
                'currency_zebra': workbook.add_format({'num_format': '$#,##0.00', 'bg_color': ZEBRA_COLOR, 'align': 'center'})
            }
            # Long text cells: wrap and top-align
            long_text_fmt = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': True})

            # Normalize percent-like columns to fractional scale for Excel percent formatting
            if PERCENT_NORMALIZE:
                try:
                    # Always divide returns-like percents by 100 for Excel percent formatting
                    returns_like = {"Price 1D %", "Price 7D %", "Momentum 30D %", "EPS Growth", "ROE", "FCF Margin", "IV Spike %"}
                    for pc in returns_like:
                        if pc in out.columns:
                            out[pc] = pd.to_numeric(out[pc], errors='coerce') / 100.0
                    # For all other percent columns, divide if values look like 0-100 range (not 0-1)
                    other_percent_cols = [c for c, hint in COLUMN_FORMAT_HINTS.items() if hint == "percent" and c in out.columns and c not in returns_like]
                    for pc in other_percent_cols:
                        s = pd.to_numeric(out[pc], errors='coerce')
                        if s.dropna().empty:
                            continue
                        q95 = s.abs().quantile(0.95)
                        if pd.notna(q95) and q95 > 1.0:  # looks like percent points
                            out[pc] = s / 100.0
                except Exception as _norm_err:
                    logging.warning(f"[EXPORT] Percent normalization skipped due to error: {_norm_err}")

            # Convert big-number columns to human-readable strings for stability
            try:
                human_cols = [c for c, hint in COLUMN_FORMAT_HINTS.items() if hint == "human" and c in out.columns]
                for hc in human_cols:
                    out[hc] = out[hc].apply(human_format)
            except Exception:
                pass

            out.to_excel(writer, sheet_name="Signals", index=False)
            for col_num, name in enumerate(out.columns):
                ws.write(0, col_num, name.upper(), header_fmt)
            ws.freeze_panes(1, 3)
            ws.autofilter(0, 0, len(out), len(out.columns) - 1)

            # Pre-build special formats
            trade_type_fmts = {k: workbook.add_format({'bg_color': v, 'align': 'center'}) for k, v in TRADE_TYPE_COLORS.items()}
            risk_level_fmts = {
                'Low': workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'align': 'center'}),
                'Moderate': workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500', 'align': 'center'}),
                'High': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'align': 'center'})
            }

            long_text_cols = [c for c in ["Reddit Summary", "AI News Summary", "AI Trends Commentary", "AI Commentary", "Score Explanation", "Risk"] if c in out.columns]
            # Set a fixed data row height to keep sheet readable while allowing wrapped text visibility
            FIXED_ROW_HEIGHT = 60  # points
            for row_idx, row in out.iterrows():
                is_even = row_idx % 2 == 1
                for col_idx, col in enumerate(out.columns):
                    val = row[col]
                    hint = COLUMN_FORMAT_HINTS.get(col, "text")
                    fmt_key = f"{hint}{'_zebra' if is_even else ''}"
                    fmt = formats.get(fmt_key, formats['text'])

                    if col in long_text_cols:
                        fmt = long_text_fmt
                    elif col == "Trade Type" and isinstance(val, str) and val in trade_type_fmts:
                        fmt = trade_type_fmts[val]
                    elif col == "Risk Level" and isinstance(val, str) and val in risk_level_fmts:
                        fmt = risk_level_fmts[val]
                    elif col == "Risk":
                        # Try to color by extracted risk level
                        level_val = row.get("Risk Level")
                        if isinstance(level_val, str) and level_val in risk_level_fmts:
                            fmt = risk_level_fmts[level_val]
                        elif isinstance(val, str):
                            for _lvl in ["High", "Moderate", "Low"]:
                                if val.startswith(_lvl):
                                    fmt = risk_level_fmts[_lvl]
                                    break
                    elif hint in {"currency", "percent", "float"} and (
                        (isinstance(val, (float, int)) and val == 0.0) or pd.isna(val)
                    ):
                        fmt = faded_fmt

                    try:
                        if col == "Ticker" and isinstance(val, str):
                            url = YAHOO_FINANCE_URL.format(ticker=val.replace("$", ""))
                            ws.write_url(row_idx + 1, col_idx, url, fmt, val)
                        else:
                            # Write full text and rely on wrapping + fixed row height
                            ws.write(row_idx + 1, col_idx, val if pd.notna(val) else '', fmt)
                        if col == "Emerging" and isinstance(val, str) and "Emerging" in val:
                            ws.write_comment(row_idx + 1, col_idx, "New ticker with sudden Reddit or comment spike.")
                    except Exception as cell_err:
                        logging.error(f"Cell write error ({row_idx}, {col}): {cell_err}")
                        raise
                # Apply fixed height to each data row
                try:
                    ws.set_row(row_idx + 1, FIXED_ROW_HEIGHT)
                except Exception:
                    pass

            # Ensure long-text columns are wider to show more content while wrapping
            try:
                for c in long_text_cols:
                    col_idx = list(out.columns).index(c)
                    ws.set_column(col_idx, col_idx, 50)
            except Exception:
                pass

            # Conditional formatting for scores (visual heatmaps)
            try:
                w_col = list(out.columns).index("Weighted Score") if "Weighted Score" in out.columns else None
                if w_col is not None:
                    ws.conditional_format(1, w_col, len(out), w_col, {
                        'type': '3_color_scale',
                        'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B'
                    })
                # Data bars for key metrics
                for metric in ["Momentum 30D %", "Relative Strength", "Volatility Rank", "Volume Spike Ratio"]:
                    if metric in out.columns:
                        col_idx = list(out.columns).index(metric)
                        ws.conditional_format(1, col_idx, len(out), col_idx, {'type': 'data_bar'})
            except Exception as cf_err:
                logging.warning(f"Conditional formatting failed: {cf_err}")

            autosize_and_center(ws, out, workbook)
            # Re-apply width for long-text columns after autosize
            try:
                for c in long_text_cols:
                    col_idx = list(out.columns).index(c)
                    ws.set_column(col_idx, col_idx, 50)
            except Exception:
                pass

            try:
                numeric_df = out.select_dtypes(include='number')
                corr = numeric_df.corr().round(2)
                corr.to_excel(writer, sheet_name="Correlations", index=True)
                autosize_and_center(writer.sheets["Correlations"], corr.reset_index(), workbook)
            except Exception as ce:
                logging.warning(f"Correlations sheet failed: {ce}")

            try:
                legend_df = pd.DataFrame(list(LEGEND.items()), columns=["Column", "Description"])
                legend_df.to_excel(writer, sheet_name="Legend", index=False)
                autosize_and_center(writer.sheets["Legend"], legend_df, workbook)
            except Exception as le:
                logging.warning(f"Legend sheet failed: {le}")

            try:
                score_cols = ["Ticker", "Company"] + [
                    c for c in out.columns if c.endswith("Score") or "Z-Score" in c
                ]
                out[score_cols].to_excel(writer, sheet_name="Score Breakdown", index=False)
                autosize_and_center(writer.sheets["Score Breakdown"], out[score_cols], workbook)
            except Exception as se:
                logging.warning(f"Score Breakdown sheet failed: {se}")

            # Backtest Metrics
            try:
                bt = compute_backtest_metrics_table(out)
                if not bt.empty:
                    bt.to_excel(writer, sheet_name="Backtest Metrics", index=False)
                    autosize_and_center(writer.sheets["Backtest Metrics"], bt, workbook)
                    # Persist key metrics to DB for the latest run, if Run ID present
                    try:
                        run_ids = out.get("Run ID") if "Run ID" in out.columns else None
                        rid = str(run_ids.iloc[0]) if run_ids is not None and len(run_ids.dropna()) else None
                        if rid:
                            ts = datetime.now(timezone.utc).isoformat()
                            for _, row in bt.iterrows():
                                ctx = {k: row[k] for k in bt.columns if k != "Window"}
                                insert_metric(rid, f"backtest_metrics:{row['Window']}", None, json.dumps(ctx), ts)
                    except Exception:
                        pass
            except Exception as bte:
                logging.warning(f"Backtest Metrics sheet failed: {bte}")

            try:
                df_notes = out[["Ticker", "Company", "Reddit Summary", "Top Factors"]].copy()
                # Safe numeric comparisons to avoid type warnings (strings vs floats)
                def _safe_gt(val, threshold: float) -> bool:
                    try:
                        v = pd.to_numeric(val, errors='coerce')
                        return bool(v > threshold) if pd.notna(v) else False
                    except Exception:
                        return False
                def _safe_lt(val, threshold: float) -> bool:
                    try:
                        v = pd.to_numeric(val, errors='coerce')
                        return bool(v < threshold) if pd.notna(v) else False
                    except Exception:
                        return False
                def _flags(row: pd.Series) -> str:
                    parts = []
                    if _safe_gt(row.get("Volatility"), 0.05):
                        parts.append("High Volatility")
                    if _safe_lt(row.get("Avg Daily Value Traded"), 1e6):
                        parts.append("Low Liquidity")
                    if _safe_gt(row.get("Short Percent Float"), 0.15):
                        parts.append("High Short Interest")
                    lw = row.get("Liquidity Warning") or row.get("Liquidity Flags")
                    if isinstance(lw, str) and lw.strip():
                        parts.append(lw.strip())
                    return ", ".join(parts)
                df_notes["Risk Flags"] = out.apply(_flags, axis=1)
                df_notes.to_excel(writer, sheet_name="Trade Notes", index=False)
                autosize_and_center(writer.sheets["Trade Notes"], df_notes, workbook)
            except Exception as tn:
                logging.warning(f"Trade Notes sheet failed: {tn}")

            try:
                if metadata:
                    md_rows = []
                    for k, v in metadata.items():
                        if isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                md_rows.append((f"{k} | {sub_k}", str(sub_v)))
                        else:
                            md_rows.append((k, str(v)))
                    md_df = pd.DataFrame(md_rows, columns=["Key", "Value"])
                    md_df.to_excel(writer, sheet_name="Run Metadata", index=False)
                    autosize_and_center(writer.sheets["Run Metadata"], md_df, workbook)
            except Exception as me:
                logging.warning(f"Run Metadata sheet failed: {me}")

            # Summary sheet (by Sector and Trade Type)
            try:
                summary_rows = []
                if "Sector" in out.columns:
                    tmp = out.groupby("Sector").agg(
                        Count=("Ticker", "count"),
                        AvgScore=("Weighted Score", "mean")
                    ).reset_index()
                    tmp["AvgScore"] = tmp["AvgScore"].round(2)
                    summary_rows.append(("By Sector", tmp))

                if "Trade Type" in out.columns:
                    tmp2 = out.groupby("Trade Type").agg(
                        Count=("Ticker", "count"),
                        AvgScore=("Weighted Score", "mean")
                    ).reset_index()
                    tmp2["AvgScore"] = tmp2["AvgScore"].round(2)
                    summary_rows.append(("By Trade Type", tmp2))

                if summary_rows:
                    start_row = 0
                    ws_sum = workbook.add_worksheet("Summary")
                    writer.sheets["Summary"] = ws_sum
                    for title, sdf in summary_rows:
                        ws_sum.write(start_row, 0, title, header_fmt)
                        sdf.to_excel(writer, sheet_name="Summary", startrow=start_row + 1, startcol=0, index=False)
                        start_row += len(sdf) + 4
                    # Autosize
                    for _, sdf in summary_rows:
                        autosize_and_center(ws_sum, sdf, workbook)
            except Exception as se2:
                logging.warning(f"Summary sheet failed: {se2}")

        logging.info(f"[EXPORT] Excel report saved to: {file_path}")
        return file_path

    except Exception as e:
        logging.warning(f"[EXPORT ERROR] Excel export failed: {e}")
        return None


def export_csv_report(df: pd.DataFrame, output_dir: str, metadata: Optional[dict] = None) -> Optional[str]:
    """Export the same Signal Report as CSVs into a References subfolder in the run directory.

    Notes:
    - No Excel-specific formatting or percent scaling.
    - Ensures FINAL_COLUMN_ORDER presence; fills missing columns with blank values.
    - Keeps preferred columns first but preserves all other columns after them.
    - Does not mutate the original DataFrame.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        references_dir = os.path.join(output_dir, "References")
        os.makedirs(references_dir, exist_ok=True)
        file_path = os.path.join(references_dir, "Signal_Report.csv")

        out = df.copy()

        # Create combined Risk column consistent with Excel
        try:
            def _build_risk(row):
                lvl = str(row.get("Risk Level") or "").strip()
                tags = str(row.get("Risk Tags") or "").strip()
                ai = str(row.get("Risk Assessment") or "").strip()
                parts = []
                if lvl:
                    parts.append(lvl)
                if tags:
                    parts.append(tags)
                if ai:
                    low_ai = ai.lower()
                    if lvl and (low_ai == f"{lvl.lower()} risk" or low_ai == lvl.lower()):
                        pass
                    else:
                        parts.append(ai)
                return " — ".join([p for p in parts if p])
            out["Risk"] = out.apply(_build_risk, axis=1)
        except Exception:
            if "Risk" not in out.columns:
                out["Risk"] = ""

        # Stable sort and rank
        sort_prim = [c for c in ["Weighted Score", "Reddit Score", "News Score", "Financial Score"] if c in out.columns]
        sort_keys = sort_prim + (["Ticker"] if "Ticker" in out.columns else [])
        if sort_keys:
            ascending = [False] * len(sort_prim) + ([True] if ("Ticker" in out.columns) else [])
            out = out.sort_values(by=sort_keys, ascending=ascending, kind="mergesort").reset_index(drop=True)
            out["Rank"] = range(1, len(out) + 1)

        out = out.loc[:, ~out.columns.duplicated()]
        for col in FINAL_COLUMN_ORDER:
            if col not in out.columns:
                out[col] = ""

        # Reorder to put preferred columns first but KEEP all other columns afterward
        ordered = [c for c in FINAL_COLUMN_ORDER if c in out.columns]
        others = [c for c in out.columns if c not in ordered]
        out = out[ordered + others]

        # Main signals CSVs
        out.to_csv(file_path, index=False, encoding="utf-8")
        out.to_csv(os.path.join(references_dir, "Signal_Report__Signals.csv"), index=False, encoding="utf-8")

        # Correlations
        try:
            numeric_df = out.select_dtypes(include='number')
            corr = numeric_df.corr().round(4)
            corr.insert(0, 'Metric', corr.index)
            corr.to_csv(os.path.join(references_dir, "Signal_Report__Correlations.csv"), index=False, encoding="utf-8")
        except Exception as ce:
            logging.warning(f"[EXPORT] Correlations CSV failed: {ce}")

        # Legend
        try:
            legend_df = pd.DataFrame(list(LEGEND.items()), columns=["Column", "Description"])
            legend_df.to_csv(os.path.join(references_dir, "Signal_Report__Legend.csv"), index=False, encoding="utf-8")
        except Exception as le:
            logging.warning(f"[EXPORT] Legend CSV failed: {le}")

        # Score Breakdown
        try:
            score_cols = ["Ticker", "Company"] + [c for c in out.columns if c.endswith("Score") or "Z-Score" in c]
            out[score_cols].to_csv(os.path.join(references_dir, "Signal_Report__Score_Breakdown.csv"), index=False, encoding="utf-8")
        except Exception as se:
            logging.warning(f"[EXPORT] Score Breakdown CSV failed: {se}")

        # Trade Notes
        try:
            tn = out[["Ticker", "Company", "Reddit Summary", "Top Factors"]].copy()
            def _safe_gt(val, threshold: float) -> bool:
                try:
                    v = pd.to_numeric(val, errors='coerce')
                    return bool(v > threshold) if pd.notna(v) else False
                except Exception:
                    return False
            def _safe_lt(val, threshold: float) -> bool:
                try:
                    v = pd.to_numeric(val, errors='coerce')
                    return bool(v < threshold) if pd.notna(v) else False
                except Exception:
                    return False
            def _flags(row: pd.Series) -> str:
                parts = []
                if _safe_gt(row.get("Volatility"), 0.05):
                    parts.append("High Volatility")
                if _safe_lt(row.get("Avg Daily Value Traded"), 1e6):
                    parts.append("Low Liquidity")
                if _safe_gt(row.get("Short Percent Float"), 0.15):
                    parts.append("High Short Interest")
                lw = row.get("Liquidity Warning") or row.get("Liquidity Flags")
                if isinstance(lw, str) and lw.strip():
                    parts.append(lw.strip())
                return ", ".join(parts)
            tn["Risk Flags"] = out.apply(_flags, axis=1)
            tn.to_csv(os.path.join(references_dir, "Signal_Report__Trade_Notes.csv"), index=False, encoding="utf-8")
        except Exception as tn_err:
            logging.warning(f"[EXPORT] Trade Notes CSV failed: {tn_err}")

        # Summaries
        try:
            if "Sector" in out.columns:
                by_sector = out.groupby("Sector").agg(
                    Count=("Ticker", "count"),
                    AvgScore=("Weighted Score", "mean")
                ).reset_index()
                by_sector["AvgScore"] = by_sector["AvgScore"].round(2)
                by_sector.to_csv(os.path.join(references_dir, "Signal_Report__Summary_By_Sector.csv"), index=False, encoding="utf-8")
            if "Trade Type" in out.columns:
                by_type = out.groupby("Trade Type").agg(
                    Count=("Ticker", "count"),
                    AvgScore=("Weighted Score", "mean")
                ).reset_index()
                by_type["AvgScore"] = by_type["AvgScore"].round(2)
                by_type.to_csv(os.path.join(references_dir, "Signal_Report__Summary_By_TradeType.csv"), index=False, encoding="utf-8")
        except Exception as sum_err:
            logging.warning(f"[EXPORT] Summary CSV failed: {sum_err}")

        # Run Metadata
        try:
            if metadata:
                md_rows = []
                for k, v in metadata.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            md_rows.append((f"{k} | {sub_k}", str(sub_v)))
                    else:
                        md_rows.append((k, str(v)))
                pd.DataFrame(md_rows, columns=["Key", "Value"]).to_csv(
                    os.path.join(references_dir, "Signal_Report__Run_Metadata.csv"), index=False, encoding="utf-8"
                )
        except Exception as me:
            logging.warning(f"[EXPORT] Run Metadata CSV failed: {me}")

        # Data Quality
        try:
            dq_rows = []
            for col in out.columns:
                null_rate = round(out[col].isna().mean() * 100, 2)
                sample_vals = out[col].dropna().astype(str).head(3).tolist()
                dq_rows.append({"Column": col, "Null %": null_rate, "Sample Values": "; ".join(sample_vals)})
            pd.DataFrame(dq_rows).to_csv(os.path.join(references_dir, "Signal_Report__Data_Quality.csv"), index=False, encoding="utf-8")
        except Exception as dq_err:
            logging.warning(f"[EXPORT] Data Quality CSV failed: {dq_err}")

        # Backtest Metrics
        try:
            bt = compute_backtest_metrics_table(out)
            if not bt.empty:
                bt.to_csv(os.path.join(references_dir, "Signal_Report__Backtest_Metrics.csv"), index=False, encoding="utf-8")
                bt.to_csv(os.path.join(references_dir, "Backtest_Metrics.csv"), index=False, encoding="utf-8")
                # Persist a simple summary row into DB (if available)
                try:
                    run_ids = out.get("Run ID") if "Run ID" in out.columns else None
                    rid = str(run_ids.iloc[0]) if run_ids is not None and len(run_ids.dropna()) else None
                    if rid:
                        ts = datetime.now(timezone.utc).isoformat()
                        for _, row in bt.iterrows():
                            ctx = {k: row[k] for k in bt.columns if k != "Window"}
                            insert_metric(rid, f"backtest_metrics:{row['Window']}", None, json.dumps(ctx), ts)
                except Exception:
                    pass
        except Exception as bt_err:
            logging.warning(f"[EXPORT] Backtest Metrics CSV failed: {bt_err}")

    # Feature Coverage (non-null %, simple stats)
        try:
            cov_rows = []
            n = max(1, len(out))
            for col in out.columns:
                series = out[col]
                nonnull = int(series.notna().sum())
                coverage = round((nonnull / n) * 100.0, 2)
                row = {"Column": col, "Non-Null Count": nonnull, "Coverage %": coverage}
                # Numeric stats when possible
                try:
                    s_num = pd.to_numeric(series, errors='coerce')
                    if s_num.notna().any():
                        row.update({
                            "Min": float(s_num.min(skipna=True)),
                            "Max": float(s_num.max(skipna=True)),
                            "Mean": float(s_num.mean(skipna=True)),
                            "Std": float(s_num.std(skipna=True))
                        })
                    else:
                        # Categorical quick view
                        vc = series.dropna().astype(str).value_counts()
                        if not vc.empty:
                            top_val = vc.index[0]
                            top_cnt = int(vc.iloc[0])
                            row.update({"Top Value": top_val, "Top Count": top_cnt})
                except Exception:
                    pass
                cov_rows.append(row)
            cov_df = pd.DataFrame(cov_rows)
            cov_df.to_csv(os.path.join(references_dir, "Signal_Report__Feature_Coverage.csv"), index=False, encoding="utf-8")
            # Also write a plain-named file for easier discovery
            cov_df.to_csv(os.path.join(references_dir, "Feature_Coverage.csv"), index=False, encoding="utf-8")
        except Exception as cov_err:
            logging.warning(f"[EXPORT] Feature Coverage CSV failed: {cov_err}")

        # Sanity checks (Weighted Score health)
        try:
            sanity = []
            if "Weighted Score" in out.columns:
                s = pd.to_numeric(out["Weighted Score"], errors='coerce')
                n = len(s)
                nulls = int(s.isna().sum())
                finite = int(np.isfinite(s).sum()) if 'np' in globals() else int(s.replace([float('inf'), float('-inf')], pd.NA).notna().sum())
                nunique = int(s.nunique(dropna=True))
                s_non = s.dropna()
                stats = {
                    "count": int(n),
                    "nulls": nulls,
                    "finite": finite,
                    "unique": nunique,
                    "min": float(s_non.min()) if not s_non.empty else None,
                    "max": float(s_non.max()) if not s_non.empty else None,
                    "mean": float(s_non.mean()) if not s_non.empty else None,
                    "std": float(s_non.std()) if not s_non.empty else None,
                }
                # Flags
                flags = []
                if nulls > 0:
                    flags.append(f"{nulls} nulls")
                if nunique <= 1:
                    flags.append("constant values")
                if stats["std"] is not None and stats["std"] == 0.0:
                    flags.append("zero std")
                # top vs median spread
                try:
                    med = float(s_non.median()) if not s_non.empty else None
                    top = float(s_non.max()) if not s_non.empty else None
                    if med is not None and top is not None and abs(top - med) < 1e-3:
                        flags.append("weak separation (top≈median)")
                except Exception:
                    pass
                sanity.append({"Metric": "Weighted Score", **stats, "flags": ", ".join(flags)})
            if sanity:
                sc_df = pd.DataFrame(sanity)
                sc_path1 = os.path.join(references_dir, "Signal_Report__Sanity_Checks.csv")
                sc_path2 = os.path.join(references_dir, "Sanity_Checks.csv")
                sc_df.to_csv(sc_path1, index=False, encoding="utf-8")
                sc_df.to_csv(sc_path2, index=False, encoding="utf-8")
                # Log any flags prominently
                for _, r in sc_df.iterrows():
                    flags_text = str(r.get("flags", "")).strip()
                    if flags_text:
                        logging.warning(f"[SANITY] {r.get('Metric')}: {flags_text}")
        except Exception as sc_err:
            logging.warning(f"[EXPORT] Sanity Checks CSV failed: {sc_err}")

        # Curated Clean CSV
        try:
            clean_cols_pref = [
                "Rank", "Ticker", "Company", "Sector", "Trade Type", "Risk Level",
                "Top Factors", "Relative Strength", "Price 1D %", "Price 7D %", "Volume Spike Ratio",
                "Market Cap", "Avg Daily Value Traded", "Short Percent Float", "EPS Growth", "ROE",
                "Reddit Sentiment", "News Sentiment", "Post Recency", "Emerging", "Thread Tag", "Reddit Summary"
            ]
            available = [c for c in clean_cols_pref if c in out.columns]
            keep = []
            for c in available:
                try:
                    null_rate = out[c].isna().mean() if c in out.columns else 1.0
                except Exception:
                    null_rate = 1.0
                if null_rate <= 0.9:
                    keep.append(c)
            if keep:
                out[keep].to_csv(os.path.join(references_dir, "Signal_Report__Clean.csv"), index=False, encoding="utf-8")
        except Exception as clean_err:
            logging.warning(f"[EXPORT] Clean CSV failed: {clean_err}")

        # Compute and persist daily IC/p@K tables (per-run + global)
        try:
            compute_and_save_daily_ic_pk(out, output_dir)
        except Exception as dm_err:
            logging.warning(f"[EXPORT] Daily IC/p@K generation failed: {dm_err}")

        logging.info(f"[EXPORT] CSV report saved to: {file_path}")
        return file_path
    except Exception as e:
        logging.warning(f"[EXPORT ERROR] CSV export failed: {e}")
        return None


def export_min_signals(df: pd.DataFrame, run_dir: str, top_n: int = None) -> Optional[dict]:
    """Export compact artifacts for web/API while keeping full reports intact.

    Writes two files into the run directory:
      - signals_min.csv
      - signals_min.json

    Fields: Rank, Ticker, Company, Sector, Trade Type, Weighted Score, Risk
    If top_n is provided, limit to that many rows after sorting by Weighted Score.
    Returns paths in a dict on success.
    """
    try:
        os.makedirs(run_dir, exist_ok=True)
        cols = [
            "Rank", "Ticker", "Company", "Sector", "Trade Type", "Weighted Score", "Risk"
        ]
        data = df.copy()
        # ensure Risk exists as in main exports
        if "Risk" not in data.columns:
            try:
                def _build_risk(row):
                    lvl = str(row.get("Risk Level") or "").strip()
                    tags = str(row.get("Risk Tags") or "").strip()
                    ai = str(row.get("Risk Assessment") or "").strip()
                    parts = []
                    if lvl:
                        parts.append(lvl)
                    if tags:
                        parts.append(tags)
                    if ai:
                        low_ai = ai.lower()
                        if lvl and (low_ai == f"{lvl.lower()} risk" or low_ai == lvl.lower()):
                            pass
                        else:
                            parts.append(ai)
                    return " — ".join([p for p in parts if p])
                data["Risk"] = data.apply(_build_risk, axis=1)
            except Exception:
                data["Risk"] = ""

        # stable sort & rank
        sort_prim = [c for c in ["Weighted Score", "Reddit Score", "News Score", "Financial Score"] if c in data.columns]
        sort_keys = sort_prim + (["Ticker"] if "Ticker" in data.columns else [])
        if sort_keys:
            ascending = [False] * len(sort_prim) + ([True] if ("Ticker" in data.columns) else [])
            data = data.sort_values(by=sort_keys, ascending=ascending, kind="mergesort").reset_index(drop=True)
        if "Rank" not in data.columns:
            data["Rank"] = range(1, len(data) + 1)

        # Default cap for web speed
        if top_n is None:
            top_n = 100
        try:
            n = int(top_n)
            data = data.head(max(n, 0))
        except Exception:
            pass

        # ensure columns exist
        for c in cols:
            if c not in data.columns:
                data[c] = ""
        compact = data[cols]

        # write csv
        csv_path = os.path.join(run_dir, "signals_min.csv")
        compact.to_csv(csv_path, index=False, encoding="utf-8")

        # write json
        try:
            import json as _json
            json_path = os.path.join(run_dir, "signals_min.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                _json.dump(compact.to_dict(orient="records"), jf, ensure_ascii=False, indent=2)
        except Exception as je:
            logging.warning(f"[EXPORT] signals_min.json failed: {je}")
            json_path = None

        logging.info(f"[EXPORT] signals_min artifacts saved: csv={csv_path}, json={json_path}")
        return {"csv": csv_path, "json": json_path}
    except Exception as e:
        logging.warning(f"[EXPORT ERROR] signals_min export failed: {e}")
        return None


def export_full_json(df: pd.DataFrame, run_dir: str) -> Optional[str]:
    """Export the full DataFrame as a JSON records file for web/API consumption.

    Writes: signals_full.json into the run directory.
    Returns the path on success.
    """
    try:
        import json as _json
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, "signals_full.json")
        # Convert any non-serializable objects to strings
        serializable = df.copy()
        for c in serializable.columns:
            try:
                serializable[c] = serializable[c].apply(lambda v: v if (pd.isna(v) or isinstance(v, (int, float, str, bool))) else str(v))
            except Exception:
                # Fallback: cast entire column to string
                serializable[c] = serializable[c].astype(str)
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(serializable.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        logging.info(f"[EXPORT] signals_full.json saved: {out_path}")
        return out_path
    except Exception as e:
        logging.warning(f"[EXPORT ERROR] signals_full export failed: {e}")
        return None


def generate_html_dashboard(df: pd.DataFrame, output_path="outputs/dashboard/dashboard.html") -> None:
    """Generate an accessible, standards-compliant HTML dashboard using Weighted Score only."""
    try:
        # Ensure output directory exists and prepare a sibling CSS file
        out_dir = os.path.dirname(output_path)
        os.makedirs(out_dir, exist_ok=True)
        css_path = os.path.join(out_dir, "styles.css")

        # Small, external stylesheet
        css = (
            "body{font-family:Arial,Helvetica,sans-serif;margin:20px;}"
            "table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:center;}"
            "th{background-color:#f2f2f2;}"
            "h2{margin-top:40px;}"
        )
        try:
            with open(css_path, "w", encoding="utf-8") as cf:
                cf.write(css)
        except Exception as css_err:
            logging.warning(f"[DASHBOARD] Failed writing CSS file: {css_err}")

        # Prepare data fragments
        top_df = df.sort_values("Weighted Score", ascending=False).head(10)
        recent_date = "N/A"
        if "Run Datetime" in df.columns:
            try:
                recent_date = pd.to_datetime(df["Run Datetime"]).max().strftime("%Y-%m-%d")
            except Exception:
                pass

        summary_stats = {
            "Total Signals": int(len(df)),
            "Unique Tickers": int(df["Ticker"].nunique()) if "Ticker" in df.columns else 0,
            "Most Recent Date": recent_date,
        }

        # Build table HTML and remove inline style attributes emitted by pandas
        display_cols = ["Ticker", "Company", "Weighted Score", "Reddit Summary"]
        display_cols = [c for c in display_cols if c in top_df.columns]
        table_html = top_df[display_cols].to_html(index=False, escape=False, classes=["dataframe", "compact"], border=0)

        import re as _re
        table_html = _re.sub(r"\sstyle=\"[^\"]*\"", "", table_html)

        # Compose HTML with proper meta tags and lang attribute
        html = [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "<title>Signal Dashboard</title>",
            "<link rel=\"stylesheet\" href=\"styles.css\">",
            "</head>",
            "<body>",
            "<h1>Signal Dashboard</h1>",
            "<h2>Summary Stats</h2>",
            "<ul>",
        ]
        for k, v in summary_stats.items():
            html.append(f"<li><b>{k}:</b> {v}</li>")
        html.extend([
            "</ul>",
            "<h2>Top 10 Signals</h2>",
            table_html,
            "</body>",
            "</html>",
        ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(html))

        logging.info(f"[DASHBOARD] Saved: {output_path}")
    except Exception as e:
        logging.warning(f"[DASHBOARD ERROR] Failed to generate HTML: {e}")


def generate_run_homepage(df: pd.DataFrame, run_dir: str) -> Optional[str]:
    """Generate a simple per-run homepage (index.html) with links to artifacts and a Top 10 table.

    Keeps it dependency-free and light; focuses on the most actionable fields and provides links
    to the full Excel/CSV/JSON artifacts to see everything else.
    """
    try:
        os.makedirs(run_dir, exist_ok=True)
        index_path = os.path.join(run_dir, "index.html")

        # Summary stats
        total = int(len(df))
        uniq = int(df["Ticker"].nunique()) if "Ticker" in df.columns else total
        recent_date = "N/A"
        if "Run Datetime" in df.columns:
            try:
                recent_date = pd.to_datetime(df["Run Datetime"]).max().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        # Top 10 compact table
        cols = [c for c in ["Rank", "Ticker", "Company", "Sector", "Trade Type", "Weighted Score", "Risk"] if c in df.columns]
        top10 = df.sort_values("Weighted Score", ascending=False).head(10)
        table_html = top10[cols].to_html(index=False, border=0)
        import re as _re
        table_html = _re.sub(r"\sstyle=\"[^\"]*\"", "", table_html)

        # Links to artifacts we expect in the run folder
        links = [
            ("Signal_Report.xlsx", "Main Excel report (all columns)"),
            ("References/Signal_Report.csv", "CSV (all columns)"),
            ("signals_min.json", "Compact JSON (homepage/API)"),
            ("signals_full.json", "Full JSON (all fields)"),
            ("References/Picks.csv", "Filtered Picks with stops & sizing"),
            ("References/Signal_Report__Backtest_Metrics.csv", "Backtest Metrics (IC, deciles, hit rate)"),
            ("References/Signal_Report__Feature_Coverage.csv", "Feature Coverage (non-null %, stats)"),
            ("References/Signal_Report__Sanity_Checks.csv", "Sanity Checks (Weighted Score health)"),
            ("References/Run_Counters.csv", "Run Counters (observability)")
        ]

        html = [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "<title>VP Investments — Run Overview</title>",
            "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px;text-align:center}th{background:#f2f2f2}h2{margin-top:28px}</style>",
            "</head>",
            "<body>",
            "<h1>Run Overview</h1>",
            f"<p><b>Total Signals:</b> {total} &nbsp; | &nbsp; <b>Unique Tickers:</b> {uniq} &nbsp; | &nbsp; <b>Most Recent:</b> {recent_date}</p>",
            "<h2>Top 10</h2>",
            table_html,
            "<h2>Artifacts</h2>",
            "<ul>",
        ]
        for href, label in links:
            html.append(f"<li><a href=\"{href}\" target=\"_blank\">{label}</a></li>")
        html.extend([
            "</ul>",
            "<p>For the complete dataset and additional analyses (Correlations, Score Breakdown, Data Quality), see the References folder.</p>",
            "</body>",
            "</html>",
        ])

        with open(index_path, "w", encoding="utf-8") as f:
            f.write("".join(html))
        logging.info(f"[DASHBOARD] Run homepage saved: {index_path}")
        return index_path
    except Exception as e:
        logging.warning(f"[DASHBOARD ERROR] Failed to generate run homepage: {e}")
        return None


def export_picks(df: pd.DataFrame, run_dir: str, top_deciles: tuple = (8, 9, 10),
                 min_adv: Optional[float] = None, exclude_earnings_days: int = 3,
                 risk_per_trade: float = 0.01) -> Optional[str]:
    """Create a weekly Picks.csv with guardrails and simple risk guidance.

    - Filters to top score deciles (default: 8–10).
    - Liquidity filter by Avg Daily Value Traded (default: max(LIQUIDITY_WARNING_ADV, $2M)).
    - Excludes names with earnings within N days (default: 3 calendar days).
    - Suggests an initial stop using volatility; sizes position by risk parity per $100k.
    """
    try:
        references_dir = os.path.join(run_dir, "References")
        os.makedirs(references_dir, exist_ok=True)
        out_path = os.path.join(references_dir, "Picks.csv")

        data = df.copy()
        # Score deciles from Weighted Score
        try:
            data["Score Decile"] = pd.qcut(data["Weighted Score"], 10, labels=list(range(1, 11)))
            data["Score Decile"] = data["Score Decile"].astype(int)
        except Exception:
            # Fallback to rank-based buckets if qcut fails
            ranks = data["Weighted Score"].rank(method="average", pct=True)
            data["Score Decile"] = (ranks * 10).clip(1, 10).round().astype(int)

        # Liquidity filter
        adv_threshold = max(LIQUIDITY_WARNING_ADV or 0, 2_000_000) if min_adv is None else float(min_adv)
        if "Avg Daily Value Traded" in data.columns:
            data = data[pd.to_numeric(data["Avg Daily Value Traded"], errors='coerce') >= adv_threshold]

        # Earnings proximity filter
        if exclude_earnings_days and "Next Earnings Date" in data.columns:
            try:
                ed = pd.to_datetime(data["Next Earnings Date"], errors='coerce')
                cutoff = datetime.now() + timedelta(days=int(exclude_earnings_days))
                data = data[(ed.isna()) | (ed >= cutoff)]
            except Exception:
                pass

        # Keep only top deciles
        data = data[data["Score Decile"].isin(list(top_deciles))]

        # Stop suggestion and size guidance
        def _stop_pct(row):
            vol = pd.to_numeric(row.get("Volatility"), errors='coerce')
            # If volatility present as fraction (e.g., 0.03), use 2x; else default 7%
            if pd.notna(vol) and 0 < vol < 0.5:
                sp = float(vol) * 2.0
            else:
                sp = 0.07
            return float(max(0.04, min(sp, 0.15)))

        data["Suggested Stop %"] = data.apply(_stop_pct, axis=1)
        cp = pd.to_numeric(data.get("Current Price", pd.Series(index=data.index)), errors='coerce')
        data["Initial Stop Price"] = (cp * (1 - data["Suggested Stop %"]).astype(float)).round(2)
        # Risk parity sizing per $100k portfolio
        try:
            risk = float(risk_per_trade)
        except Exception:
            risk = 0.01
        # position_value_for_100k = 100000 * risk / stop_pct
        data["Size @ $100k (USD)"] = (100000.0 * risk / data["Suggested Stop %"]).round(0)

        keep_cols = [c for c in [
            "Ticker", "Company", "Sector", "Trade Type", "Risk Level", "Score Decile",
            "Weighted Score", "Current Price", "Suggested Stop %", "Initial Stop Price", "Size @ $100k (USD)",
            "Avg Daily Value Traded", "Market Cap", "Volatility", "Momentum 30D %", "Relative Strength",
            "Short Percent Float", "EPS Growth", "ROE", "Reddit Sentiment", "News Sentiment", "Top Factors",
            "Liquidity Warning", "Risk Tags", "Emerging/Thread", "Reddit Summary"
        ] if c in data.columns]

        picks = data.sort_values(["Score Decile", "Weighted Score"], ascending=[False, False])[keep_cols]

        # Pretty percent for output readability
        for pc in ["Suggested Stop %", "Momentum 30D %", "Short Percent Float", "EPS Growth", "ROE"]:
            if pc in picks.columns:
                s = pd.to_numeric(picks[pc], errors='coerce')
                # If looks like fraction, format as percent string
                picks[pc] = s.apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) and abs(x) <= 1 else (f"{x:.2f}%" if pd.notna(x) else ""))

        picks.to_csv(out_path, index=False, encoding="utf-8")
        logging.info(f"[PICKS] Saved: {out_path} (min ADV={adv_threshold:,}, top deciles={top_deciles})")
        return out_path
    except Exception as e:
        logging.warning(f"[PICKS ERROR] Failed to export Picks: {e}")
        return None


def write_run_readme(run_dir: str) -> Optional[str]:
    """Write a brief README.md into the run directory describing key artifacts."""
    try:
        os.makedirs(run_dir, exist_ok=True)
        readme_path = os.path.join(run_dir, "README.md")
        content: list[str] = []
        content.append(f"# VP Investments Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        content.append("Artifacts in this folder:")
        content.append("- Signal_Report.xlsx — main report with multiple sheets.")
        content.append("- References/ — CSV versions, correlations, legends, and clean extracts.")
        content.append("- References/Picks.csv — filtered candidates with stops and size guidance.")
        content.append("- signals_min.json — compact JSON for homepage/API.")
        content.append("- signals_full.json — full JSON (all columns) for programmatic use.")
        content.append("")
        content.append("Quick tips:")
        content.append("- Picks.csv defaults to deciles 8–10, ADV >= $2M, excludes earnings within 3 days.")
        content.append("- Size @ $100k column uses 1% risk per trade and the suggested stop.")
        content.append("- Adjust filters later in code if your workflow differs.")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        return readme_path
    except Exception as e:
        logging.warning(f"[README ERROR] Failed to write run README: {e}")
        return None
