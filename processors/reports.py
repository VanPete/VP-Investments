import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import xlsxwriter
from typing import Optional

from config.labels import FINAL_COLUMN_ORDER, COLUMN_FORMAT_HINTS
from config.config import PERCENT_NORMALIZE, LIQUIDITY_WARNING_ADV
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

def export_excel_report(df: pd.DataFrame, output_dir: str, metadata: Optional[dict] = None) -> Optional[str]:
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "Signal_Report.xlsx")
        logging.info("Preparing to export Excel report...")

        # Work on a copy to avoid mutating the pipeline DataFrame
        out = df.copy(deep=True)

    # Remove legacy normalized score; rely on Weighted Score directly

        # Sort for presentation and compute a stable unique Rank
        sort_keys = [c for c in ["Weighted Score", "Reddit Score", "News Score", "Financial Score"] if c in out.columns]
        if sort_keys:
            out = out.sort_values(by=sort_keys, ascending=[False] * len(sort_keys), kind="mergesort").reset_index(drop=True)
            out["Rank"] = range(1, len(out) + 1)

        out = out.loc[:, ~out.columns.duplicated()]
        for col in FINAL_COLUMN_ORDER:
            if col not in out.columns:
                out[col] = ""
        out = out[FINAL_COLUMN_ORDER]

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

            long_text_cols = [c for c in ["Reddit Summary", "AI News Summary", "AI Trends Commentary", "AI Commentary", "Score Explanation", "Risk Assessment"] if c in out.columns]
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

            try:
                df_notes = out[["Ticker", "Company", "Reddit Summary", "Top Factors"]].copy()
                df_notes["Risk Flags"] = out.apply(lambda row: ", ".join(filter(None, [
                    "High Volatility" if row.get("Volatility", 0) > 0.05 else "",
                    "Low Liquidity" if row.get("Avg Daily Value Traded", 1e9) < 1e6 else "",
                    "High Short Interest" if row.get("Short Percent Float", 0) > 0.15 else "",
                    row.get("Liquidity Warning") if isinstance(row.get("Liquidity Warning"), str) else ""
                ])), axis=1)
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
    - Does not mutate the original DataFrame.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        references_dir = os.path.join(output_dir, "References")
        os.makedirs(references_dir, exist_ok=True)
        file_path = os.path.join(references_dir, "Signal_Report.csv")

        out = df.copy()
        sort_keys = [c for c in ["Weighted Score", "Reddit Score", "News Score", "Financial Score"] if c in out.columns]
        if sort_keys:
            out = out.sort_values(by=sort_keys, ascending=[False] * len(sort_keys), kind="mergesort").reset_index(drop=True)
            out["Rank"] = range(1, len(out) + 1)
        out = out.loc[:, ~out.columns.duplicated()]

    # Drop legacy normalized score; rely on Weighted Score only

        for col in FINAL_COLUMN_ORDER:
            if col not in out.columns:
                out[col] = ""
        out = out[FINAL_COLUMN_ORDER]

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
            tn["Risk Flags"] = out.apply(lambda row: ", ".join(filter(None, [
                "High Volatility" if row.get("Volatility", 0) > 0.05 else "",
                "Low Liquidity" if row.get("Avg Daily Value Traded", 1e9) < 1e6 else "",
                "High Short Interest" if row.get("Short Percent Float", 0) > 0.15 else "",
                row.get("Liquidity Warning") if isinstance(row.get("Liquidity Warning"), str) else ""
            ])), axis=1)
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

        logging.info(f"[EXPORT] CSV report saved to: {file_path}")
        return file_path
    except Exception as e:
        logging.warning(f"[EXPORT ERROR] CSV export failed: {e}")
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
        content = []
        content.append(f"# VP Investments Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        content.append("Artifacts in this folder:")
        content.append("- Signal_Report.xlsx — main report with multiple sheets.")
        content.append("- References/ — CSV versions, correlations, legends, and clean extracts.")
        content.append("- References/Picks.csv — filtered candidates with stops and size guidance.")
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
