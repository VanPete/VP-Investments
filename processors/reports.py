import os
import logging
import pandas as pd
import xlsxwriter
from typing import Optional

from config.labels import FINAL_COLUMN_ORDER, COLUMN_FORMAT_HINTS
from report_legend import LEGEND

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
    "Reddit Sentiment Z-Score": "float",
    "Price 7D % Z-Score": "float",
    "Volume Spike Ratio Z-Score": "float",
    "Liquidity Warning": "text"
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

        if "Weighted Score" in df.columns and not df["Weighted Score"].isna().all():
            min_score = df["Weighted Score"].min()
            max_score = df["Weighted Score"].max()
            df["Score (0–100)"] = df["Weighted Score"].apply(
                lambda x: round(100 * (x - min_score) / (max_score - min_score), 2)
                if max_score > min_score else 0.0
            )

        for col, hint in COLUMN_FORMAT_HINTS.items():
            if col in df.columns:
                if hint == "percent":
                    df[col] = pd.to_numeric(df[col], errors="coerce") / 100
                elif hint == "human":
                    df[col] = df[col].apply(human_format)

        df = df.loc[:, ~df.columns.duplicated()]
        for col in FINAL_COLUMN_ORDER:
            if col not in df.columns:
                df[col] = ""
        df = df[FINAL_COLUMN_ORDER]

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
                'text': workbook.add_format({'num_format': '@', 'align': 'center'}),
                'text_zebra': workbook.add_format({'num_format': '@', 'bg_color': ZEBRA_COLOR, 'align': 'center'}),
                'float': workbook.add_format({'num_format': '0.00', 'align': 'center'}),
                'float_zebra': workbook.add_format({'num_format': '0.00', 'bg_color': ZEBRA_COLOR, 'align': 'center'}),
                'percent': workbook.add_format({'num_format': '0.00%', 'align': 'center'}),
                'percent_zebra': workbook.add_format({'num_format': '0.00%', 'bg_color': ZEBRA_COLOR, 'align': 'center'}),
                'currency': workbook.add_format({'num_format': '$#,##0.00', 'align': 'center'}),
                'currency_zebra': workbook.add_format({'num_format': '$#,##0.00', 'bg_color': ZEBRA_COLOR, 'align': 'center'})
            }

            df.to_excel(writer, sheet_name="Signals", index=False)
            for col_num, name in enumerate(df.columns):
                ws.write(0, col_num, name.upper(), header_fmt)
            ws.freeze_panes(1, 3)
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)

            for row_idx, row in df.iterrows():
                is_even = row_idx % 2 == 1
                for col_idx, col in enumerate(df.columns):
                    val = row[col]
                    hint = COLUMN_FORMAT_HINTS.get(col, "text")
                    fmt_key = f"{hint}{'_zebra' if is_even else ''}"
                    fmt = formats.get(fmt_key, formats['text'])

                    if col in {"Reddit Summary", "Top Factors"}:
                        fmt = subdued_fmt
                    elif hint in {"currency", "percent", "float"} and (
                        (isinstance(val, (float, int)) and val == 0.0) or pd.isna(val)
                    ):
                        fmt = faded_fmt

                    try:
                        if col == "Ticker" and isinstance(val, str):
                            url = YAHOO_FINANCE_URL.format(ticker=val.replace("$", ""))
                            ws.write_url(row_idx + 1, col_idx, url, fmt, val)
                        else:
                            ws.write(row_idx + 1, col_idx, val if pd.notna(val) else '', fmt)
                        if col == "Emerging" and isinstance(val, str) and "Emerging" in val:
                            ws.write_comment(row_idx + 1, col_idx, "New ticker with sudden Reddit or comment spike.")
                    except Exception as cell_err:
                        logging.error(f"Cell write error ({row_idx}, {col}): {cell_err}")
                        raise

            autosize_and_center(ws, df, workbook)

            try:
                numeric_df = df.select_dtypes(include='number')
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
                    c for c in df.columns if c.endswith("Score") or "Z-Score" in c
                ]
                df[score_cols].to_excel(writer, sheet_name="Score Breakdown", index=False)
                autosize_and_center(writer.sheets["Score Breakdown"], df[score_cols], workbook)
            except Exception as se:
                logging.warning(f"Score Breakdown sheet failed: {se}")

            try:
                df_notes = df[["Ticker", "Company", "Score (0–100)", "Reddit Summary", "Top Factors"]].copy()
                df_notes["Risk Flags"] = df.apply(lambda row: ", ".join(filter(None, [
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

        logging.info(f"[EXPORT] Excel report saved to: {file_path}")
        return file_path

    except Exception as e:
        logging.warning(f"[EXPORT ERROR] Excel export failed: {e}")
        return None


def generate_html_dashboard(df: pd.DataFrame, output_path="outputs/dashboard/dashboard.html") -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        top_df = df.sort_values("Weighted Score", ascending=False).head(10)
        recent_date = "N/A"
        if "Run Datetime" in df.columns:
            recent_date = pd.to_datetime(df["Run Datetime"]).max().strftime("%Y-%m-%d")

        summary_stats = {
            "Total Signals": len(df),
            "Unique Tickers": df["Ticker"].nunique(),
            "Most Recent Date": recent_date
        }

        html = "<html><head><title>Signal Dashboard</title><style>"
        html += "body { font-family: Arial; margin: 20px; } table { border-collapse: collapse; width: 100%; }"
        html += "th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }"
        html += "th { background-color: #f2f2f2; } h2 { margin-top: 40px; }"
        html += "</style></head><body>"
        html += "<h1>Signal Dashboard</h1>"

        html += "<h2>Summary Stats</h2><ul>"
        for k, v in summary_stats.items():
            html += f"<li><b>{k}:</b> {v}</li>"
        html += "</ul>"

        html += "<h2>Top 10 Signals</h2>"
        html += top_df[["Ticker", "Company", "Weighted Score", "Score (0–100)", "Reddit Summary"]].to_html(index=False, escape=False)

        html += "</body></html>"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logging.info(f"[DASHBOARD] Saved: {output_path}")
    except Exception as e:
        logging.warning(f"[DASHBOARD ERROR] Failed to generate HTML: {e}")
