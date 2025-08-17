import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import setup_logging
from config.labels import COLUMN_FORMAT_HINTS
from collections import defaultdict
from functools import lru_cache
from dotenv import load_dotenv
from utils.observability import emit_metric
from config.config import (
    FEATURE_TOGGLES, FMP_API_KEY,
    TECH_VOLATILITY_WINDOW, TECH_RSI_PERIOD, TECH_BB_PERIOD,
    TECH_MOMENTUM_DAYS, TECH_VOL_SPIKE_WINDOW, BETA_WINDOW,
    YF_MAX_WORKERS, YF_BATCH_SIZE, YF_BATCH_PAUSE_SEC
)
from typing import Tuple as _Tuple

# === Setup ===
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# Shared HTTP session for external APIs (FMP endpoints)
from utils.http_client import build_session, get_token_bucket, CircuitBreaker
_sess = build_session(
    cache_name=os.path.join(os.path.dirname(__file__), "..", "cache", "finance_cache"),
    cache_expire_seconds=600,
    timeout=20,
)

# Separate SEC session (respect SEC fair access & user agent requirements)
_sec_sess = build_session(
    cache_name=os.path.join(os.path.dirname(__file__), "..", "cache", "sec_cache"),
    cache_expire_seconds=3600,
    timeout=20,
)
try:
    _ua = os.getenv("SEC_USER_AGENT") or "VP-Investments/1.0 (contact: please-set-SEC_USER_AGENT)"
    _sec_sess.headers.update({
        "User-Agent": _ua,
        "Accept-Encoding": "gzip, deflate",
    })
except Exception:
    pass

# EDGAR dynamic state/toggles (robust env parsing)
def _env_int_safe(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return int(default)
    try:
        # Accept integers or floats, ignore trailing text
        # e.g., "600 optional" -> 600
        token = str(val).strip().split()[0]
        return int(float(token))
    except Exception:
        return int(default)

def _env_bool_safe(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def _env_choice_safe(name: str, choices: set[str], default: str = "") -> str:
    val = (os.getenv(name) or "").strip().lower()
    return val if val in choices else default

_EDGAR_COOLDOWN_SEC = _env_int_safe("SEC_EDGAR_COOLDOWN_SEC", 600)  # 10 minutes default
_EDGAR_FORCE = _env_choice_safe("SEC_EDGAR_FORCE", {"fmp-only", "edgar-only", "hybrid"}, default="")
_edgar_state = {
    "enabled": not _env_bool_safe("SEC_EDGAR_DISABLED", False),
    "disabled_until": 0.0,
    "last_error": None,
}

def _edgar_enabled() -> bool:
    """Return True if EDGAR requests are allowed right now (cooldown expired and not forced off)."""
    # Force switches override
    if _EDGAR_FORCE == "fmp-only":
        return False
    if _EDGAR_FORCE == "edgar-only":
        return True
    # Cooldown based
    now = time.time()
    if _edgar_state["disabled_until"] and now < _edgar_state["disabled_until"]:
        return False
    return bool(_edgar_state["enabled"])  # hybrid/default path

def _disable_edgar_temporarily(reason: str, status: int | None = None) -> None:
    try:
        _edgar_state["enabled"] = False
        _edgar_state["disabled_until"] = time.time() + max(60, _EDGAR_COOLDOWN_SEC)
        _edgar_state["last_error"] = {"reason": reason, "status": status}
        logger.warning(f"EDGAR access disabled for cooldown ({_EDGAR_COOLDOWN_SEC}s). Reason: {reason} status={status}")
    except Exception:
        pass

def _maybe_reenable_edgar() -> None:
    try:
        if _EDGAR_FORCE in ("fmp-only",):
            return
        if _edgar_state["disabled_until"] and time.time() >= _edgar_state["disabled_until"]:
            _edgar_state["enabled"] = True
            _edgar_state["disabled_until"] = 0.0
            _edgar_state["last_error"] = None
            logger.info("EDGAR access re-enabled after cooldown")
    except Exception:
        pass

def _sec_get(url: str, expect_json: bool = True):
    """Wrapper for SEC GET requests with token bucket and dynamic disable on 403/429.

    Returns parsed JSON when expect_json=True; otherwise returns Response.text.
    Returns None on hard errors or when EDGAR is disabled.
    """
    try:
        if not _edgar_enabled():
            try:
                emit_metric("edgar", {"ok": 0, "disabled": True, "url": url})
            except Exception:
                pass
            return None
        _maybe_reenable_edgar()
        _bucket.take(1.0)
        _start = time.time()
        r = _sec_sess.get(url)
        # Handle throttling / forbidden
        if getattr(r, "status_code", 200) in (403, 429):
            retry_after = r.headers.get("Retry-After")
            reason = "rate-limited" if r.status_code == 429 else "forbidden"
            if retry_after:
                try:
                    # Honor Retry-After if provided
                    delay = int(retry_after)
                    _edgar_state["disabled_until"] = time.time() + delay
                except Exception:
                    pass
            _disable_edgar_temporarily(reason, r.status_code)
            try:
                emit_metric("edgar", {"ok": 0, "status": r.status_code, "latency_ms": int((time.time()-_start)*1000), "url": url})
            except Exception:
                pass
            return None
        try:
            emit_metric("edgar", {"ok": 1, "status": getattr(r, "status_code", 200), "latency_ms": int((time.time()-_start)*1000), "url": url})
        except Exception:
            pass
        if expect_json:
            return r.json()
        return r.text
    except Exception as e:
        # Network or parse error — do not permanently disable, but log.
        logger.warning(f"SEC GET failed for {url}: {e}")
        try:
            emit_metric("edgar", {"ok": 0, "error": str(e)[:200], "url": url})
        except Exception:
            pass
        return None

def sec_self_check() -> tuple[bool, str]:
    """Lightweight startup self-check for SEC access and User-Agent.

    Returns (ok, message). On repeated 403/429, EDGAR will be put on cooldown.
    """
    try:
        ua = _sec_sess.headers.get("User-Agent", "")
        if not ua or "please-set-SEC_USER_AGENT" in ua:
            return False, "SEC_USER_AGENT not set or placeholder; set SEC_USER_AGENT in environment/.env"
        # Ping a small, public JSON (Apple submissions)
        data = _sec_get("https://data.sec.gov/submissions/CIK0000320193.json", expect_json=True)
        if data is None:
            if not _edgar_enabled():
                return False, "EDGAR temporarily disabled due to prior 403/429 or cooldown"
            return False, "Unable to reach data.sec.gov or parse response"
        # Basic sanity check
        if isinstance(data, dict) and data.get("cik") == "0000320193":
            return True, "SEC connectivity OK"
        return True, "SEC connectivity OK (generic)"
    except Exception as e:
        return False, f"SEC self-check error: {e}"

# Global token bucket and circuit breaker for external calls
from config.config import TOKEN_BUCKET_RATE, TOKEN_BUCKET_BURST, YF_PER_TICKER_TIMEOUT_SEC, BREAKER_FAIL_THRESHOLD, BREAKER_RESET_AFTER_SEC
_bucket = get_token_bucket(TOKEN_BUCKET_RATE, TOKEN_BUCKET_BURST)
_breaker = CircuitBreaker(fail_threshold=BREAKER_FAIL_THRESHOLD, reset_after_sec=BREAKER_RESET_AFTER_SEC)

# Perform a light self-check once on import; non-fatal but logs guidance
try:
    ok, msg = sec_self_check()
    if ok:
        logger.info(f"[SEC] {msg}")
    else:
        logger.warning(f"[SEC] {msg}")
except Exception:
    pass

# === Utility Functions ===
def human_format(num) -> str:
    if num is None or pd.isna(num):
        return ""
    num = float(num)
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:.1f}{unit}" if unit else f"{num:.0f}"
        num /= 1000.0
    return f"{num:.1f}P"

def load_company_data() -> pd.DataFrame:
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        COMPANY_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "company_names.csv")
        df = pd.read_csv(COMPANY_DATA_PATH)
        df["Ticker"] = df["Ticker"].apply(lambda x: f"${str(x).upper()}")
        df["Company"] = df["Company"].astype(str)
        return df.drop_duplicates(subset="Ticker")
    except FileNotFoundError:
        raise FileNotFoundError(f"{COMPANY_DATA_PATH} not found. Ensure it exists.")

@lru_cache(maxsize=256)
def get_yf_ticker(ticker: str):
    return yf.Ticker(ticker.replace("$", ""))

@lru_cache(maxsize=1)
def _get_spy_closes_1y() -> Optional[pd.Series]:
    try:
        spy = get_yf_ticker("SPY")
        hist = spy.history(period="1y")
        return hist["Close"].dropna()
    except Exception:
        return None

def _compute_beta_from_returns(closes: pd.Series, spy_closes: Optional[pd.Series], window: int = BETA_WINDOW) -> float:
    try:
        if spy_closes is None:
            return np.nan
        r1 = closes.pct_change().dropna()
        r2 = spy_closes.pct_change().dropna()
        df = pd.concat([r1, r2], axis=1, join='inner').dropna()
        df = df.tail(window)
        if len(df) < 30:
            return np.nan
        cov = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)[0, 1]
        var = np.var(df.iloc[:, 1], ddof=1)
        return float(cov / var) if var else np.nan
    except Exception:
        return np.nan

def _derive_eps_growth(yt, info: Dict[str, Any]) -> float:
    """Best-effort EPS Growth proxy to reduce nulls.
    Preference order:
    - info['earningsQuarterlyGrowth']
    - info['earningsGrowth']
    - pct change of quarterly net earnings (last vs prev)
    - pct change of annual net earnings (last vs prev)
    """
    try:
        val = info.get("earningsQuarterlyGrowth", np.nan)
        if pd.notna(val):
            return float(val)
    except Exception:
        pass
    try:
        val = info.get("earningsGrowth", np.nan)
        if pd.notna(val):
            return float(val)
    except Exception:
        pass
    # Quarterly earnings proxy
    try:
        q = yt.quarterly_earnings
        if isinstance(q, pd.DataFrame) and not q.empty and "Earnings" in q.columns:
            s = q["Earnings"].dropna()
            if len(s) >= 2 and s.iloc[-2] not in (0, np.nan):
                base = float(s.iloc[-2])
                if base != 0:
                    return float((float(s.iloc[-1]) - base) / abs(base))
    except Exception:
        pass
    # Annual earnings proxy via income statement (Net Income) — replaces deprecated yt.earnings
    try:
        inc = getattr(yt, "income_stmt", None)
        if isinstance(inc, pd.DataFrame) and not inc.empty:
            # income_stmt is indexed by line items; columns are dates/periods
            # Prefer Net Income row
            if "Net Income" in inc.index:
                s = inc.loc["Net Income"].dropna()
                # s is a Series indexed by period; compare last vs previous
                if s.shape[0] >= 2:
                    last = float(s.iloc[0]) if hasattr(s, 'iloc') else float(list(s)[-1])
                    prev = float(s.iloc[1]) if hasattr(s, 'iloc') else float(list(s)[-2])
                    if prev not in (0, np.nan) and prev != 0:
                        return float((last - prev) / abs(prev))
    except Exception:
        pass
    # Quarterly income statement fallback
    try:
        qinc = getattr(yt, "quarterly_income_stmt", None)
        if isinstance(qinc, pd.DataFrame) and not qinc.empty and "Net Income" in qinc.index:
            s = qinc.loc["Net Income"].dropna()
            if s.shape[0] >= 2:
                last = float(s.iloc[0])
                prev = float(s.iloc[1])
                if prev not in (0, np.nan) and prev != 0:
                    return float((last - prev) / abs(prev))
    except Exception:
        pass
    return np.nan

# === Feature Extractors ===
def fetch_options_sentiment(ticker: str) -> Tuple[float, float, float, float, float]:
    try:
        yt = get_yf_ticker(ticker)
        opt_dates = yt.options
        if not opt_dates:
            return (np.nan,) * 5
        chain = yt.option_chain(opt_dates[0])
        calls = chain.calls
        puts = chain.puts
        call_oi_ratio = puts["openInterest"].sum() / calls["openInterest"].sum() if calls["openInterest"].sum() else np.nan
        call_vol_ratio = puts["volume"].sum() / calls["volume"].sum() if calls["volume"].sum() else np.nan
        skew = calls["impliedVolatility"].mean() - puts["impliedVolatility"].mean()
        call_vol_spike = np.nan
        if len(calls["volume"]) >= 5:
            rolling_mean = calls["volume"].rolling(5).mean().iloc[-1]
            call_vol_spike = calls["volume"].iloc[-1] / rolling_mean if rolling_mean else np.nan
        iv_spike = (
            calls["impliedVolatility"].pct_change().iloc[-1] * 100
            if len(calls["impliedVolatility"]) > 1 else np.nan
        )
        return call_oi_ratio, call_vol_ratio, skew, call_vol_spike, iv_spike
    except Exception as e:
        logger.warning(f"{ticker} options fetch failed: {e}")
        return (np.nan,) * 5

def fetch_insider_buys(ticker: str) -> Tuple[int, int, str, str]:
    try:
        symbol = ticker.replace("$", "")
        resp = _sess.get(
            f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={symbol}&apikey={FMP_API_KEY}"
        )
        data = resp.json()
        df = pd.DataFrame(data if isinstance(data, list) else [data])
        if df.empty or 'transactionDate' not in df.columns:
            # Fallback to EDGAR if FMP missing
            return _fetch_insider_buys_edgar(ticker)
        df['transactionDate'] = pd.to_datetime(df['transactionDate'], errors='coerce')
        recent = df[(df['transactionType'] == "Buy") & (df['transactionDate'] >= datetime.utcnow() - timedelta(days=30))]
        count = len(recent)
        volume = int(recent['securitiesTransacted'].sum()) if 'securitiesTransacted' in recent else 0
        last_date = recent['transactionDate'].max().strftime("%Y-%m-%d") if count else ""
        signal = "Yes" if count >= 1 else "No"
        # If FMP yields nothing, try EDGAR to reduce blanks
        if count == 0 and volume == 0:
            return _fetch_insider_buys_edgar(ticker)
        return count, volume, last_date, signal
    except Exception as e:
        logger.warning(f"{ticker} insider fetch failed: {e}")
        # Try EDGAR fallback regardless of error
        try:
            return _fetch_insider_buys_edgar(ticker)
        except Exception:
            return 0, 0, "", "No"

@lru_cache(maxsize=1)
def _get_sec_ticker_map() -> dict:
    """Build a map of TICKER -> 10-digit CIK from SEC files.

    Handles multiple JSON shapes:
    - Dict with numeric keys: {"0": {"ticker": "AAPL", "cik_str": 320193, ...}, ...}
    - Dict with 'data': [{"ticker": "AAPL", "cik_str": 320193, ...}, ...]
    - List of records: [{"ticker": "AAPL", "cik_str": 320193, ...}, ...]
    Falls back to company_tickers_exchange.json if needed.
    """
    def parse_mapping(obj) -> dict:
        mapping: dict[str, str] = {}
        try:
            if isinstance(obj, dict):
                # Shape B: {fields: [...], data: [[...], ...]}
                if ("fields" in obj and isinstance(obj.get("data"), list)):
                    # Shape B: {fields: [...], data: [[...], ...]}
                    fields = obj.get("fields", [])
                    data = obj.get("data", [])
                    if isinstance(fields, list) and isinstance(data, list):
                        try:
                            idx_t = fields.index("ticker") if "ticker" in fields else None
                            # cik field name varies
                            for cand in ("cik_str", "cik", "ciknumber"):
                                if cand in fields:
                                    idx_c = fields.index(cand)
                                    break
                            else:
                                idx_c = None
                            if idx_t is not None and idx_c is not None:
                                for row in data:
                                    try:
                                        t = str(row[idx_t]).upper()
                                        cik_int = row[idx_c]
                                        if t and cik_int is not None:
                                            mapping[t] = str(int(cik_int)).zfill(10)
                                    except Exception:
                                        continue
                        except Exception:
                            pass
                else:
                    # Shape A: numeric-keyed dict of records
                    for rec in list(obj.values()):
                        try:
                            t = str(rec.get("ticker", "")).upper()
                            cik_int = rec.get("cik_str") or rec.get("cik") or rec.get("ciknumber")
                            if t and cik_int is not None:
                                mapping[t] = str(int(cik_int)).zfill(10)
                        except Exception:
                            continue
            elif isinstance(obj, list):
                for rec in obj:
                    try:
                        t = str(rec.get("ticker", "")).upper()
                        cik_int = rec.get("cik_str") or rec.get("cik") or rec.get("ciknumber")
                        if t and cik_int is not None:
                            mapping[t] = str(int(cik_int)).zfill(10)
                    except Exception:
                        continue
        except Exception:
            return mapping
        return mapping

    # Primary
    m1 = _sec_get("https://www.sec.gov/files/company_tickers.json", expect_json=True)
    mapping = parse_mapping(m1) if m1 is not None else {}
    if mapping:
        return mapping
    # Secondary
    m2 = _sec_get("https://www.sec.gov/files/company_tickers_exchange.json", expect_json=True)
    mapping = parse_mapping(m2) if m2 is not None else {}
    return mapping

@lru_cache(maxsize=4096)
def _cik_for_ticker(ticker: str) -> Optional[str]:
    try:
        t = str(ticker).strip().lstrip("$").upper()
        mapping = _get_sec_ticker_map()
        return mapping.get(t)
    except Exception:
        return None

def _discover_form4_xml(cik_no_zeros: str, accession_nodash: str) -> Optional[str]:
    """Find a Form 4 XML file in an accession directory via index.json."""
    try:
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_nodash}/index.json"
        idx = _sec_get(idx_url, expect_json=True)
        if idx is None:
            return None
        files = idx.get("directory", {}).get("item", []) if isinstance(idx, dict) else []
        # Prefer files that look like Form 4 XML
        for cand in files:
            name = str(cand.get("name", ""))
            if name.lower().endswith(".xml") and ("form4" in name.lower() or "doc4" in name.lower() or "f345" in name.lower()):
                return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_nodash}/{name}"
        # fallback: first xml
        for cand in files:
            name = str(cand.get("name", ""))
            if name.lower().endswith(".xml"):
                return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_nodash}/{name}"
    except Exception:
        return None
    return None

def _fetch_insider_buys_edgar(ticker: str) -> Tuple[int, int, str, str]:
    """Best-effort EDGAR fallback: count purchase transactions (code 'P') in Form 4 within 30D.

    Returns: (count_transactions, shares_bought_sum, last_date, signal)
    """
    try:
        # If EDGAR disabled (cooldown or forced), bail out early
        if not _edgar_enabled():
            return 0, 0, "", "No"
        import xml.etree.ElementTree as ET
        # Ticker -> CIK
        cik = _cik_for_ticker(ticker)
        if not cik:
            return 0, 0, "", "No"
        cik_no_zeros = str(int(cik))  # drop leading zeros for path segment
        subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        subs = _sec_get(subs_url, expect_json=True)
        if subs is None:
            return 0, 0, "", "No"
        recent = subs.get("filings", {}).get("recent", {}) if isinstance(subs, dict) else {}
        forms = list(recent.get("form", []))
        dates = list(recent.get("filingDate", []))
        accs = list(recent.get("accessionNumber", []))
        prims = list(recent.get("primaryDocument", []))
        if not forms:
            return 0, 0, "", "No"
        cutoff = datetime.utcnow().date() - timedelta(days=30)
        count_tx = 0
        vol_sum = 0
        last_dt: Optional[str] = ""
        for i, f in enumerate(forms):
            try:
                if not str(f).startswith("4"):
                    continue
                fdate = pd.to_datetime(dates[i], errors='coerce')
                if pd.isna(fdate) or fdate.date() < cutoff:
                    continue
                acc = str(accs[i])
                accession_nodash = acc.replace("-", "")
                # Discover XML to parse
                xml_url = _discover_form4_xml(cik_no_zeros, accession_nodash)
                if not xml_url:
                    # Try primary document if looks xml
                    prim = str(prims[i]) if i < len(prims) else ""
                    if prim.lower().endswith('.xml'):
                        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_nodash}/{prim}"
                if not xml_url:
                    continue
                xml_text = _sec_get(xml_url, expect_json=False)
                if not xml_text:
                    continue
                tree = ET.fromstring(xml_text)
                # Namespaces vary; search by tag suffix
                def _iter(tx_tag: str):
                    for el in tree.iter():
                        if el.tag.endswith(tx_tag):
                            yield el
                for tx in _iter('nonDerivativeTransaction'):
                    code = None
                    shares = 0
                    tdate = None
                    for el in tx.iter():
                        tag = el.tag.split('}')[-1]
                        if tag == 'transactionCode':
                            code = (el.text or '').strip()
                        elif tag == 'transactionShares':
                            # value is nested <value>
                            for sub in el.iter():
                                if sub.tag.split('}')[-1] == 'value':
                                    try:
                                        shares = int(float((sub.text or '0').replace(',', '')))
                                    except Exception:
                                        shares = shares
                        elif tag == 'transactionDate':
                            for sub in el.iter():
                                if sub.tag.split('}')[-1] == 'value':
                                    tdate = (sub.text or '').strip()
                    if code == 'P':
                        count_tx += 1
                        vol_sum += shares
                        if tdate:
                            if not last_dt or tdate > last_dt:
                                last_dt = tdate
            except Exception:
                continue
        signal = "Yes" if count_tx > 0 else "No"
        return count_tx, int(vol_sum), (last_dt or ""), signal
    except Exception:
        return 0, 0, "", "No"

def _proxy_sector_from_info(info: Dict[str, Any]) -> Optional[str]:
    try:
        s = info.get("sector")
        return str(s) if isinstance(s, str) and s else None
    except Exception:
        return None

def _etf_for_sector(sector: str) -> Optional[str]:
    # Map common sectors to popular SPDR ETFs
    mapping = {
        "Technology": "XLK", "Information Technology": "XLK",
        "Health Care": "XLV", "Healthcare": "XLV",
        "Financial": "XLF", "Financial Services": "XLF",
        "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
        "Energy": "XLE", "Utilities": "XLU",
        "Industrials": "XLI", "Materials": "XLB",
        "Real Estate": "XLRE", "Communication Services": "XLC"
    }
    return mapping.get(sector)

def _etf_flow_proxy(etf_symbol: str) -> _Tuple[float, float, str]:
    try:
        yt = get_yf_ticker(etf_symbol)
        hist = yt.history(period="6mo")
        if hist.empty:
            return 0.0, 0.0, "No"
        # Proxy flows using price*volume spike vs 30D avg
        pv = (hist["Close"] * hist["Volume"]).dropna()
        spike_ratio = float(pv.iloc[-1] / pv.rolling(30).mean().iloc[-1]) if len(pv) >= 30 else 0.0
        inflow = float(pv.pct_change().iloc[-1] * 1e6) if len(pv) >= 2 else 0.0
        signal = "Yes" if spike_ratio > 1.5 else "No"
        return inflow, spike_ratio, signal
    except Exception:
        return 0.0, 0.0, "No"

def fetch_sector_flows(ticker: str) -> Tuple[float, float, str]:
    try:
        resp = _sess.get(
            f"https://financialmodelingprep.com/api/v4/etf-sector-weightings?symbol=SPY&apikey={FMP_API_KEY}"
        )
        data = resp.json()
        if not data or not isinstance(data, list):
            return 0.0, 0.0, "No"
        current_weights = {item["sector"]: float(item["weightPercentage"]) for item in data}
        sector = max(current_weights, key=current_weights.get)
        inflow = round(current_weights[sector] - current_weights[sector] * 0.97, 2)
        spike_ratio = round(current_weights[sector] / (current_weights[sector] * 0.95), 2)
        signal = "Yes" if inflow > 0.5 or spike_ratio > 1.5 else "No"
        return inflow, spike_ratio, signal
    except Exception as e:
        # Fallback: approximate using sector ETF
        try:
            symbol = ticker.replace("$", "")
            info = get_yf_ticker(symbol).info
            sector = _proxy_sector_from_info(info)
            etf = _etf_for_sector(sector) if sector else None
            if etf:
                return _etf_flow_proxy(etf)
        except Exception:
            pass
        logger.warning(f"{ticker} ETF flow fetch failed: {e}")
        return 0.0, 0.0, "No"

# === Feature Builders ===
def calculate_technicals(closes: pd.Series) -> Dict[str, float]:
    current = closes.iloc[-1]
    volatility = closes.rolling(TECH_VOLATILITY_WINDOW).std().iloc[-1]
    ma_50 = closes.rolling(50).mean().iloc[-1]
    ma_200 = closes.rolling(200).mean().iloc[-1]
    price_vs_50 = ((current - ma_50) / ma_50 * 100) if ma_50 else np.nan
    price_vs_200 = ((current - ma_200) / ma_200 * 100) if ma_200 else np.nan

    delta = closes.diff()
    avg_gain = delta.clip(lower=0).rolling(TECH_RSI_PERIOD).mean()
    avg_loss = -delta.clip(upper=0).rolling(TECH_RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).dropna().iloc[-1] if not rs.dropna().empty else np.nan

    exp1 = closes.ewm(span=12, adjust=False).mean()
    exp2 = closes.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = (macd - signal).iloc[-1] if len(macd) > 0 else np.nan

    bb_mean = closes.rolling(TECH_BB_PERIOD).mean()
    bb_std = closes.rolling(TECH_BB_PERIOD).std()
    upper = bb_mean + 2 * bb_std
    lower = bb_mean - 2 * bb_std
    bollinger_width = ((upper - lower) / closes).iloc[-1] if len(closes) >= TECH_BB_PERIOD else np.nan

    return {
        "Volatility": volatility,
        "Above 50-Day MA %": price_vs_50,
        "Above 200-Day MA %": price_vs_200,
        "RSI": rsi,
        "MACD Histogram": macd_hist,
        "Bollinger Width": bollinger_width
    }

# === Main Pull Function ===
def fetch_single_ticker(ticker: str) -> Tuple[str, List]:
    try:
        # Rate limit across threads and short-circuit if breaker is open
        if not _breaker.allow():
            logger.warning(f"{ticker} skipped due to open circuit breaker")
            try:
                emit_metric("yfinance", {"ticker": ticker, "ok": 0, "breaker_open": True})
            except Exception:
                pass
            return ticker, [ticker] + [np.nan] * 43
        waited = _bucket.take(1.0)
        if waited > 0.5:
            logger.debug(f"Rate-limited {ticker}, waited {waited:.2f}s")

        # Per-ticker soft timeout guard for yfinance calls
        start_ts = time.time()
        yt = get_yf_ticker(ticker)
        info = yt.info
        hist = yt.history(period="1y")
        closes = hist["Close"].dropna()
        if closes.empty:
            raise ValueError("No historical close data")

        current = closes.iloc[-1]
        change_1d = ((current - closes.iloc[-2]) / closes.iloc[-2] * 100) if len(closes) >= 2 else np.nan
        change_7d = ((current - closes.iloc[-7]) / closes.iloc[-7] * 100) if len(closes) >= 7 else np.nan
        # Relative strength vs SPY (7D)
        spy_closes = _get_spy_closes_1y()
        if spy_closes is not None and len(spy_closes) >= 7 and len(closes) >= 7:
            spy_current = spy_closes.iloc[-1]
            spy_prev = spy_closes.iloc[-7]
            spy_change_7d = ((spy_current - spy_prev) / spy_prev * 100) if spy_prev else np.nan
            rel_strength = (change_7d - spy_change_7d) if pd.notna(change_7d) and pd.notna(spy_change_7d) else np.nan
        else:
            rel_strength = np.nan
        volume_spike = info.get("volume", 0) / hist["Volume"].rolling(TECH_VOL_SPIKE_WINDOW).mean().iloc[-1] if len(hist) >= TECH_VOL_SPIKE_WINDOW else np.nan

        tech = calculate_technicals(closes)

        # EPS growth: derive robustly using multiple fallbacks
        eps_growth = _derive_eps_growth(yt, info)
        fundamentals = [
            info.get("trailingPE", np.nan), eps_growth,
            info.get("returnOnEquity", np.nan), info.get("debtToEquity", np.nan),
            (info.get("freeCashflow", 0) / info.get("totalRevenue", 1)) if info.get("totalRevenue") else np.nan
        ]

        # Short interest and fallbacks
        shares_short = info.get("sharesShort", np.nan)
        short_ratio = info.get("shortRatio", np.nan)
        short_pct_float = info.get("shortPercentOfFloat", np.nan)
        short_pct_out = info.get("shortPercentOfSharesOutstanding", np.nan)
        shares_outstanding = info.get("sharesOutstanding", np.nan)
        # Fallback for Short Percent Outstanding = sharesShort / sharesOutstanding
        try:
            if (pd.isna(short_pct_out) or short_pct_out is None) and pd.notna(shares_short) and pd.notna(shares_outstanding) and float(shares_outstanding) != 0:
                short_pct_out = float(shares_short) / float(shares_outstanding)
        except Exception:
            pass
        short_interest = [
            shares_short, short_ratio,
            short_pct_float, short_pct_out
        ]

        # Optionally fetch expensive/spotty data depending on feature toggles
        if any(FEATURE_TOGGLES.get(k, False) for k in [
            "Put/Call OI Ratio", "Put/Call Volume Ratio", "Options Skew", "Call Volume Spike Ratio", "IV Spike %"]):
            options = fetch_options_sentiment(ticker)
        else:
            options = (np.nan,) * 5

        if any(FEATURE_TOGGLES.get(k, False) for k in [
            "Insider Buys 30D", "Insider Buy Volume", "Last Insider Buy Date", "Insider Signal"]):
            insiders = fetch_insider_buys(ticker)
        else:
            insiders = (0, 0, "", "No")

        if any(FEATURE_TOGGLES.get(k, False) for k in [
            "Sector Inflows", "ETF Flow Spike Ratio", "ETF Flow Signal"]):
            etf = fetch_sector_flows(ticker)
        else:
            etf = (0.0, 0.0, "No")

        momentum_30d = ((current - closes.iloc[-TECH_MOMENTUM_DAYS]) / closes.iloc[-TECH_MOMENTUM_DAYS] * 100) if len(closes) >= TECH_MOMENTUM_DAYS else np.nan
        beta = info.get("beta", np.nan)
        if pd.isna(beta):
            beta = _compute_beta_from_returns(closes, _get_spy_closes_1y())

        try:
            earnings_date = yt.calendar.loc["Earnings Date"][0].strftime("%Y-%m-%d")
        except Exception:
            earnings_date = ""

        try:
            earn = yt.earnings_dates.index[-1]
            pre = closes[closes.index < earn].iloc[-1]
            post = closes[closes.index >= earn].iloc[0]
            earnings_gap = ((post - pre) / pre * 100)
        except Exception:
            earnings_gap = np.nan

        avg_val = (hist["Close"] * hist["Volume"]).rolling(30).mean().iloc[-1] if len(hist) >= 30 else np.nan

        # Retail holding approximation = 1 - institutions - insiders, clamped to [0,1]
        held_inst = info.get("heldPercentInstitutions", np.nan)
        held_ins = info.get("heldPercentInsiders", np.nan)
        try:
            retail_pct = float(1 - float(held_inst or 0) - float(held_ins or 0))
            retail_pct = max(0.0, min(1.0, retail_pct))
        except Exception:
            retail_pct = np.nan

        # Success — record breaker
        _breaker.record(True)
        # Defensive timeout log
        elapsed = time.time() - start_ts
        if elapsed > YF_PER_TICKER_TIMEOUT_SEC:
            logger.warning(f"{ticker} exceeded per-ticker timeout ({elapsed:.1f}s > {YF_PER_TICKER_TIMEOUT_SEC}s)")
        try:
            emit_metric("yfinance", {"ticker": ticker, "ok": 1, "latency_ms": int(elapsed*1000)})
        except Exception:
            pass

        return ticker, [
            ticker, info.get("shortName", ""), current, change_1d, change_7d, info.get("marketCap"),
            info.get("volume"), info.get("sector", ""), volume_spike, rel_strength,
            tech["Above 50-Day MA %"], tech["Above 200-Day MA %"], tech["RSI"], tech["MACD Histogram"],
            tech["Bollinger Width"], tech["Volatility"], *fundamentals, *short_interest, *options,
            *insiders, *etf, momentum_30d, beta, earnings_date, earnings_gap,
            retail_pct, held_inst, avg_val
        ]
    except Exception as e:
        logger.warning(f"{ticker} failed: {e}")
        # Failure — inform breaker
        try:
            _breaker.record(False)
        except Exception:
            pass
        try:
            emit_metric("yfinance", {"ticker": ticker, "ok": 0})
        except Exception:
            pass
        return ticker, [ticker] + [np.nan] * 43

def fetch_yf_data(ticker_list: List[str], max_workers: int = YF_MAX_WORKERS) -> pd.DataFrame:
    """Fetch finance features from yfinance with gentle batching.

    - Processes tickers in batches of YF_BATCH_SIZE with a short pause between batches
      to avoid rate limiting and be a better API citizen.
    - Uses up to max_workers threads per batch.
    """
    results: Dict[str, List] = {}
    delisted: List[str] = []
    tickers = [str(t) for t in ticker_list]

    if not tickers:
        return pd.DataFrame()

    batch_size = max(1, int(YF_BATCH_SIZE))
    pause_sec = max(0.0, float(YF_BATCH_PAUSE_SEC))

    with tqdm(total=len(tickers), desc="📊 Fetching yfinance data", unit="ticker") as pbar:
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            # Cap workers to batch size to avoid spawning idle threads
            workers = min(max_workers, len(batch)) if max_workers else len(batch)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(fetch_single_ticker, t): t for t in batch}
                for future in as_completed(futures):
                    try:
                        ticker, data = future.result()
                        results[ticker] = data
                        if len(data) != 44 or all(pd.isna(val) for val in data[1:]):
                            delisted.append(ticker)
                    except Exception:
                        t = futures[future]
                        results[t] = [t] + [np.nan] * 43
                        delisted.append(t)
                    finally:
                        pbar.update(1)
            # Gentle pause between batches, but not after the last batch
            if i + batch_size < len(tickers) and pause_sec > 0:
                time.sleep(pause_sec)

    col_labels = [
        "Ticker", "Company", "Current Price", "Price 1D %", "Price 7D %", "Market Cap", "Volume", "Sector",
        "Volume Spike Ratio", "Relative Strength", "Above 50-Day MA %", "Above 200-Day MA %", "RSI", "MACD Histogram",
        "Bollinger Width", "Volatility", "P/E Ratio", "EPS Growth", "ROE", "Debt/Equity", "FCF Margin",
        "Shares Short", "Short Ratio", "Short Percent Float", "Short Percent Outstanding",
        "Put/Call OI Ratio", "Put/Call Volume Ratio", "Options Skew", "Call Volume Spike Ratio", "IV Spike %",
        "Insider Buys 30D", "Insider Buy Volume", "Last Insider Buy Date", "Insider Signal",
        "Sector Inflows", "ETF Flow Spike Ratio", "ETF Flow Signal",
        "Momentum 30D %", "Beta vs SPY", "Next Earnings Date", "Earnings Gap %",
        "Retail Holding %", "Float % Held by Institutions", "Avg Daily Value Traded"
    ]

    df = pd.DataFrame([results[t] for t in tickers], columns=col_labels)

    if "Volatility" in df.columns and not df["Volatility"].isnull().all():
        df["Volatility Rank"] = df["Volatility"].rank(pct=True)

    return df
