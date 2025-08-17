"""
universe_fetcher.py
Builds an expanded ticker universe from multiple public sources:
- FMP top gainers/losers/actives (requires FMP_API_KEY)
- Stocktwits trending symbols (public)
- Yahoo Finance trending tickers (public)

Returns a set of bare tickers (e.g., NVDA) to union with Reddit-derived tickers.
"""
import os
import logging
from typing import Set, List
import requests
import time
import json

from dotenv import load_dotenv
from config.config import UNIVERSE_SOURCES, UNIVERSE_LIMIT, TOKEN_BUCKET_RATE, TOKEN_BUCKET_BURST, BREAKER_FAIL_THRESHOLD, BREAKER_RESET_AFTER_SEC
from utils.http_client import build_session, get_token_bucket, CircuitBreaker
from utils.observability import emit_metric

load_dotenv()
logger = logging.getLogger(__name__)

# Shared HTTP session with small cache to avoid hammering sources
_sess = build_session(cache_name=os.path.join(os.path.dirname(__file__), "..", "cache", "universe_cache"), cache_expire_seconds=600, timeout=15)
_bucket = get_token_bucket(TOKEN_BUCKET_RATE, TOKEN_BUCKET_BURST)
_breaker = CircuitBreaker(BREAKER_FAIL_THRESHOLD, BREAKER_RESET_AFTER_SEC)


def _sanitize(symbol: str) -> str:
    s = (symbol or "").upper().strip().replace("$", "")
    # Basic filter: 1-5 letters, numbers allowed for some tickers
    return s if 1 <= len(s) <= 6 else ""


def fetch_fmp_movers() -> List[str]:
    if not UNIVERSE_SOURCES.get("FMP_MOVERS", True):
        return []
    key = os.getenv("FMP_API_KEY")
    if not key:
        logger.info("FMP_API_KEY missing; skipping FMP movers")
        return []
    base = "https://financialmodelingprep.com/api/v3/stock_market"
    endpoints = ["gainers", "losers", "actives"]
    results: List[str] = []
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache", "ai")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.abspath(os.path.join(cache_dir, "fmp_movers_cache.json"))

    for ep in endpoints:
        url = f"{base}/{ep}?apikey={key}"
        for attempt in range(3):
            try:
                if not _breaker.allow():
                    logger.info("FMP circuit open; skipping %s", ep)
                    break
                _bucket.take(1.0)
                _start = time.time()
                resp = _sess.get(url)
                resp.raise_for_status()
                data = resp.json() or []
                # cache latest
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump({"endpoint": ep, "data": data, "ts": time.time()}, f)
                except Exception:
                    pass
                for row in data[:200]:
                    sym = _sanitize(row.get("symbol", ""))
                    if sym:
                        results.append(sym)
                _breaker.record(True)
                try:
                    emit_metric("universe_fmp", {"ok": 1, "endpoint": ep, "items": len(data), "latency_ms": int((time.time()-_start)*1000)})
                except Exception:
                    pass
                break
            except Exception as e:
                # On final attempt, try cache fallback
                if attempt == 2:
                    try:
                        if os.path.exists(cache_file):
                            with open(cache_file, "r", encoding="utf-8") as f:
                                cached = json.load(f)
                                data = cached.get("data", [])
                                for row in data[:100]:
                                    sym = _sanitize(row.get("symbol", ""))
                                    if sym:
                                        results.append(sym)
                                logger.info("Using cached FMP movers due to rate-limit/error")
                                break
                    except Exception:
                        pass
                # small backoff with jitter
                _breaker.record(False)
                try:
                    emit_metric("universe_fmp", {"ok": 0, "endpoint": ep})
                except Exception:
                    pass
                time.sleep(0.7 + 0.3 * attempt)
                if attempt == 2:
                    logger.warning(f"FMP {ep} failed: {e}")
    return results


def fetch_stocktwits_trending() -> List[str]:
    if not UNIVERSE_SOURCES.get("STOCKTWITS_TRENDING", True):
        return []
    try:
        url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        if not _breaker.allow():
            logger.info("Stocktwits circuit open; skipping trending")
            return []
        _bucket.take(1.0)
        _start = time.time()
        resp = _sess.get(url)
        resp.raise_for_status()
        data = resp.json() or {}
        syms = [s.get("symbol", "") for s in (data.get("symbols", []) or [])]
        out: List[str] = []
        for s in syms:
            sym = _sanitize(s)
            if sym:
                out.append(sym)
        _breaker.record(True)
        try:
            emit_metric("universe_stocktwits", {"ok": 1, "items": len(syms), "latency_ms": int((time.time()-_start)*1000)})
        except Exception:
            pass
        return out
    except Exception as e:
        # 403s are common without proper client context; treat as informational.
        logger.info(f"Stocktwits trending unavailable ({e}); continuing with other sources")
        _breaker.record(False)
        try:
            emit_metric("universe_stocktwits", {"ok": 0})
        except Exception:
            pass
        return []


def fetch_yahoo_trending() -> List[str]:
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        if not _breaker.allow():
            logger.info("Yahoo trending circuit open; skipping")
            return []
        _bucket.take(1.0)
        _start = time.time()
        resp = _sess.get(url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json() or {}
        results: List[str] = []
        for item in (data.get("finance", {}).get("result", []) or []):
            for q in (item.get("quotes", []) or []):
                sym = _sanitize(q.get("symbol", ""))
                if sym:
                    results.append(sym)
        _breaker.record(True)
        try:
            emit_metric("universe_yahoo", {"ok": 1, "items": len(results), "latency_ms": int((time.time()-_start)*1000)})
        except Exception:
            pass
        return results
    except Exception as e:
        logger.warning(f"Yahoo trending failed: {e}")
        _breaker.record(False)
        try:
            emit_metric("universe_yahoo", {"ok": 0})
        except Exception:
            pass
        return []


def build_trending_universe(limit: int = None) -> Set[str]:
    lim = limit or UNIVERSE_LIMIT
    pool: List[str] = []
    pool.extend(fetch_fmp_movers())
    pool.extend(fetch_stocktwits_trending())
    pool.extend(fetch_yahoo_trending())
    uniq = []
    seen = set()
    for s in pool:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
        if len(uniq) >= lim:
            break
    out = set(uniq)
    try:
        emit_metric("universe_builder", {"ok": 1, "count": len(out)})
    except Exception:
        pass
    return out
