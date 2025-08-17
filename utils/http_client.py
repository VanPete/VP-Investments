"""
Unified HTTP client with retries, timeouts, and optional on-disk caching.
Use this for all network calls (news, trends, FMP, etc.) to improve reliability.
"""
from __future__ import annotations

import os
from typing import Optional
import threading
import time
import random

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import requests_cache  # type: ignore
except Exception:  # pragma: no cover - optional
    requests_cache = None

DEFAULT_TIMEOUT = 15


class TimeoutSession(requests.Session):
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        super().__init__()
        self._timeout = timeout

    def request(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        return super().request(*args, **kwargs)


def _retry_adapter(total: int = 3, backoff: float = 0.5) -> HTTPAdapter:
    # Add a little jitter to backoff to avoid thundering herds
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    return HTTPAdapter(max_retries=retry)


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.capacity = max(1, int(burst))
        self.tokens = float(self.capacity)
        self.rate = max(0.1, float(rate_per_sec))
        self.timestamp = time.time()
        self.lock = threading.Lock()

    def take(self, amount: float = 1.0) -> float:
        """Block until tokens are available; returns sleep time actually waited."""
        waited = 0.0
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.timestamp
                self.timestamp = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= amount:
                    self.tokens -= amount
                    return waited
            sleep_for = max(0.0, amount / self.rate)
            # small jitter to avoid sync
            sleep_for *= (0.9 + 0.2 * random.random())
            time.sleep(sleep_for)
            waited += sleep_for


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, reset_after_sec: int = 60):
        self.fail_threshold = max(1, int(fail_threshold))
        self.reset_after = max(5, int(reset_after_sec))
        self.fail_count = 0
        self.opened_at: Optional[float] = None
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            if self.opened_at is None:
                return True
            if (time.time() - self.opened_at) >= self.reset_after:
                # half-open
                self.fail_count = 0
                self.opened_at = None
                return True
            return False

    def record(self, success: bool) -> None:
        with self.lock:
            if success:
                self.fail_count = 0
                self.opened_at = None
            else:
                self.fail_count += 1
                if self.fail_count >= self.fail_threshold:
                    self.opened_at = time.time()


def build_session(cache_name: Optional[str] = None, cache_expire_seconds: int = 900, timeout: int = DEFAULT_TIMEOUT) -> requests.Session:
    """Create a Session with retries and optional caching.

    cache_name: file path stem for requests-cache; if None, no caching.
    cache_expire_seconds: TTL for cached responses.
    timeout: default per-request timeout seconds.
    """
    # Optional global override for cache TTL via env
    try:
        _override = int(os.environ.get("GLOBAL_CACHE_TTL_SEC", "0"))
        if _override > 0:
            cache_expire_seconds = _override
    except Exception:
        pass
    if cache_name and requests_cache is not None:
        # Ensure parent folder exists
        os.makedirs(os.path.dirname(cache_name), exist_ok=True)
        sess = requests_cache.CachedSession(
            cache_name=cache_name,
            backend="sqlite",
            expire_after=cache_expire_seconds,
        )
    else:
        sess = TimeoutSession(timeout=timeout)

    adapter = _retry_adapter()
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


# Shared, process-wide token bucket (opt-in by callers)
_GLOBAL_BUCKET: Optional[TokenBucket] = None


def get_token_bucket(rate_per_sec: float, burst: int) -> TokenBucket:
    global _GLOBAL_BUCKET
    if _GLOBAL_BUCKET is None:
        _GLOBAL_BUCKET = TokenBucket(rate_per_sec, burst)
    return _GLOBAL_BUCKET
