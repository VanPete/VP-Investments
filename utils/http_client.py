"""
Unified HTTP client with retries, timeouts, and optional on-disk caching.
Use this for all network calls (news, trends, FMP, etc.) to improve reliability.
"""
from __future__ import annotations

import os
from typing import Optional

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


def build_session(cache_name: Optional[str] = None, cache_expire_seconds: int = 900, timeout: int = DEFAULT_TIMEOUT) -> requests.Session:
    """Create a Session with retries and optional caching.

    cache_name: file path stem for requests-cache; if None, no caching.
    cache_expire_seconds: TTL for cached responses.
    timeout: default per-request timeout seconds.
    """
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
