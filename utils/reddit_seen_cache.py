"""
Simple JSON-based seen-post cache for Reddit to reduce rescanning the same posts.
Entries expire after TTL seconds.
"""
from __future__ import annotations

import os
import json
import time
from typing import Dict

_CACHE_PATH = os.path.join("cache", "reddit_seen.json")
os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)

def _load() -> Dict[str, float]:
    try:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    return {}

def _save(d: Dict[str, float]) -> None:
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass

class SeenCache:
    def __init__(self, ttl_seconds: float):
        self.ttl = float(ttl_seconds)
        self.data: Dict[str, float] = _load()

    def has(self, pid: str) -> bool:
        now = time.time()
        # drop expired lazily
        expired = [k for k, ts in self.data.items() if now - ts > self.ttl]
        for k in expired:
            self.data.pop(k, None)
        ts = self.data.get(pid)
        return ts is not None and (now - ts) <= self.ttl

    def add(self, pid: str) -> None:
        self.data[pid] = time.time()

    def flush(self) -> None:
        _save(self.data)
