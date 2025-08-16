"""
Simple file-based cache for AI responses to reduce duplicate API calls.
Keyed by a stable string; values stored as JSON in cache/ai/*.json
"""
import os
import json
import hashlib
from typing import Any, Optional

CACHE_DIR = os.path.join("cache", "ai")
os.makedirs(CACHE_DIR, exist_ok=True)


def _key_to_path(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.json")


def get(key: str) -> Optional[Any]:
    path = _key_to_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def set(key: str, value: Any) -> None:
    path = _key_to_path(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
    except Exception:
        pass
