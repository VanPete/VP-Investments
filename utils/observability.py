import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional


class JsonlLogger:
    def __init__(self, run_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.path = os.path.join(run_dir, "logs", f"events_{int(time.time())}.jsonl")
        if self.enabled:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def emit(self, event: str, payload: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        row = {
            "ts": datetime.utcnow().isoformat(),
            "event": event,
        }
        if payload:
            row.update(payload)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            # Non-fatal
            pass


class RunCounters:
    def __init__(self, run_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.path = os.path.join(run_dir, "logs", "counters.csv")
        self._counts: Dict[str, int] = {}
        if self.enabled:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def inc(self, key: str, n: int = 1):
        if not self.enabled:
            return
        self._counts[key] = self._counts.get(key, 0) + int(n)

    def flush(self):
        if not self.enabled:
            return
        try:
            # Write simple key,count CSV
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("key,count\n")
                for k, v in sorted(self._counts.items()):
                    f.write(f"{k},{v}\n")
        except Exception:
            pass


def emit_metric(component: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Write a single JSON line metric under the active run logs.

    Uses RUN_DIR environment variable; if not set, no-ops. File: logs/{component}_metrics.jsonl
    """
    try:
        run_dir = os.getenv("RUN_DIR")
        if not run_dir:
            return
        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, f"{component}_metrics.jsonl")
        row = {"ts": datetime.utcnow().isoformat(), "component": component}
        if payload:
            row.update(payload)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Non-fatal
        pass
