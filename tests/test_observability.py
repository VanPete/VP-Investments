import os
import json
import shutil
from pathlib import Path

from utils.observability import emit_metric

def test_emit_metric_writes_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    os.environ["RUN_DIR"] = str(run_dir)
    emit_metric("unittest", {"ok": 1, "msg": "hello"})
    log_file = run_dir / "logs" / "unittest_metrics.jsonl"
    assert log_file.exists(), "metrics file should be created"
    content = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) >= 1
    row = json.loads(content[-1])
    assert row.get("component") == "unittest"
    assert row.get("ok") == 1
    assert row.get("msg") == "hello"
