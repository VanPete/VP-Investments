from processors.error_budget_trend import compute_error_budget_trend
from utils.db import ensure_schema, insert_metric
from datetime import datetime, timezone


def test_error_budget_trend_smoke(tmp_path):
    ensure_schema()
    ts = datetime.now(timezone.utc).isoformat()
    insert_metric(None, 'fetch_success_news', 0.99, None, ts)
    insert_metric(None, 'fetch_success_reddit', 0.97, None, ts)
    df = compute_error_budget_trend(7)
    assert not df.empty
    assert set(['day','component','success_rate']).issubset(df.columns)
    assert {'news','reddit'}.issubset(set(df['component']))
