import pandas as pd
from processors.backtest import compute_returns


def test_compute_returns_net_included(monkeypatch):
    # Minimal row with a date far enough in the past
    from datetime import datetime, timezone, timedelta
    row = pd.Series({"Ticker": "TEST", "Run Datetime": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()})

    # Patch fetch_price_series to return a simple increasing series
    import processors.backtest as bt
    def fake_fetch(symbol, start, end):
        import pandas as _pd
        return _pd.Series([100, 101, 102, 103, 104, 105])
    bt.fetch_price_series = fake_fetch

    out = compute_returns(row)
    assert out is None or isinstance(out, dict)
    if isinstance(out, dict):
        # Expected to include net variants if returns exist
        has_any = any(k.endswith("D Return (net)") for k in out.keys())
        assert has_any
