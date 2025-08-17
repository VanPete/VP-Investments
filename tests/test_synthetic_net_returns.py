import math
from processors.backtest import net_returns_from_series


def test_net_returns_from_series_basic(monkeypatch):
    # Force simple cost model for deterministic test
    import processors.backtest as bt
    bt.SLIPPAGE_BPS = 10.0
    bt.FEES_BPS = 5.0  # total 15 bps = 0.15%

    series = [100, 101, 102, 103, 104, 105]
    out = net_returns_from_series(series, [1, 3, 5])
    assert out["1D Return"] == 1.0
    assert out["1D Return (net)"] == 0.85  # 1.00 - 0.15
    assert out["3D Return"] == 3.0
    assert out["3D Return (net)"] == 2.85
    assert out["5D Return"] == 5.0
    assert out["5D Return (net)"] == 4.85
