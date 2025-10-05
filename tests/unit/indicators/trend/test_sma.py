"""Unit tests for SMA indicator."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.api import SMA, Context
from qtrader.models.bar import Bar


def create_bar(symbol: str, close: float, ts: datetime | None = None) -> Bar:
    """Helper to create test bar."""
    if ts is None:
        ts = datetime.now(timezone.utc)
    return Bar(
        ts=ts,
        symbol=symbol,
        open=Decimal(str(close)),
        high=Decimal(str(close * 1.01)),
        low=Decimal(str(close * 0.99)),
        close=Decimal(str(close)),
        volume=1000,
    )


def test_sma_initialization():
    """Test SMA initializes correctly."""
    sma = SMA(period=20)
    assert sma.period == 20
    assert sma.field == "close"

    sma = SMA(period=10, field="open")
    assert sma.period == 10
    assert sma.field == "open"


def test_sma_insufficient_data():
    """Test SMA returns None when insufficient data."""
    sma = SMA(period=5)
    ctx = Context()

    # Add 4 bars (need 5)
    for i in range(4):
        bar = create_bar("TEST", 100.0 + i)
        ctx._add_bar_to_history(bar)
        result = sma.compute("TEST", ctx)
        assert result is None


def test_sma_exact_period():
    """Test SMA computes correctly with exact period."""
    sma = SMA(period=5)
    ctx = Context()

    # Add 5 bars: 100, 101, 102, 103, 104
    # Call compute for each bar
    result = None
    for i in range(5):
        bar = create_bar("TEST", 100.0 + i)
        ctx._add_bar_to_history(bar)
        result = sma.compute("TEST", ctx)

    expected = (100 + 101 + 102 + 103 + 104) / 5  # 102.0
    assert result == pytest.approx(expected, rel=1e-6)


def test_sma_rolling_window():
    """Test SMA maintains rolling window correctly."""
    sma = SMA(period=3)
    ctx = Context()

    # Add bars: 100, 110, 120, 130
    values = [100.0, 110.0, 120.0, 130.0]
    results = []

    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = sma.compute("TEST", ctx)
        results.append(result)

    # First two should be None
    assert results[0] is None
    assert results[1] is None

    # Third should be (100 + 110 + 120) / 3 = 110
    assert results[2] == pytest.approx(110.0, rel=1e-6)

    # Fourth should be (110 + 120 + 130) / 3 = 120
    assert results[3] == pytest.approx(120.0, rel=1e-6)


def test_sma_different_fields():
    """Test SMA can compute on different bar fields."""
    from datetime import datetime, timezone
    from decimal import Decimal

    sma_close = SMA(period=3, field="close")
    sma_open = SMA(period=3, field="open")
    ctx = Context()

    # Add bars with different OHLC values
    # Call compute for each bar
    result_close = None
    result_open = None
    for i in range(3):
        bar = Bar(
            ts=datetime.now(timezone.utc),
            symbol="TEST",
            open=Decimal(str(100 + i)),
            high=Decimal(str(105 + i)),
            low=Decimal(str(95 + i)),
            close=Decimal(str(102 + i)),
            volume=1000,
        )
        ctx._add_bar_to_history(bar)
        result_close = sma_close.compute("TEST", ctx)
        result_open = sma_open.compute("TEST", ctx)

    # Close: (102 + 103 + 104) / 3 = 103
    assert result_close == pytest.approx(103.0, rel=1e-6)

    # Open: (100 + 101 + 102) / 3 = 101
    assert result_open == pytest.approx(101.0, rel=1e-6)


def test_sma_multiple_symbols():
    """Test SMA handles multiple symbols independently."""
    sma = SMA(period=3)
    ctx = Context()

    # Add bars for AAPL: 100, 110, 120
    result_aapl = None
    for val in [100.0, 110.0, 120.0]:
        bar = create_bar("AAPL", val)
        ctx._add_bar_to_history(bar)
        result_aapl = sma.compute("AAPL", ctx)

    # Add bars for MSFT: 200, 210, 220
    result_msft = None
    for val in [200.0, 210.0, 220.0]:
        bar = create_bar("MSFT", val)
        ctx._add_bar_to_history(bar)
        result_msft = sma.compute("MSFT", ctx)

    assert result_aapl == pytest.approx(110.0, rel=1e-6)
    assert result_msft == pytest.approx(210.0, rel=1e-6)


def test_sma_reset():
    """Test SMA reset clears state."""
    sma = SMA(period=3)
    ctx = Context()

    # Add 3 bars
    result1 = None
    for val in [100.0, 110.0, 120.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result1 = sma.compute("TEST", ctx)

    assert result1 is not None

    # Reset
    sma.reset("TEST")

    # Should need 3 bars again
    bar = create_bar("TEST", 130.0)
    ctx._add_bar_to_history(bar)
    result2 = sma.compute("TEST", ctx)
    assert result2 is None


def test_sma_accuracy_vs_manual():
    """Test SMA accuracy against manual calculation."""
    sma = SMA(period=5)
    ctx = Context()

    # Real-world-ish prices
    prices = [150.25, 151.75, 149.50, 152.00, 153.25, 154.00, 152.75]

    results = []
    for price in prices:
        bar = create_bar("TEST", price)
        ctx._add_bar_to_history(bar)
        result = sma.compute("TEST", ctx)
        results.append(result)

    # First 4 should be None
    assert all(r is None for r in results[:4])

    # 5th: (150.25 + 151.75 + 149.50 + 152.00 + 153.25) / 5
    assert results[4] == pytest.approx(151.35, rel=1e-6)

    # 6th: (151.75 + 149.50 + 152.00 + 153.25 + 154.00) / 5
    assert results[5] == pytest.approx(152.10, rel=1e-6)

    # 7th: (149.50 + 152.00 + 153.25 + 154.00 + 152.75) / 5
    assert results[6] == pytest.approx(152.30, rel=1e-6)
