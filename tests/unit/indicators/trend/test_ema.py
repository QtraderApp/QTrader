"""Unit tests for EMA indicator."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.api import EMA, Context
from qtrader.models.bar import Bar


def create_bar(symbol: str, close: float) -> Bar:
    """Helper to create test bar."""
    return Bar(
        ts=datetime.now(timezone.utc),
        symbol=symbol,
        open=Decimal(str(close)),
        high=Decimal(str(close * 1.01)),
        low=Decimal(str(close * 0.99)),
        close=Decimal(str(close)),
        volume=1000,
    )


def test_ema_initialization():
    """Test EMA initializes correctly."""
    ema = EMA(period=12)
    assert ema.period == 12
    assert ema.field == "close"
    assert ema.alpha == pytest.approx(2.0 / 13, rel=1e-6)  # 2/(period+1)

    ema = EMA(period=26, field="open")
    assert ema.period == 26
    assert ema.field == "open"
    assert ema.alpha == pytest.approx(2.0 / 27, rel=1e-6)


def test_ema_insufficient_data():
    """Test EMA returns None when insufficient data."""
    ema = EMA(period=5)
    ctx = Context()

    # Add 4 bars (need 5 for first EMA)
    for i in range(4):
        bar = create_bar("TEST", 100.0 + i)
        ctx._add_bar_to_history(bar)
        result = ema.compute("TEST", ctx)
        assert result is None


def test_ema_first_value_is_sma():
    """Test first EMA value equals SMA."""
    ema = EMA(period=5)
    ctx = Context()

    # Add 5 bars: 100, 102, 104, 106, 108
    # Must call compute() for each bar as it's added
    values = [100.0, 102.0, 104.0, 106.0, 108.0]
    result = None
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = ema.compute("TEST", ctx)

    # First EMA should be SMA
    expected_sma = sum(values) / len(values)  # 104.0
    assert result == pytest.approx(expected_sma, rel=1e-6)


def test_ema_exponential_weighting():
    """Test EMA applies exponential weighting correctly."""
    period = 5
    ema = EMA(period=period)
    ctx = Context()

    # Add initial bars for SMA initialization
    # Must call compute() for each bar
    initial_values = [100.0, 102.0, 104.0, 106.0, 108.0]
    first_ema = None
    for val in initial_values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        first_ema = ema.compute("TEST", ctx)

    # First EMA (SMA)
    assert first_ema == pytest.approx(104.0, rel=1e-6)

    # Add another bar: 110
    bar = create_bar("TEST", 110.0)
    ctx._add_bar_to_history(bar)
    second_ema = ema.compute("TEST", ctx)

    # Calculate expected: alpha * 110 + (1 - alpha) * 104
    alpha = 2.0 / (period + 1)
    expected = alpha * 110.0 + (1 - alpha) * 104.0
    assert second_ema == pytest.approx(expected, rel=1e-6)


def test_ema_vs_manual_calculation():
    """Test EMA accuracy against manual calculation."""
    period = 3
    ema = EMA(period=period)
    ctx = Context()

    # Add bars: 100, 110, 120, 130, 140
    values = [100.0, 110.0, 120.0, 130.0, 140.0]
    results = []

    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = ema.compute("TEST", ctx)
        results.append(result)

    # First 2 should be None
    assert results[0] is None
    assert results[1] is None

    # 3rd: SMA = (100 + 110 + 120) / 3 = 110
    assert results[2] == pytest.approx(110.0, rel=1e-6)

    # 4th: EMA = 0.5 * 130 + 0.5 * 110 = 120
    assert results[3] == pytest.approx(120.0, rel=1e-6)

    # 5th: EMA = 0.5 * 140 + 0.5 * 120 = 130
    assert results[4] == pytest.approx(130.0, rel=1e-6)


def test_ema_reacts_faster_than_sma():
    """Test EMA reacts faster to price changes than SMA."""
    from qtrader.api import SMA

    period = 5
    ema = EMA(period=period)
    sma = SMA(period=period)
    ctx = Context()

    # Add stable prices: 100, 100, 100, 100, 100
    # Call compute() for each bar
    ema_before = None
    sma_before = None
    for _ in range(5):
        bar = create_bar("TEST", 100.0)
        ctx._add_bar_to_history(bar)
        ema_before = ema.compute("TEST", ctx)
        sma_before = sma.compute("TEST", ctx)

    # Both should be 100
    assert ema_before == pytest.approx(100.0, rel=1e-6)
    assert sma_before == pytest.approx(100.0, rel=1e-6)

    # Add sudden spike: 150
    bar = create_bar("TEST", 150.0)
    ctx._add_bar_to_history(bar)

    ema_after = ema.compute("TEST", ctx)
    sma_after = sma.compute("TEST", ctx)

    # EMA should react more than SMA
    # EMA: alpha * 150 + (1-alpha) * 100, alpha = 2/6 = 0.333
    # SMA: (100 + 100 + 100 + 100 + 150) / 5 = 110
    assert ema_after is not None
    assert sma_after is not None
    assert ema_after > sma_after


def test_ema_multiple_symbols():
    """Test EMA handles multiple symbols independently."""
    ema = EMA(period=3)
    ctx = Context()

    # Add bars for AAPL: 100, 110, 120
    # Call compute for each bar
    result_aapl = None
    for val in [100.0, 110.0, 120.0]:
        bar = create_bar("AAPL", val)
        ctx._add_bar_to_history(bar)
        result_aapl = ema.compute("AAPL", ctx)

    # Add bars for MSFT: 200, 210, 220
    result_msft = None
    for val in [200.0, 210.0, 220.0]:
        bar = create_bar("MSFT", val)
        ctx._add_bar_to_history(bar)
        result_msft = ema.compute("MSFT", ctx)

    # Both should be SMA on first computation
    assert result_aapl == pytest.approx(110.0, rel=1e-6)
    assert result_msft == pytest.approx(210.0, rel=1e-6)


def test_ema_reset():
    """Test EMA reset clears state."""
    ema = EMA(period=3)
    ctx = Context()

    # Add 3 bars, calling compute for each
    result1 = None
    for val in [100.0, 110.0, 120.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result1 = ema.compute("TEST", ctx)

    assert result1 is not None

    # Reset
    ema.reset("TEST")

    # Should need initialization again
    bar = create_bar("TEST", 130.0)
    ctx._add_bar_to_history(bar)
    result2 = ema.compute("TEST", ctx)
    assert result2 is None


def test_ema_different_fields():
    """Test EMA can compute on different bar fields."""
    ema_close = EMA(period=3, field="close")
    ema_high = EMA(period=3, field="high")
    ctx = Context()

    # Add bars with different OHLC values
    # Call compute for each bar
    result_close = None
    result_high = None
    for i in range(3):
        bar = Bar(
            ts=datetime.now(timezone.utc),
            symbol="TEST",
            open=Decimal(str(100 + i)),
            high=Decimal(str(110 + i)),
            low=Decimal(str(95 + i)),
            close=Decimal(str(105 + i)),
            volume=1000,
        )
        ctx._add_bar_to_history(bar)
        result_close = ema_close.compute("TEST", ctx)
        result_high = ema_high.compute("TEST", ctx)

    # Close SMA: (105 + 106 + 107) / 3 = 106
    assert result_close == pytest.approx(106.0, rel=1e-6)

    # High SMA: (110 + 111 + 112) / 3 = 111
    assert result_high == pytest.approx(111.0, rel=1e-6)
