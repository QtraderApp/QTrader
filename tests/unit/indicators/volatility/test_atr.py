"""Unit tests for ATR (Average True Range) indicator."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.api import ATR, Context
from qtrader.models.bar import Bar


def create_bar(symbol: str, high: float, low: float, close: float) -> Bar:
    """Helper to create a bar with specified HLC values."""
    return Bar(
        ts=datetime.now(timezone.utc),
        symbol=symbol,
        open=Decimal(str(close)),  # Not used by ATR
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=1000,
    )


def test_atr_initialization():
    """Test ATR initializes correctly."""
    atr = ATR(period=14)
    assert atr.period == 14
    # ATR computes on high/low/close, no single field


def test_atr_insufficient_data():
    """Test ATR returns None until period bars received."""
    atr = ATR(period=3)
    ctx = Context()

    # Add 2 bars (need 3 for ATR(3))
    result = None
    for vals in [(10.0, 8.0, 9.0), (11.0, 9.0, 10.0)]:
        bar = create_bar("TEST", *vals)
        ctx._add_bar_to_history(bar)
        result = atr.compute("TEST", ctx)

    assert result is None


def test_atr_first_value():
    """Test ATR first value is SMA of True Range."""
    atr = ATR(period=3)
    ctx = Context()

    # Bar 1: high=10, low=8, close=9
    # TR = max(10-8, |10-9|, |8-9|) = 2.0
    bar1 = create_bar("TEST", 10.0, 8.0, 9.0)
    ctx._add_bar_to_history(bar1)
    _ = atr.compute("TEST", ctx)

    # Bar 2: high=11, low=9, close=10, prev_close=9
    # TR = max(11-9, |11-9|, |9-9|) = 2.0
    bar2 = create_bar("TEST", 11.0, 9.0, 10.0)
    ctx._add_bar_to_history(bar2)
    _ = atr.compute("TEST", ctx)

    # Bar 3: high=12, low=10, close=11, prev_close=10
    # TR = max(12-10, |12-10|, |10-10|) = 2.0
    bar3 = create_bar("TEST", 12.0, 10.0, 11.0)
    ctx._add_bar_to_history(bar3)
    result3 = atr.compute("TEST", ctx)

    # First ATR = SMA(TR) = (2.0 + 2.0 + 2.0) / 3 = 2.0
    assert result3 == pytest.approx(2.0, rel=1e-6)


def test_atr_smoothing():
    """Test ATR smoothing after first value."""
    atr = ATR(period=3)
    ctx = Context()

    # First 3 bars all have TR = 2.0
    result = None
    for vals in [(10.0, 8.0, 9.0), (11.0, 9.0, 10.0), (12.0, 10.0, 11.0)]:
        bar = create_bar("TEST", *vals)
        ctx._add_bar_to_history(bar)
        result = atr.compute("TEST", ctx)

    # First ATR = 2.0
    assert result == pytest.approx(2.0, rel=1e-6)

    # Bar 4: high=14, low=11, close=12, prev_close=11
    # TR = max(14-11, |14-11|, |11-11|) = 3.0
    # ATR = ((3-1) * 2.0 + 3.0) / 3 = (4.0 + 3.0) / 3 = 2.333...
    bar4 = create_bar("TEST", 14.0, 11.0, 12.0)
    ctx._add_bar_to_history(bar4)
    result4 = atr.compute("TEST", ctx)

    assert result4 == pytest.approx(2.333333, rel=1e-5)


def test_atr_true_range_high_to_low():
    """Test True Range is high - low when largest."""
    atr = ATR(period=2)
    ctx = Context()

    # Bar 1: TR = 10 - 5 = 5
    bar1 = create_bar("TEST", 10.0, 5.0, 8.0)
    ctx._add_bar_to_history(bar1)
    _ = atr.compute("TEST", ctx)

    # Bar 2: TR should also be around 5 (depending on prev_close)
    bar2 = create_bar("TEST", 15.0, 10.0, 13.0)
    ctx._add_bar_to_history(bar2)
    result2 = atr.compute("TEST", ctx)

    # First ATR should be close to 5 (average of TRs)
    assert result2 is not None
    assert result2 > 4.0  # Should be around 5


def test_atr_true_range_gap_up():
    """Test True Range includes gap up from previous close."""
    atr = ATR(period=2)
    ctx = Context()

    # Bar 1: close=10
    bar1 = create_bar("TEST", 12.0, 8.0, 10.0)
    ctx._add_bar_to_history(bar1)
    atr.compute("TEST", ctx)

    # Bar 2: Gaps up to 20 (high=22, low=20, close=21)
    # TR = max(22-20, |22-10|, |20-10|) = max(2, 12, 10) = 12
    bar2 = create_bar("TEST", 22.0, 20.0, 21.0)
    ctx._add_bar_to_history(bar2)
    result2 = atr.compute("TEST", ctx)

    # ATR should reflect the large gap
    assert result2 > 6.0  # Should be high due to gap


def test_atr_true_range_gap_down():
    """Test True Range includes gap down from previous close."""
    atr = ATR(period=2)
    ctx = Context()

    # Bar 1: close=20
    bar1 = create_bar("TEST", 22.0, 18.0, 20.0)
    ctx._add_bar_to_history(bar1)
    atr.compute("TEST", ctx)

    # Bar 2: Gaps down to 10 (high=12, low=10, close=11)
    # TR = max(12-10, |12-20|, |10-20|) = max(2, 8, 10) = 10
    bar2 = create_bar("TEST", 12.0, 10.0, 11.0)
    ctx._add_bar_to_history(bar2)
    result2 = atr.compute("TEST", ctx)

    # ATR should reflect the large gap
    assert result2 > 5.0  # Should be high due to gap


def test_atr_multiple_symbols():
    """Test ATR handles multiple symbols independently."""
    atr = ATR(period=2)
    ctx = Context()

    # AAPL: Bars with TR=2.0 each
    result_aapl = None
    for vals in [(10.0, 8.0, 9.0), (11.0, 9.0, 10.0)]:
        bar = create_bar("AAPL", *vals)
        ctx._add_bar_to_history(bar)
        result_aapl = atr.compute("AAPL", ctx)

    # MSFT: Bars with TR=4.0 each
    result_msft = None
    for vals in [(20.0, 16.0, 18.0), (22.0, 18.0, 20.0)]:
        bar = create_bar("MSFT", *vals)
        ctx._add_bar_to_history(bar)
        result_msft = atr.compute("MSFT", ctx)

    # AAPL ATR should be around 2.0
    assert result_aapl == pytest.approx(2.0, rel=1e-5)

    # MSFT ATR should be around 4.0
    assert result_msft == pytest.approx(4.0, rel=1e-5)


def test_atr_reset():
    """Test ATR reset clears state."""
    atr = ATR(period=2)
    ctx = Context()

    # Add 2 bars
    result1 = None
    for vals in [(10.0, 8.0, 9.0), (11.0, 9.0, 10.0)]:
        bar = create_bar("TEST", *vals)
        ctx._add_bar_to_history(bar)
        result1 = atr.compute("TEST", ctx)

    assert result1 is not None

    # Reset
    atr.reset("TEST")

    # Should need 2 bars again
    bar = create_bar("TEST", 12.0, 10.0, 11.0)
    ctx._add_bar_to_history(bar)
    result2 = atr.compute("TEST", ctx)
    assert result2 is None
