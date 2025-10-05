"""Unit tests for RSI (Relative Strength Index) indicator."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.api import RSI, Context
from qtrader.models.bar import Bar


def create_bar(symbol: str, close: float) -> Bar:
    """Helper to create a bar with specified close price."""
    return Bar(
        ts=datetime.now(timezone.utc),
        symbol=symbol,
        open=Decimal(str(close)),
        high=Decimal(str(close + 1)),
        low=Decimal(str(close - 1)),
        close=Decimal(str(close)),
        volume=1000,
    )


def test_rsi_initialization():
    """Test RSI initializes correctly."""
    rsi = RSI(period=14)
    assert rsi.period == 14
    # RSI always computes on close price


def test_rsi_insufficient_data():
    """Test RSI returns None until period+1 bars received."""
    rsi = RSI(period=3)
    ctx = Context()

    # Add 3 bars (need 4 for RSI(3) - period + 1)
    result = None
    for val in [100.0, 110.0, 105.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = rsi.compute("TEST", ctx)

    # Should still be None (need period+1 bars)
    assert result is None


def test_rsi_first_value():
    """Test RSI calculates correctly after warmup."""
    rsi = RSI(period=3)
    ctx = Context()

    # Add 4 bars: 100, 110 (+10), 105 (-5), 115 (+10)
    result = None
    for val in [100.0, 110.0, 105.0, 115.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = rsi.compute("TEST", ctx)

    # First 3 changes: +10, -5, +10
    # Avg Gain = (10 + 0 + 10) / 3 = 6.666...
    # Avg Loss = (0 + 5 + 0) / 3 = 1.666...
    # RS = 6.666... / 1.666... = 4.0
    # RSI = 100 - (100 / (1 + 4)) = 100 - 20 = 80
    assert result is not None
    assert result == pytest.approx(80.0, rel=1e-5)


def test_rsi_range_0_to_100():
    """Test RSI stays within 0-100 range."""
    rsi = RSI(period=3)
    ctx = Context()

    # Add bars with extreme movements
    values = [100.0, 150.0, 200.0, 250.0, 50.0, 25.0, 10.0]
    result = None
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = rsi.compute("TEST", ctx)
        if result is not None:
            assert 0.0 <= result <= 100.0


def test_rsi_overbought_oversold():
    """Test RSI detects overbought/oversold conditions."""
    rsi = RSI(period=3)
    ctx = Context()

    # Strong uptrend - should produce high RSI (overbought)
    result_up = None
    for val in [100.0, 110.0, 120.0, 130.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result_up = rsi.compute("TEST", ctx)

    # RSI should be high (> 70 typically considered overbought)
    assert result_up is not None
    assert result_up > 70.0

    # Reset and test downtrend
    rsi.reset("TEST")
    ctx = Context()

    # Strong downtrend - should produce low RSI (oversold)
    result_down = None
    for val in [130.0, 120.0, 110.0, 100.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result_down = rsi.compute("TEST", ctx)

    # RSI should be low (< 30 typically considered oversold)
    assert result_down is not None
    assert result_down < 30.0


def test_rsi_neutral():
    """Test RSI around 50 for neutral/sideways market."""
    rsi = RSI(period=5)
    ctx = Context()

    # Alternating up/down movements (balanced)
    values = [100.0, 105.0, 100.0, 105.0, 100.0, 105.0]
    result = None
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = rsi.compute("TEST", ctx)

    # Equal gains and losses should produce RSI around 50
    # Note: Due to Wilder's smoothing, might not be exactly 50
    assert result is not None
    assert 35.0 < result < 65.0  # Wider range to account for smoothing


def test_rsi_smoothing_wilders():
    """Test RSI uses Wilder's smoothing method."""
    rsi = RSI(period=3)
    ctx = Context()

    # First set of bars for initialization
    result1 = None
    for val in [100.0, 110.0, 105.0, 115.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result1 = rsi.compute("TEST", ctx)

    # Add another bar with gain
    bar = create_bar("TEST", 120.0)
    ctx._add_bar_to_history(bar)
    result2 = rsi.compute("TEST", ctx)

    # Result should be smoothed using Wilder's method
    # New avg = (old_avg * (period - 1) + new_value) / period
    assert result2 is not None
    assert result1 is not None
    assert result2 != result1  # Should change
    assert result2 > result1  # Should increase with gain


def test_rsi_vs_manual_calculation():
    """Test RSI matches manual calculation."""
    rsi = RSI(period=2)
    ctx = Context()

    # Simple sequence: 100, 105 (+5), 110 (+5)
    # First avg gain = (5 + 5) / 2 = 5.0
    # First avg loss = 0
    # RS = 5.0 / (very small number to avoid div by 0)
    # RSI should be very close to 100
    result = None
    for val in [100.0, 105.0, 110.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = rsi.compute("TEST", ctx)

    assert result is not None
    assert result > 90.0  # All gains, no losses


def test_rsi_different_fields():
    """Test RSI computes on close price only."""
    rsi = RSI(period=3)
    ctx = Context()

    # Add bars with different OHLC values
    # RSI only uses close price
    result = None
    for i in range(4):
        bar = Bar(
            ts=datetime.now(timezone.utc),
            symbol="TEST",
            open=Decimal(str(90 + i * 5)),  # 90, 95, 100, 105
            high=Decimal(str(110 + i)),
            low=Decimal(str(95 + i)),
            close=Decimal(str(100 + i * 10)),  # 100, 110, 120, 130 (strong uptrend)
            volume=1000,
        )
        ctx._add_bar_to_history(bar)
        result = rsi.compute("TEST", ctx)

    # Should calculate based on close prices (strong gains)
    assert result is not None
    assert result > 70.0  # Should be overbought


def test_rsi_multiple_symbols():
    """Test RSI handles multiple symbols independently."""
    rsi = RSI(period=3)
    ctx = Context()

    # AAPL: Strong uptrend
    result_aapl = None
    for val in [100.0, 110.0, 120.0, 130.0]:
        bar = create_bar("AAPL", val)
        ctx._add_bar_to_history(bar)
        result_aapl = rsi.compute("AAPL", ctx)

    # MSFT: Strong downtrend
    result_msft = None
    for val in [200.0, 190.0, 180.0, 170.0]:
        bar = create_bar("MSFT", val)
        ctx._add_bar_to_history(bar)
        result_msft = rsi.compute("MSFT", ctx)

    # AAPL should be overbought, MSFT oversold
    assert result_aapl is not None
    assert result_msft is not None
    assert result_aapl > 70.0
    assert result_msft < 30.0


def test_rsi_reset():
    """Test RSI reset clears state."""
    rsi = RSI(period=3)
    ctx = Context()

    # Add 4 bars
    result1 = None
    for val in [100.0, 110.0, 105.0, 115.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result1 = rsi.compute("TEST", ctx)

    assert result1 is not None

    # Reset
    rsi.reset("TEST")

    # Should need 4 bars again (period + 1)
    bar = create_bar("TEST", 120.0)
    ctx._add_bar_to_history(bar)
    result2 = rsi.compute("TEST", ctx)
    assert result2 is None


def test_rsi_no_movement():
    """Test RSI with no price movement."""
    rsi = RSI(period=3)
    ctx = Context()

    # All same price - no gains or losses
    result = None
    for _ in range(5):
        bar = create_bar("TEST", 100.0)
        ctx._add_bar_to_history(bar)
        result = rsi.compute("TEST", ctx)

    # With no movement, RSI behavior depends on implementation
    # Some implementations return 100 (no losses), some return 50 (neutral)
    # Our implementation returns 100 when there are no losses
    assert result is not None
    assert result == pytest.approx(100.0, rel=1e-5)
