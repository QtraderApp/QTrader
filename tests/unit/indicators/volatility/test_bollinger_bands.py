"""Unit tests for Bollinger Bands indicator."""

import math
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.api import BollingerBandsIndicator, Context
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


def test_bb_initialization():
    """Test Bollinger Bands initializes correctly."""
    bb = BollingerBandsIndicator(period=20, num_std=2.0)
    assert bb.period == 20
    assert bb.num_std == 2.0
    assert bb.field == "close"


def test_bb_insufficient_data():
    """Test BB returns None until period bars received."""
    bb = BollingerBandsIndicator(period=3, num_std=2.0)
    ctx = Context()

    # Add 2 bars (need 3 for BB(3))
    result = None
    for val in [100.0, 110.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = bb.compute("TEST", ctx)

    assert result is None


def test_bb_basic_calculation():
    """Test BB calculates bands correctly."""
    bb = BollingerBandsIndicator(period=3, num_std=2.0)
    ctx = Context()

    # Add bars: 100, 110, 120
    result = None
    for val in [100.0, 110.0, 120.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = bb.compute("TEST", ctx)

    assert result is not None
    upper, middle, lower = result

    # Middle = SMA = (100 + 110 + 120) / 3 = 110
    assert middle == pytest.approx(110.0, rel=1e-6)

    # Std dev = sqrt(((100-110)^2 + (110-110)^2 + (120-110)^2) / 3)
    #         = sqrt((100 + 0 + 100) / 3)
    #         = sqrt(200/3)
    #         = sqrt(66.666...)
    #         = 8.1649...
    std = math.sqrt(((100 - 110) ** 2 + (110 - 110) ** 2 + (120 - 110) ** 2) / 3)

    # Upper = middle + 2*std = 110 + 2*8.1649 = 126.33
    # Lower = middle - 2*std = 110 - 2*8.1649 = 93.67
    assert upper == pytest.approx(110 + 2 * std, rel=1e-5)
    assert lower == pytest.approx(110 - 2 * std, rel=1e-5)


def test_bb_rolling_window():
    """Test BB uses rolling window correctly."""
    bb = BollingerBandsIndicator(period=3, num_std=2.0)
    ctx = Context()

    # Add 4 bars: 100, 110, 120, 130
    result = None
    for val in [100.0, 110.0, 120.0, 130.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = bb.compute("TEST", ctx)

    # Should use last 3: 110, 120, 130
    upper, middle, lower = result

    # Middle = (110 + 120 + 130) / 3 = 120
    assert middle == pytest.approx(120.0, rel=1e-6)

    # Std dev with last 3 values
    std = math.sqrt(((110 - 120) ** 2 + (120 - 120) ** 2 + (130 - 120) ** 2) / 3)
    assert upper == pytest.approx(120 + 2 * std, rel=1e-5)
    assert lower == pytest.approx(120 - 2 * std, rel=1e-5)


def test_bb_different_num_std():
    """Test BB with different number of standard deviations."""
    bb1 = BollingerBandsIndicator(period=3, num_std=1.0)
    bb2 = BollingerBandsIndicator(period=3, num_std=3.0)
    ctx = Context()

    # Add bars: 100, 110, 120
    result1 = None
    result2 = None
    for val in [100.0, 110.0, 120.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result1 = bb1.compute("TEST", ctx)
        result2 = bb2.compute("TEST", ctx)

    upper1, middle1, lower1 = result1
    upper2, middle2, lower2 = result2

    # Both should have same middle
    assert middle1 == pytest.approx(middle2, rel=1e-6)

    # BB with 3 std should have wider bands than 1 std
    assert (upper2 - middle2) > (upper1 - middle1)
    assert (middle2 - lower2) > (middle1 - lower1)


def test_bb_zero_volatility():
    """Test BB when volatility is zero (all same values)."""
    bb = BollingerBandsIndicator(period=3, num_std=2.0)
    ctx = Context()

    # Add bars with same value: 100, 100, 100
    result = None
    for val in [100.0, 100.0, 100.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = bb.compute("TEST", ctx)

    upper, middle, lower = result

    # Middle = 100, std = 0
    # All bands should be equal (no volatility)
    assert middle == pytest.approx(100.0, rel=1e-6)
    assert upper == pytest.approx(100.0, rel=1e-6)
    assert lower == pytest.approx(100.0, rel=1e-6)


def test_bb_high_volatility():
    """Test BB with highly volatile prices."""
    bb = BollingerBandsIndicator(period=3, num_std=2.0)
    ctx = Context()

    # Add bars: 50, 100, 150 (high variance)
    result = None
    for val in [50.0, 100.0, 150.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = bb.compute("TEST", ctx)

    upper, middle, lower = result

    # Middle = (50 + 100 + 150) / 3 = 100
    assert middle == pytest.approx(100.0, rel=1e-6)

    # Bands should be wide due to high volatility
    std = math.sqrt(((50 - 100) ** 2 + (100 - 100) ** 2 + (150 - 100) ** 2) / 3)
    band_width = upper - lower
    expected_width = 4 * std  # 2*std on each side

    assert band_width == pytest.approx(expected_width, rel=1e-5)
    assert band_width > 100.0  # Should be wide


def test_bb_different_fields():
    """Test BB can compute on different bar fields."""
    bb_close = BollingerBandsIndicator(period=3, num_std=2.0, field="close")
    bb_high = BollingerBandsIndicator(period=3, num_std=2.0, field="high")
    ctx = Context()

    # Add bars with different high/close values
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
        result_close = bb_close.compute("TEST", ctx)
        result_high = bb_high.compute("TEST", ctx)

    _, middle_close, _ = result_close
    _, middle_high, _ = result_high

    # Close: (105 + 106 + 107) / 3 = 106
    assert middle_close == pytest.approx(106.0, rel=1e-6)

    # High: (110 + 111 + 112) / 3 = 111
    assert middle_high == pytest.approx(111.0, rel=1e-6)


def test_bb_multiple_symbols():
    """Test BB handles multiple symbols independently."""
    bb = BollingerBandsIndicator(period=3, num_std=2.0)
    ctx = Context()

    # AAPL: 100, 110, 120
    result_aapl = None
    for val in [100.0, 110.0, 120.0]:
        bar = create_bar("AAPL", val)
        ctx._add_bar_to_history(bar)
        result_aapl = bb.compute("AAPL", ctx)

    # MSFT: 200, 210, 220
    result_msft = None
    for val in [200.0, 210.0, 220.0]:
        bar = create_bar("MSFT", val)
        ctx._add_bar_to_history(bar)
        result_msft = bb.compute("MSFT", ctx)

    _, middle_aapl, _ = result_aapl
    _, middle_msft, _ = result_msft

    assert middle_aapl == pytest.approx(110.0, rel=1e-6)
    assert middle_msft == pytest.approx(210.0, rel=1e-6)


def test_bb_reset():
    """Test BB reset clears state."""
    bb = BollingerBandsIndicator(period=3, num_std=2.0)
    ctx = Context()

    # Add 3 bars
    result1 = None
    for val in [100.0, 110.0, 120.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result1 = bb.compute("TEST", ctx)

    assert result1 is not None

    # Reset
    bb.reset("TEST")

    # Should need 3 bars again
    bar = create_bar("TEST", 130.0)
    ctx._add_bar_to_history(bar)
    result2 = bb.compute("TEST", ctx)
    assert result2 is None
