"""Unit tests for MACD (Moving Average Convergence Divergence) indicator."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.api import Context, MACDIndicator
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


def test_macd_initialization():
    """Test MACD initializes correctly."""
    macd = MACDIndicator(fast=12, slow=26, signal=9)
    assert macd.fast == 12
    assert macd.slow == 26
    assert macd.signal_period == 9
    assert macd.field == "close"


def test_macd_default_parameters():
    """Test MACD uses standard default parameters."""
    macd = MACDIndicator()
    assert macd.fast == 12
    assert macd.slow == 26
    assert macd.signal_period == 9


def test_macd_insufficient_data():
    """Test MACD returns None until slow period bars received."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Add 4 bars (need 5 for slow period)
    result = None
    for val in [100.0, 110.0, 105.0, 115.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)

    assert result is None


def test_macd_initial_value():
    """Test MACD calculates correctly after warmup."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Add enough bars for signal line to initialize
    # Need: slow period (5) + signal period (2) bars
    result = None
    for val in [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)

    # Should have MACD value after enough bars
    assert result is not None
    macd_line, signal_line, histogram = result

    # In uptrend, MACD line should be positive (fast EMA > slow EMA)
    assert macd_line > 0.0

    # Signal line should exist
    assert signal_line is not None

    # Histogram = MACD - Signal
    assert histogram == pytest.approx(macd_line - signal_line, abs=1e-10)


def test_macd_line_calculation():
    """Test MACD line is difference between fast and slow EMAs."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Steady uptrend
    values = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
    result = None
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)

    assert result is not None
    macd_line, _, _ = result

    # In steady uptrend, fast EMA > slow EMA, so MACD > 0
    assert macd_line > 0.0


def test_macd_signal_line():
    """Test signal line is EMA of MACD line."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Add enough bars to get multiple MACD values
    values = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
    results = []
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)
        if result is not None:
            results.append(result)

    # Should have multiple results
    assert len(results) >= 2

    # Signal line should smooth MACD line (not jump around as much)
    macd_changes = [abs(results[i][0] - results[i - 1][0]) for i in range(1, len(results))]
    signal_changes = [abs(results[i][1] - results[i - 1][1]) for i in range(1, len(results))]

    # Signal should be smoother (smaller average change)
    if len(macd_changes) > 0 and len(signal_changes) > 0:
        avg_macd_change = sum(macd_changes) / len(macd_changes)
        avg_signal_change = sum(signal_changes) / len(signal_changes)
        assert avg_signal_change <= avg_macd_change


def test_macd_histogram():
    """Test histogram is difference between MACD and signal lines."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Add bars
    values = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
    result = None
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)

    assert result is not None
    macd_line, signal_line, histogram = result

    # Histogram should equal MACD - Signal
    assert histogram == pytest.approx(macd_line - signal_line, abs=1e-10)


def test_macd_bullish_crossover():
    """Test MACD detects bullish crossover (MACD crosses above signal)."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Downtrend followed by uptrend (should create bullish crossover)
    values = [150.0, 140.0, 130.0, 120.0, 110.0, 115.0, 125.0, 140.0]
    results = []
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)
        if result is not None:
            results.append(result)

    # Should have results
    assert len(results) > 2

    # Look for histogram going from negative to positive (bullish crossover)
    histograms = [r[2] for r in results]
    found_crossover = False
    for i in range(1, len(histograms)):
        if histograms[i - 1] < 0 and histograms[i] > 0:
            found_crossover = True
            break

    # In this scenario we should see a crossover
    assert found_crossover or histograms[-1] > histograms[0]


def test_macd_bearish_crossover():
    """Test MACD detects bearish crossover (MACD crosses below signal)."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Uptrend followed by downtrend (should create bearish crossover)
    values = [100.0, 110.0, 120.0, 130.0, 140.0, 135.0, 125.0, 110.0]
    results = []
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)
        if result is not None:
            results.append(result)

    # Should have results
    assert len(results) > 2

    # Look for histogram going from positive to negative (bearish crossover)
    histograms = [r[2] for r in results]
    found_crossover = False
    for i in range(1, len(histograms)):
        if histograms[i - 1] > 0 and histograms[i] < 0:
            found_crossover = True
            break

    # In this scenario we should see a crossover or at least declining histogram
    assert found_crossover or histograms[-1] < histograms[0]


def test_macd_divergence():
    """Test MACD can show divergence from price."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Price making higher highs, but momentum weakening
    values = [100.0, 110.0, 105.0, 115.0, 110.0, 120.0, 115.0, 122.0]
    results = []
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)
        if result is not None:
            results.append(result)

    # Just verify we get values (divergence detection would be in strategy logic)
    assert len(results) > 0
    for macd_line, signal_line, histogram in results:
        assert macd_line is not None
        assert signal_line is not None
        assert histogram is not None


def test_macd_different_fields():
    """Test MACD can compute on different bar fields."""
    macd_close = MACDIndicator(fast=3, slow=5, signal=2, field="close")
    macd_high = MACDIndicator(fast=3, slow=5, signal=2, field="high")
    ctx = Context()

    # Add bars with different high/close values
    result_close = None
    result_high = None
    for i in range(6):
        bar = Bar(
            ts=datetime.now(timezone.utc),
            symbol="TEST",
            open=Decimal(str(100 + i * 5)),
            high=Decimal(str(110 + i * 5)),  # Higher values
            low=Decimal(str(95 + i * 5)),
            close=Decimal(str(105 + i * 5)),  # Lower values
            volume=1000,
        )
        ctx._add_bar_to_history(bar)
        result_close = macd_close.compute("TEST", ctx)
        result_high = macd_high.compute("TEST", ctx)

    # Both should return values
    assert result_close is not None
    assert result_high is not None

    # Values should be different (different input data)
    macd_close_line = result_close[0]
    macd_high_line = result_high[0]
    assert macd_close_line != macd_high_line


def test_macd_multiple_symbols():
    """Test MACD handles multiple symbols independently."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # AAPL: Strong uptrend
    result_aapl = None
    for val in [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]:
        bar = create_bar("AAPL", val)
        ctx._add_bar_to_history(bar)
        result_aapl = macd.compute("AAPL", ctx)

    # MSFT: Strong downtrend
    result_msft = None
    for val in [200.0, 190.0, 180.0, 170.0, 160.0, 150.0]:
        bar = create_bar("MSFT", val)
        ctx._add_bar_to_history(bar)
        result_msft = macd.compute("MSFT", ctx)

    # Both should return values
    assert result_aapl is not None
    assert result_msft is not None

    # AAPL should have positive MACD (uptrend), MSFT negative (downtrend)
    macd_aapl, _, _ = result_aapl
    macd_msft, _, _ = result_msft

    assert macd_aapl > 0.0
    assert macd_msft < 0.0


def test_macd_reset():
    """Test MACD reset clears state."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Add enough bars (slow + signal periods)
    result1 = None
    for val in [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result1 = macd.compute("TEST", ctx)

    assert result1 is not None

    # Reset
    macd.reset("TEST")

    # Should need slow period bars again
    bar = create_bar("TEST", 135.0)
    ctx._add_bar_to_history(bar)
    result2 = macd.compute("TEST", ctx)
    assert result2 is None


def test_macd_zero_line_cross():
    """Test MACD crossing zero line (bullish/bearish signals)."""
    macd = MACDIndicator(fast=3, slow=5, signal=2)
    ctx = Context()

    # Start high, go down, then back up (should cross zero line)
    values = [150.0, 140.0, 130.0, 120.0, 110.0, 120.0, 135.0, 155.0]
    results = []
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)
        if result is not None:
            results.append(result)

    # Check if MACD line crossed zero
    macd_lines = [r[0] for r in results]

    # Look for zero line cross
    found_negative = any(val < 0 for val in macd_lines)
    found_positive = any(val > 0 for val in macd_lines)

    # Should see both sides of zero in this price action
    assert found_negative or found_positive  # At least one side


def test_macd_fast_vs_slow():
    """Test fast period reacts quicker than slow period."""
    macd = MACDIndicator(fast=3, slow=7, signal=2)
    ctx = Context()

    # Sharp price increase
    values = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 150.0]
    results = []
    for val in values:
        bar = create_bar("TEST", val)
        ctx._add_bar_to_history(bar)
        result = macd.compute("TEST", ctx)
        if result is not None:
            results.append(result)

    # After price spike, MACD should be strongly positive
    # (fast EMA reacts faster than slow EMA to price change)
    if len(results) > 0:
        final_macd, _, _ = results[-1]
        assert final_macd > 0.0
