"""
Integration test: MACD histogram strategy with helper functions.

Tests:
- MACD computes correctly (macd_line, signal_line, histogram)
- Histogram zero-cross detection
- Helper functions for histogram analysis
- Full integration: MACD → histogram helpers → signals
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from qtrader.api import Context
from qtrader.indicators import histogram_flipped_negative, histogram_flipped_positive
from qtrader.models.bar import Bar


@pytest.fixture
def macd_test_bars():
    """
    Create bars with trend changes to generate MACD histogram crosses.

    Pattern:
    - Downtrend → histogram negative
    - Reversal → histogram crosses zero to positive
    - Uptrend → histogram positive
    - Reversal → histogram crosses zero to negative
    """
    base_ts = datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc)

    prices = [
        # Initial stable
        100.0,
        100.5,
        101.0,
        # Downtrend (histogram should go negative)
        100.0,
        99.0,
        98.0,
        97.0,
        96.0,
        95.0,
        94.0,
        93.0,
        92.0,
        91.0,
        # Reversal point
        90.5,
        90.0,
        90.5,
        # Uptrend (histogram should cross to positive)
        91.0,
        92.0,
        93.0,
        94.0,
        95.0,
        96.0,
        97.0,
        98.0,
        99.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        # Peak
        105.5,
        106.0,
        105.5,
        # Downtrend again (histogram should cross to negative)
        105.0,
        104.0,
        103.0,
        102.0,
        101.0,
        100.0,
        99.0,
        98.0,
    ]

    bars = []
    for i, price in enumerate(prices):
        bar = Bar(
            ts=base_ts + timedelta(days=i),
            symbol="TEST",
            open=Decimal(str(price)),
            high=Decimal(str(price + 0.5)),
            low=Decimal(str(price - 0.5)),
            close=Decimal(str(price)),
            volume=1_000_000,
        )
        bars.append(bar)

    return bars


def test_macd_structure(macd_test_bars):
    """Test that MACD returns correct structure."""
    ctx = Context()

    # Add bars and compute incrementally
    for bar in macd_test_bars[:35]:
        ctx._add_bar_to_history(bar)
        ctx.ind.macd("TEST", fast=5, slow=10, signal=3)
        ctx._save_indicator_state()  # Simulate engine behavior

    macd = ctx.ind.macd("TEST", fast=5, slow=10, signal=3)

    assert macd is not None
    assert hasattr(macd, "macd_line")
    assert hasattr(macd, "signal_line")
    assert hasattr(macd, "histogram")

    # All values should be floats
    assert isinstance(macd.macd_line, float)
    assert isinstance(macd.signal_line, float)
    assert isinstance(macd.histogram, float)


def test_macd_histogram_calculation(macd_test_bars):
    """Test that histogram equals macd_line - signal_line."""
    ctx = Context()

    for bar in macd_test_bars[:35]:
        ctx._add_bar_to_history(bar)
        ctx.ind.macd("TEST", fast=5, slow=10, signal=3)
        ctx._save_indicator_state()  # Simulate engine behavior

    macd = ctx.ind.macd("TEST", fast=5, slow=10, signal=3)

    assert macd is not None

    # Histogram should equal macd_line - signal_line
    expected_histogram = macd.macd_line - macd.signal_line
    assert abs(macd.histogram - expected_histogram) < 0.0001, (
        f"Histogram mismatch: {macd.histogram} != {expected_histogram}"
    )


def test_macd_histogram_zero_crosses(macd_test_bars):
    """Test detection of histogram crossing zero."""
    ctx = Context()

    histogram_values = []
    zero_crosses = []

    for bar in macd_test_bars:
        ctx._add_bar_to_history(bar)
        macd = ctx.ind.macd("TEST", fast=5, slow=10, signal=3)

        if macd is not None:
            histogram_values.append(
                {
                    "ts": bar.ts,
                    "histogram": macd.histogram,
                }
            )

            # Detect zero crosses
            if len(histogram_values) >= 2:
                prev_hist = histogram_values[-2]["histogram"]
                curr_hist = histogram_values[-1]["histogram"]

                # Cross from negative to positive
                if prev_hist <= 0 < curr_hist:
                    zero_crosses.append(
                        {
                            "ts": bar.ts,
                            "type": "bullish",
                            "prev": prev_hist,
                            "curr": curr_hist,
                        }
                    )

                # Cross from positive to negative
                elif prev_hist >= 0 > curr_hist:
                    zero_crosses.append(
                        {
                            "ts": bar.ts,
                            "type": "bearish",
                            "prev": prev_hist,
                            "curr": curr_hist,
                        }
                    )

        ctx._save_indicator_state()  # Simulate engine behavior

    # Should detect zero crosses in both directions
    bullish_crosses = [c for c in zero_crosses if c["type"] == "bullish"]
    bearish_crosses = [c for c in zero_crosses if c["type"] == "bearish"]

    assert len(bullish_crosses) >= 1, "Should detect bullish zero cross"
    assert len(bearish_crosses) >= 1, "Should detect bearish zero cross"


def test_histogram_flip_helpers():
    """Test histogram flip helper functions."""
    ctx = Context()

    # Create specific price pattern
    base_ts = datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc)

    # Downtrend then uptrend to create histogram flip
    prices = [100.0] + [100.0 - i for i in range(1, 12)]  # Down
    prices += [prices[-1] + i for i in range(1, 12)]  # Up

    bars = []
    for i, price in enumerate(prices):
        bar = Bar(
            ts=base_ts + timedelta(days=i),
            symbol="TEST",
            open=Decimal(str(price)),
            high=Decimal(str(price + 0.5)),
            low=Decimal(str(price - 0.5)),
            close=Decimal(str(price)),
            volume=1_000_000,
        )
        bars.append(bar)
        ctx._add_bar_to_history(bar)
        ctx.ind.macd("TEST", fast=4, slow=8, signal=3)
        ctx._save_indicator_state()  # Simulate engine behavior

    # Process bars and detect flips
    prev_histogram = None
    flips = []

    for bar in bars:
        macd = ctx.ind.macd("TEST", fast=4, slow=8, signal=3)

        if macd is not None and prev_histogram is not None:
            # Check for histogram flips
            if histogram_flipped_positive(macd.histogram, prev_histogram):
                flips.append({"type": "positive", "histogram": macd.histogram})

            elif histogram_flipped_negative(macd.histogram, prev_histogram):
                flips.append({"type": "negative", "histogram": macd.histogram})

        if macd is not None:
            prev_histogram = macd.histogram

    # Should detect at least one flip
    assert len(flips) >= 1, "Should detect histogram flips"


def test_macd_strategy_workflow(macd_test_bars):
    """
    Test complete MACD histogram zero-cross strategy.

    Strategy logic:
    - BUY when histogram crosses above zero (bullish momentum)
    - SELL when histogram crosses below zero (bearish momentum)
    """
    ctx = Context()

    signals = []

    for bar in macd_test_bars:
        ctx._add_bar_to_history(bar)
        macd = ctx.ind.macd("TEST", fast=5, slow=10, signal=3)

        if macd is None:
            continue

        # Track histogram for zero-cross detection
        ctx._track_indicator("TEST", "macd_histogram", macd.histogram)

        # Generate signals on zero crosses
        if ctx.crossed_above_threshold("TEST", "macd_histogram", 0.0):
            signals.append(
                {
                    "ts": bar.ts,
                    "type": "BUY",
                    "reason": "MACD histogram crossed above zero",
                    "histogram": macd.histogram,
                    "macd_line": macd.macd_line,
                    "signal_line": macd.signal_line,
                    "price": float(bar.close),
                }
            )

        elif ctx.crossed_below_threshold("TEST", "macd_histogram", 0.0):
            signals.append(
                {
                    "ts": bar.ts,
                    "type": "SELL",
                    "reason": "MACD histogram crossed below zero",
                    "histogram": macd.histogram,
                    "macd_line": macd.macd_line,
                    "signal_line": macd.signal_line,
                    "price": float(bar.close),
                }
            )

        ctx._save_indicator_state()  # Simulate engine behavior

    # Should generate signals
    assert len(signals) >= 2, f"Expected at least 2 signals, got {len(signals)}"

    # Verify signal data
    for signal in signals:
        assert signal["type"] in ("BUY", "SELL")
        assert signal["price"] > 0

        # Verify histogram sign matches signal type
        if signal["type"] == "BUY":
            assert signal["histogram"] > 0, "BUY signal should have positive histogram"
        else:
            assert signal["histogram"] < 0, "SELL signal should have negative histogram"


def test_macd_different_parameters(macd_test_bars):
    """Test MACD with different parameter sets."""
    ctx = Context()

    # Add bars and compute incrementally
    for bar in macd_test_bars[:35]:
        ctx._add_bar_to_history(bar)
        ctx.ind.macd("TEST", fast=5, slow=10, signal=3)
        ctx.ind.macd("TEST", fast=12, slow=26, signal=9)
        ctx._save_indicator_state()  # Simulate engine behavior

    # Compute MACD with different parameters
    macd_fast = ctx.ind.macd("TEST", fast=5, slow=10, signal=3)
    macd_standard = ctx.ind.macd("TEST", fast=12, slow=26, signal=9)

    assert macd_fast is not None
    assert macd_standard is not None

    # Different parameters should give different results
    # (though both should be valid)
    assert macd_fast.macd_line != macd_standard.macd_line or macd_fast.signal_line != macd_standard.signal_line


def test_macd_with_insufficient_data():
    """Test MACD returns None with insufficient data."""
    ctx = Context()

    # Add only 5 bars (need more for MACD)
    for i in range(5):
        bar = Bar(
            ts=datetime(2023, 1, i + 1, tzinfo=timezone.utc),
            symbol="TEST",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=1_000_000,
        )
        ctx._add_bar_to_history(bar)
        ctx.ind.macd("TEST", fast=12, slow=26, signal=9)
        ctx._save_indicator_state()  # Simulate engine behavior

    macd = ctx.ind.macd("TEST", fast=12, slow=26, signal=9)
    assert macd is None


def test_macd_caching():
    """Test that MACD values compute correctly."""
    ctx = Context()

    # Add bars
    for i in range(35):
        bar = Bar(
            ts=datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
            symbol="TEST",
            open=Decimal("100") + Decimal(i * 0.5),
            high=Decimal("102") + Decimal(i * 0.5),
            low=Decimal("98") + Decimal(i * 0.5),
            close=Decimal("100") + Decimal(i * 0.5),
            volume=1_000_000,
        )
        ctx._add_bar_to_history(bar)
        _ = ctx.ind.macd("TEST", fast=5, slow=10, signal=3)
        ctx._save_indicator_state()  # Simulate engine behavior

    # Verify MACD computes
    macd = ctx.ind.macd("TEST", fast=5, slow=10, signal=3)

    assert macd is not None
    assert hasattr(macd, "macd_line")
    assert hasattr(macd, "signal_line")
    assert hasattr(macd, "histogram")


def test_macd_histogram_trend_analysis(macd_test_bars):
    """Test analyzing histogram trend (increasing/decreasing)."""
    ctx = Context()

    histogram_history = []

    for bar in macd_test_bars[:35]:
        ctx._add_bar_to_history(bar)
        macd = ctx.ind.macd("TEST", fast=5, slow=10, signal=3)

        if macd is not None:
            histogram_history.append(macd.histogram)

        ctx._save_indicator_state()  # Simulate engine behavior

    # Should have enough data
    assert len(histogram_history) > 10

    # Histogram should change over time (not flat)
    unique_values = len(set(histogram_history))
    assert unique_values > 5, "Histogram should vary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
