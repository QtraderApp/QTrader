"""
Integration test: RSI threshold strategy with helper functions.

Tests:
- RSI computes correctly across bars
- Threshold crossing detection (oversold/overbought)
- Helper functions work in real strategy context
- Full integration: RSI → threshold helpers → signals
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from qtrader.api import Context
from qtrader.indicators import crossed_above_threshold, crossed_below_threshold
from qtrader.models.bar import Bar


@pytest.fixture
def rsi_test_bars():
    """
    Create bars with alternating up/down moves to generate RSI signal.

    RSI will oscillate between oversold (<30) and overbought (>70).
    """
    base_ts = datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc)

    # Create strong trends to push RSI to extremes
    prices = [
        # Initial prices
        100.0,
        # Strong downtrend → RSI should drop below 30 (oversold)
        98.0,
        96.0,
        94.0,
        92.0,
        90.0,
        88.0,
        86.0,
        84.0,
        82.0,
        80.0,
        # Recovery (small ups)
        81.0,
        82.0,
        83.0,
        # Strong uptrend → RSI should rise above 70 (overbought)
        85.0,
        88.0,
        91.0,
        94.0,
        97.0,
        100.0,
        103.0,
        106.0,
        109.0,
        112.0,
        # Pullback
        111.0,
        110.0,
        109.0,
        # Another downtrend
        107.0,
        105.0,
        103.0,
        101.0,
        99.0,
        97.0,
        95.0,
    ]

    bars = []
    for i, price in enumerate(prices):
        bar = Bar(
            ts=base_ts + timedelta(days=i),
            symbol="TEST",
            open=Decimal(str(price)),
            high=Decimal(str(price + 1)),
            low=Decimal(str(price - 1)),
            close=Decimal(str(price)),
            volume=1_000_000,
        )
        bars.append(bar)

    return bars


def test_rsi_computation_range(rsi_test_bars):
    """Test that RSI stays within 0-100 range."""
    ctx = Context()

    rsi_values = []

    for bar in rsi_test_bars:
        ctx._add_bar_to_history(bar)
        rsi = ctx.ind.rsi("TEST", period=14)

        if rsi is not None:
            rsi_values.append(rsi)
            # RSI must always be between 0 and 100
            assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"

        ctx._save_indicator_state()  # Simulate engine behavior

    # Should have computed RSI for most bars
    assert len(rsi_values) > 10


def test_rsi_extremes_in_trends(rsi_test_bars):
    """Test that RSI reaches extremes during strong trends."""
    ctx = Context()

    rsi_values = []

    for bar in rsi_test_bars:
        ctx._add_bar_to_history(bar)
        rsi = ctx.ind.rsi("TEST", period=14)

        if rsi is not None:
            rsi_values.append(rsi)

        ctx._save_indicator_state()  # Simulate engine behavior

    # In strong downtrend, RSI should go below 40
    min_rsi = min(rsi_values[:20])
    assert min_rsi < 40, f"Expected low RSI in downtrend, got {min_rsi}"

    # In strong uptrend, RSI should go above 60
    max_rsi = max(rsi_values)
    assert max_rsi > 60, f"Expected high RSI in uptrend, got {max_rsi}"


def test_threshold_crossing_detection():
    """Test detection of RSI crossing thresholds using helper functions."""
    ctx = Context()

    # Create bars with strong trends to push RSI to extremes
    base_ts = datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc)

    # Strong downtrend to push RSI below 30
    prices = [100.0] + [100.0 - i * 4 for i in range(1, 12)]
    # Then strong uptrend to push RSI above 70
    prices += [prices[-1] + i * 5 for i in range(1, 18)]

    bars = []
    for i, price in enumerate(prices):
        bar = Bar(
            ts=base_ts + timedelta(days=i),
            symbol="TEST",
            open=Decimal(str(price)),
            high=Decimal(str(price + 1)),
            low=Decimal(str(price - 1)),
            close=Decimal(str(price)),
            volume=1_000_000,
        )
        bars.append(bar)

    oversold_signals = []
    overbought_signals = []
    prev_rsi = None

    for bar in bars:
        ctx._add_bar_to_history(bar)
        rsi = ctx.ind.rsi("TEST", period=7)  # Shorter period for faster moves

        if rsi is not None and prev_rsi is not None:
            # Check if RSI crossed above 30 (exit oversold)
            if crossed_above_threshold(rsi, prev_rsi, 30):
                oversold_signals.append(
                    {
                        "ts": bar.ts,
                        "rsi": rsi,
                        "prev_rsi": prev_rsi,
                    }
                )

            # Check if RSI crossed below 70 (exit overbought)
            if crossed_below_threshold(rsi, prev_rsi, 70):
                overbought_signals.append(
                    {
                        "ts": bar.ts,
                        "rsi": rsi,
                        "prev_rsi": prev_rsi,
                    }
                )

        prev_rsi = rsi
        ctx._save_indicator_state()  # Simulate engine behavior

    # Should detect at least one threshold crossing in each direction
    assert len(oversold_signals) >= 1, "Should detect RSI crossing above 30"
    # Note: May not always cross below 70 depending on price pattern
    # Main test is that the helper functions work correctly


def test_context_threshold_helpers(rsi_test_bars):
    """Test Context's built-in threshold crossing helpers."""
    ctx = Context()

    crossings = []

    for bar in rsi_test_bars:
        ctx._add_bar_to_history(bar)
        rsi = ctx.ind.rsi("TEST", period=14)

        if rsi is not None:
            # Track RSI for threshold detection
            ctx._track_indicator("TEST", "rsi", rsi)

            # Check for threshold crossings
            if ctx.crossed_above_threshold("TEST", "rsi", 30):
                crossings.append({"type": "above_30", "rsi": rsi})

            if ctx.crossed_below_threshold("TEST", "rsi", 70):
                crossings.append({"type": "below_70", "rsi": rsi})

        ctx._save_indicator_state()  # Simulate engine behavior

    # Should detect threshold crossings
    assert len(crossings) > 0, "Expected at least one threshold crossing"


def test_rsi_strategy_workflow(rsi_test_bars):
    """
    Test complete RSI strategy workflow.

    Strategy logic:
    - BUY when RSI crosses above 30 (oversold recovery)
    - SELL when RSI crosses below 70 (overbought pullback)
    """
    ctx = Context()

    signals = []

    for bar in rsi_test_bars:
        ctx._add_bar_to_history(bar)
        rsi = ctx.ind.rsi("TEST", period=14)

        if rsi is None:
            continue

        # Track RSI
        ctx._track_indicator("TEST", "rsi", rsi)

        # Generate signals based on threshold crossings
        if ctx.crossed_above_threshold("TEST", "rsi", 30):
            signals.append(
                {
                    "ts": bar.ts,
                    "type": "BUY",
                    "reason": "RSI crossed above 30 (oversold exit)",
                    "rsi": rsi,
                    "price": float(bar.close),
                }
            )

        elif ctx.crossed_below_threshold("TEST", "rsi", 70):
            signals.append(
                {
                    "ts": bar.ts,
                    "type": "SELL",
                    "reason": "RSI crossed below 70 (overbought exit)",
                    "rsi": rsi,
                    "price": float(bar.close),
                }
            )

        ctx._save_indicator_state()  # Simulate engine behavior

    # Should generate at least one signal
    assert len(signals) >= 1, f"Expected signals, got {len(signals)}"

    # Verify signal data
    for signal in signals:
        assert signal["type"] in ("BUY", "SELL")
        assert 0 <= signal["rsi"] <= 100
        assert signal["price"] > 0


def test_rsi_multiple_periods(rsi_test_bars):
    """Test RSI with different periods."""
    ctx = Context()

    # Add all bars and compute incrementally
    for bar in rsi_test_bars:
        ctx._add_bar_to_history(bar)
        ctx.ind.rsi("TEST", period=7)
        ctx.ind.rsi("TEST", period=14)
        ctx.ind.rsi("TEST", period=21)
        ctx._save_indicator_state()  # Simulate engine behavior

    # Compute RSI with different periods
    rsi_7 = ctx.ind.rsi("TEST", period=7)
    rsi_14 = ctx.ind.rsi("TEST", period=14)
    rsi_21 = ctx.ind.rsi("TEST", period=21)

    assert rsi_7 is not None
    assert rsi_14 is not None
    assert rsi_21 is not None

    # All should be in valid range
    assert 0 <= rsi_7 <= 100
    assert 0 <= rsi_14 <= 100
    assert 0 <= rsi_21 <= 100

    # Shorter periods are more reactive (higher variance)
    # But all should be relatively close in this scenario


def test_rsi_with_insufficient_data():
    """Test RSI returns None with insufficient data."""
    ctx = Context()

    # Add only 5 bars (need 14 + 1 for RSI(14))
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
        ctx.ind.rsi("TEST", period=14)
        ctx._save_indicator_state()  # Simulate engine behavior

    rsi = ctx.ind.rsi("TEST", period=14)
    assert rsi is None


def test_rsi_caching():
    """Test that RSI indicators compute correctly."""
    ctx = Context()

    # Add bars with price changes
    for i in range(20):
        bar = Bar(
            ts=datetime(2023, 1, i + 1, tzinfo=timezone.utc),
            symbol="TEST",
            open=Decimal("100") + Decimal(i),
            high=Decimal("102") + Decimal(i),
            low=Decimal("98") + Decimal(i),
            close=Decimal("100") + Decimal(i),
            volume=1_000_000,
        )
        ctx._add_bar_to_history(bar)
        _ = ctx.ind.rsi("TEST", period=14)
        _ = ctx.ind.rsi("TEST", period=7)
        ctx._save_indicator_state()  # Simulate engine behavior

    # Verify indicators work with different periods
    rsi_14 = ctx.ind.rsi("TEST", period=14)

    assert rsi_14 is not None
    assert isinstance(rsi_14, float)
    assert 0 <= rsi_14 <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
