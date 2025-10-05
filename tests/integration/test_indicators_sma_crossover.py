"""
Integration test: SMA Crossover strategy with full workflow.

Tests:
- Indicators compute correctly across multiple bars
- Crossover detection works end-to-end
- Strategy generates signals based on indicators
- Full integration: indicators → signals → (future: orders → fills)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from qtrader.api import Context
from qtrader.models.bar import Bar


@pytest.fixture
def sma_test_bars():
    """
    Create bars that produce a clear SMA crossover.

    Prices designed to create:
    - Initial downtrend
    - Clear bullish crossover (fast crosses above slow)
    - Uptrend
    - Clear bearish crossover (fast crosses below slow)
    """
    base_ts = datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc)

    # Prices: downtrend → crossover → uptrend → crossover → downtrend
    prices = [
        # Downtrend (10 bars)
        110.0,
        108.0,
        106.0,
        104.0,
        102.0,
        100.0,
        98.0,
        96.0,
        94.0,
        92.0,
        # Bottom and reversal (5 bars)
        91.0,
        90.0,
        91.0,
        93.0,
        95.0,
        # Uptrend (10 bars) - fast SMA will cross above slow here
        97.0,
        99.0,
        101.0,
        103.0,
        105.0,
        107.0,
        109.0,
        111.0,
        113.0,
        115.0,
        # Top and reversal (5 bars)
        116.0,
        117.0,
        116.0,
        114.0,
        112.0,
        # Downtrend (10 bars) - fast SMA will cross below slow here
        110.0,
        108.0,
        106.0,
        104.0,
        102.0,
        100.0,
        98.0,
        96.0,
        94.0,
        92.0,
    ]

    bars = []
    for i, price in enumerate(prices):
        bar = Bar(
            ts=base_ts + timedelta(days=i),
            symbol="TEST",
            open=Decimal(str(price)),
            high=Decimal(str(price + 2)),
            low=Decimal(str(price - 2)),
            close=Decimal(str(price)),
            volume=1_000_000,
        )
        bars.append(bar)

    return bars


def test_sma_computation_across_bars(sma_test_bars):
    """Test that SMAs compute correctly as bars are added."""
    ctx = Context()

    # Add bars and compute SMA for each
    sma_values = []
    for bar in sma_test_bars[:10]:  # First 10 bars
        ctx._add_bar_to_history(bar)
        sma = ctx.ind.sma("TEST", period=5)
        sma_values.append(sma)
        ctx._save_indicator_state()  # Simulate engine behavior

    # First 4 bars should return None (insufficient data)
    assert sma_values[0] is None
    assert sma_values[1] is None
    assert sma_values[2] is None
    assert sma_values[3] is None

    # 5th bar onwards should have values
    assert sma_values[4] is not None

    # SMA should be decreasing (prices are decreasing)
    for i in range(5, 9):
        assert sma_values[i] < sma_values[i - 1], (
            f"SMA should decrease in downtrend: {sma_values[i]} >= {sma_values[i - 1]}"
        )


def test_sma_crossover_detection(sma_test_bars):
    """Test detection of SMA crossovers (fast crosses above/below slow)."""
    ctx = Context()

    fast_period = 5
    slow_period = 10

    crossovers = []

    # Process all bars
    for bar in sma_test_bars:
        ctx._add_bar_to_history(bar)

        fast_sma = ctx.ind.sma("TEST", period=fast_period)
        slow_sma = ctx.ind.sma("TEST", period=slow_period)

        if fast_sma is not None and slow_sma is not None:
            # Track for crossover detection
            ctx._track_indicator("TEST", "fast_sma", fast_sma)
            ctx._track_indicator("TEST", "slow_sma", slow_sma)

            # Check for crossovers
            if ctx.crossed_above("TEST", "fast_sma", "slow_sma"):
                crossovers.append(
                    {
                        "bar_idx": sma_test_bars.index(bar),
                        "type": "bullish",
                        "fast": fast_sma,
                        "slow": slow_sma,
                    }
                )
            elif ctx.crossed_below("TEST", "fast_sma", "slow_sma"):
                crossovers.append(
                    {
                        "bar_idx": sma_test_bars.index(bar),
                        "type": "bearish",
                        "fast": fast_sma,
                        "slow": slow_sma,
                    }
                )

        ctx._save_indicator_state()  # Simulate engine behavior

    # Should detect at least one bullish and one bearish crossover
    assert len(crossovers) >= 2, f"Expected at least 2 crossovers, got {len(crossovers)}"

    bullish = [c for c in crossovers if c["type"] == "bullish"]
    bearish = [c for c in crossovers if c["type"] == "bearish"]

    assert len(bullish) >= 1, "Should detect at least 1 bullish crossover"
    assert len(bearish) >= 1, "Should detect at least 1 bearish crossover"

    # Bullish should come before bearish (prices go down → up → down)
    assert bullish[0]["bar_idx"] < bearish[0]["bar_idx"], "First bullish crossover should come before first bearish"


def test_sma_caching_performance(sma_test_bars):
    """Test that indicators compute correctly and consistently."""
    ctx = Context()

    # Add bars and compute indicators incrementally
    for bar in sma_test_bars[:20]:
        ctx._add_bar_to_history(bar)
        _ = ctx.ind.sma("TEST", period=5)
        _ = ctx.ind.sma("TEST", period=10)
        _ = ctx.ind.sma("TEST", period=20)
        ctx._save_indicator_state()  # Simulate engine behavior

    # Verify indicators work
    sma_10 = ctx.ind.sma("TEST", period=10)
    sma_5 = ctx.ind.sma("TEST", period=5)

    assert sma_10 is not None
    assert sma_5 is not None
    assert isinstance(sma_10, float)
    assert isinstance(sma_5, float)


def test_multiple_symbols_independent(sma_test_bars):
    """Test that indicators for different symbols are independent."""
    ctx = Context()

    # Add bars for two symbols with different prices
    for i, bar in enumerate(sma_test_bars[:15]):
        # Symbol 1: use original prices
        ctx._add_bar_to_history(bar)

        # Symbol 2: use higher prices
        bar2 = Bar(
            ts=bar.ts,
            symbol="TEST2",
            open=bar.open + Decimal("50"),
            high=bar.high + Decimal("50"),
            low=bar.low + Decimal("50"),
            close=bar.close + Decimal("50"),
            volume=bar.volume,
        )
        ctx._add_bar_to_history(bar2)

        # Compute indicators incrementally for both symbols
        ctx.ind.sma("TEST", period=5)
        ctx.ind.sma("TEST2", period=5)
        ctx._save_indicator_state()  # Simulate engine behavior

    # Get SMAs for both symbols
    sma_test1 = ctx.ind.sma("TEST", period=5)
    sma_test2 = ctx.ind.sma("TEST2", period=5)

    assert sma_test1 is not None
    assert sma_test2 is not None

    # TEST2 should have higher SMA (prices are 50 points higher)
    assert sma_test2 > sma_test1 + 45


def test_indicator_with_field_parameter(sma_test_bars):
    """Test that indicators can compute on different fields."""
    ctx = Context()

    # Modify bars to have different open/close prices
    modified_bars = []
    for bar in sma_test_bars[:10]:
        modified = Bar(
            ts=bar.ts,
            symbol="TEST",
            open=bar.close - Decimal("5"),  # Open lower than close
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
        modified_bars.append(modified)
        ctx._add_bar_to_history(modified)

        # Compute indicators incrementally
        ctx.ind.sma("TEST", period=5, field="close")
        ctx.ind.sma("TEST", period=5, field="open")
        ctx._save_indicator_state()  # Simulate engine behavior

    # Compute SMA on both fields
    sma_close = ctx.ind.sma("TEST", period=5, field="close")
    sma_open = ctx.ind.sma("TEST", period=5, field="open")

    assert sma_close is not None
    assert sma_open is not None

    # SMA on close should be higher (close is 5 points above open)
    assert sma_close > sma_open


def test_sma_with_insufficient_data():
    """Test that SMA returns None when insufficient data."""
    ctx = Context()

    # Add only 3 bars
    for i in range(3):
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

    # Request SMA(10) with only 3 bars
    sma = ctx.ind.sma("TEST", period=10)
    assert sma is None


def test_full_strategy_workflow(sma_test_bars):
    """
    Test complete workflow simulating a strategy using SMA crossover.

    This mimics what a real strategy would do:
    1. Compute indicators
    2. Track them
    3. Detect crossovers
    4. Generate signals
    """
    ctx = Context()

    signals_generated = []

    for bar in sma_test_bars:
        # Add bar to history
        ctx._add_bar_to_history(bar)

        # Compute indicators (strategy logic)
        fast_sma = ctx.ind.sma("TEST", period=5)
        slow_sma = ctx.ind.sma("TEST", period=10)

        # Skip if insufficient data
        if fast_sma is None or slow_sma is None:
            continue

        # Track indicators
        ctx._track_indicator("TEST", "fast_sma", fast_sma)
        ctx._track_indicator("TEST", "slow_sma", slow_sma)

        # Detect crossovers and generate signals
        if ctx.crossed_above("TEST", "fast_sma", "slow_sma"):
            signals_generated.append(
                {
                    "ts": bar.ts,
                    "type": "BUY",
                    "fast": fast_sma,
                    "slow": slow_sma,
                    "price": float(bar.close),
                }
            )
        elif ctx.crossed_below("TEST", "fast_sma", "slow_sma"):
            signals_generated.append(
                {
                    "ts": bar.ts,
                    "type": "SELL",
                    "fast": fast_sma,
                    "slow": slow_sma,
                    "price": float(bar.close),
                }
            )

        ctx._save_indicator_state()  # Simulate engine behavior

    # Should generate signals
    assert len(signals_generated) >= 2, f"Expected at least 2 signals, got {len(signals_generated)}"

    # First signal should be BUY (uptrend starts first)
    assert signals_generated[0]["type"] == "BUY", "First signal should be BUY after downtrend reversal"

    # Verify signal data is complete
    for signal in signals_generated:
        assert "ts" in signal
        assert "type" in signal
        assert "fast" in signal
        assert "slow" in signal
        assert "price" in signal
        assert signal["price"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
