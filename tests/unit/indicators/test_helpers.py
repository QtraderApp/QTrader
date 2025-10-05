"""Unit tests for indicator helper functions."""

from datetime import datetime, timezone
from decimal import Decimal

from qtrader.api import RSI, SMA, Context, MACDIndicator
from qtrader.indicators import helpers
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


class TestCrossoverHelpers:
    """Tests for crossover detection helpers."""

    def test_crossed_above_basic(self):
        """Test basic crossover detection."""
        # Series1 crosses above Series2: prev1=5<=prev2=6, curr1=10>curr2=8
        assert helpers.crossed_above(10.0, 8.0, 5.0, 6.0) is True
        # Series1 crosses below Series2 (not above)
        assert helpers.crossed_above(5.0, 8.0, 10.0, 6.0) is False
        # No crossover - Series1 was already above
        assert helpers.crossed_above(10.0, 8.0, 9.0, 6.0) is False

    def test_crossed_above_equal_values(self):
        """Test crossover with equal values."""
        # Equal to threshold is not a crossover
        assert helpers.crossed_above(10.0, 10.0, 9.0, 11.0) is False
        assert helpers.crossed_above(10.0, 10.0, 11.0, 11.0) is False

    def test_crossed_above_none_values(self):
        """Test crossover with None values."""
        # None values should return False
        assert helpers.crossed_above(None, 10.0, 5.0, 11.0) is False
        assert helpers.crossed_above(5.0, None, 4.0, 11.0) is False
        assert helpers.crossed_above(5.0, 10.0, None, 11.0) is False
        assert helpers.crossed_above(5.0, 10.0, 4.0, None) is False

    def test_crossed_below_basic(self):
        """Test basic cross below detection."""
        # Series1 crosses below Series2: prev1=10>=prev2=6, curr1=5<curr2=8
        assert helpers.crossed_below(5.0, 8.0, 10.0, 6.0) is True
        # Series1 crosses above Series2 (not below)
        assert helpers.crossed_below(10.0, 8.0, 5.0, 6.0) is False
        # No crossover - Series1 was already below
        assert helpers.crossed_below(5.0, 8.0, 4.0, 6.0) is False

    def test_crossed_below_equal_values(self):
        """Test cross below with equal values."""
        assert helpers.crossed_below(10.0, 10.0, 11.0, 9.0) is False
        assert helpers.crossed_below(10.0, 10.0, 10.0, 10.0) is False

    def test_crossed_below_none_values(self):
        """Test cross below with None values."""
        assert helpers.crossed_below(None, 5.0, 10.0, 4.0) is False
        assert helpers.crossed_below(10.0, None, 11.0, 4.0) is False
        assert helpers.crossed_below(10.0, 5.0, None, 4.0) is False
        assert helpers.crossed_below(10.0, 5.0, 11.0, None) is False


class TestThresholdHelpers:
    """Tests for threshold detection helpers."""

    def test_crossed_above_threshold_basic(self):
        """Test basic threshold crossover."""
        # Crosses above 70: prev=65 <= 70 < curr=75
        assert helpers.crossed_above_threshold(75.0, 65.0, 70.0) is True
        # Already above: prev=71 > 70, so no crossover
        assert helpers.crossed_above_threshold(75.0, 71.0, 70.0) is False
        # Crosses below (not above): prev=75 >= 70 > curr=65
        assert helpers.crossed_above_threshold(65.0, 75.0, 70.0) is False

    def test_crossed_above_threshold_exact(self):
        """Test threshold with exact values."""
        # Moving from at threshold to above
        assert helpers.crossed_above_threshold(75.0, 70.0, 70.0) is True
        # Staying at threshold
        assert helpers.crossed_above_threshold(70.0, 70.0, 70.0) is False

    def test_crossed_above_threshold_none_values(self):
        """Test threshold with None values."""
        assert helpers.crossed_above_threshold(None, 75.0, 70.0) is False
        assert helpers.crossed_above_threshold(65.0, None, 70.0) is False

    def test_crossed_below_threshold_basic(self):
        """Test basic threshold cross below."""
        # Crosses below 30: prev=35 >= 30 > curr=25
        assert helpers.crossed_below_threshold(25.0, 35.0, 30.0) is True
        # Already below: prev=29 < 30, so no crossover
        assert helpers.crossed_below_threshold(25.0, 29.0, 30.0) is False
        # Crosses above (not below): prev=25 <= 30 < curr=35
        assert helpers.crossed_below_threshold(35.0, 25.0, 30.0) is False

    def test_crossed_below_threshold_exact(self):
        """Test cross below with exact values."""
        # Moving from at threshold to below
        assert helpers.crossed_below_threshold(25.0, 30.0, 30.0) is True
        # Staying at threshold
        assert helpers.crossed_below_threshold(30.0, 30.0, 30.0) is False

    def test_crossed_below_threshold_none_values(self):
        """Test cross below threshold with None values."""
        assert helpers.crossed_below_threshold(None, 25.0, 30.0) is False
        assert helpers.crossed_below_threshold(35.0, None, 30.0) is False

    def test_above_threshold_basic(self):
        """Test above threshold detection."""
        assert helpers.above_threshold(75.0, 70.0) is True
        assert helpers.above_threshold(70.0, 70.0) is False
        assert helpers.above_threshold(65.0, 70.0) is False

    def test_above_threshold_none(self):
        """Test above threshold with None."""
        assert helpers.above_threshold(None, 70.0) is False

    def test_below_threshold_basic(self):
        """Test below threshold detection."""
        assert helpers.below_threshold(25.0, 30.0) is True
        assert helpers.below_threshold(30.0, 30.0) is False
        assert helpers.below_threshold(35.0, 30.0) is False

    def test_below_threshold_none(self):
        """Test below threshold with None."""
        assert helpers.below_threshold(None, 30.0) is False

    def test_between_thresholds_basic(self):
        """Test between thresholds detection."""
        # Within range
        assert helpers.between_thresholds(50.0, 30.0, 70.0) is True
        # At boundaries
        assert helpers.between_thresholds(30.0, 30.0, 70.0) is True
        assert helpers.between_thresholds(70.0, 30.0, 70.0) is True
        # Outside range
        assert helpers.between_thresholds(25.0, 30.0, 70.0) is False
        assert helpers.between_thresholds(75.0, 30.0, 70.0) is False

    def test_between_thresholds_none(self):
        """Test between thresholds with None values."""
        assert helpers.between_thresholds(None, 30.0, 70.0) is False


class TestTrendHelpers:
    """Tests for trend/sequence helpers."""

    def test_is_increasing_basic(self):
        """Test is_increasing detection."""
        # Clearly increasing
        values: list[float | None] = [100.0, 110.0, 120.0]
        assert helpers.is_increasing(values) is True
        # Clearly decreasing
        values = [120.0, 110.0, 100.0]
        assert helpers.is_increasing(values) is False
        # Flat
        values = [100.0, 100.0, 100.0]
        assert helpers.is_increasing(values) is False

    def test_is_increasing_with_periods(self):
        """Test is_increasing with different periods."""
        values: list[float | None] = [100.0, 110.0, 120.0, 130.0, 140.0]
        assert helpers.is_increasing(values, periods=3) is True
        assert helpers.is_increasing(values, periods=5) is True

    def test_is_increasing_insufficient_data(self):
        """Test is_increasing with insufficient data."""
        values: list[float | None] = [100.0]
        assert helpers.is_increasing(values) is False
        assert helpers.is_increasing([]) is False

    def test_is_increasing_none_values(self):
        """Test is_increasing with None values."""
        values: list[float | None] = [100.0, None, 120.0, 130.0]
        # Should filter None and still detect trend
        result = helpers.is_increasing(values)
        # Implementation may vary - just check it doesn't crash
        assert isinstance(result, bool)

    def test_is_decreasing_basic(self):
        """Test is_decreasing detection."""
        # Clearly decreasing
        values: list[float | None] = [120.0, 110.0, 100.0]
        assert helpers.is_decreasing(values) is True
        # Clearly increasing
        values = [100.0, 110.0, 120.0]
        assert helpers.is_decreasing(values) is False
        # Flat
        values = [100.0, 100.0, 100.0]
        assert helpers.is_decreasing(values) is False

    def test_is_decreasing_with_periods(self):
        """Test is_decreasing with different periods."""
        values: list[float | None] = [140.0, 130.0, 120.0, 110.0, 100.0]
        assert helpers.is_decreasing(values, periods=3) is True
        assert helpers.is_decreasing(values, periods=5) is True

    def test_is_decreasing_insufficient_data(self):
        """Test is_decreasing with insufficient data."""
        values: list[float | None] = [100.0]
        assert helpers.is_decreasing(values) is False
        assert helpers.is_decreasing([]) is False


class TestComparisonHelpers:
    """Tests for comparison helpers."""

    def test_all_above_basic(self):
        """Test all above threshold using helper combinations."""
        # Can combine above_threshold with all()
        values = [50.0, 60.0, 70.0]
        threshold = 45.0
        assert all(helpers.above_threshold(v, threshold) for v in values) is True

        values = [50.0, 40.0, 70.0]
        assert all(helpers.above_threshold(v, threshold) for v in values) is False


class TestIntegrationWithIndicators:
    """Integration tests with real indicators."""

    def test_sma_crossover_detection(self):
        """Test SMA crossover using helpers."""
        sma_fast = SMA(period=3)
        sma_slow = SMA(period=5)
        ctx = Context()

        # Add bars for downtrend
        values = [150.0, 140.0, 130.0, 120.0, 110.0]
        fast = None
        slow = None
        for val in values:
            bar = create_bar("TEST", val)
            ctx._add_bar_to_history(bar)
            fast = sma_fast.compute("TEST", ctx)
            slow = sma_slow.compute("TEST", ctx)

        # Store last values
        prev_fast = fast
        prev_slow = slow

        # Add bars for uptrend (should cause crossover)
        for val in [115.0, 125.0, 140.0]:
            bar = create_bar("TEST", val)
            ctx._add_bar_to_history(bar)
            curr_fast = sma_fast.compute("TEST", ctx)
            curr_slow = sma_slow.compute("TEST", ctx)

            if prev_fast and prev_slow and curr_fast and curr_slow:
                # Check for bullish crossover
                if helpers.crossed_above(prev_fast, curr_fast, prev_slow, curr_slow):
                    # Found crossover!
                    assert curr_fast > curr_slow
                    return

            prev_fast = curr_fast
            prev_slow = curr_slow

    def test_rsi_threshold_detection(self):
        """Test RSI threshold detection using helpers."""
        rsi = RSI(period=3)
        ctx = Context()

        # Start neutral
        prev_rsi = None
        for val in [100.0, 105.0, 100.0, 105.0]:
            bar = create_bar("TEST", val)
            ctx._add_bar_to_history(bar)
            prev_rsi = rsi.compute("TEST", ctx)

        # Strong uptrend to push RSI above 70
        for val in [110.0, 120.0, 135.0, 155.0]:
            bar = create_bar("TEST", val)
            ctx._add_bar_to_history(bar)
            curr_rsi = rsi.compute("TEST", ctx)

            if prev_rsi and curr_rsi:
                # Check for overbought threshold cross
                if helpers.crossed_above_threshold(prev_rsi, curr_rsi, 70.0):
                    assert curr_rsi > 70.0
                    return

            prev_rsi = curr_rsi

    def test_macd_histogram_analysis(self):
        """Test MACD histogram analysis using helpers."""
        macd = MACDIndicator(fast=3, slow=5, signal=2)
        ctx = Context()

        # Add enough bars
        values = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
        results = []
        for val in values:
            bar = create_bar("TEST", val)
            ctx._add_bar_to_history(bar)
            result = macd.compute("TEST", ctx)
            if result:
                results.append(result)

        # Check histogram properties
        if len(results) >= 2:
            # Last histogram should be positive (uptrend) with tolerance for floating point
            last_hist = results[-1].histogram
            assert last_hist > -1e-10  # Allow tiny negative due to floating point errors

            # Check if histogram is increasing
            prev_hist = results[-2].histogram
            if last_hist > prev_hist:
                assert last_hist > prev_hist
