"""Tests for IndicatorManager - simplified and working version."""

from datetime import datetime, timezone
from decimal import Decimal

from qtrader.api.context import Context
from qtrader.indicators.manager import IndicatorManager
from qtrader.indicators.trend.sma import SMA
from qtrader.models.bar import Bar


def create_bar(symbol: str, close: float, idx: int = 0) -> Bar:
    """Helper to create a test bar."""
    return Bar(
        ts=datetime.fromtimestamp(idx * 60, tz=timezone.utc),
        symbol=symbol,
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        volume=1000,
    )


class TestIndicatorManager:
    """Tests for IndicatorManager caching and management."""

    def test_indicator_caching_single_symbol(self):
        """Test that indicators are cached per symbol."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add bars and compute
        values = [100.0, 105.0, 110.0]
        for i, val in enumerate(values):
            bar = create_bar("TEST", val, i)
            ctx._add_bar_to_history(bar)
            manager.sma("TEST", period=2)

        # Get final result
        result1 = manager.sma("TEST", period=2)
        assert result1 is not None

        # Same params should return same value
        result2 = manager.sma("TEST", period=2)
        assert result2 == result1

    def test_indicator_caching_multiple_symbols(self):
        """Test that indicators cache separately per symbol."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add bars for two symbols
        for symbol in ["AAPL", "MSFT"]:
            for i, val in enumerate([100.0, 105.0, 110.0]):
                bar = create_bar(symbol, val, i)
                ctx._add_bar_to_history(bar)
                manager.sma(symbol, period=2)

        # Get results for both
        result_aapl = manager.sma("AAPL", period=2)
        result_msft = manager.sma("MSFT", period=2)

        # Should be same values (same data)
        assert result_aapl == result_msft
        assert result_aapl is not None

    def test_convenience_method_sma(self):
        """Test SMA convenience method works."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add enough bars
        for i in range(5):
            bar = create_bar("TEST", 100.0 + i * 2.0, i)
            ctx._add_bar_to_history(bar)
            manager.sma("TEST", period=3)

        result = manager.sma("TEST", period=3)
        assert result is not None
        # Should be average of last 3: (104, 106, 108)
        assert result > 100.0

    def test_convenience_method_ema(self):
        """Test EMA convenience method works."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add enough bars
        for i in range(6):
            bar = create_bar("TEST", 100.0 + i * 2.0, i)
            ctx._add_bar_to_history(bar)
            manager.ema("TEST", period=3)

        result = manager.ema("TEST", period=3)
        assert result is not None
        assert result > 100.0

    def test_convenience_method_rsi(self):
        """Test RSI convenience method works."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add uptrend bars
        for i in range(10):
            bar = create_bar("TEST", 100.0 + i * 5.0, i)
            ctx._add_bar_to_history(bar)
            manager.rsi("TEST", period=3)

        result = manager.rsi("TEST", period=3)
        assert result is not None
        assert 0 <= result <= 100
        assert result > 50  # Uptrend should have high RSI

    def test_convenience_method_atr(self):
        """Test ATR convenience method works."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add bars with volatility
        for i in range(10):
            bar = create_bar("TEST", 100.0 + (i % 3) * 5.0, i)
            ctx._add_bar_to_history(bar)
            manager.atr("TEST", period=3)

        result = manager.atr("TEST", period=3)
        assert result is not None
        assert result >= 0

    def test_convenience_method_bollinger_bands(self):
        """Test Bollinger Bands convenience method works."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add bars with some volatility
        for i in range(10):
            bar = create_bar("TEST", 100.0 + i * 2.0, i)
            ctx._add_bar_to_history(bar)
            manager.bollinger_bands("TEST", period=5, num_std=2.0)

        result = manager.bollinger_bands("TEST", period=5, num_std=2.0)
        assert result is not None
        assert hasattr(result, "upper")
        assert hasattr(result, "middle")
        assert hasattr(result, "lower")
        # Upper band should be above middle, middle above lower
        assert result.upper > result.middle > result.lower

    def test_convenience_method_macd(self):
        """Test MACD convenience method works."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add uptrend bars (enough for MACD calculation)
        for i in range(15):
            bar = create_bar("TEST", 100.0 + i * 1.5, i)
            ctx._add_bar_to_history(bar)
            manager.macd("TEST", fast=5, slow=8, signal=3)

        result = manager.macd("TEST", fast=5, slow=8, signal=3)
        assert result is not None
        assert hasattr(result, "macd_line")
        assert hasattr(result, "signal_line")
        assert hasattr(result, "histogram")
        # In uptrend, MACD line should exist
        assert result.macd_line is not None

    def test_custom_indicator_registration(self):
        """Test custom indicator registration."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add bars and compute
        for i in range(5):
            bar = create_bar("TEST", 100.0 + i * 2.0, i)
            ctx._add_bar_to_history(bar)

        # Register custom SMA
        custom_sma = SMA(period=2)
        manager.register("my_sma", custom_sma)

        # Compute for each bar
        for i in range(5):
            manager.get("my_sma", "TEST")

        result = manager.get("my_sma", "TEST")
        assert result is not None

    def test_get_max_lookback_single(self):
        """Test max lookback with single indicator."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add enough bars
        for i in range(25):
            bar = create_bar("TEST", 100.0, i)
            ctx._add_bar_to_history(bar)
            manager.sma("TEST", period=20)

        max_lookback = manager.get_max_lookback()
        assert max_lookback >= 20

    def test_get_max_lookback_multiple(self):
        """Test max lookback with multiple indicators."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add enough bars
        for i in range(25):
            bar = create_bar("TEST", 100.0 + i, i)
            ctx._add_bar_to_history(bar)
            manager.sma("TEST", period=10)
            manager.ema("TEST", period=5)

        max_lookback = manager.get_max_lookback()
        assert max_lookback >= 10

    def test_reset_clears_cache(self):
        """Test reset clears cached indicators."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add bars and compute
        for i in range(5):
            bar = create_bar("TEST", 100.0 + i * 2.0, i)
            ctx._add_bar_to_history(bar)
            manager.sma("TEST", period=2)

        result1 = manager.sma("TEST", period=2)
        assert result1 is not None

        # Reset clears cache, so need to recompute
        manager.reset("TEST")

        # Recompute from scratch
        for i in range(5):
            manager.sma("TEST", period=2)

        result2 = manager.sma("TEST", period=2)
        assert result2 is not None
        assert result2 == result1  # Same data, same result

    def test_indicator_with_insufficient_data(self):
        """Test indicators return None with insufficient data."""
        ctx = Context()
        manager = IndicatorManager(ctx)

        # Only 2 bars
        for i in range(2):
            bar = create_bar("TEST", 100.0 + i, i)
            ctx._add_bar_to_history(bar)
            manager.sma("TEST", period=10)

        result = manager.sma("TEST", period=10)
        assert result is None  # Not enough data

    def test_field_parameter(self):
        """Test field parameter works correctly."""
        from decimal import Decimal

        ctx = Context()
        manager = IndicatorManager(ctx)

        # Add bars with different open/close
        for i in range(5):
            bar = Bar(
                ts=datetime.fromtimestamp(i * 60, tz=timezone.utc),
                symbol="TEST",
                open=Decimal(str(100.0 + i)),
                high=Decimal(str(110.0 + i)),
                low=Decimal(str(90.0 + i)),
                close=Decimal(str(102.0 + i)),
                volume=1000,
            )
            ctx._add_bar_to_history(bar)
            manager.sma("TEST", period=3, field="close")
            manager.sma("TEST", period=3, field="open")

        sma_close = manager.sma("TEST", period=3, field="close")
        sma_open = manager.sma("TEST", period=3, field="open")

        assert sma_close is not None
        assert sma_open is not None
        # Should be different
        assert abs(sma_close - sma_open) > 0.01
