"""
Integration test: Backtest runner with warmup system.

Tests complete backtest lifecycle including:
- Strategy initialization (on_init)
- Warmup phase with indicator building
- Post-warmup start (on_start)
- Main trading loop (on_bar)
- Cleanup (on_end)
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.execution.config import ExecutionConfig
from qtrader.models.bar import Bar
from qtrader.models.portfolio import Portfolio
from qtrader.risk import Signal, SignalDirection, SignalType


class TrackingStrategy:
    """Strategy that tracks lifecycle calls for testing."""

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        """
        Initialize strategy.

        Args:
            fast_period: Fast SMA period
            slow_period: Slow SMA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Track lifecycle calls
        self.on_init_called = False
        self.on_start_called = False
        self.on_bar_count = 0
        self.on_end_called = False

        # Track warmup state at on_start
        self.indicators_valid_at_start = False
        self.fast_sma_at_start = None
        self.slow_sma_at_start = None

    def on_init(self, ctx: Context) -> None:
        """Called before warmup - register indicators."""
        self.on_init_called = True

        # Register indicators that require warmup
        _ = ctx.ind.sma("AAPL", self.fast_period)
        _ = ctx.ind.sma("AAPL", self.slow_period)

    def on_start(self, ctx: Context) -> None:
        """Called after warmup - verify indicators are ready."""
        self.on_start_called = True

        # Check if indicators have values
        fast = ctx.ind.sma("AAPL", self.fast_period)
        slow = ctx.ind.sma("AAPL", self.slow_period)

        self.fast_sma_at_start = fast
        self.slow_sma_at_start = slow
        self.indicators_valid_at_start = (fast is not None) and (slow is not None)

    def on_bar(self, bar: Bar, ctx: Context) -> Optional[List[Signal]]:
        """Called for each trading bar."""
        self.on_bar_count += 1

        # Compute indicators
        fast = ctx.ind.sma(bar.symbol, self.fast_period)
        slow = ctx.ind.sma(bar.symbol, self.slow_period)

        # Generate signal if both valid
        if fast is not None and slow is not None and fast > slow:
            return [
                Signal(
                    signal_id=f"test_{self.on_bar_count}",
                    strategy_ts=bar.ts,
                    symbol=bar.symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    direction=SignalDirection.LONG,
                )
            ]

        return None

    def on_end(self, ctx: Context) -> None:
        """Called after last bar."""
        self.on_end_called = True

    def on_fill(self, fill, ctx: Context) -> None:
        """Called after each fill - required by Strategy protocol."""
        pass


class MinimalStrategyNoInit:
    """Strategy without on_init - should work with 0 warmup bars."""

    def __init__(self):
        self.on_bar_count = 0

    def on_init(self, ctx: Context) -> None:
        """Empty on_init."""
        pass

    def on_start(self, ctx: Context) -> None:
        """Empty on_start."""
        pass

    def on_bar(self, bar: Bar, ctx: Context) -> Optional[List[Signal]]:
        """Called for each bar."""
        self.on_bar_count += 1
        return None

    def on_fill(self, fill, ctx: Context) -> None:
        """Empty on_fill."""
        pass

    def on_end(self, ctx: Context) -> None:
        """Empty on_end."""
        pass


def create_test_bars(symbol: str, count: int, start_price: float = 100.0) -> List[Bar]:
    """
    Create test bars with realistic OHLC.

    Args:
        symbol: Symbol for bars
        count: Number of bars to create
        start_price: Starting price

    Returns:
        List of Bar objects
    """
    bars = []
    base_date = datetime(2024, 1, 1)
    price = Decimal(str(start_price))

    for i in range(count):
        # Add some price variation
        daily_change = Decimal(str((i % 5) - 2))  # -2, -1, 0, 1, 2 pattern

        open_price = price
        high_price = price + abs(daily_change) + Decimal("0.5")
        low_price = price - abs(daily_change) - Decimal("0.5")
        close_price = price + daily_change

        bars.append(
            Bar(
                symbol=symbol,
                ts=base_date + timedelta(days=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=1000000,
            )
        )

        price = close_price

    return bars


class TestBacktestWarmupIntegration:
    """Test backtest runner with warmup system."""

    def test_full_lifecycle_with_warmup_auto_detect(self):
        """Test complete lifecycle with auto-detected warmup period."""
        # Setup
        strategy = TrackingStrategy(fast_period=10, slow_period=20)
        config = ExecutionConfig(warmup=True, warmup_bars=None)  # Auto-detect
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        ctx = Context(portfolio=portfolio)

        # Create 30 bars (20 for warmup, 10 for trading)
        bars = create_test_bars("AAPL", count=30)

        # Add bars to context history
        for bar in bars:
            ctx._bar_history["AAPL"].append(bar)

        # Run backtest
        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, bars, ["AAPL"], out_dir=None)  # type: ignore

        # Verify lifecycle sequence
        assert strategy.on_init_called, "on_init should be called"
        assert strategy.on_start_called, "on_start should be called"
        assert strategy.on_bar_count == 10, f"Expected 10 on_bar calls, got {strategy.on_bar_count}"
        assert strategy.on_end_called, "on_end should be called"

        # Verify indicators valid at start
        assert strategy.indicators_valid_at_start, "Indicators should be valid after warmup"
        assert strategy.fast_sma_at_start is not None
        assert strategy.slow_sma_at_start is not None

        # Verify metadata
        assert "warmup" in metadata
        assert metadata["warmup"]["enabled"] is True
        assert metadata["warmup"]["warmup_bars"] == 20  # Max of (10, 20)
        assert metadata["warmup"]["bars_processed"] == 20
        assert metadata["warmup"]["complete"] is True
        assert metadata["total_bars"] == 30
        assert metadata["trading_bars"] == 10

    def test_full_lifecycle_with_explicit_warmup(self):
        """Test lifecycle with explicit warmup period."""
        strategy = TrackingStrategy(fast_period=10, slow_period=20)
        config = ExecutionConfig(warmup=True, warmup_bars=15)  # Explicit override
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        ctx = Context(portfolio=portfolio)

        bars = create_test_bars("AAPL", count=25)
        for bar in bars:
            ctx._bar_history["AAPL"].append(bar)

        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, bars, ["AAPL"], out_dir=None)  # type: ignore

        # Verify explicit period used
        assert metadata["warmup"]["warmup_bars"] == 15
        assert metadata["warmup"]["bars_processed"] == 15
        assert strategy.on_bar_count == 10  # 25 - 15 = 10 trading bars

    def test_full_lifecycle_without_warmup(self):
        """Test lifecycle with warmup disabled."""
        strategy = TrackingStrategy(fast_period=10, slow_period=20)
        config = ExecutionConfig(warmup=False)  # Warmup disabled
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        ctx = Context(portfolio=portfolio)

        bars = create_test_bars("AAPL", count=30)
        for bar in bars:
            ctx._bar_history["AAPL"].append(bar)

        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, bars, ["AAPL"], out_dir=None)  # type: ignore

        # All bars should be trading bars
        assert strategy.on_bar_count == 30
        assert "warmup" not in metadata
        assert metadata["trading_bars"] == 30

    def test_strategy_without_on_init(self):
        """Test strategy without on_init - should work with 0 warmup."""
        strategy = MinimalStrategyNoInit()
        config = ExecutionConfig(warmup=True, warmup_bars=None)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        ctx = Context(portfolio=portfolio)

        bars = create_test_bars("AAPL", count=20)
        for bar in bars:
            ctx._bar_history["AAPL"].append(bar)

        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, bars, ["AAPL"], out_dir=None)  # type: ignore

        # No warmup needed (no indicators)
        assert strategy.on_bar_count == 20
        assert metadata["warmup"]["warmup_bars"] == 0
        assert metadata["trading_bars"] == 20

    def test_multi_symbol_warmup(self):
        """Test warmup with multiple symbols."""
        _ = TrackingStrategy(fast_period=10, slow_period=20)
        config = ExecutionConfig(warmup=True, warmup_bars=None)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        ctx = Context(portfolio=portfolio)

        # Create bars for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        all_bars = []

        for symbol in symbols:
            bars = create_test_bars(symbol, count=30, start_price=100 + len(symbol) * 10)
            for bar in bars:
                ctx._bar_history[symbol].append(bar)
                all_bars.append(bar)

        # Register indicators for all symbols in on_init
        class MultiSymbolStrategy(TrackingStrategy):
            def on_init(self, ctx: Context) -> None:
                super().on_init(ctx)
                for sym in ["MSFT", "GOOGL"]:
                    _ = ctx.ind.sma(sym, self.fast_period)
                    _ = ctx.ind.sma(sym, self.slow_period)

            def on_fill(self, fill, ctx: Context) -> None:
                """Required by Strategy protocol."""
                pass

        strategy_multi = MultiSymbolStrategy(fast_period=10, slow_period=20)
        backtest = Backtest(config, strategy_multi)
        metadata = backtest.run(ctx, all_bars, symbols, out_dir=None)  # type: ignore

        # Verify warmup applied
        assert metadata["warmup"]["warmup_bars"] == 20
        assert metadata["warmup"]["complete"] is True

    def test_warmup_with_macd(self):
        """Test warmup auto-detection with MACD indicator."""

        class MACDStrategy:
            def __init__(self):
                self.on_bar_count = 0
                self.macd_valid_at_start = False

            def on_init(self, ctx: Context) -> None:
                # MACD requires slow + signal_period = 26 + 9 = 35 bars
                _ = ctx.ind.macd("AAPL", fast=12, slow=26, signal=9)

            def on_start(self, ctx: Context) -> None:
                macd = ctx.ind.macd("AAPL", fast=12, slow=26, signal=9)
                self.macd_valid_at_start = macd is not None

            def on_bar(self, bar: Bar, ctx: Context) -> Optional[List[Signal]]:
                self.on_bar_count += 1
                return None

            def on_fill(self, fill, ctx: Context) -> None:
                """Required by Strategy protocol."""
                pass

            def on_end(self, ctx: Context) -> None:
                """Required by Strategy protocol."""
                pass

        strategy = MACDStrategy()
        config = ExecutionConfig(warmup=True, warmup_bars=None)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        ctx = Context(portfolio=portfolio)

        bars = create_test_bars("AAPL", count=45)
        for bar in bars:
            ctx._bar_history["AAPL"].append(bar)

        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, bars, ["AAPL"], out_dir=None)  # type: ignore

        # Verify MACD warmup detected correctly
        assert metadata["warmup"]["warmup_bars"] == 35  # 26 + 9
        assert strategy.macd_valid_at_start, "MACD should be valid after warmup"
        assert strategy.on_bar_count == 10  # 45 - 35 = 10

    def test_warmup_metadata_structure(self):
        """Test warmup metadata has correct structure."""
        strategy = TrackingStrategy(fast_period=10, slow_period=20)
        config = ExecutionConfig(warmup=True, warmup_bars=None)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        ctx = Context(portfolio=portfolio)

        bars = create_test_bars("AAPL", count=30)
        for bar in bars:
            ctx._bar_history["AAPL"].append(bar)

        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, bars, ["AAPL"], out_dir=None)  # type: ignore

        # Verify metadata structure
        assert isinstance(metadata, dict)
        assert "warmup" in metadata
        assert "total_bars" in metadata
        assert "trading_bars" in metadata

        warmup = metadata["warmup"]
        assert "enabled" in warmup
        assert "warmup_bars" in warmup
        assert "bars_processed" in warmup
        assert "complete" in warmup

        # Verify types
        assert isinstance(warmup["enabled"], bool)
        assert isinstance(warmup["warmup_bars"], int)
        assert isinstance(warmup["bars_processed"], int)
        assert isinstance(warmup["complete"], bool)

    def test_insufficient_bars_for_warmup(self):
        """Test behavior when dataset has fewer bars than warmup period."""
        _strategy = TrackingStrategy(fast_period=10, slow_period=20)
        config = ExecutionConfig(warmup=True, warmup_bars=None)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        ctx = Context(portfolio=portfolio)

        # Only 15 bars, but need 20 for warmup
        bars = create_test_bars("AAPL", count=15)
        for bar in bars:
            ctx._bar_history["AAPL"].append(bar)

        backtest = Backtest(config, _strategy)
        metadata = backtest.run(ctx, bars, ["AAPL"], out_dir=None)  # type: ignore

        # All bars used for warmup, no trading bars
        assert metadata["warmup"]["warmup_bars"] == 20
        assert metadata["warmup"]["bars_processed"] == 15  # Only 15 available
        assert _strategy.on_bar_count == 0  # No trading bars
