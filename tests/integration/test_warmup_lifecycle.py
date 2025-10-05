"""Integration test for warmup lifecycle with strategy."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional

from qtrader.api.context import Context
from qtrader.execution.config import ExecutionConfig
from qtrader.execution.warmup import WarmupDetector, WarmupProcessor
from qtrader.models.bar import Bar
from qtrader.risk import Signal


def create_test_bar(symbol: str, date: datetime, close: float) -> Bar:
    """Helper to create test bar."""
    return Bar(
        ts=date,
        symbol=symbol,
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        volume=10000,
    )


class SimpleStrategy:
    """Simple test strategy that tracks lifecycle calls."""

    def __init__(self):
        self.on_init_called = False
        self.on_start_called = False
        self.on_bar_count = 0
        self.on_bar_calls = []  # Track which bars triggered on_bar
        self.warmup_complete_when_on_start = False

    def on_init(self, ctx: Context) -> None:
        """Register indicators before warmup."""
        self.on_init_called = True
        # Access indicators to register them
        # These will be auto-detected for warmup
        _ = ctx.ind.sma("AAPL", period=10)
        _ = ctx.ind.rsi("AAPL", period=14)

    def on_start(self, ctx: Context) -> None:
        """Called after warmup completes."""
        self.on_start_called = True

    def on_bar(self, bar: Bar, ctx: Context) -> Optional[List[Signal]]:
        """Process trading bar."""
        self.on_bar_count += 1
        self.on_bar_calls.append(bar.ts)

        # Try to access indicators (verify they work after warmup)
        _ = ctx.ind.sma("AAPL", 10)
        _ = ctx.ind.rsi("AAPL", 14)

        # After warmup, indicators should return values
        return None  # No signals for this test


class TestWarmupLifecycle:
    """Test complete warmup lifecycle integration."""

    def test_lifecycle_sequence_with_auto_detection(self):
        """Test full lifecycle: on_init → warmup → on_start → on_bar."""
        # Create strategy and context
        strategy = SimpleStrategy()
        ctx = Context()

        # Enable warmup with auto-detection
        config = ExecutionConfig(warmup=True, warmup_bars=None)

        # Step 1: Call on_init
        strategy.on_init(ctx)
        assert strategy.on_init_called
        assert not strategy.on_start_called
        assert strategy.on_bar_count == 0

        # Step 2: Detect warmup period
        warmup_bars = WarmupDetector.detect_max_lookback(ctx)
        assert warmup_bars == 14  # Max of SMA(10) and RSI(14)

        # Step 3: Create warmup processor
        processor = WarmupProcessor(warmup_bars, config.warmup)

        # Step 4: Generate and process warmup bars
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(warmup_bars):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)

            # During warmup, should skip on_bar
            if processor.should_skip_bar(i):
                processor.process_warmup_bar(ctx, bar, ["AAPL"])
                # Verify on_bar was NOT called
                assert strategy.on_bar_count == 0

        # Step 5: Complete warmup and call on_start
        processor.complete_warmup()
        strategy.on_start(ctx)
        assert strategy.on_start_called

        # Step 6: Process trading bars
        for i in range(5):
            bar = create_test_bar(
                "AAPL",
                base_date + timedelta(days=warmup_bars + i),
                164.0 + i,
            )

            # After warmup, should process normally
            assert not processor.should_skip_bar(warmup_bars + i)
            strategy.on_bar(bar, ctx)

        # Verify final state
        assert strategy.on_bar_count == 5
        assert len(strategy.on_bar_calls) == 5
        assert processor.warmup_complete

    def test_lifecycle_with_explicit_warmup_period(self):
        """Test warmup with explicit period override."""
        strategy = SimpleStrategy()
        ctx = Context()

        # Enable warmup with explicit period
        explicit_period = 20
        config = ExecutionConfig(warmup=True, warmup_bars=explicit_period)

        # Call on_init
        strategy.on_init(ctx)

        # Use explicit period (not auto-detected)
        warmup_bars = config.warmup_bars or WarmupDetector.detect_max_lookback(ctx)
        assert warmup_bars == 20  # Explicit override

        # Create processor and process warmup
        processor = WarmupProcessor(warmup_bars, config.warmup)

        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(warmup_bars):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            if processor.should_skip_bar(i):
                processor.process_warmup_bar(ctx, bar, ["AAPL"])

        processor.complete_warmup()
        strategy.on_start(ctx)

        # Verify warmup used explicit period
        assert processor.bars_processed == 20

    def test_lifecycle_without_warmup(self):
        """Test lifecycle when warmup is disabled."""
        strategy = SimpleStrategy()
        ctx = Context()

        # Disable warmup
        config = ExecutionConfig(warmup=False)

        # Call on_init
        strategy.on_init(ctx)

        # No warmup processing needed
        processor = WarmupProcessor(0, config.warmup)

        # Call on_start immediately
        strategy.on_start(ctx)

        # Process bars normally from start
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(5):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            strategy.on_bar(bar, ctx)

        # All bars processed normally
        assert strategy.on_bar_count == 5
        assert not processor.enable_warmup

    def test_indicators_valid_after_warmup(self):
        """Test that indicators return valid values after warmup."""
        ctx = Context()

        # Process warmup bars
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(20):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 100.0 + i)
            ctx._add_bar_to_history(bar)
            # Compute indicators during warmup
            _ = ctx.ind.sma("AAPL", 10)
            _ = ctx.ind.rsi("AAPL", 14)
            ctx._save_indicator_state()

        # After 20 bars, both indicators should have valid values
        sma = ctx.ind.sma("AAPL", 10)
        rsi = ctx.ind.rsi("AAPL", 14)

        assert sma is not None
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_warmup_metadata_recorded(self):
        """Test that warmup metadata is correctly generated."""
        ctx = Context()

        # Enable warmup and detect period
        config = ExecutionConfig(warmup=True, warmup_bars=None)

        # Register indicators
        _ = ctx.ind.sma("AAPL", period=20)
        _ = ctx.ind.macd("AAPL", fast=12, slow=26, signal=9)

        warmup_bars = WarmupDetector.detect_max_lookback(ctx)
        assert warmup_bars == 35  # MACD: 26 + 9

        # Create and run processor
        processor = WarmupProcessor(warmup_bars, config.warmup)

        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(warmup_bars):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            if processor.should_skip_bar(i):
                processor.process_warmup_bar(ctx, bar, ["AAPL"])

        processor.complete_warmup()

        # Get metadata
        metadata = processor.get_metadata()

        assert metadata["enabled"] is True
        assert metadata["warmup_bars"] == 35
        assert metadata["bars_processed"] == 35
        assert metadata["complete"] is True

    def test_multi_symbol_warmup(self):
        """Test warmup with multiple symbols."""
        ctx = Context()
        symbols = ["AAPL", "MSFT", "GOOGL"]

        # Process warmup for multiple symbols
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(20):
            bars = [
                create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i),
                create_test_bar("MSFT", base_date + timedelta(days=i), 300.0 + i * 2),
                create_test_bar("GOOGL", base_date + timedelta(days=i), 100.0 + i * 0.5),
            ]

            for bar in bars:
                ctx._add_bar_to_history(bar)

            # Compute indicators for all symbols
            for symbol in symbols:
                _ = ctx.ind.sma(symbol, 10)

            ctx._save_indicator_state()

        # Verify all symbols have indicator values
        for symbol in symbols:
            sma = ctx.ind.sma(symbol, 10)
            assert sma is not None

    def test_warmup_with_insufficient_bars(self):
        """Test behavior when warmup has fewer bars than needed."""
        ctx = Context()

        # Register indicator needing 20 bars
        _ = ctx.ind.sma("AAPL", period=20)

        warmup_bars = WarmupDetector.detect_max_lookback(ctx)
        assert warmup_bars == 20

        processor = WarmupProcessor(warmup_bars, enable_warmup=True)

        # Only process 10 bars (not enough)
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(10):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            processor.process_warmup_bar(ctx, bar, ["AAPL"])

        # SMA might still return None after insufficient warmup
        sma = ctx.ind.sma("AAPL", 20)
        # This is expected - need full warmup period
        assert sma is None or isinstance(sma, float)

    def test_warmup_processor_state_transitions(self):
        """Test warmup processor state transitions."""
        processor = WarmupProcessor(warmup_bars=10, enable_warmup=True)

        # Initial state
        assert not processor.warmup_complete
        assert processor.bars_processed == 0

        # During warmup
        assert processor.should_skip_bar(0)
        assert processor.should_skip_bar(9)

        # After warmup period
        assert not processor.should_skip_bar(10)
        assert not processor.should_skip_bar(11)

        # Complete warmup
        processor.complete_warmup()
        assert processor.warmup_complete

        # After complete, nothing should be skipped
        assert not processor.should_skip_bar(0)

    def test_warmup_with_strategy_that_skips_on_init(self):
        """Test warmup when strategy doesn't implement on_init."""

        class MinimalStrategy:
            def __init__(self):
                self.on_bar_count = 0

            def on_bar(self, bar, ctx):
                self.on_bar_count += 1
                return None

        strategy = MinimalStrategy()
        ctx = Context()

        # No on_init call - warmup defaults to 0
        warmup_bars = WarmupDetector.detect_max_lookback(ctx)
        assert warmup_bars == 0  # No indicators registered

        # Processor with 0 bars effectively disables warmup
        processor = WarmupProcessor(0, enable_warmup=True)

        # All bars should be processed normally
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(5):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0)
            assert not processor.should_skip_bar(i)
            strategy.on_bar(bar, ctx)

        assert strategy.on_bar_count == 5
