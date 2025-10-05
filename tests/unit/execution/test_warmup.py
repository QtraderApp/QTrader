"""Unit tests for warmup system."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from qtrader.api.context import Context
from qtrader.execution.warmup import WarmupDetector, WarmupProcessor
from qtrader.models.bar import Bar


def create_test_bar(symbol: str, timestamp: datetime, close: float) -> Bar:
    """Helper to create test bar."""
    return Bar(
        ts=timestamp,
        symbol=symbol,
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        volume=1000,
    )


class TestWarmupDetector:
    """Test warmup detector auto-detection."""

    def test_detect_no_indicators(self):
        """Should return 0 when no indicators registered."""
        ctx = Context()

        lookback = WarmupDetector.detect_max_lookback(ctx)

        assert lookback == 0

    def test_detect_indicators_after_usage(self):
        """Should detect indicators after they've been used."""
        ctx = Context()

        # Add bars
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(30):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            ctx._add_bar_to_history(bar)

        # Access indicators - this registers them in the manager
        _ = ctx.ind.sma("AAPL", period=10)
        _ = ctx.ind.sma("AAPL", period=20)
        _ = ctx.ind.rsi("AAPL", period=14)

        ctx._save_indicator_state()

        # Now detect lookback - should find the longest period
        lookback = WarmupDetector.detect_max_lookback(ctx)

        # Should be 20 (longest SMA period)
        assert lookback == 20

    def test_detect_macd(self):
        """Should detect MACD lookback as slow + signal."""
        ctx = Context()

        # Add bars
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(60):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            ctx._add_bar_to_history(bar)

        # Access MACD - this registers it
        _ = ctx.ind.macd("AAPL", fast=12, slow=26, signal=9)

        ctx._save_indicator_state()

        lookback = WarmupDetector.detect_max_lookback(ctx)

        # Should be 35 (slow 26 + signal 9)
        assert lookback == 35

    def test_detect_mixed_indicators(self):
        """Should handle mix of different indicator types."""
        ctx = Context()

        # Add bars
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(50):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            ctx._add_bar_to_history(bar)

        # Access various indicators
        _ = ctx.ind.sma("AAPL", period=20)
        _ = ctx.ind.rsi("AAPL", period=14)
        _ = ctx.ind.macd("AAPL", fast=12, slow=26, signal=9)  # 35
        _ = ctx.ind.atr("AAPL", period=10)

        ctx._save_indicator_state()

        lookback = WarmupDetector.detect_max_lookback(ctx)

        # MACD has longest lookback (26 + 9 = 35)
        assert lookback == 35


class TestWarmupProcessor:
    """Test warmup processor logic."""

    def test_disabled_warmup(self):
        """Should skip all warmup logic when disabled."""
        processor = WarmupProcessor(warmup_bars=20, enable_warmup=False)

        assert not processor.should_skip_bar(0)
        assert not processor.should_skip_bar(10)
        assert not processor.should_skip_bar(19)
        assert not processor.should_skip_bar(20)

        assert not processor.warmup_complete

    def test_should_skip_bar_during_warmup(self):
        """Should skip bars during warmup period."""
        processor = WarmupProcessor(warmup_bars=20, enable_warmup=True)

        # Should skip bars 0-19
        for i in range(20):
            assert processor.should_skip_bar(i)

        # Should not skip bar 20
        assert not processor.should_skip_bar(20)

    def test_should_skip_bar_after_complete(self):
        """Should not skip bars after warmup completes."""
        processor = WarmupProcessor(warmup_bars=20, enable_warmup=True)

        processor.complete_warmup()

        # Should not skip any bars after complete
        assert not processor.should_skip_bar(0)
        assert not processor.should_skip_bar(10)
        assert not processor.should_skip_bar(19)
        assert not processor.should_skip_bar(20)

    def test_process_warmup_bar(self):
        """Should process warmup bar and update context."""
        processor = WarmupProcessor(warmup_bars=20, enable_warmup=True)

        ctx = Context()

        # Initialize with some bars first
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(11):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            ctx._add_bar_to_history(bar)

        # Access indicator to register it
        _ = ctx.ind.sma("AAPL", period=10)
        ctx._save_indicator_state()

        # Process first warmup bar
        bar = create_test_bar(
            "AAPL",
            base_date + timedelta(days=11),
            161.0,
        )

        processor.process_warmup_bar(ctx, bar, ["AAPL"])

        # Bar should be in history (check per-symbol list)
        assert len(ctx._bar_history["AAPL"]) == 12
        assert processor.bars_processed == 1
        assert not processor.warmup_complete

    def test_process_multiple_warmup_bars(self):
        """Should process multiple warmup bars."""
        processor = WarmupProcessor(warmup_bars=5, enable_warmup=True)

        ctx = Context()

        # Initialize with some bars first
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(4):
            bar = create_test_bar("AAPL", base_date + timedelta(days=i), 150.0 + i)
            ctx._add_bar_to_history(bar)

        # Access indicator to register it
        _ = ctx.ind.sma("AAPL", period=3)
        ctx._save_indicator_state()

        # Process 5 warmup bars
        for i in range(5):
            bar = create_test_bar(
                "AAPL",
                base_date + timedelta(days=4 + i),
                154.0 + i,
            )
            processor.process_warmup_bar(ctx, bar, ["AAPL"])

        # All bars should be in history
        assert len(ctx._bar_history["AAPL"]) == 9
        assert processor.bars_processed == 5
        assert not processor.warmup_complete

    def test_complete_warmup(self):
        """Should mark warmup as complete."""
        processor = WarmupProcessor(warmup_bars=20, enable_warmup=True)

        assert not processor.warmup_complete

        processor.complete_warmup()

        assert processor.warmup_complete

    def test_get_metadata_disabled(self):
        """Should return correct metadata when disabled."""
        processor = WarmupProcessor(warmup_bars=20, enable_warmup=False)

        metadata = processor.get_metadata()

        assert metadata == {
            "enabled": False,
            "warmup_bars": 20,
            "bars_processed": 0,
            "complete": False,
        }

    def test_get_metadata_in_progress(self):
        """Should return correct metadata during warmup."""
        processor = WarmupProcessor(warmup_bars=20, enable_warmup=True)

        ctx = Context()

        # Process 5 bars
        for i in range(5):
            bar = create_test_bar(
                "AAPL",
                datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                150.0 + i,
            )
            processor.process_warmup_bar(ctx, bar, ["AAPL"])

        metadata = processor.get_metadata()

        assert metadata == {
            "enabled": True,
            "warmup_bars": 20,
            "bars_processed": 5,
            "complete": False,
        }

    def test_get_metadata_complete(self):
        """Should return correct metadata after complete."""
        processor = WarmupProcessor(warmup_bars=20, enable_warmup=True)

        ctx = Context()

        # Process 20 bars
        for i in range(20):
            bar = create_test_bar(
                "AAPL",
                datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                150.0 + i,
            )
            processor.process_warmup_bar(ctx, bar, ["AAPL"])

        processor.complete_warmup()

        metadata = processor.get_metadata()

        assert metadata == {
            "enabled": True,
            "warmup_bars": 20,
            "bars_processed": 20,
            "complete": True,
        }

    def test_warmup_builds_indicator_state(self):
        """Should build indicator state during warmup."""
        processor = WarmupProcessor(warmup_bars=10, enable_warmup=True)

        ctx = Context()

        # Process 10 warmup bars
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i, price in enumerate(prices):
            bar = create_test_bar(
                "AAPL",
                base_date + timedelta(days=i),
                price,
            )
            processor.process_warmup_bar(ctx, bar, ["AAPL"])

        # After warmup, verify we can compute indicators
        # The key test is that warmup processed without error
        _ = ctx.ind.sma("AAPL", 3)

        assert processor.bars_processed == 10
        assert len(ctx._bar_history["AAPL"]) == 10


class TestExecutionConfigWarmup:
    """Test warmup configuration."""

    def test_warmup_disabled_by_default(self):
        """Warmup should be disabled by default."""
        from qtrader.execution.config import ExecutionConfig

        config = ExecutionConfig()

        assert config.warmup is False
        assert config.warmup_bars is None

    def test_warmup_enabled(self):
        """Can enable warmup."""
        from qtrader.execution.config import ExecutionConfig

        config = ExecutionConfig(warmup=True)

        assert config.warmup is True

    def test_warmup_bars_explicit(self):
        """Can set explicit warmup bars."""
        from qtrader.execution.config import ExecutionConfig

        config = ExecutionConfig(warmup=True, warmup_bars=50)

        assert config.warmup is True
        assert config.warmup_bars == 50

    def test_warmup_bars_none_valid(self):
        """warmup_bars=None should be valid (auto-detect)."""
        from qtrader.execution.config import ExecutionConfig

        # Should not raise
        config = ExecutionConfig(warmup=True, warmup_bars=None)
        assert config.warmup_bars is None
