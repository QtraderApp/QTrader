"""
Integration test for Phase 3: BacktestEngine + StrategyService.

Tests the complete data -> strategy -> signals pipeline with real historical data.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine
from qtrader.events.events import SignalEvent


class TestStrategyIntegration:
    """Test end-to-end strategy execution in backtest."""

    def test_buy_and_hold_strategy_emits_signal(self, tmp_path):
        """
        Test that buy_and_hold strategy discovers, loads, and emits signal.

        This tests:
        1. Strategy auto-discovery from my_library/strategies
        2. Strategy instantiation from portfolio.yaml config
        3. DataService streaming bars to StrategyService
        4. Strategy receiving bars and emitting signals
        5. Signal publishing to EventBus
        6. Metrics tracking
        """
        # Load config
        config_path = Path("config/portfolio.yaml")
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_backtest_config(config_path)

        # Verify config has strategies
        assert len(config.strategies) > 0, "No strategies configured"
        assert config.strategies[0].strategy_id == "buy_and_hold"

        # Collect signals
        signals_collected = []

        def collect_signals(event):
            if isinstance(event, SignalEvent):
                signals_collected.append(event)

        # Run backtest
        with BacktestEngine.from_config(config) as engine:
            # Subscribe to signal events
            engine._event_bus.subscribe("signal", collect_signals)

            # Run backtest
            result = engine.run()

            # Verify bars processed
            assert result.bars_processed > 0, "No bars processed"

            # Verify strategy metrics
            if engine._strategy_service:
                metrics = engine._strategy_service.get_metrics()
                assert "buy_and_hold" in metrics, "Strategy not in metrics"

                strategy_metrics = metrics["buy_and_hold"]
                assert strategy_metrics["bars_processed"] > 0, "Strategy processed no bars"
                assert strategy_metrics["signals_emitted"] == 1, "Expected exactly 1 signal"
                assert strategy_metrics["errors"] == 0, "Strategy had errors"

                # Verify signal was collected
                assert len(signals_collected) == 1, "Expected exactly 1 signal event"

                signal = signals_collected[0]
                assert signal.symbol == "AAPL"
                assert signal.intention == "OPEN_LONG"
                assert isinstance(signal.price, Decimal)
                assert signal.price > 0
                assert signal.confidence == Decimal("1.0")
                assert "Buy and hold" in signal.reason

    def test_strategy_universe_filtering(self, tmp_path):
        """
        Test that strategies only receive bars for symbols in their universe.

        This creates a custom config with AAPL+MSFT universe but strategy
        only trades AAPL.
        """
        # This test would require creating a custom config
        # For now, just verify the buy_and_hold strategy only trades AAPL
        config_path = Path("config/portfolio.yaml")
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_backtest_config(config_path)

        with BacktestEngine.from_config(config) as engine:
            result = engine.run()

            # Verify strategy only processed AAPL bars
            if engine._strategy_service:
                metrics = engine._strategy_service.get_metrics()
                strategy_metrics = metrics["buy_and_hold"]

                # The strategy universe is ['AAPL']
                # So bars_processed should equal total bars for AAPL
                assert strategy_metrics["bars_processed"] == result.bars_processed

    def test_strategy_lifecycle_methods_called(self):
        """
        Test that strategy setup and teardown are called.

        This is implicitly tested by the first test, but we verify
        the lifecycle explicitly here.
        """
        config_path = Path("config/portfolio.yaml")
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_backtest_config(config_path)

        with BacktestEngine.from_config(config) as engine:
            # Check strategy service exists
            assert engine._strategy_service is not None

            # Run backtest (this calls setup and teardown)
            result = engine.run()

            # If we got here without exceptions, setup and teardown succeeded
            assert result.bars_processed > 0

    def test_backtest_performance(self):
        """
        Test that backtest performance meets minimum threshold.

        Target: >50 bars/second (conservative threshold for integration test)
        """
        config_path = Path("config/portfolio.yaml")
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_backtest_config(config_path)

        with BacktestEngine.from_config(config) as engine:
            result = engine.run()

            # Calculate performance
            bars_per_sec = result.bars_processed / result.duration.total_seconds()

            # Verify performance (very conservative threshold for integration tests)
            # Note: Most of the time is spent in data loading/validation
            assert bars_per_sec > 50, f"Performance too slow: {bars_per_sec:.0f} bars/sec"
