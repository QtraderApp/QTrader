"""Integration tests for BacktestEngine.run() with full event flow.

These tests verify the complete backtest execution with real data loading,
event publishing, and results collection.

NOTE: These tests are SKIPPED pending engine refactor to new architecture.
The engine implementation still uses old BacktestConfig API (warmup_bars, dataset, etc.).
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from qtrader.contracts.data import Bar
from qtrader.engine.config import BacktestConfig
from qtrader.engine.engine import BacktestEngine, BacktestResult
from qtrader.events.event_bus import EventBus
from qtrader.events.events import PriceBarEvent, RiskEvaluationTriggerEvent, ValuationTriggerEvent

pytestmark = pytest.mark.skip(reason="Engine not yet refactored for new config architecture")


@pytest.fixture
def minimal_risk_config():
    """Minimal risk configuration for tests."""
    return {
        "budgets": [],
        "sizing": {},
        "concentration": {"max_position_pct": 0.10},
        "leverage": {"max_gross": 1.0, "max_net": 1.0},
    }


class TestBacktestEngineRun:
    """Test BacktestEngine.run() method."""

    def test_run_returns_backtest_result(self, minimal_risk_config):
        """Test that run() returns a BacktestResult."""
        # Create minimal config
        config_dict = {
            "start_date": "2020-01-02",
            "end_date": "2020-01-05",
            "initial_equity": 100000,
            "warmup_bars": 0,
            "universe": ["AAPL"],
            "data": {
                "source": "algoseek",
                "data_path": "data/",
                "dataset": "algoseek-us-equity-1d-unadjusted",
            },
            "portfolio": {
                "initial_equity": 100000,
                "base_currency": "USD",
                "accounting_method": "average_cost",
            },
            "execution": {
                "fill_model": "close",
                "commission_model": "fixed",
                "slippage_model": "none",
            },
            "risk": minimal_risk_config,
            "strategies": [],
        }

        config = BacktestConfig(**config_dict)

        # Mock DataService.stream_universe to simulate bar publishing
        with patch.object(BacktestEngine, "__init__", lambda self, *args, **kwargs: None):
            engine = BacktestEngine(
                config=config,
                event_bus=Mock(),
                data_service=Mock(),
                portfolio_service=Mock(),
                execution_service=Mock(),
                risk_service=Mock(),
                strategy_service=Mock(),
            )

            # Mock the services
            engine.config = config
            engine._event_bus = EventBus()
            engine._data_service = Mock()
            engine._portfolio_service = Mock()
            engine._execution_service = Mock()
            engine._risk_service = Mock()
            engine._strategy_service = Mock()

            # Mock stream_universe to do nothing (no data available)
            engine._data_service.stream_universe = Mock()

            # Mock get_equity
            engine._portfolio_service.get_equity = Mock(return_value=Decimal("105000"))

            # Mock get_filled_orders
            engine._execution_service.get_filled_orders = Mock(return_value=[])

            # Run backtest
            result = engine.run()

            # Verify result
            assert isinstance(result, BacktestResult)
            assert result.start_date == config.start_date
            assert result.end_date == config.end_date
            assert result.initial_equity == 100000.0
            assert result.final_capital == 105000.0
            assert result.total_return == pytest.approx(0.05)  # 5% return
            assert result.num_trades == 0

    def test_run_with_warmup_phase(self, minimal_risk_config):
        """Test that run() executes warmup phase when warmup_bars > 0."""
        config_dict = {
            "start_date": "2020-01-10",
            "end_date": "2020-01-15",
            "initial_equity": 100000,
            "warmup_bars": 5,  # 5 warmup bars
            "universe": ["AAPL"],
            "data": {
                "source": "algoseek",
                "data_path": "data/",
                "dataset": "algoseek-us-equity-1d-unadjusted",
            },
            "portfolio": {
                "initial_equity": 100000,
                "base_currency": "USD",
                "accounting_method": "average_cost",
            },
            "execution": {
                "fill_model": "close",
                "commission_model": "fixed",
                "slippage_model": "none",
            },
            "risk": minimal_risk_config,
            "strategies": [],
        }

        config = BacktestConfig(**config_dict)

        with patch.object(BacktestEngine, "__init__", lambda self, *args, **kwargs: None):
            engine = BacktestEngine(
                config=config,
                event_bus=Mock(),
                data_service=Mock(),
                portfolio_service=Mock(),
                execution_service=Mock(),
                risk_service=Mock(),
                strategy_service=Mock(),
            )

            # Setup mocks
            engine.config = config
            engine._event_bus = EventBus()
            engine._data_service = Mock()
            engine._portfolio_service = Mock()
            engine._execution_service = Mock()
            engine._risk_service = Mock()
            engine._strategy_service = Mock()

            # Mock services
            engine._data_service.stream_universe = Mock()
            engine._portfolio_service.get_equity = Mock(return_value=Decimal("100000"))
            engine._execution_service.get_filled_orders = Mock(return_value=[])

            # Run backtest
            result = engine.run()

            # Verify stream_universe called twice (warmup + main)
            assert engine._data_service.stream_universe.call_count == 2

            # First call should be warmup (is_warmup=True)
            warmup_call = engine._data_service.stream_universe.call_args_list[0]
            assert warmup_call.kwargs["is_warmup"] is True

            # Second call should be main phase (is_warmup=False)
            main_call = engine._data_service.stream_universe.call_args_list[1]
            assert main_call.kwargs["is_warmup"] is False

            # Verify result is returned
            assert isinstance(result, BacktestResult)

    def test_run_publishes_trigger_events(self, minimal_risk_config):
        """Test that run() publishes ValuationTriggerEvent and RiskEvaluationTriggerEvent."""
        config_dict = {
            "start_date": "2020-01-02",
            "end_date": "2020-01-03",
            "initial_equity": 100000,
            "warmup_bars": 0,
            "universe": ["AAPL"],
            "data": {
                "source": "algoseek",
                "data_path": "data/",
                "dataset": "algoseek-us-equity-1d-unadjusted",
            },
            "portfolio": {
                "initial_equity": 100000,
                "base_currency": "USD",
                "accounting_method": "average_cost",
            },
            "execution": {
                "fill_model": "close",
                "commission_model": "fixed",
                "slippage_model": "none",
            },
            "risk": minimal_risk_config,
            "strategies": [],
        }

        config = BacktestConfig(**config_dict)

        with patch.object(BacktestEngine, "__init__", lambda self, *args, **kwargs: None):
            engine = BacktestEngine(
                config=config,
                event_bus=Mock(),
                data_service=Mock(),
                portfolio_service=Mock(),
                execution_service=Mock(),
                risk_service=Mock(),
                strategy_service=Mock(),
            )

            # Setup mocks
            engine.config = config
            event_bus = EventBus()
            engine._event_bus = event_bus
            engine._data_service = Mock()
            engine._portfolio_service = Mock()
            engine._execution_service = Mock()

            # Track published events
            valuation_events = []
            risk_events = []

            def track_valuation(event):
                valuation_events.append(event)

            def track_risk(event):
                risk_events.append(event)

            event_bus.subscribe("valuation_trigger", track_valuation)
            event_bus.subscribe("risk_evaluation_trigger", track_risk)

            # Mock stream_universe to publish some bars
            def mock_stream(symbols, start_date, end_date, is_warmup):
                if not is_warmup:
                    # Simulate 2 bars for AAPL on different days
                    bar1 = Bar(
                        trade_datetime=datetime(2020, 1, 2, 16, 0),
                        open=100.0,
                        high=101.0,
                        low=99.0,
                        close=100.5,
                        volume=1000000,
                    )
                    bar2 = Bar(
                        trade_datetime=datetime(2020, 1, 3, 16, 0),
                        open=100.5,
                        high=102.0,
                        low=100.0,
                        close=101.5,
                        volume=1100000,
                    )

                    # Publish events
                    event_bus.publish(
                        PriceBarEvent(
                            symbol="AAPL",
                            bar=bar1,
                            timestamp=datetime(2020, 1, 2, 16, 0),
                            is_warmup=False,
                        )
                    )
                    event_bus.publish(
                        PriceBarEvent(
                            symbol="AAPL",
                            bar=bar2,
                            timestamp=datetime(2020, 1, 3, 16, 0),
                            is_warmup=False,
                        )
                    )

            engine._data_service.stream_universe = mock_stream
            engine._portfolio_service.get_equity = Mock(return_value=Decimal("100000"))
            engine._execution_service.get_filled_orders = Mock(return_value=[])

            # Run backtest
            result = engine.run()

            # Verify trigger events were published
            # Should have 2 valuation triggers (one per timestamp)
            assert len(valuation_events) == 2
            assert all(isinstance(e, ValuationTriggerEvent) for e in valuation_events)

            # Should have 2 risk evaluation triggers (one per timestamp)
            assert len(risk_events) == 2
            assert all(isinstance(e, RiskEvaluationTriggerEvent) for e in risk_events)

            # Verify result is returned
            assert isinstance(result, BacktestResult)

    def test_run_collects_results_from_services(self, minimal_risk_config):
        """Test that run() collects final results from services."""
        config_dict = {
            "start_date": "2020-01-02",
            "end_date": "2020-01-05",
            "initial_equity": 100000,
            "warmup_bars": 0,
            "universe": ["AAPL"],
            "data": {
                "source": "algoseek",
                "data_path": "data/",
                "dataset": "algoseek-us-equity-1d-unadjusted",
            },
            "portfolio": {
                "initial_equity": 100000,
                "base_currency": "USD",
                "accounting_method": "average_cost",
            },
            "execution": {
                "fill_model": "close",
                "commission_model": "fixed",
                "slippage_model": "none",
            },
            "risk": minimal_risk_config,
            "strategies": [],
        }

        config = BacktestConfig(**config_dict)

        with patch.object(BacktestEngine, "__init__", lambda self, *args, **kwargs: None):
            engine = BacktestEngine(
                config=config,
                event_bus=Mock(),
                data_service=Mock(),
                portfolio_service=Mock(),
                execution_service=Mock(),
                risk_service=Mock(),
                strategy_service=Mock(),
            )

            # Setup mocks
            engine.config = config
            engine._event_bus = EventBus()
            engine._data_service = Mock()
            engine._portfolio_service = Mock()
            engine._execution_service = Mock()

            # Mock services
            engine._data_service.stream_universe = Mock()

            # Mock portfolio service to return specific equity
            engine._portfolio_service.get_equity = Mock(return_value=Decimal("125000.50"))

            # Mock execution service to return some fills
            mock_fills = [Mock(), Mock(), Mock()]  # 3 fills
            engine._execution_service.get_filled_orders = Mock(return_value=mock_fills)

            # Run backtest
            result = engine.run()

            # Verify results collected from services
            assert result.final_capital == 125000.50
            assert result.num_trades == 3
            assert result.total_return == pytest.approx(0.250005)  # 25% return

    def test_run_handles_errors_gracefully(self, minimal_risk_config):
        """Test that run() handles errors and raises RuntimeError."""
        config_dict = {
            "start_date": "2020-01-02",
            "end_date": "2020-01-05",
            "initial_equity": 100000,
            "warmup_bars": 0,
            "universe": ["AAPL"],
            "data": {
                "source": "algoseek",
                "data_path": "data/",
                "dataset": "algoseek-us-equity-1d-unadjusted",
            },
            "portfolio": {
                "initial_equity": 100000,
                "base_currency": "USD",
                "accounting_method": "average_cost",
            },
            "execution": {
                "fill_model": "close",
                "commission_model": "fixed",
                "slippage_model": "none",
            },
            "risk": minimal_risk_config,
            "strategies": [],
        }

        config = BacktestConfig(**config_dict)

        with patch.object(BacktestEngine, "__init__", lambda self, *args, **kwargs: None):
            engine = BacktestEngine(
                config=config,
                event_bus=Mock(),
                data_service=Mock(),
                portfolio_service=Mock(),
                execution_service=Mock(),
                risk_service=Mock(),
                strategy_service=Mock(),
            )

            # Setup mocks
            engine.config = config
            engine._event_bus = EventBus()
            engine._data_service = Mock()

            # Make stream_universe raise an exception
            engine._data_service.stream_universe = Mock(side_effect=ValueError("Test error"))

            # Verify RuntimeError is raised
            with pytest.raises(RuntimeError, match="Backtest execution failed"):
                engine.run()


class TestBacktestEngineEndToEnd:
    """End-to-end tests with minimal mocking."""

    def test_engine_can_be_created_and_run(self, minimal_risk_config):
        """Test that engine can be created from config and run (with mocked data)."""
        config_dict = {
            "start_date": "2020-01-02",
            "end_date": "2020-01-05",
            "initial_equity": 100000,
            "warmup_bars": 0,
            "universe": ["AAPL"],
            "data": {
                "source": "algoseek",
                "data_path": "data/",
                "dataset": "algoseek-us-equity-1d-unadjusted",
            },
            "portfolio": {
                "initial_equity": 100000,
                "base_currency": "USD",
                "accounting_method": "average_cost",
            },
            "execution": {
                "fill_model": "close",
                "commission_model": "fixed",
                "slippage_model": "none",
            },
            "risk": minimal_risk_config,
            "strategies": [],
        }

        config = BacktestConfig(**config_dict)

        # Create engine from config
        engine = BacktestEngine.from_config(config)

        # Verify engine created
        assert engine is not None
        assert engine.config == config

        # Mock data service to avoid loading real data
        engine._data_service.stream_universe = Mock()
        engine._portfolio_service.get_equity = Mock(return_value=Decimal("100000"))
        engine._execution_service.get_filled_orders = Mock(return_value=[])

        # Run backtest
        result = engine.run()

        # Verify result
        assert isinstance(result, BacktestResult)
        assert result.initial_equity == 100000.0
        assert result.final_capital > 0  # Should have some value
