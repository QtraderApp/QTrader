"""
Tests for BacktestEngine.
"""

from qtrader.backtest.config import (
    BacktestConfig,
    ConcentrationLimitConfig,
    DataConfig,
    ExecutionConfig,
    LeverageLimitConfig,
    PortfolioConfig,
    RiskConfig,
)
from qtrader.backtest.engine import BacktestEngine, BacktestResult
from qtrader.events.event_bus import EventBus
from qtrader.services.risk.service import RiskService
from qtrader.services.strategy.service import StrategyService


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_can_create_result(self):
        """Should create result with all fields."""
        from datetime import date, timedelta

        result = BacktestResult(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            final_capital=110000.0,
            total_return=0.10,
            num_trades=42,
            duration=timedelta(seconds=5.5),
        )

        assert result.start_date == date(2024, 1, 1)
        assert result.end_date == date(2024, 12, 31)
        assert result.initial_capital == 100000.0
        assert result.final_capital == 110000.0
        assert result.total_return == 0.10
        assert result.num_trades == 42
        assert result.duration == timedelta(seconds=5.5)


class TestBacktestEngineInit:
    """Test BacktestEngine initialization."""

    def test_init_stores_config_and_services(self):
        """Should store configuration and all service references."""
        from datetime import datetime
        from decimal import Decimal
        from unittest.mock import Mock

        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=Decimal("100000"),
            warmup_bars=20,
            universe=["AAPL", "MSFT"],
            data=DataConfig(source="schwab", data_path="/data/test", dataset="schwab-us-equity-1d-adjusted"),
            portfolio=PortfolioConfig(initial_capital=Decimal("100000")),
            risk=RiskConfig(
                budgets=[],
                sizing={},
                concentration=ConcentrationLimitConfig(max_position_pct=0.1),
                leverage=LeverageLimitConfig(max_gross=1.0, max_net=1.0),
            ),
            execution=ExecutionConfig(),
            strategies=[],
        )

        event_bus = EventBus()

        # Create mock services
        data_service = Mock()
        portfolio_service = Mock()
        execution_service = Mock()

        risk_service = RiskService.from_config(config_dict=config.risk.model_dump(), event_bus=event_bus)

        strategy_service = StrategyService(event_bus=event_bus)

        engine = BacktestEngine(
            config=config,
            event_bus=event_bus,
            data_service=data_service,
            portfolio_service=portfolio_service,
            execution_service=execution_service,
            risk_service=risk_service,
            strategy_service=strategy_service,
        )

        assert engine.config == config
        assert engine._event_bus == event_bus
        assert engine._risk_service == risk_service
        assert engine._strategy_service == strategy_service


class TestBacktestEngineFromConfig:
    """Test BacktestEngine.from_config() factory method."""

    def test_creates_engine_with_all_services(self):
        """Should create engine with all services from configuration."""
        from datetime import datetime
        from decimal import Decimal

        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=Decimal("100000"),
            warmup_bars=20,
            universe=["AAPL", "MSFT"],
            data=DataConfig(source="schwab", data_path="/data/test", dataset="schwab-us-equity-1d-adjusted"),
            portfolio=PortfolioConfig(initial_capital=Decimal("100000")),
            risk=RiskConfig(
                budgets=[],
                sizing={},
                concentration=ConcentrationLimitConfig(max_position_pct=0.1),
                leverage=LeverageLimitConfig(max_gross=1.0, max_net=1.0),
            ),
            execution=ExecutionConfig(),
            strategies=[],
        )

        # This should not raise and should create all services
        engine = BacktestEngine.from_config(config)

        assert engine.config == config
        assert engine._event_bus is not None
        assert engine._data_service is not None
        assert engine._portfolio_service is not None
        assert engine._execution_service is not None
        assert engine._risk_service is not None
        assert engine._strategy_service is not None


class TestWarmupPhase:
    """Test warmup phase behavior."""

    def test_price_bar_event_has_is_warmup_field(self):
        """Should support is_warmup field in PriceBarEvent."""
        from datetime import datetime

        from qtrader.contracts.data import Bar
        from qtrader.events.events import PriceBarEvent

        bar = Bar(
            trade_datetime=datetime(2024, 1, 2, 16, 0),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
        )

        # Test warmup event
        warmup_event = PriceBarEvent(symbol="AAPL", bar=bar, is_warmup=True)
        assert warmup_event.is_warmup is True

        # Test normal event
        normal_event = PriceBarEvent(symbol="AAPL", bar=bar, is_warmup=False)
        assert normal_event.is_warmup is False

        # Test default is False
        default_event = PriceBarEvent(symbol="AAPL", bar=bar)
        assert default_event.is_warmup is False


# TODO: Day 5 - Add tests for warmup phase
# TODO: Day 9 - Add tests for main event loop
# TODO: Day 12 - Add end-to-end integration tests
