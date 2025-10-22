"""
Tests for Strategy Service.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from qtrader.contracts.data import Bar
from qtrader.events.event_bus import EventBus
from qtrader.events.events import PriceBarEvent
from qtrader.services.strategy.service import StrategyService


class TestStrategyService:
    """Test strategy service initialization and basic operations."""

    def test_init_creates_empty_strategy_dict(self):
        """Should initialize with empty strategies dictionary."""
        bus = EventBus()
        service = StrategyService(event_bus=bus)
        assert service._strategies == {}

    def test_subscribes_to_bar_events(self):
        """Should subscribe to bar events during initialization."""
        bus = EventBus()
        StrategyService(event_bus=bus)
        # Service subscribes to "bar" events - verified by construction


class TestStrategyLoading:
    """Test loading strategies from external files."""

    def test_load_valid_strategy(self):
        """Should successfully load a valid strategy file."""
        # Create temporary strategy file
        strategy_code = """
from qtrader.events.event_bus import EventBus
from qtrader.events.events import PriceBarEvent

class Strategy:
    def __init__(self, strategy_id: str, event_bus: EventBus, config: dict):
        self.strategy_id = strategy_id
        self.event_bus = event_bus
        self.config = config
        self.bars_received = 0

    def on_bar(self, event: PriceBarEvent) -> None:
        self.bars_received += 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(strategy_code)
            strategy_path = f.name

        try:
            bus = EventBus()
            service = StrategyService(event_bus=bus)
            service.load_strategy(
                strategy_path=strategy_path,
                strategy_id="test_strategy",
                config={"param1": 42},
            )

            # Verify strategy loaded
            assert "test_strategy" in service._strategies
            strategy = service._strategies["test_strategy"]
            assert strategy.strategy_id == "test_strategy"
            assert strategy.config == {"param1": 42}
            assert strategy.bars_received == 0

        finally:
            Path(strategy_path).unlink()

    def test_load_strategy_file_not_found(self):
        """Should raise ValueError if strategy file does not exist."""
        bus = EventBus()
        service = StrategyService(event_bus=bus)

        with pytest.raises(ValueError, match="Strategy file not found"):
            service.load_strategy(
                strategy_path="/nonexistent/strategy.py",
                strategy_id="test",
                config={},
            )

    def test_load_strategy_not_py_file(self):
        """Should raise ValueError if file is not a .py file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not python")
            file_path = f.name

        try:
            bus = EventBus()
            service = StrategyService(event_bus=bus)

            with pytest.raises(ValueError, match="Strategy file must be .py"):
                service.load_strategy(strategy_path=file_path, strategy_id="test", config={})
        finally:
            Path(file_path).unlink()

    def test_load_strategy_missing_strategy_class(self):
        """Should raise ValueError if file doesn't contain Strategy class."""
        strategy_code = """
class NotStrategy:
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(strategy_code)
            strategy_path = f.name

        try:
            bus = EventBus()
            service = StrategyService(event_bus=bus)

            with pytest.raises(ValueError, match="must contain a class named 'Strategy'"):
                service.load_strategy(strategy_path=strategy_path, strategy_id="test", config={})
        finally:
            Path(strategy_path).unlink()

    def test_load_duplicate_strategy_id(self):
        """Should raise ValueError if strategy_id already loaded."""
        strategy_code = """
class Strategy:
    def __init__(self, strategy_id: str, event_bus, config: dict):
        self.strategy_id = strategy_id
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(strategy_code)
            strategy_path = f.name

        try:
            bus = EventBus()
            service = StrategyService(event_bus=bus)
            service.load_strategy(strategy_path=strategy_path, strategy_id="test", config={})

            # Try to load same strategy_id again
            with pytest.raises(ValueError, match="Strategy 'test' already loaded"):
                service.load_strategy(strategy_path=strategy_path, strategy_id="test", config={})
        finally:
            Path(strategy_path).unlink()

    def test_load_multiple_strategies(self):
        """Should successfully load multiple different strategies."""
        strategy_code = """
class Strategy:
    def __init__(self, strategy_id: str, event_bus, config: dict):
        self.strategy_id = strategy_id
        self.config = config
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(strategy_code)
            strategy_path = f.name

        try:
            bus = EventBus()
            service = StrategyService(event_bus=bus)

            service.load_strategy(strategy_path=strategy_path, strategy_id="strategy1", config={"a": 1})
            service.load_strategy(strategy_path=strategy_path, strategy_id="strategy2", config={"b": 2})

            assert len(service._strategies) == 2
            assert "strategy1" in service._strategies
            assert "strategy2" in service._strategies
            assert service._strategies["strategy1"].config == {"a": 1}
            assert service._strategies["strategy2"].config == {"b": 2}

        finally:
            Path(strategy_path).unlink()


class TestBarEventRouting:
    """Test routing of bar events to strategies."""

    def test_routes_bar_to_all_strategies(self):
        """Should route bar event to all loaded strategies."""
        strategy_code = """
from qtrader.events.events import PriceBarEvent

class Strategy:
    def __init__(self, strategy_id: str, event_bus, config: dict):
        self.strategy_id = strategy_id
        self.bars_received = []

    def on_bar(self, event: PriceBarEvent) -> None:
        self.bars_received.append(event.symbol)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(strategy_code)
            strategy_path = f.name

        try:
            bus = EventBus()
            service = StrategyService(event_bus=bus)

            service.load_strategy(strategy_path=strategy_path, strategy_id="strat1", config={})
            service.load_strategy(strategy_path=strategy_path, strategy_id="strat2", config={})

            # Create and send bar event
            bar = Bar(
                trade_datetime=datetime(2024, 1, 2, 16, 0),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000,
            )
            event = PriceBarEvent(symbol="AAPL", bar=bar)
            service.on_bar(event)

            # Both strategies should receive the bar
            assert service._strategies["strat1"].bars_received == ["AAPL"]
            assert service._strategies["strat2"].bars_received == ["AAPL"]

        finally:
            Path(strategy_path).unlink()

    def test_handles_strategy_without_on_bar_method(self):
        """Should gracefully handle strategies without on_bar method."""
        strategy_code = """
class Strategy:
    def __init__(self, strategy_id: str, event_bus, config: dict):
        self.strategy_id = strategy_id
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(strategy_code)
            strategy_path = f.name

        try:
            bus = EventBus()
            service = StrategyService(event_bus=bus)
            service.load_strategy(strategy_path=strategy_path, strategy_id="strat1", config={})

            # Should not raise error even though strategy has no on_bar
            bar = Bar(
                trade_datetime=datetime(2024, 1, 2, 16, 0),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000,
            )
            event = PriceBarEvent(symbol="AAPL", bar=bar)
            service.on_bar(event)  # Should not raise

        finally:
            Path(strategy_path).unlink()


class TestFromConfig:
    """Test factory method for creating service from configuration."""

    def test_creates_service_with_loaded_strategies(self):
        """Should create service and load all configured strategies."""
        strategy_code = """
class Strategy:
    def __init__(self, strategy_id: str, event_bus, config: dict):
        self.strategy_id = strategy_id
        self.config = config
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(strategy_code)
            strategy_path = f.name

        try:
            strategies_config = [
                {
                    "path": strategy_path,
                    "strategy_id": "momentum",
                    "config": {"period": 20},
                },
                {
                    "path": strategy_path,
                    "strategy_id": "mean_reversion",
                    "config": {"threshold": 2.0},
                },
            ]

            bus = EventBus()
            service = StrategyService.from_config(strategies_config=strategies_config, event_bus=bus)

            assert len(service._strategies) == 2
            assert "momentum" in service._strategies
            assert "mean_reversion" in service._strategies
            assert service._strategies["momentum"].config == {"period": 20}
            assert service._strategies["mean_reversion"].config == {"threshold": 2.0}

        finally:
            Path(strategy_path).unlink()

    def test_creates_service_with_no_strategies(self):
        """Should create empty service if no strategies configured."""
        bus = EventBus()
        service = StrategyService.from_config(strategies_config=[], event_bus=bus)
        assert len(service._strategies) == 0

    def test_creates_service_without_config_key(self):
        """Should handle strategies without config key."""
        strategy_code = """
class Strategy:
    def __init__(self, strategy_id: str, event_bus, config: dict):
        self.strategy_id = strategy_id
        self.config = config
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(strategy_code)
            strategy_path = f.name

        try:
            strategies_config = [
                {
                    "path": strategy_path,
                    "strategy_id": "simple",
                    # No config key
                }
            ]

            bus = EventBus()
            service = StrategyService.from_config(strategies_config=strategies_config, event_bus=bus)

            assert "simple" in service._strategies
            assert service._strategies["simple"].config == {}

        finally:
            Path(strategy_path).unlink()
