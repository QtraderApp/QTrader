"""Unit tests for StrategyService - Phase 2 implementation."""

from decimal import Decimal

from qtrader.events.event_bus import EventBus
from qtrader.events.events import PriceBarEvent
from qtrader.libraries.strategies import Context, Strategy, StrategyConfig
from qtrader.services.strategy.service import StrategyService


class MockStrategyConfig(StrategyConfig):
    """Mock strategy config for testing."""

    name: str = "mock_strategy"
    display_name: str = "Mock Strategy"
    warmup_bars: int = 0
    universe: list[str] = []


class MockStrategy(Strategy):
    """Mock strategy for testing."""

    def __init__(self, config: MockStrategyConfig):
        self.config = config
        self.setup_called = False
        self.teardown_called = False
        self.bars_received: list[PriceBarEvent] = []

    def setup(self, context: Context) -> None:
        self.setup_called = True

    def teardown(self, context: Context) -> None:
        self.teardown_called = True

    def on_bar(self, event: PriceBarEvent, context: Context) -> None:
        self.bars_received.append(event)


def test_service_initialization():
    """Service can initialize with strategies."""
    event_bus = EventBus()
    config = MockStrategyConfig()
    strategy = MockStrategy(config)
    service = StrategyService(event_bus=event_bus, strategies={"mock": strategy})
    assert len(service._strategies) == 1


def test_service_routes_bars_to_strategy():
    """Service routes bars to strategies."""
    event_bus = EventBus()
    config = MockStrategyConfig()
    strategy = MockStrategy(config)
    service = StrategyService(event_bus=event_bus, strategies={"mock": strategy})

    event = PriceBarEvent(
        symbol="AAPL",
        timestamp="2024-01-01T10:00:00Z",
        open=Decimal("100.0"),
        high=Decimal("102.0"),
        low=Decimal("99.0"),
        close=Decimal("101.0"),
        volume=1000,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="test",
        interval="1d",
    )

    service.on_bar(event)
    assert len(strategy.bars_received) == 1


def test_universe_filtering():
    """Strategy with universe filters bars."""
    event_bus = EventBus()
    config = MockStrategyConfig(universe=["AAPL"])
    strategy = MockStrategy(config)
    service = StrategyService(event_bus=event_bus, strategies={"mock": strategy})

    aapl_bar = PriceBarEvent(
        symbol="AAPL",
        timestamp="2024-01-01T10:00:00",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1000,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="test",
        interval="1d",
    )

    msft_bar = PriceBarEvent(
        symbol="MSFT",
        timestamp="2024-01-01T10:00:00",
        open=Decimal("200"),
        high=Decimal("201"),
        low=Decimal("199"),
        close=Decimal("200.5"),
        volume=2000,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="test",
        interval="1d",
    )

    service.on_bar(aapl_bar)
    service.on_bar(msft_bar)

    # Only AAPL bar should be received
    assert len(strategy.bars_received) == 1
    assert strategy.bars_received[0].symbol == "AAPL"


def test_lifecycle_methods():
    """Service calls setup and teardown."""
    event_bus = EventBus()
    config = MockStrategyConfig()
    strategy = MockStrategy(config)
    service = StrategyService(event_bus=event_bus, strategies={"mock": strategy})

    service.setup()
    assert strategy.setup_called

    service.teardown()
    assert strategy.teardown_called
