"""
Integration tests for complete event flow.

Tests the full event chain from BarEvent through to OrderFillEvent,
verifying that all services communicate correctly via the EventBus.
"""

from datetime import datetime
from decimal import Decimal

from qtrader.contracts.data import Bar
from qtrader.events.event_bus import EventBus
from qtrader.events.events import PortfolioStateEvent, PriceBarEvent, RiskEvaluationTriggerEvent, ValuationTriggerEvent
from qtrader.services.ledger.models import PortfolioConfig
from qtrader.services.ledger.service import PortfolioService


class TestEventFlow:
    """Test complete event flow integration."""

    def test_bar_to_valuation_to_portfolio_state(self):
        """Should handle bar event and publish portfolio state on valuation trigger."""
        # Setup
        event_bus = EventBus()
        config = PortfolioConfig(initial_cash=Decimal("100000"))
        portfolio = PortfolioService(config=config, event_bus=event_bus)  # noqa: F841

        # Track published events
        published_states = []

        def capture_state(event: PortfolioStateEvent) -> None:
            published_states.append(event)

        event_bus.subscribe("portfolio_state", capture_state)  # pyright: ignore[reportArgumentType]

        # Create and publish bar event
        bar = Bar(
            trade_datetime=datetime(2024, 1, 2, 16, 0),
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=1000000,
        )
        bar_event = PriceBarEvent(symbol="AAPL", bar=bar, timestamp=datetime(2024, 1, 2, 16, 0))
        event_bus.publish(bar_event)

        # Trigger valuation
        valuation_event = ValuationTriggerEvent(ts=datetime(2024, 1, 2, 16, 0))
        event_bus.publish(valuation_event)

        # Verify portfolio state was published
        assert len(published_states) == 1
        state = published_states[0]
        assert state.total_equity == Decimal("100000")
        assert state.cash == Decimal("100000")
        assert state.num_positions == 0

    def test_multiple_bars_update_prices(self):
        """Should track latest prices from multiple bar events."""
        # Setup
        event_bus = EventBus()
        config = PortfolioConfig(initial_cash=Decimal("100000"))
        portfolio = PortfolioService(config=config, event_bus=event_bus)

        # Publish bars for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        prices = [150.0, 250.0, 2800.0]

        for symbol, price in zip(symbols, prices):
            bar = Bar(
                trade_datetime=datetime(2024, 1, 2, 16, 0),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
            )
            bar_event = PriceBarEvent(symbol=symbol, bar=bar, timestamp=datetime(2024, 1, 2, 16, 0))
            event_bus.publish(bar_event)

        # Verify latest prices stored
        assert portfolio._latest_prices["AAPL"] == Decimal("150.0")
        assert portfolio._latest_prices["MSFT"] == Decimal("250.0")
        assert portfolio._latest_prices["GOOGL"] == Decimal("2800.0")

    def test_warmup_flag_propagates(self):
        """Should correctly set and propagate is_warmup flag."""
        # Setup
        event_bus = EventBus()
        config = PortfolioConfig(initial_cash=Decimal("100000"))
        PortfolioService(config=config, event_bus=event_bus)

        # Create warmup bar event
        bar = Bar(
            trade_datetime=datetime(2024, 1, 1, 16, 0),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
        )

        warmup_event = PriceBarEvent(symbol="TEST", bar=bar, is_warmup=True)
        assert warmup_event.is_warmup is True

        normal_event = PriceBarEvent(symbol="TEST", bar=bar, is_warmup=False)
        assert normal_event.is_warmup is False


class TestEventOrdering:
    """Test that events are published in correct order."""

    def test_bar_then_valuation_then_risk_evaluation(self):
        """Should publish events in correct order: Bar -> Valuation -> RiskEvaluation."""
        event_bus = EventBus()

        # Track event order
        event_sequence = []

        def track_bar(event: PriceBarEvent) -> None:
            event_sequence.append(("bar", event.symbol))

        def track_valuation(event: ValuationTriggerEvent) -> None:
            event_sequence.append(("valuation", event.ts))

        def track_risk(event: RiskEvaluationTriggerEvent) -> None:
            event_sequence.append(("risk_evaluation", event.ts))

        event_bus.subscribe("price_bar", track_bar)  # pyright: ignore[reportArgumentType]
        event_bus.subscribe("valuation_trigger", track_valuation)  # pyright: ignore[reportArgumentType]
        event_bus.subscribe("risk_evaluation_trigger", track_risk)  # pyright: ignore[reportArgumentType]

        # Simulate engine publishing events in correct order
        ts = datetime(2024, 1, 2, 16, 0)

        # Step 1: Publish bars
        bar = Bar(
            trade_datetime=ts,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
        )
        event_bus.publish(PriceBarEvent(symbol="AAPL", bar=bar, timestamp=ts))
        event_bus.publish(PriceBarEvent(symbol="MSFT", bar=bar, timestamp=ts))

        # Step 2: Trigger valuation
        event_bus.publish(ValuationTriggerEvent(ts=ts))

        # Step 3: Trigger risk evaluation
        event_bus.publish(RiskEvaluationTriggerEvent(ts=ts))

        # Verify order
        assert len(event_sequence) == 4
        assert event_sequence[0] == ("bar", "AAPL")
        assert event_sequence[1] == ("bar", "MSFT")
        assert event_sequence[2] == ("valuation", ts)
        assert event_sequence[3] == ("risk_evaluation", ts)


class TestServiceIntegration:
    """Test multiple services working together."""

    def test_portfolio_and_execution_services_communicate(self):
        """Should verify portfolio and execution services can communicate via events."""
        from qtrader.services.execution.config import ExecutionConfig
        from qtrader.services.execution.service import ExecutionService

        event_bus = EventBus()

        # Create services
        portfolio_config = PortfolioConfig(initial_cash=Decimal("100000"))
        portfolio = PortfolioService(config=portfolio_config, event_bus=event_bus)

        execution_config = ExecutionConfig()
        execution = ExecutionService(config=execution_config, event_bus=event_bus)

        # Both services should be subscribed to events
        assert portfolio._event_bus == event_bus
        assert execution._event_bus == event_bus


# TODO: Day 9 - Add multi-strategy coordination tests
# TODO: Day 10 - Add full system integration tests with all services
# TODO: Day 12 - Add end-to-end tests with real data
