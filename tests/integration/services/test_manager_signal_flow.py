"""
Integration test for ManagerService signal-to-order flow.

Tests the complete Phase 3 architecture:
- SignalEvent → ManagerService → OrderEvent
- Risk library integration (sizing, limits)
- Audit trail (intent_id, idempotency_key)
- Intention → side mapping
"""

from decimal import Decimal

import pytest

from qtrader.events.event_bus import EventBus
from qtrader.events.events import OrderEvent, PortfolioStateEvent, SignalEvent
from qtrader.services.manager import ManagerService


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus()


@pytest.fixture
def manager_service(event_bus):
    """Create ManagerService with test configuration."""
    config_dict = {
        "name": "naive",  # Use builtin naive policy
        "config": {},  # No overrides
    }
    service = ManagerService.from_config(config_dict, event_bus)

    # Emit initial PortfolioStateEvent so Manager has cached equity
    # (Manager requires this before processing signals)
    portfolio_state = PortfolioStateEvent(
        portfolio_id="test-portfolio",
        start_datetime="2020-01-01T00:00:00Z",
        snapshot_datetime="2020-01-02T00:00:00Z",
        reporting_currency="USD",
        initial_portfolio_equity=Decimal("100000.00"),
        cash_balance=Decimal("100000.00"),
        current_portfolio_equity=Decimal("100000.00"),
        total_market_value=Decimal("0.00"),
        total_unrealized_pl=Decimal("0.00"),
        total_realized_pl=Decimal("0.00"),
        total_pl=Decimal("0.00"),
        long_exposure=Decimal("0.00"),
        short_exposure=Decimal("0.00"),
        net_exposure=Decimal("0.00"),
        gross_exposure=Decimal("0.00"),
        leverage=Decimal("0.00"),
        strategies_groups=[],
    )
    service.on_portfolio_state(portfolio_state)

    return service


class TestManagerServiceSignalToOrder:
    """Test signal processing and order emission."""

    def test_open_long_signal_emits_buy_order(self, manager_service, event_bus):
        """Test OPEN_LONG intention maps to buy order."""
        # Arrange
        orders_received = []

        def capture_order(event: OrderEvent):
            orders_received.append(event)

        event_bus.subscribe("order", capture_order)

        signal = SignalEvent(
            signal_id="sig-001",
            timestamp="2020-01-02T16:00:00Z",
            strategy_id="test_strategy",
            symbol="AAPL",
            intention="OPEN_LONG",
            price=Decimal("150.00"),
            confidence=Decimal("0.80"),
            metadata={"equity": 100000.0},
        )

        # Act
        manager_service.on_signal(signal)

        # Assert
        assert len(orders_received) == 1
        order = orders_received[0]
        assert order.symbol == "AAPL"
        assert order.side == "buy"
        assert order.intent_id == "sig-001"
        assert "test_strategy-sig-001" in order.idempotency_key
        assert order.quantity > 0

    def test_close_long_signal_emits_sell_order(self, manager_service, event_bus):
        """Test CLOSE_LONG intention maps to sell order and uses position size."""
        # Arrange
        orders_received = []

        def capture_order(event: OrderEvent):
            orders_received.append(event)

        event_bus.subscribe("order", capture_order)

        # First, publish portfolio state with a long position
        from qtrader.events import PortfolioPosition, PortfolioStateEvent, StrategyGroup

        portfolio_state = PortfolioStateEvent(
            portfolio_id="test_portfolio",
            start_datetime="2020-01-01T00:00:00Z",
            snapshot_datetime="2020-01-03T15:00:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000"),
            cash_balance=Decimal("85000"),
            current_portfolio_equity=Decimal("101000"),
            total_market_value=Decimal("16000"),
            total_unrealized_pl=Decimal("1000"),
            total_realized_pl=Decimal("0"),
            total_pl=Decimal("1000"),
            long_exposure=Decimal("16000"),
            short_exposure=Decimal("0"),
            net_exposure=Decimal("16000"),
            gross_exposure=Decimal("16000"),
            leverage=Decimal("0.16"),
            strategies_groups=[
                StrategyGroup(
                    strategy_id="test_strategy",
                    positions=[
                        PortfolioPosition(
                            symbol="AAPL",
                            side="long",
                            open_quantity=100,  # Strategy has 100 shares long
                            average_fill_price=Decimal("150.00"),
                            commission_paid=Decimal("10.00"),
                            cost_basis=Decimal("15010.00"),
                            market_price=Decimal("160.00"),
                            gross_market_value=Decimal("16000.00"),
                            unrealized_pl=Decimal("990.00"),
                            realized_pl=Decimal("0"),
                            dividends_received=Decimal("0"),
                            dividends_paid=Decimal("0"),
                            total_position_value=Decimal("16000.00"),
                            currency="USD",
                            last_updated="2020-01-03T15:00:00Z",
                        )
                    ],
                )
            ],
        )
        manager_service.on_portfolio_state(portfolio_state)

        signal = SignalEvent(
            signal_id="sig-002",
            timestamp="2020-01-03T16:00:00Z",
            strategy_id="test_strategy",
            symbol="AAPL",
            intention="CLOSE_LONG",
            price=Decimal("155.00"),
            confidence=Decimal("0.90"),  # 90% confidence = close 90% of position
            metadata={"equity": 100000.0},
        )

        # Act
        manager_service.on_signal(signal)

        # Assert
        assert len(orders_received) == 1
        order = orders_received[0]
        assert order.symbol == "AAPL"
        assert order.side == "sell"
        assert order.intent_id == "sig-002"
        # Quantity should be 90 (90% of 100 shares due to confidence=0.90)
        assert order.quantity == Decimal("90")

    def test_open_short_signal_emits_sell_order(self, manager_service, event_bus):
        """Test OPEN_SHORT intention maps to sell order."""
        # Arrange
        orders_received = []

        def capture_order(event: OrderEvent):
            orders_received.append(event)

        event_bus.subscribe("order", capture_order)
        signal = SignalEvent(
            signal_id="sig-003",
            timestamp="2020-01-04T16:00:00Z",
            strategy_id="test_strategy",
            symbol="TSLA",
            intention="OPEN_SHORT",
            price=Decimal("500.00"),
            confidence=Decimal("0.75"),
            metadata={"equity": 100000.0},
        )

        # Act
        manager_service.on_signal(signal)

        # Assert
        assert len(orders_received) == 1
        order = orders_received[0]
        assert order.symbol == "TSLA"
        assert order.side == "sell"

    def test_close_short_signal_emits_buy_order(self, manager_service, event_bus):
        """Test CLOSE_SHORT intention maps to buy order and uses position size."""
        # Arrange
        orders_received = []

        def capture_order(event: OrderEvent):
            orders_received.append(event)

        event_bus.subscribe("order", capture_order)

        # First, publish portfolio state with a short position
        from qtrader.events import PortfolioPosition, PortfolioStateEvent, StrategyGroup

        portfolio_state = PortfolioStateEvent(
            portfolio_id="test_portfolio",
            start_datetime="2020-01-01T00:00:00Z",
            snapshot_datetime="2020-01-05T15:00:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000"),
            cash_balance=Decimal("150000"),  # Cash increased from short sale
            current_portfolio_equity=Decimal("102000"),
            total_market_value=Decimal("-48000"),  # Short position negative value
            total_unrealized_pl=Decimal("2000"),
            total_realized_pl=Decimal("0"),
            total_pl=Decimal("2000"),
            long_exposure=Decimal("0"),
            short_exposure=Decimal("48000"),
            net_exposure=Decimal("-48000"),
            gross_exposure=Decimal("48000"),
            leverage=Decimal("0.48"),
            strategies_groups=[
                StrategyGroup(
                    strategy_id="test_strategy",
                    positions=[
                        PortfolioPosition(
                            symbol="TSLA",
                            side="short",
                            open_quantity=-100,  # Strategy has 100 shares short
                            average_fill_price=Decimal("500.00"),
                            commission_paid=Decimal("10.00"),
                            cost_basis=Decimal("50010.00"),
                            market_price=Decimal("480.00"),
                            gross_market_value=Decimal("-48000.00"),
                            unrealized_pl=Decimal("1990.00"),
                            realized_pl=Decimal("0"),
                            dividends_received=Decimal("0"),
                            dividends_paid=Decimal("0"),
                            total_position_value=Decimal("-48000.00"),
                            currency="USD",
                            last_updated="2020-01-05T15:00:00Z",
                        )
                    ],
                )
            ],
        )
        manager_service.on_portfolio_state(portfolio_state)

        signal = SignalEvent(
            signal_id="sig-004",
            timestamp="2020-01-05T16:00:00Z",
            strategy_id="test_strategy",
            symbol="TSLA",
            intention="CLOSE_SHORT",
            price=Decimal("480.00"),
            confidence=Decimal("0.85"),  # 85% confidence = close 85% of position
            metadata={"equity": 100000.0},
        )

        # Act
        manager_service.on_signal(signal)

        # Assert
        assert len(orders_received) == 1
        order = orders_received[0]
        assert order.symbol == "TSLA"
        assert order.side == "buy"
        assert order.quantity == Decimal("85")  # 85% of 100 share short position

    def test_signal_without_equity_rejected(self, manager_service, event_bus):
        """Test signal is rejected when no portfolio state cached (equity unknown).

        This tests the scenario where Manager receives a signal before any
        PortfolioStateEvent has been published. In this case, Manager doesn't
        know the current equity and must reject the signal.
        """
        # Arrange - Create fresh ManagerService WITHOUT portfolio state
        config_dict = {
            "name": "naive",
            "config": {},
        }
        fresh_service = ManagerService.from_config(config_dict, event_bus)
        # Note: NO portfolio state emitted

        orders_received = []

        def capture_order(event: OrderEvent):
            orders_received.append(event)

        event_bus.subscribe("order", capture_order)

        signal = SignalEvent(
            signal_id="sig-005",
            timestamp="2020-01-06T16:00:00Z",
            strategy_id="test_strategy",
            symbol="AAPL",
            intention="OPEN_LONG",
            price=Decimal("150.00"),
            confidence=Decimal("0.80"),
            metadata={"some": "data"},
        )

        # Act
        fresh_service.on_signal(signal)

        # Assert
        assert len(orders_received) == 0  # No order emitted (no cached equity)

    def test_zero_confidence_signal_no_order(self, manager_service, event_bus):
        """Test signal with zero confidence results in zero quantity (no order)."""
        # Arrange
        orders_received = []

        def capture_order(event: OrderEvent):
            orders_received.append(event)

        event_bus.subscribe("order", capture_order)

        signal = SignalEvent(
            signal_id="sig-006",
            timestamp="2020-01-07T16:00:00Z",
            strategy_id="test_strategy",
            symbol="AAPL",
            intention="OPEN_LONG",
            price=Decimal("150.00"),
            confidence=Decimal("0.00"),  # Zero confidence
            metadata={"equity": 100000.0},
        )

        # Act
        manager_service.on_signal(signal)

        # Assert
        assert len(orders_received) == 0  # No order (quantity rounds to zero)

    def test_audit_trail_fields_present(self, manager_service, event_bus):
        """Test OrderEvent includes complete audit trail fields."""
        # Arrange
        orders_received = []

        def capture_order(event: OrderEvent):
            orders_received.append(event)

        event_bus.subscribe("order", capture_order)

        signal = SignalEvent(
            signal_id="sig-audit-001",
            timestamp="2020-01-08T16:00:00Z",
            strategy_id="momentum",
            symbol="AAPL",
            intention="OPEN_LONG",
            price=Decimal("150.00"),
            confidence=Decimal("0.80"),
            metadata={"equity": 100000.0},
        )

        # Act
        manager_service.on_signal(signal)

        # Assert
        assert len(orders_received) == 1
        order = orders_received[0]

        # Audit trail fields
        assert order.intent_id == "sig-audit-001"
        assert order.idempotency_key == "momentum-sig-audit-001-2020-01-08T16:00:00Z"
        assert order.timestamp == "2020-01-08T16:00:00Z"
        assert order.source_strategy_id == "momentum"

    def test_position_sizing_with_confidence(self, manager_service, event_bus):
        """Test position size scales with signal confidence."""
        # Arrange
        orders_received = []

        def capture_order(event: OrderEvent):
            orders_received.append(event)

        event_bus.subscribe("order", capture_order)

        # High confidence signal
        signal_high = SignalEvent(
            signal_id="sig-high",
            timestamp="2020-01-09T16:00:00Z",
            strategy_id="test_strategy",
            symbol="AAPL",
            intention="OPEN_LONG",
            price=Decimal("150.00"),
            confidence=Decimal("1.00"),  # Maximum confidence
            metadata={"equity": 100000.0},
        )

        # Act
        manager_service.on_signal(signal_high)

        # Assert
        assert len(orders_received) == 1
        high_quantity = orders_received[0].quantity

        # Reset
        orders_received.clear()

        # Low confidence signal (same symbol, price, equity)
        signal_low = SignalEvent(
            signal_id="sig-low",
            timestamp="2020-01-10T16:00:00Z",
            strategy_id="test_strategy",
            symbol="AAPL",
            intention="OPEN_LONG",
            price=Decimal("150.00"),
            confidence=Decimal("0.50"),  # Half confidence
            metadata={"equity": 100000.0},
        )

        manager_service.on_signal(signal_low)

        assert len(orders_received) == 1
        low_quantity = orders_received[0].quantity

        # Higher confidence → larger position
        assert high_quantity > low_quantity
