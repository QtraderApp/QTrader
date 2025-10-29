"""Tests for OrderEvent - validates against manager/order.v1.json schema."""

from decimal import Decimal

import pytest

from qtrader.events import OrderEvent


class TestOrderEventCreation:
    """Test OrderEvent creation and initialization."""

    def test_create_with_required_fields_only(self) -> None:
        """OrderEvent can be created with only required fields."""
        # Arrange & Act
        event = OrderEvent(
            order_id="order-001",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
            source_service="manager_service",
        )

        # Assert
        assert event.order_id == "order-001"
        assert event.symbol == "AAPL"
        assert event.side == "buy"
        assert event.quantity == Decimal("100")
        assert event.order_type == "market"
        assert event.time_in_force == "GTC"  # default
        assert event.limit_price is None
        assert event.stop_price is None
        assert event.source_strategy_id is None
        assert event.stop_loss is None
        assert event.take_profit is None

    def test_create_limit_order(self) -> None:
        """OrderEvent can be created as limit order with limit_price."""
        # Arrange & Act
        event = OrderEvent(
            order_id="order-002",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="MSFT",
            side="sell",
            quantity=Decimal("50"),
            order_type="limit",
            limit_price=Decimal("350.00"),
            time_in_force="DAY",
            source_service="manager_service",
        )

        # Assert
        assert event.order_type == "limit"
        assert event.limit_price == Decimal("350.00")
        assert event.time_in_force == "DAY"

    def test_create_stop_order(self) -> None:
        """OrderEvent can be created as stop order with stop_price."""
        # Arrange & Act
        event = OrderEvent(
            order_id="order-003",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="GOOGL",
            side="buy",
            quantity=Decimal("25"),
            order_type="stop",
            stop_price=Decimal("2800.00"),
            source_service="manager_service",
        )

        # Assert
        assert event.order_type == "stop"
        assert event.stop_price == Decimal("2800.00")

    def test_create_with_all_fields(self) -> None:
        """OrderEvent can be created with all optional fields."""
        # Arrange & Act
        event = OrderEvent(
            order_id="order-004",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="stop_limit",
            limit_price=Decimal("145.50"),
            stop_price=Decimal("145.00"),
            time_in_force="GTC",
            source_strategy_id="sma_crossover",
            source_signal_id="signal-abc-123",
            stop_loss=Decimal("140.00"),
            take_profit=Decimal("150.00"),
            source_service="manager_service",
        )

        # Assert
        assert event.order_type == "stop_limit"
        assert event.limit_price == Decimal("145.50")
        assert event.stop_price == Decimal("145.00")
        assert event.source_strategy_id == "sma_crossover"
        assert event.source_signal_id == "signal-abc-123"
        assert event.stop_loss == Decimal("140.00")
        assert event.take_profit == Decimal("150.00")


class TestOrderEventSerialization:
    """Test OrderEvent serialization to wire format."""

    def test_serialize_decimals_to_strings(self) -> None:
        """OrderEvent serializes Decimal fields to strings for wire format."""
        # Arrange
        event = OrderEvent(
            order_id="order-005",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100.5"),
            order_type="limit",
            limit_price=Decimal("145.75"),
            stop_loss=Decimal("140.25"),
            take_profit=Decimal("150.50"),
            source_service="manager_service",
        )

        # Act
        serialized = event.model_dump()

        # Assert - decimals are strings
        assert serialized["quantity"] == "100.5"
        assert serialized["limit_price"] == "145.75"
        assert serialized["stop_loss"] == "140.25"
        assert serialized["take_profit"] == "150.50"

    def test_serialize_null_fields(self) -> None:
        """OrderEvent serializes None fields as null."""
        # Arrange
        event = OrderEvent(
            order_id="order-006",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
            source_service="manager_service",
        )

        # Act
        serialized = event.model_dump()

        # Assert - optional fields are None
        assert serialized["limit_price"] is None
        assert serialized["stop_price"] is None
        assert serialized["source_strategy_id"] is None
        assert serialized["source_signal_id"] is None
        assert serialized["stop_loss"] is None
        assert serialized["take_profit"] is None


class TestOrderEventValidation:
    """Test OrderEvent validation against schema."""

    def test_validates_against_schema(self) -> None:
        """OrderEvent validates payload against manager/order.v1.json."""
        # Arrange & Act - should not raise
        event = OrderEvent(
            order_id="order-007",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="limit",
            limit_price=Decimal("145.50"),
            source_service="manager_service",
        )

        # Assert - validation passed
        assert event.event_type == "order"
        assert event.SCHEMA_BASE == "manager/order"

    def test_invalid_side_fails(self) -> None:
        """OrderEvent with invalid side value fails validation."""
        # Arrange & Act & Assert
        with pytest.raises(Exception):  # ValidationError from schema
            OrderEvent(
                order_id="order-008",
                timestamp="2024-03-15T14:30:15.123Z",
                symbol="AAPL",
                side="invalid_side",
                quantity=Decimal("100"),
                order_type="market",
                source_service="manager_service",
            )

    def test_invalid_order_type_fails(self) -> None:
        """OrderEvent with invalid order_type fails validation."""
        # Arrange & Act & Assert
        with pytest.raises(Exception):  # ValidationError from schema
            OrderEvent(
                order_id="order-009",
                timestamp="2024-03-15T14:30:15.123Z",
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="invalid_type",
                source_service="manager_service",
            )

    def test_invalid_time_in_force_fails(self) -> None:
        """OrderEvent with invalid time_in_force fails validation."""
        # Arrange & Act & Assert
        with pytest.raises(Exception):  # ValidationError from schema
            OrderEvent(
                order_id="order-010",
                timestamp="2024-03-15T14:30:15.123Z",
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
                time_in_force="INVALID",
                source_service="manager_service",
            )

    def test_missing_required_field_fails(self) -> None:
        """OrderEvent without required field fails."""
        # Arrange & Act & Assert
        with pytest.raises(Exception):  # Pydantic ValidationError
            OrderEvent(
                order_id="order-011",
                timestamp="2024-03-15T14:30:15.123Z",
                symbol="AAPL",
                side="buy",
                # Missing quantity - required field
                order_type="market",
                source_service="manager_service",
            )  # type: ignore


class TestOrderEventEnvelope:
    """Test OrderEvent envelope fields."""

    def test_has_envelope_fields(self) -> None:
        """OrderEvent has all envelope fields."""
        # Arrange & Act
        event = OrderEvent(
            order_id="order-012",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
            source_service="manager_service",
        )

        # Assert
        assert hasattr(event, "event_id")
        assert hasattr(event, "event_type")
        assert hasattr(event, "event_version")
        assert hasattr(event, "occurred_at")
        assert hasattr(event, "correlation_id")
        assert hasattr(event, "causation_id")
        assert hasattr(event, "source_service")

    def test_envelope_validates(self) -> None:
        """OrderEvent envelope validates against envelope schema."""
        # Arrange & Act
        event = OrderEvent(
            order_id="order-013",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
            source_service="manager_service",
            correlation_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            causation_id="f1e2d3c4-b5a6-7890-1234-567890abcdef",
        )

        # Assert
        assert event.source_service == "manager_service"
        assert event.correlation_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert event.causation_id == "f1e2d3c4-b5a6-7890-1234-567890abcdef"
        assert event.event_type == "order"
        assert event.event_version == 1


class TestOrderEventImmutability:
    """Test OrderEvent immutability."""

    def test_event_is_frozen(self) -> None:
        """OrderEvent is immutable after creation."""
        # Arrange
        event = OrderEvent(
            order_id="order-014",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
            source_service="manager_service",
        )

        # Act & Assert - should raise ValidationError
        with pytest.raises(Exception):
            event.order_id = "modified"


class TestOrderEventTypes:
    """Test different order types."""

    @pytest.mark.parametrize(
        "order_type",
        ["market", "limit", "stop", "stop_limit"],
    )
    def test_all_order_types_valid(self, order_type: str) -> None:
        """OrderEvent accepts all valid order types."""
        # Arrange & Act
        event = OrderEvent(
            order_id=f"order-{order_type}",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type=order_type,
            limit_price=Decimal("145.00") if order_type in ["limit", "stop_limit"] else None,
            stop_price=Decimal("145.00") if order_type in ["stop", "stop_limit"] else None,
            source_service="manager_service",
        )

        # Assert
        assert event.order_type == order_type

    @pytest.mark.parametrize(
        "side",
        ["buy", "sell"],
    )
    def test_all_sides_valid(self, side: str) -> None:
        """OrderEvent accepts all valid sides."""
        # Arrange & Act
        event = OrderEvent(
            order_id=f"order-{side}",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side=side,
            quantity=Decimal("100"),
            order_type="market",
            source_service="manager_service",
        )

        # Assert
        assert event.side == side

    @pytest.mark.parametrize(
        "time_in_force",
        ["GTC", "DAY", "IOC", "FOK"],
    )
    def test_all_time_in_force_valid(self, time_in_force: str) -> None:
        """OrderEvent accepts all valid time_in_force values."""
        # Arrange & Act
        event = OrderEvent(
            order_id=f"order-{time_in_force}",
            timestamp="2024-03-15T14:30:15.123Z",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
            time_in_force=time_in_force,
            source_service="manager_service",
        )

        # Assert
        assert event.time_in_force == time_in_force
