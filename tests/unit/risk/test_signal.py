"""Tests for Signal model."""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.models.order import OrderType, TimeInForce
from qtrader.risk.signal import Signal, SignalDirection, SignalType


def test_signal_creation():
    """Test basic signal creation."""
    signal = Signal(
        signal_id="sig-001",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
    )

    assert signal.signal_id == "sig-001"
    assert signal.symbol == "AAPL"
    assert signal.signal_type == SignalType.ENTRY_LONG
    assert signal.direction == SignalDirection.LONG
    assert signal.conviction == Decimal("1.0")
    assert signal.urgency == "normal"


def test_signal_with_sizing_hints():
    """Test signal with various sizing hints."""
    signal = Signal(
        signal_id="sig-002",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="MSFT",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_qty=100,
        target_weight=Decimal("0.10"),
        target_value=Decimal("10000.00"),
    )

    assert signal.target_qty == 100
    assert signal.target_weight == Decimal("0.10")
    assert signal.target_value == Decimal("10000.00")


def test_signal_validation_conviction_range():
    """Test signal validation enforces conviction 0.0-1.0."""
    signal = Signal(
        signal_id="sig-003",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        conviction=Decimal("1.5"),  # Invalid
    )

    with pytest.raises(ValueError, match="Conviction must be 0.0-1.0"):
        signal.validate()


def test_signal_validation_urgency():
    """Test signal validation enforces valid urgency values."""
    signal = Signal(
        signal_id="sig-004",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        urgency="urgent",  # Invalid
    )

    with pytest.raises(ValueError, match="Urgency must be"):
        signal.validate()


def test_signal_validation_type_direction_consistency():
    """Test signal validation enforces type/direction consistency."""
    # ENTRY_LONG must have LONG direction
    signal = Signal(
        signal_id="sig-005",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.SHORT,  # Invalid
    )

    with pytest.raises(ValueError, match="ENTRY_LONG requires LONG direction"):
        signal.validate()


def test_signal_validation_exit_requires_flat():
    """Test exit signals must have FLAT direction."""
    signal = Signal(
        signal_id="sig-006",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.EXIT_LONG,
        direction=SignalDirection.LONG,  # Invalid
    )

    with pytest.raises(ValueError, match="EXIT signals should have FLAT direction"):
        signal.validate()


def test_signal_validation_limit_order_requires_price():
    """Test LIMIT orders require limit_price."""
    signal = Signal(
        signal_id="sig-007",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        order_type=OrderType.LIMIT,
        # Missing limit_price
    )

    with pytest.raises(ValueError, match="LIMIT orders require limit_price"):
        signal.validate()


def test_signal_validation_stop_order_requires_price():
    """Test STOP orders require stop_price."""
    signal = Signal(
        signal_id="sig-008",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        order_type=OrderType.STOP,
        # Missing stop_price
    )

    with pytest.raises(ValueError, match="STOP orders require stop_price"):
        signal.validate()


def test_signal_validation_positive_quantities():
    """Test signal validation enforces positive quantities."""
    signal = Signal(
        signal_id="sig-009",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_qty=-100,  # Invalid
    )

    with pytest.raises(ValueError, match="target_qty must be positive"):
        signal.validate()


def test_signal_with_all_fields():
    """Test signal with all optional fields populated."""
    signal = Signal(
        signal_id="sig-010",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_qty=100,
        target_weight=Decimal("0.10"),
        target_value=Decimal("15000.00"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00"),
        stop_price=Decimal("145.00"),
        tif=TimeInForce.GTC,
        conviction=Decimal("0.8"),
        urgency="high",
        metadata={"reason": "breakout", "confidence": 0.85},
    )

    signal.validate()  # Should not raise

    assert signal.target_qty == 100
    assert signal.order_type == OrderType.LIMIT
    assert signal.limit_price == Decimal("150.00")
    assert signal.conviction == Decimal("0.8")
    assert signal.urgency == "high"
    assert signal.metadata["reason"] == "breakout"
