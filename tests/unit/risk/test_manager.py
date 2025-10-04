"""Tests for RiskManager."""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.models.order import OrderSide, OrderState
from qtrader.models.portfolio import Portfolio
from qtrader.risk.manager import RiskDecision, RiskManager
from qtrader.risk.policy import RiskPolicy, SizingMethod
from qtrader.risk.signal import Signal, SignalDirection, SignalType


@pytest.fixture
def portfolio():
    """Create portfolio with $100,000 cash."""
    return Portfolio(initial_cash=Decimal("100000.00"))


@pytest.fixture
def policy():
    """Create default risk policy."""
    return RiskPolicy(
        sizing_method=SizingMethod.PORTFOLIO_PERCENT,
        default_position_size=Decimal("0.05"),  # 5% per position
        max_position_pct=Decimal("0.20"),  # Max 20% per position
        max_positions=10,
        allow_shorting=False,
        cash_reserve_pct=Decimal("0.05"),  # Keep 5% cash reserve
    )


@pytest.fixture
def signal_long():
    """Create basic LONG entry signal."""
    return Signal(
        signal_id="sig-001",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
    )


def test_risk_manager_initialization(portfolio, policy):
    """Test RiskManager initialization."""
    manager = RiskManager(policy=policy, portfolio=portfolio)

    assert manager.policy == policy
    assert manager.portfolio == portfolio
    assert manager.signal_count == 0
    assert manager.approved_count == 0
    assert manager.rejected_count == 0


def test_approve_valid_signal(portfolio, policy, signal_long):
    """Test RiskManager approves valid signal."""
    manager = RiskManager(policy=policy, portfolio=portfolio)
    current_price = Decimal("100.00")

    decision = manager.evaluate_signal(signal_long, current_price)

    assert decision.approved is True
    assert decision.sized_qty > 0
    assert decision.reason == "Signal approved"
    assert manager.approved_count == 1
    assert manager.rejected_count == 0


def test_reject_invalid_signal(portfolio, policy):
    """Test RiskManager rejects invalid signal."""
    manager = RiskManager(policy=policy, portfolio=portfolio)

    # Invalid signal (conviction out of range)
    signal = Signal(
        signal_id="sig-002",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        conviction=Decimal("1.5"),  # Invalid
    )

    decision = manager.evaluate_signal(signal, Decimal("100.00"))

    assert decision.approved is False
    assert decision.sized_qty == 0
    assert "validation failed" in decision.reason.lower()
    assert manager.rejected_count == 1


def test_reject_short_when_not_allowed(portfolio, policy):
    """Test RiskManager rejects SHORT signal when shorting disabled."""
    manager = RiskManager(policy=policy, portfolio=portfolio)

    signal = Signal(
        signal_id="sig-003",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_SHORT,
        direction=SignalDirection.SHORT,
    )

    decision = manager.evaluate_signal(signal, Decimal("100.00"))

    assert decision.approved is False
    assert "Shorting not allowed" in decision.reason
    assert manager.rejected_count == 1


def test_apply_max_position_pct_limit(portfolio, policy, signal_long):
    """Test RiskManager applies max_position_pct limit."""
    manager = RiskManager(policy=policy, portfolio=portfolio)
    current_price = Decimal("100.00")

    # Request 30% but max is 20%
    signal = signal_long._replace(target_weight=Decimal("0.30"))

    decision = manager.evaluate_signal(signal, current_price)

    assert decision.approved is True
    # Position value should be capped at 20% of equity
    max_value = portfolio.get_equity() * Decimal("0.20")
    max_qty = int(max_value / current_price)
    assert decision.sized_qty == max_qty
    assert "max_position_pct" in decision.applied_limits[0]


def test_reject_when_max_positions_reached(portfolio, policy, signal_long):
    """Test RiskManager rejects signal when max positions reached."""
    # Set max_positions to 1
    policy_limited = policy._replace(max_positions=1)
    manager = RiskManager(policy=policy_limited, portfolio=portfolio)

    # First signal should be approved
    decision1 = manager.evaluate_signal(signal_long, Decimal("100.00"))
    assert decision1.approved is True

    # Add a position to portfolio by simulating a fill
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=decision1.sized_qty,
        fill_price=Decimal("100.00"),
        commission=Decimal("1.00"),
        ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        order_id="ord-001",
        fill_id="fill-001",
    )

    # Second signal for different symbol should be rejected
    signal2 = Signal(
        signal_id="sig-002",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="MSFT",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
    )

    decision2 = manager.evaluate_signal(signal2, Decimal("200.00"))

    assert decision2.approved is False
    assert "max_positions" in decision2.reason.lower() or "concentration" in decision2.reason.lower()


def test_reject_insufficient_cash(portfolio, signal_long):
    """Test RiskManager rejects signal when insufficient cash."""
    # Use FIXED_QUANTITY sizing with a huge quantity
    # Set max_position_pct very high so concentration doesn't interfere
    policy_fixed_qty = RiskPolicy(
        sizing_method=SizingMethod.FIXED_QUANTITY,
        default_position_size=Decimal("2000"),  # 2000 shares
        max_position_pct=Decimal("1.0"),  # Allow 100% (no concentration limit)
        cash_reserve_pct=Decimal("0.05"),  # Need 5% reserve
        reject_on_insufficient_cash=True,
    )
    manager = RiskManager(policy=policy_fixed_qty, portfolio=portfolio)

    # Portfolio has 100,000 equity, requesting 2000 shares @ $100 = $200,000
    # After 5% reserve, only 95,000 available
    # Request 200,000 > 95,000 available
    current_price = Decimal("100.00")

    decision = manager.evaluate_signal(signal_long, current_price)

    assert decision.approved is False
    assert "Insufficient cash" in decision.reason


def test_signal_to_order_conversion(portfolio, policy, signal_long):
    """Test signal_to_order converts approved signal to Order."""
    manager = RiskManager(policy=policy, portfolio=portfolio)
    current_price = Decimal("100.00")

    decision = manager.evaluate_signal(signal_long, current_price)
    assert decision.approved is True

    order = manager.signal_to_order(signal_long, decision, current_price)

    assert order.order_id == f"ord-{signal_long.signal_id}"
    assert order.symbol == signal_long.symbol
    assert order.side == OrderSide.BUY
    assert order.qty == decision.sized_qty
    assert order.order_type == signal_long.order_type
    assert order.state == OrderState.SUBMITTED


def test_signal_to_order_rejects_rejected_signal(portfolio, policy):
    """Test signal_to_order raises error for rejected signal."""
    manager = RiskManager(policy=policy, portfolio=portfolio)

    # Create rejected decision
    decision = RiskDecision(
        approved=False,
        sized_qty=0,
        reason="Test rejection",
    )

    signal = Signal(
        signal_id="sig-004",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
    )

    with pytest.raises(ValueError, match="Cannot convert rejected signal"):
        manager.signal_to_order(signal, decision, Decimal("100.00"))


def test_short_signal_creates_sell_order(portfolio, signal_long):
    """Test SHORT signal creates SELL order."""
    policy_short = RiskPolicy(allow_shorting=True)
    manager = RiskManager(policy=policy_short, portfolio=portfolio)

    signal = Signal(
        signal_id="sig-005",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_SHORT,
        direction=SignalDirection.SHORT,
        target_qty=100,
    )

    decision = manager.evaluate_signal(signal, Decimal("100.00"))
    assert decision.approved is True

    order = manager.signal_to_order(signal, decision, Decimal("100.00"))

    assert order.side == OrderSide.SELL


def test_get_stats(portfolio, policy, signal_long):
    """Test get_stats returns correct statistics."""
    manager = RiskManager(policy=policy, portfolio=portfolio)

    # Approve one signal
    manager.evaluate_signal(signal_long, Decimal("100.00"))

    # Reject one signal (shorting not allowed)
    signal_short = Signal(
        signal_id="sig-006",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_SHORT,
        direction=SignalDirection.SHORT,
    )
    manager.evaluate_signal(signal_short, Decimal("100.00"))

    stats = manager.get_stats()

    assert stats["signals_total"] == 2
    assert stats["signals_approved"] == 1
    assert stats["signals_rejected"] == 1
    assert stats["approval_rate"] == "50.0%"


def test_zero_qty_rejection(portfolio, policy):
    """Test RiskManager rejects when calculated size is zero."""
    # Use tiny default position size
    policy_tiny = policy._replace(default_position_size=Decimal("0.00001"))
    manager = RiskManager(policy=policy_tiny, portfolio=portfolio)

    signal = Signal(
        signal_id="sig-007",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
    )

    # Very high price will result in zero shares
    decision = manager.evaluate_signal(signal, Decimal("1000000.00"))

    assert decision.approved is False
    assert "zero or negative" in decision.reason.lower()
