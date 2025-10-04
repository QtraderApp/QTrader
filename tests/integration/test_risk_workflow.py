"""
Integration tests for Risk Management workflow.

Tests end-to-end flows from Signal generation through RiskManager evaluation
to Order creation, including multi-symbol portfolio scenarios.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.models import OrderBase, OrderSide, OrderState, OrderType, Portfolio, TimeInForce
from qtrader.risk import RiskManager, RiskPolicy, Signal, SignalDirection, SignalType, SizingMethod

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def policy_portfolio_pct():
    """Risk policy using portfolio percentage sizing."""
    return RiskPolicy(
        sizing_method=SizingMethod.PORTFOLIO_PERCENT,
        default_position_size=Decimal("0.10"),  # 10% of portfolio per position
        max_position_pct=Decimal("0.15"),  # Max 15% in any single position
        cash_reserve_pct=Decimal("0.05"),  # 5% cash reserve
        max_positions=10,
        allow_shorting=True,
        max_gross_exposure=Decimal("2.0"),  # 200% gross leverage
        max_net_exposure=Decimal("1.0"),  # 100% net leverage
        reject_on_insufficient_cash=True,
    )


@pytest.fixture
def policy_fixed_value():
    """Risk policy using fixed value sizing."""
    return RiskPolicy(
        sizing_method=SizingMethod.FIXED_VALUE,
        default_position_size=Decimal("5000.00"),  # $5,000 per position
        max_position_pct=Decimal("0.20"),  # Max 20% in any position
        cash_reserve_pct=Decimal("0.05"),
        max_positions=20,
        allow_shorting=False,
        reject_on_insufficient_cash=True,
    )


@pytest.fixture
def portfolio_100k():
    """Portfolio with $100,000 initial capital."""
    return Portfolio(initial_cash=Decimal("100000.00"))


@pytest.fixture
def signal_aapl_long():
    """Signal to enter long AAPL position."""
    return Signal(
        signal_id="signal_001",
        strategy_ts=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_qty=None,
        target_weight=None,
        target_value=None,
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_price=None,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.85"),
        urgency="normal",
        metadata={"notes": "Strong momentum signal"},
    )


@pytest.fixture
def signal_msft_long():
    """Signal to enter long MSFT position."""
    return Signal(
        signal_id="signal_002",
        strategy_ts=datetime(2024, 1, 15, 9, 31, 0, tzinfo=timezone.utc),
        symbol="MSFT",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_qty=None,
        target_weight=None,
        target_value=None,
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_price=None,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.75"),
        urgency="normal",
        metadata={"notes": "Breakout signal"},
    )


@pytest.fixture
def signal_googl_short():
    """Signal to enter short GOOGL position."""
    return Signal(
        signal_id="signal_003",
        strategy_ts=datetime(2024, 1, 15, 9, 32, 0, tzinfo=timezone.utc),
        symbol="GOOGL",
        signal_type=SignalType.ENTRY_SHORT,
        direction=SignalDirection.SHORT,
        target_qty=None,
        target_weight=None,
        target_value=None,
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_price=None,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.70"),
        urgency="normal",
        metadata={"notes": "Reversal signal"},
    )


# ============================================================================
# Test 1-3: Signal → Order Workflow
# ============================================================================


def test_signal_to_order_approved_creates_order(portfolio_100k, policy_portfolio_pct, signal_aapl_long):
    """Test that an approved signal creates a valid order."""
    manager = RiskManager(policy=policy_portfolio_pct, portfolio=portfolio_100k)
    current_price = Decimal("150.00")

    # Evaluate signal
    decision = manager.evaluate_signal(signal_aapl_long, current_price)
    assert decision.approved is True
    assert decision.sized_qty > 0

    # Convert to order
    order = manager.signal_to_order(signal_aapl_long, decision, current_price)

    # Verify order properties
    assert isinstance(order, OrderBase)
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.MARKET
    assert order.qty == decision.sized_qty
    assert order.state == OrderState.SUBMITTED
    assert order.limit_price is None  # Market order


def test_signal_to_order_rejected_raises_error(portfolio_100k, policy_portfolio_pct, signal_aapl_long):
    """Test that attempting to create order from rejected signal raises error."""
    # Create policy that doesn't allow shorts
    policy_no_short = RiskPolicy(
        sizing_method=SizingMethod.PORTFOLIO_PERCENT,
        default_position_size=Decimal("0.10"),
        allow_shorting=False,
    )
    manager = RiskManager(policy=policy_no_short, portfolio=portfolio_100k)

    # Create short signal (will be rejected)
    signal_short = Signal(
        signal_id="signal_short",
        strategy_ts=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_SHORT,
        direction=SignalDirection.SHORT,
        target_qty=None,
        target_weight=None,
        target_value=None,
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_price=None,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.80"),
        urgency="normal",
        metadata={"notes": "Short signal"},
    )

    decision = manager.evaluate_signal(signal_short, Decimal("150.00"))
    assert decision.approved is False

    # Should raise error when trying to convert rejected signal
    with pytest.raises(ValueError, match="Cannot convert rejected signal to order"):
        manager.signal_to_order(signal_short, decision, Decimal("150.00"))


def test_multiple_signals_sequential_processing(portfolio_100k, policy_fixed_value, signal_aapl_long, signal_msft_long):
    """Test processing multiple signals sequentially affects portfolio state."""
    manager = RiskManager(policy=policy_fixed_value, portfolio=portfolio_100k)

    # Process first signal (AAPL)
    decision1 = manager.evaluate_signal(signal_aapl_long, Decimal("150.00"))
    assert decision1.approved is True
    _ = manager.signal_to_order(signal_aapl_long, decision1, Decimal("150.00"))

    # Simulate order fill - update portfolio
    portfolio_100k.positions.update_position(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=decision1.sized_qty,
        price=Decimal("150.00"),
    )
    # Reduce cash
    portfolio_100k.cash.debit(
        amount=decision1.sized_qty * Decimal("150.00"),
        timestamp=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc).isoformat(),
        transaction_type="TRADE",
        description="AAPL purchase",
    )

    # Process second signal (MSFT) - portfolio state has changed
    decision2 = manager.evaluate_signal(signal_msft_long, Decimal("300.00"))
    assert decision2.approved is True
    _ = manager.signal_to_order(signal_msft_long, decision2, Decimal("300.00"))

    # Verify stats updated
    stats = manager.get_stats()
    assert stats["signals_total"] == 2
    assert stats["signals_approved"] == 2
    assert stats["signals_rejected"] == 0


# ============================================================================
# Test 4-6: Multi-Symbol Portfolio Scenarios
# ============================================================================


def test_concentration_limit_across_symbols(portfolio_100k, policy_portfolio_pct, signal_aapl_long, signal_msft_long):
    """Test concentration limits are applied per symbol."""
    manager = RiskManager(policy=policy_portfolio_pct, portfolio=portfolio_100k)

    # Both signals should be approved up to concentration limit
    decision_aapl = manager.evaluate_signal(signal_aapl_long, Decimal("150.00"))
    decision_msft = manager.evaluate_signal(signal_msft_long, Decimal("300.00"))

    assert decision_aapl.approved is True
    assert decision_msft.approved is True

    # Calculate expected max position value (15% of $100k = $15,000)
    max_position_value = Decimal("100000.00") * Decimal("0.15")

    # Verify AAPL position doesn't exceed limit
    aapl_value = decision_aapl.sized_qty * Decimal("150.00")
    assert aapl_value <= max_position_value

    # Verify MSFT position doesn't exceed limit
    msft_value = decision_msft.sized_qty * Decimal("300.00")
    assert msft_value <= max_position_value

    # Verify concentration limit was applied if position exceeds default size
    # Note: If default size (10%) doesn't exceed max (15%), no limit is applied
    # This is correct behavior - limits only apply when they're actually breached


def test_max_positions_limit_rejects_new_symbol(portfolio_100k):
    """Test max_positions limit rejects signal when limit reached."""
    # Create policy with max 2 positions
    policy_limited = RiskPolicy(
        sizing_method=SizingMethod.FIXED_VALUE,
        default_position_size=Decimal("1000.00"),
        max_positions=2,
        reject_on_insufficient_cash=False,
    )
    manager = RiskManager(policy=policy_limited, portfolio=portfolio_100k)

    # Add 2 existing positions to portfolio
    portfolio_100k.positions.update_position(symbol="EXISTING1", side=OrderSide.BUY, qty=10, price=Decimal("100.00"))
    portfolio_100k.positions.update_position(symbol="EXISTING2", side=OrderSide.BUY, qty=20, price=Decimal("50.00"))

    # Create signal for new symbol
    signal_new = Signal(
        signal_id="signal_new",
        strategy_ts=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
        symbol="NEWSYMBOL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_qty=None,
        target_weight=None,
        target_value=None,
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_price=None,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.80"),
        urgency="normal",
        metadata={"notes": "New position"},
    )

    # Should reject due to max_positions
    decision = manager.evaluate_signal(signal_new, Decimal("150.00"))
    assert decision.approved is False
    assert "concentration limits" in decision.reason  # Technical error format


def test_cash_depletion_across_multiple_signals(portfolio_100k, signal_aapl_long, signal_msft_long):
    """
    Test that evaluating multiple signals works correctly when cash is depleted.

    This test uses check_cash_before_concentration=False (default behavior) which allows
    concentration limits to reduce position size to fit available cash. This is standard
    risk management practice but may not match multi-strategy allocation expectations.

    For multi-strategy fairness (rejecting signals that can't afford the calculated size),
    set check_cash_before_concentration=True in the policy.

    Note: Due to the interaction between concentration limits (based on equity) and
    cash availability (based on ledger balance), the concentration limit may reduce the
    position size to fit within available cash before the cash check occurs. This is
    conservative risk management behavior - the system prevents oversized positions
    relative to current capital.
    """
    policy = RiskPolicy(
        sizing_method=SizingMethod.FIXED_QUANTITY,
        default_position_size=Decimal("10"),  # Small quantity
        cash_reserve_pct=Decimal("0.0"),
        reject_on_insufficient_cash=True,
        check_cash_before_concentration=False,  # Default: allow concentration to adjust
    )
    manager = RiskManager(policy=policy, portfolio=portfolio_100k)

    # First signal should be approved
    decision1 = manager.evaluate_signal(signal_aapl_long, Decimal("150.00"))
    assert decision1.approved is True
    assert decision1.sized_qty == 10

    # Deplete most cash
    portfolio_100k.cash.debit(
        amount=Decimal("99500.00"),  # Leave only $500
        timestamp=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc).isoformat(),
        transaction_type="TRADE",
        description="Simulate depleted cash",
    )

    # Second signal: 10 shares @ $300 = $3,000 needed
    # Only $500 available, so should be rejected
    decision2 = manager.evaluate_signal(signal_msft_long, Decimal("300.00"))
    assert decision2.approved is False
    # May be rejected by either cash check or concentration limit - both are valid
    assert "Insufficient cash" in decision2.reason or "concentration limits" in decision2.reason


def test_cash_first_check_for_multi_strategy_fairness(portfolio_100k):
    """
    Test cash-first check for multi-strategy allocation fairness.

    When check_cash_before_concentration=True, signals are rejected BEFORE concentration
    adjustment if they can't afford the calculated size. This ensures fair allocation
    across multiple strategies, preventing later strategies from benefiting from
    concentration limits that "rescue" under-funded signals.

    Scenario:
    - 3 strategies, each requesting $40k positions
    - Only $100k available
    - First 2 strategies approved and filled ($80k spent)
    - 3rd strategy rejected due to insufficient cash ($40k needed, $20k available)
    - Even though concentration could reduce it to fit, we reject for fairness
    """
    # Policy with cash-first check enabled
    policy = RiskPolicy(
        sizing_method=SizingMethod.FIXED_VALUE,
        default_position_size=Decimal("40000.00"),  # $40k per position
        max_position_pct=Decimal("0.50"),  # 50% max concentration
        cash_reserve_pct=Decimal("0.0"),
        reject_on_insufficient_cash=True,
        check_cash_before_concentration=True,  # Enable cash-first check
    )

    manager = RiskManager(policy=policy, portfolio=portfolio_100k)

    # Strategy 1: AAPL signal
    signal1 = Signal(
        signal_id="strat1_signal",
        strategy_ts=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_value=Decimal("40000.00"),
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.80"),
    )

    decision1 = manager.evaluate_signal(signal1, Decimal("150.00"))
    assert decision1.approved is True

    # Simulate fill
    portfolio_100k.positions.update_position(
        symbol="AAPL", side=OrderSide.BUY, qty=decision1.sized_qty, price=Decimal("150.00")
    )
    portfolio_100k.cash.debit(
        amount=Decimal(decision1.sized_qty) * Decimal("150.00"),
        timestamp=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc).isoformat(),
        transaction_type="TRADE",
        description="AAPL purchase",
    )

    # Strategy 2: MSFT signal
    signal2 = Signal(
        signal_id="strat2_signal",
        strategy_ts=datetime(2024, 1, 15, 9, 31, 0, tzinfo=timezone.utc),
        symbol="MSFT",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_value=Decimal("40000.00"),
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.80"),
    )

    decision2 = manager.evaluate_signal(signal2, Decimal("300.00"))
    assert decision2.approved is True

    # Simulate fill
    portfolio_100k.positions.update_position(
        symbol="MSFT", side=OrderSide.BUY, qty=decision2.sized_qty, price=Decimal("300.00")
    )
    portfolio_100k.cash.debit(
        amount=Decimal(decision2.sized_qty) * Decimal("300.00"),
        timestamp=datetime(2024, 1, 15, 9, 31, 0, tzinfo=timezone.utc).isoformat(),
        transaction_type="TRADE",
        description="MSFT purchase",
    )

    # Now cash should be ~$20k
    remaining_cash = portfolio_100k.cash.get_balance()
    assert remaining_cash < Decimal("40000.00")

    # Strategy 3: GOOGL signal (should be rejected)
    signal3 = Signal(
        signal_id="strat3_signal",
        strategy_ts=datetime(2024, 1, 15, 9, 32, 0, tzinfo=timezone.utc),
        symbol="GOOGL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_value=Decimal("40000.00"),  # Wants $40k but only ~$20k available
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.80"),
    )

    decision3 = manager.evaluate_signal(signal3, Decimal("140.00"))

    # Should be REJECTED due to insufficient cash (before concentration adjustment)
    assert decision3.approved is False
    assert "Insufficient cash" in decision3.reason

    # Verify stats
    stats = manager.get_stats()
    assert stats["signals_total"] == 3
    assert stats["signals_approved"] == 2
    assert stats["signals_rejected"] == 1


# ============================================================================
# Test 8: Strategy Integration (continued)
# ============================================================================


def test_long_short_portfolio_net_exposure(portfolio_100k, policy_portfolio_pct, signal_aapl_long, signal_googl_short):
    """Test long and short positions contribute to net exposure calculation."""
    manager = RiskManager(policy=policy_portfolio_pct, portfolio=portfolio_100k)

    # Process long signal
    decision_long = manager.evaluate_signal(signal_aapl_long, Decimal("150.00"))
    assert decision_long.approved is True

    # Add long position
    portfolio_100k.positions.update_position(
        symbol="AAPL", side=OrderSide.BUY, qty=decision_long.sized_qty, price=Decimal("150.00")
    )

    # Process short signal
    decision_short = manager.evaluate_signal(signal_googl_short, Decimal("140.00"))
    assert decision_short.approved is True

    # Both should be approved as they offset each other in net exposure
    # Long: ~$15k (15% of $100k)
    # Short: ~$15k (15% of $100k)
    # Net exposure: ~0%
    # Gross exposure: ~30%

    # Verify orders can be created
    order_long = manager.signal_to_order(signal_aapl_long, decision_long, Decimal("150.00"))
    order_short = manager.signal_to_order(signal_googl_short, decision_short, Decimal("140.00"))

    assert order_long.side == OrderSide.BUY
    assert order_short.side == OrderSide.SELL  # Short = SELL


def test_full_workflow_with_limit_order(portfolio_100k, policy_portfolio_pct):
    """Test complete workflow with limit order signal."""
    manager = RiskManager(policy=policy_portfolio_pct, portfolio=portfolio_100k)

    # Create signal with limit order
    signal_limit = Signal(
        signal_id="signal_limit",
        strategy_ts=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_qty=None,
        target_weight=Decimal("0.12"),  # 12% of portfolio
        target_value=None,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("148.50"),  # Limit order at $148.50
        stop_price=None,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.90"),
        urgency="normal",
        metadata={"notes": "Buy on dip"},
    )

    # Validate signal
    signal_limit.validate()

    # Evaluate signal
    current_price = Decimal("150.00")  # Current market price
    decision = manager.evaluate_signal(signal_limit, current_price)
    assert decision.approved is True

    # Convert to order
    order = manager.signal_to_order(signal_limit, decision, current_price)

    # Verify limit order properties
    assert order.order_type == OrderType.LIMIT
    assert order.limit_price == Decimal("148.50")
    assert order.qty == decision.sized_qty

    # Expected quantity: 12% of $100k = $12,000 / $150 = 80 shares
    # But limited by max_position_pct (15%) = $15,000 / $150 = 100 shares
    assert decision.sized_qty <= 100  # Within limit

    # Verify stats
    stats = manager.get_stats()
    assert stats["signals_total"] == 1
    assert stats["signals_approved"] == 1
