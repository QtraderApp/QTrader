"""Tests for Order models."""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.models.order import Fill, Order, OrderSide, OrderState, OrderType, TimeInForce


@pytest.fixture
def strategy_ts():
    """Fixed strategy timestamp."""
    return datetime(2023, 1, 15, 9, 30, tzinfo=pytz.timezone("America/New_York"))


def test_market_order_creation(strategy_ts):
    """Market order should be created with correct defaults."""
    order = Order(
        order_id="O001",
        strategy_ts=strategy_ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        state=OrderState.SUBMITTED,
    )

    assert order.order_id == "O001"
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.qty == 100
    assert order.order_type == OrderType.MARKET
    assert order.state == OrderState.SUBMITTED
    assert order.remaining_qty == 100  # Auto-set for SUBMITTED orders
    assert order.filled_qty == 0
    assert order.tif == TimeInForce.DAY  # Default
    assert order.limit_price is None
    assert order.stop_price is None


def test_limit_order_requires_limit_price(strategy_ts):
    """LIMIT orders must have limit_price."""
    with pytest.raises(ValueError, match="LIMIT orders must have limit_price"):
        Order(
            order_id="O002",
            strategy_ts=strategy_ts,
            symbol="MSFT",
            side=OrderSide.BUY,
            qty=50,
            order_type=OrderType.LIMIT,
            state=OrderState.SUBMITTED,
            # Missing limit_price
        )


def test_limit_order_creation(strategy_ts):
    """LIMIT order should be created with limit_price."""
    order = Order(
        order_id="O002",
        strategy_ts=strategy_ts,
        symbol="MSFT",
        side=OrderSide.BUY,
        qty=50,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("250.00"),
    )

    assert order.limit_price == Decimal("250.00")
    assert order.stop_price is None


def test_stop_order_requires_stop_price(strategy_ts):
    """STOP orders must have stop_price."""
    with pytest.raises(ValueError, match="STOP orders must have stop_price"):
        Order(
            order_id="O003",
            strategy_ts=strategy_ts,
            symbol="AMZN",
            side=OrderSide.SELL,
            qty=25,
            order_type=OrderType.STOP,
            state=OrderState.SUBMITTED,
            # Missing stop_price
        )


def test_stop_order_creation(strategy_ts):
    """STOP order should be created with stop_price."""
    order = Order(
        order_id="O003",
        strategy_ts=strategy_ts,
        symbol="AMZN",
        side=OrderSide.SELL,
        qty=25,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("3000.00"),
    )

    assert order.stop_price == Decimal("3000.00")
    assert order.limit_price is None


def test_order_qty_must_be_positive(strategy_ts):
    """Order qty must be > 0."""
    with pytest.raises(ValueError, match="Order qty must be > 0"):
        Order(
            order_id="O004",
            strategy_ts=strategy_ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=0,  # Invalid
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
        )


def test_order_state_transition(strategy_ts):
    """Order state should transition correctly."""
    order = Order(
        order_id="O001",
        strategy_ts=strategy_ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        state=OrderState.SUBMITTED,
    )

    # Transition to FILLED
    filled_order = order.with_state(OrderState.FILLED)
    assert filled_order.state == OrderState.FILLED
    assert filled_order.order_id == order.order_id  # Other fields unchanged
    assert order.state == OrderState.SUBMITTED  # Original unchanged (immutable)


def test_order_partial_fill(strategy_ts):
    """Order should track partial fills correctly."""
    order = Order(
        order_id="O001",
        strategy_ts=strategy_ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=1000,
        order_type=OrderType.MARKET,
        state=OrderState.SUBMITTED,
    )

    # First partial: 400 @ $150
    order1 = order.with_partial_fill(fill_qty=400, fill_price=Decimal("150.00"), remaining=600)
    assert order1.filled_qty == 400
    assert order1.remaining_qty == 600
    assert order1.avg_fill_price == Decimal("150.00")
    assert order1.state == OrderState.PARTIALLY_FILLED

    # Second partial: 300 @ $151
    order2 = order1.with_partial_fill(fill_qty=300, fill_price=Decimal("151.00"), remaining=300)
    assert order2.filled_qty == 700
    assert order2.remaining_qty == 300
    # Weighted avg: (400*150 + 300*151) / 700 = 105300 / 700 = 150.428...
    expected_avg = (Decimal("150.00") * 400 + Decimal("151.00") * 300) / 700
    assert order2.avg_fill_price == expected_avg
    assert order2.state == OrderState.PARTIALLY_FILLED

    # Final fill: 300 @ $152
    order3 = order2.with_partial_fill(fill_qty=300, fill_price=Decimal("152.00"), remaining=0)
    assert order3.filled_qty == 1000
    assert order3.remaining_qty == 0
    assert order3.state == OrderState.FILLED  # Auto-transitioned


def test_order_is_terminal():
    """Terminal states should be identified correctly."""
    strategy_ts = datetime(2023, 1, 15, tzinfo=pytz.UTC)

    # Non-terminal states
    submitted = Order("O1", strategy_ts, "AAPL", OrderSide.BUY, 100, OrderType.MARKET, OrderState.SUBMITTED)
    assert not submitted.is_terminal()

    partial = submitted.with_state(OrderState.PARTIALLY_FILLED)
    assert not partial.is_terminal()

    # Terminal states
    filled = submitted.with_state(OrderState.FILLED)
    assert filled.is_terminal()

    expired = submitted.with_state(OrderState.EXPIRED)
    assert expired.is_terminal()

    canceled = submitted.with_state(OrderState.CANCELED)
    assert canceled.is_terminal()


def test_order_is_fillable():
    """Fillable states should be identified correctly."""
    strategy_ts = datetime(2023, 1, 15, tzinfo=pytz.UTC)
    order = Order("O1", strategy_ts, "AAPL", OrderSide.BUY, 100, OrderType.MARKET, OrderState.SUBMITTED)

    # Fillable states
    assert order.is_fillable()

    triggered = order.with_state(OrderState.TRIGGERED)
    assert triggered.is_fillable()

    partial = order.with_state(OrderState.PARTIALLY_FILLED)
    assert partial.is_fillable()

    # Non-fillable (terminal) states
    filled = order.with_state(OrderState.FILLED)
    assert not filled.is_fillable()

    expired = order.with_state(OrderState.EXPIRED)
    assert not expired.is_fillable()


def test_fill_creation():
    """Fill should be created correctly."""
    execution_ts = datetime(2023, 1, 15, 9, 31, tzinfo=pytz.UTC)

    fill = Fill(
        fill_id="F001",
        order_id="O001",
        execution_ts=execution_ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        price=Decimal("150.00"),
        slippage_bps=0,
        fees=Decimal("0.50"),
        participation=0.01,
        partial_index=0,
    )

    assert fill.fill_id == "F001"
    assert fill.order_id == "O001"
    assert fill.symbol == "AAPL"
    assert fill.side == OrderSide.BUY
    assert fill.qty == 100
    assert fill.price == Decimal("150.00")
    assert fill.fees == Decimal("0.50")


def test_fill_gross_value():
    """Fill gross value should be calculated correctly."""
    execution_ts = datetime(2023, 1, 15, tzinfo=pytz.UTC)

    fill = Fill(
        "F001",
        "O001",
        execution_ts,
        "AAPL",
        OrderSide.BUY,
        100,
        Decimal("150.00"),
        0,
        Decimal("0.50"),
        0.01,
        0,
    )

    # Gross value = price * qty
    assert fill.gross_value() == Decimal("15000.00")


def test_fill_net_value_buy():
    """Fill net value for BUY should be negative (cash outflow)."""
    execution_ts = datetime(2023, 1, 15, tzinfo=pytz.UTC)

    fill = Fill(
        "F001",
        "O001",
        execution_ts,
        "AAPL",
        OrderSide.BUY,
        100,
        Decimal("150.00"),
        0,
        Decimal("0.50"),
        0.01,
        0,
    )

    # Net value = -(gross + fees) for BUY
    # = -(15000.00 + 0.50) = -15000.50
    assert fill.net_value() == Decimal("-15000.50")


def test_fill_net_value_sell():
    """Fill net value for SELL should be positive (cash inflow)."""
    execution_ts = datetime(2023, 1, 15, tzinfo=pytz.UTC)

    fill = Fill(
        "F001",
        "O001",
        execution_ts,
        "AAPL",
        OrderSide.SELL,
        100,
        Decimal("150.00"),
        0,
        Decimal("0.50"),
        0.01,
        0,
    )

    # Net value = gross - fees for SELL
    # = 15000.00 - 0.50 = 14999.50
    assert fill.net_value() == Decimal("14999.50")


def test_moc_order_creation(strategy_ts):
    """Market-on-close order should be created correctly."""
    order = Order(
        order_id="O005",
        strategy_ts=strategy_ts,
        symbol="MSFT",
        side=OrderSide.SELL,
        qty=200,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
        tif=TimeInForce.IOC,  # MOC is effectively IOC
    )

    assert order.order_type == OrderType.MARKET_ON_CLOSE
    assert order.tif == TimeInForce.IOC
