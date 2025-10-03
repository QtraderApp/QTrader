"""Integration tests for ExecutionEngine - tests end-to-end order flow."""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.execution.config import ExecutionConfig
from qtrader.execution.engine import ExecutionEngine
from qtrader.models.bar import Bar
from qtrader.models.order import Order, OrderSide, OrderState, OrderType
from qtrader.models.portfolio import Portfolio

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def engine():
    """Engine with $100k starting capital."""
    portfolio = Portfolio(initial_cash=Decimal("100000.00"))
    config = ExecutionConfig()
    return ExecutionEngine(portfolio=portfolio, config=config)


@pytest.fixture
def aapl_bar():
    """Sample AAPL bar."""
    return Bar(
        ts=datetime(2023, 1, 15, 9, 30, tzinfo=ET),
        symbol="AAPL",
        open=Decimal("150.00"),
        high=Decimal("152.00"),
        low=Decimal("149.00"),
        close=Decimal("151.00"),
        volume=1000000,
    )


@pytest.fixture
def aapl_bar_2():
    """Second AAPL bar."""
    return Bar(
        ts=datetime(2023, 1, 15, 9, 31, tzinfo=ET),
        symbol="AAPL",
        open=Decimal("151.50"),
        high=Decimal("153.00"),
        low=Decimal("150.50"),
        close=Decimal("152.00"),
        volume=1200000,
    )


def test_engine_initialization(engine):
    """Engine initializes with correct values."""
    assert engine.portfolio.cash.get_balance() == Decimal("100000.00")
    assert len(engine.portfolio.positions) == 0
    assert len(engine.pending_orders) == 0


def test_submit_moc_order(engine, aapl_bar):
    """Submitting MOC order adds to pending queue and fills immediately."""
    order = Order(
        order_id="order-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    engine.submit_order(order)
    assert "order-1" in engine.pending_orders

    # Process bar - MOC should fill
    fills = engine.on_bar(aapl_bar)
    assert len(fills) == 1
    assert fills[0].symbol == "AAPL"
    assert fills[0].qty == 100

    # Order moved to filled
    assert "order-1" not in engine.pending_orders
    assert "order-1" in engine.filled_orders


def test_submit_market_order_waits_for_next_bar(engine, aapl_bar, aapl_bar_2):
    """Market orders don't fill on submission bar."""
    order = Order(
        order_id="order-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        state=OrderState.SUBMITTED,
        submission_bar_ts=aapl_bar.ts,
    )

    engine.submit_order(order)

    # First bar: no fill
    fills1 = engine.on_bar(aapl_bar)
    assert len(fills1) == 0
    assert "order-1" in engine.pending_orders

    # Second bar with next_bar: fills
    fills2 = engine.on_bar(aapl_bar_2, next_bar=aapl_bar_2)
    assert len(fills2) == 1
    assert "order-1" not in engine.pending_orders


def test_portfolio_updated_after_fill(engine, aapl_bar):
    """Portfolio should reflect fill."""
    initial_cash = engine.portfolio.cash.get_balance()

    order = Order(
        order_id="order-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    engine.submit_order(order)
    fills = engine.on_bar(aapl_bar)

    # Cash decreased
    assert engine.portfolio.cash.get_balance() < initial_cash

    # Position created
    position = engine.portfolio.positions.get("AAPL")
    assert position is not None
    assert position.qty == 100

    # Cash change = (fill_price * qty) + commission
    fill = fills[0]
    expected_cash = initial_cash - (fill.price * fill.qty) - fill.commission
    assert engine.portfolio.cash.get_balance() == expected_cash


def test_buy_then_sell_round_trip(engine, aapl_bar):
    """Buy then sell should close position."""
    initial_cash = engine.portfolio.cash.get_balance()

    # Buy
    buy_order = Order(
        order_id="buy-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )
    engine.submit_order(buy_order)
    engine.on_bar(aapl_bar)

    # Sell
    sell_order = Order(
        order_id="sell-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )
    engine.submit_order(sell_order)
    engine.on_bar(aapl_bar)

    # Position flat
    position = engine.portfolio.positions.get("AAPL")
    assert position is None or position.qty == 0

    # Lost money due to slippage + commissions
    assert engine.portfolio.cash.get_balance() < initial_cash


def test_short_position(engine, aapl_bar):
    """Can create short positions."""
    order = Order(
        order_id="short-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    engine.submit_order(order)
    engine.on_bar(aapl_bar)

    position = engine.portfolio.positions.get("AAPL")
    assert position is not None
    assert position.qty == -100


def test_multiple_fills_accumulate(engine, aapl_bar):
    """Multiple fills should accumulate position."""
    # First buy
    order1 = Order(
        order_id="buy-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )
    engine.submit_order(order1)
    engine.on_bar(aapl_bar)

    assert engine.portfolio.positions.get("AAPL").qty == 100

    # Second buy
    order2 = Order(
        order_id="buy-2",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=50,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )
    engine.submit_order(order2)
    engine.on_bar(aapl_bar)

    assert engine.portfolio.positions.get("AAPL").qty == 150


def test_commission_deducted_from_cash(engine, aapl_bar):
    """Commission should be deducted from cash."""
    initial_cash = engine.portfolio.cash.get_balance()

    order = Order(
        order_id="order-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    engine.submit_order(order)
    fills = engine.on_bar(aapl_bar)

    fill = fills[0]
    # Cash = initial - (price * qty) - commission
    expected = initial_cash - (fill.price * fill.qty) - fill.commission
    assert engine.portfolio.cash.get_balance() == expected


def test_order_state_transitions(engine, aapl_bar):
    """Orders should transition to FILLED state."""
    order = Order(
        order_id="order-1",
        strategy_ts=aapl_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    engine.submit_order(order)
    engine.on_bar(aapl_bar)

    filled_order = engine.filled_orders.get("order-1")
    assert filled_order is not None
    assert filled_order.state == OrderState.FILLED
    assert filled_order.filled_qty == 100
