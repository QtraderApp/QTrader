"""Integration tests for Stage 4: Limit & Stop orders with conservative touch rules."""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.execution.config import ExecutionConfig
from qtrader.execution.engine import ExecutionEngine
from qtrader.execution.fill_policy import FillPolicy
from qtrader.models.bar import Bar
from qtrader.models.order import Order, OrderSide, OrderState, OrderType, TimeInForce
from qtrader.models.portfolio import Portfolio

ET = pytz.timezone("US/Eastern")

# Test timestamps
BAR_TS_1 = datetime(2023, 1, 15, 9, 30, tzinfo=ET)
BAR_TS_DAY1_END = datetime(2023, 1, 15, 16, 0, tzinfo=ET)
BAR_TS_DAY2_START = datetime(2023, 1, 16, 9, 30, tzinfo=ET)
BAR_TS_DAY1_LATER = datetime(2023, 1, 15, 10, 0, tzinfo=ET)


@pytest.fixture
def engine():
    """Engine with $100k starting capital."""
    portfolio = Portfolio(initial_cash=Decimal("100000.00"))
    config = ExecutionConfig()
    return ExecutionEngine(portfolio=portfolio, config=config)


@pytest.fixture
def fill_policy():
    """Fill policy with conservative rules."""
    return FillPolicy(
        moc_slip_bps=5,
        stop_slip_bps=5,
        limit_mode="conservative",
        stop_mode="conservative",
    )


@pytest.fixture
def bar_high():
    """Bar with high price reached."""
    return Bar(
        trade_datetime=BAR_TS_1.isoformat(),
        open=150.00,
        high=155.00,
        low=149.00,
        close=152.00,
        volume=1000000,
    )


@pytest.fixture
def bar_low():
    """Bar with low price reached."""
    return Bar(
        trade_datetime=BAR_TS_1.isoformat(),
        open=150.00,
        high=151.00,
        low=145.00,
        close=148.00,
        volume=1000000,
    )


@pytest.fixture
def bar_no_touch():
    """Bar that doesn't touch limit/stop levels."""
    return Bar(
        trade_datetime=BAR_TS_1.isoformat(),
        open=150.00,
        high=151.00,
        low=149.00,
        close=150.50,
        volume=1000000,
    )


# ============================================================================
# Limit Buy Tests - Conservative Touch Rules
# ============================================================================


def test_limit_buy_touched_fills_at_min_limit_close(fill_policy, bar_low):
    """Conservative Limit Buy: if low ≤ limit, fill at min(limit, close)."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("148.00"),  # limit = close, low < limit
    )

    decision = fill_policy.evaluate_limit_order(order, bar_low)
    assert decision.should_fill is True
    # bar_low: low=145, close=148, limit=148
    # Touched: low(145) ≤ limit(148) ✓
    # Fill at min(limit, close) = min(148, 148) = 148
    assert decision.fill_price == Decimal("148.00")


def test_limit_buy_touched_fills_at_limit_when_close_higher(fill_policy, bar_high):
    """Conservative Limit Buy: fill at limit when close > limit."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("150.00"),
    )

    decision = fill_policy.evaluate_limit_order(order, bar_high)
    assert decision.should_fill is True
    # bar_high: low=149, close=152, limit=150
    # Touched: low(149) ≤ limit(150) ✓
    # Fill at min(limit, close) = min(150, 152) = 150
    assert decision.fill_price == Decimal("150.00")


def test_limit_buy_not_touched(fill_policy, bar_no_touch):
    """Limit Buy does not fill if low > limit."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("148.00"),
    )

    decision = fill_policy.evaluate_limit_order(order, bar_no_touch)
    assert decision.should_fill is False
    # bar_no_touch: low=149, limit=148
    # Not touched: low(149) > limit(148)


# ============================================================================
# Limit Sell Tests - Conservative Touch Rules
# ============================================================================


def test_limit_sell_touched_fills_at_max_limit_close(fill_policy, bar_high):
    """Conservative Limit Sell: if high ≥ limit, fill at max(limit, close)."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("152.00"),  # limit = close, high > limit
    )

    decision = fill_policy.evaluate_limit_order(order, bar_high)
    assert decision.should_fill is True
    # bar_high: high=155, close=152, limit=152
    # Touched: high(155) ≥ limit(152) ✓
    # Fill at max(limit, close) = max(152, 152) = 152
    assert decision.fill_price == Decimal("152.00")


def test_limit_sell_touched_fills_at_limit_when_close_lower(fill_policy, bar_low):
    """Conservative Limit Sell: fill at limit when close < limit."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("150.00"),
    )

    decision = fill_policy.evaluate_limit_order(order, bar_low)
    assert decision.should_fill is True
    # bar_low: high=151, close=148, limit=150
    # Touched: high(151) ≥ limit(150) ✓
    # Fill at max(limit, close) = max(150, 148) = 150
    assert decision.fill_price == Decimal("150.00")


def test_limit_sell_not_touched(fill_policy, bar_no_touch):
    """Limit Sell does not fill if high < limit."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("152.00"),
    )

    decision = fill_policy.evaluate_limit_order(order, bar_no_touch)
    assert decision.should_fill is False
    # bar_no_touch: high=151, limit=152
    # Not touched: high(151) < limit(152)


# ============================================================================
# Stop Buy Tests - Conservative Touch Rules
# ============================================================================


def test_stop_buy_triggered_fills_at_max_with_slippage(fill_policy, bar_high):
    """Conservative Stop Buy: if high ≥ stop, fill at max(stop, close) + slippage."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("152.00"),
    )

    decision = fill_policy.evaluate_stop_order(order, bar_high)
    assert decision.should_fill is True
    # bar_high: high=155, close=152, stop=152
    # Triggered: high(155) ≥ stop(152) ✓
    # Fill at max(stop, close) + slip = max(152, 152) * 1.0005 = 152.076
    expected = Decimal("152.00") * Decimal("1.0005")
    assert decision.fill_price == expected


def test_stop_buy_triggered_at_stop_below_close(fill_policy, bar_low):
    """Stop Buy fills at close + slippage when stop < close."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("146.00"),
    )

    decision = fill_policy.evaluate_stop_order(order, bar_low)
    assert decision.should_fill is True
    # bar_low: high=151, close=148, stop=146
    # Triggered: high(151) ≥ stop(146) ✓
    # Fill at max(stop, close) + slip = max(146, 148) * 1.0005 = 148.074
    expected = Decimal("148.00") * Decimal("1.0005")
    assert decision.fill_price == expected


def test_stop_buy_not_triggered(fill_policy, bar_no_touch):
    """Stop Buy does not trigger if high < stop."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("152.00"),
    )

    decision = fill_policy.evaluate_stop_order(order, bar_no_touch)
    assert decision.should_fill is False
    # bar_no_touch: high=151, stop=152
    # Not triggered: high(151) < stop(152)


# ============================================================================
# Stop Sell Tests - Conservative Touch Rules
# ============================================================================


def test_stop_sell_triggered_fills_at_min_with_slippage(fill_policy, bar_low):
    """Conservative Stop Sell: if low ≤ stop, fill at min(stop, close) - slippage."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("148.00"),
    )

    decision = fill_policy.evaluate_stop_order(order, bar_low)
    assert decision.should_fill is True
    # bar_low: low=145, close=148, stop=148
    # Triggered: low(145) ≤ stop(148) ✓
    # Fill at min(stop, close) - slip = min(148, 148) * 0.9995 = 147.926
    expected = Decimal("148.00") * Decimal("0.9995")
    assert decision.fill_price == expected


def test_stop_sell_triggered_at_stop_above_close(fill_policy, bar_high):
    """Stop Sell fills at close - slippage when stop > close."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("154.00"),
    )

    decision = fill_policy.evaluate_stop_order(order, bar_high)
    assert decision.should_fill is True
    # bar_high: low=149, close=152, stop=154
    # Triggered: low(149) ≤ stop(154) ✓
    # Fill at min(stop, close) - slip = min(154, 152) * 0.9995 = 151.924
    expected = Decimal("152.00") * Decimal("0.9995")
    assert decision.fill_price == expected


def test_stop_sell_not_triggered(fill_policy, bar_no_touch):
    """Stop Sell does not trigger if low > stop."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("148.00"),
    )

    decision = fill_policy.evaluate_stop_order(order, bar_no_touch)
    assert decision.should_fill is False
    # bar_no_touch: low=149, stop=148
    # Not triggered: low(149) > stop(148)


# ============================================================================
# Close-Only Bar Tests
# ============================================================================


def test_close_only_bar_skips_limit_orders(fill_policy, bar_high):
    """Limit orders are not evaluated on close-only bars (malformed OHLC)."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("150.00"),
    )

    decision = fill_policy.evaluate_order(order, bar_high, next_bar=None, is_close_only=True)
    assert decision.should_fill is False
    assert "Close-only bar" in decision.reason


def test_close_only_bar_skips_stop_orders(fill_policy, bar_low):
    """Stop orders are not evaluated on close-only bars (malformed OHLC)."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("148.00"),
    )

    decision = fill_policy.evaluate_order(order, bar_low, next_bar=None, is_close_only=True)
    assert decision.should_fill is False
    assert "Close-only bar" in decision.reason


def test_close_only_bar_allows_moc_orders(fill_policy, bar_high):
    """MOC orders still work on close-only bars (only need close price)."""
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    decision = fill_policy.evaluate_order(order, bar_high, next_bar=None, is_close_only=True)
    assert decision.should_fill is True


# ============================================================================
# DAY Order Expiration Tests
# ============================================================================


def test_day_order_expires_next_day(engine):
    """DAY orders expire when a new day arrives."""
    # First day bar
    bar1 = Bar(
        trade_datetime=BAR_TS_DAY1_END.isoformat(),
        open=150.00,
        high=151.00,
        low=149.00,
        close=150.50,
        volume=1000000,
    )

    # Submit limit order that doesn't fill
    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_DAY1_END,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("148.00"),  # Below low, won't fill
        tif=TimeInForce.DAY,
        submission_bar_ts=BAR_TS_DAY1_END,  # Set explicitly for test
    )

    engine.submit_order(order)
    fills1 = engine.on_bar(bar1, symbol="AAPL", ts=BAR_TS_DAY1_END)
    assert len(fills1) == 0
    assert "order-1" in engine.pending_orders

    # Next day bar - order should expire
    bar2 = Bar(
        trade_datetime=BAR_TS_DAY2_START.isoformat(),  # New day
        open=151.00,
        high=152.00,
        low=150.00,
        close=151.50,
        volume=1200000,
    )

    fills2 = engine.on_bar(bar2, symbol="AAPL", ts=BAR_TS_DAY2_START)
    assert len(fills2) == 0
    assert "order-1" not in engine.pending_orders
    assert "order-1" in engine.expired_orders


def test_day_order_survives_same_day_bars(engine):
    """DAY orders persist across multiple bars on the same day."""
    # First bar
    bar1 = Bar(
        trade_datetime=BAR_TS_1.isoformat(),
        open=150.00,
        high=151.00,
        low=149.00,
        close=150.50,
        volume=1000000,
    )

    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_DAY1_END,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("148.00"),
        tif=TimeInForce.DAY,
        submission_bar_ts=BAR_TS_DAY1_END,  # Set explicitly for test
    )

    engine.submit_order(order)
    engine.on_bar(bar1, symbol="AAPL", ts=BAR_TS_DAY1_END)

    # Second bar same day - order should still be pending
    bar2 = Bar(
        trade_datetime=BAR_TS_DAY1_LATER.isoformat(),  # Same day, later time
        open=150.50,
        high=151.50,
        low=150.00,
        close=151.00,
        volume=1100000,
    )

    engine.on_bar(bar2, symbol="AAPL", ts=BAR_TS_DAY1_LATER)
    assert "order-1" in engine.pending_orders
    assert "order-1" not in engine.expired_orders


# ============================================================================
# ExecutionEngine Integration Tests
# ============================================================================


def test_engine_fills_limit_buy_and_updates_portfolio(engine):
    """Engine correctly fills limit buy and updates portfolio."""
    initial_cash = engine.portfolio.cash.get_balance()

    bar = Bar(
        trade_datetime=BAR_TS_1.isoformat(),
        open=150.00,
        high=152.00,
        low=148.00,
        close=151.00,
        volume=1000000,
    )

    order = Order(
        order_id="order-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("150.00"),
    )

    engine.submit_order(order)
    fills = engine.on_bar(bar, symbol="AAPL", ts=BAR_TS_1)

    assert len(fills) == 1
    assert fills[0].price == Decimal("150.00")

    # Check portfolio
    position = engine.portfolio.positions.get_position("AAPL")
    assert position.qty == 100
    assert engine.portfolio.cash.get_balance() < initial_cash


def test_engine_fills_stop_sell_and_updates_portfolio(engine):
    """Engine correctly fills stop sell and updates portfolio."""
    # First create a position
    bar1 = Bar(
        trade_datetime=BAR_TS_1.isoformat(),
        open=150.00,
        high=152.00,
        low=149.00,
        close=151.00,
        volume=1000000,
    )

    buy_order = Order(
        order_id="buy-1",
        strategy_ts=BAR_TS_1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    engine.submit_order(buy_order)
    engine.on_bar(bar1, symbol="AAPL", ts=BAR_TS_1)

    # Now submit stop sell
    bar2 = Bar(
        trade_datetime=BAR_TS_DAY1_LATER.isoformat(),
        open=148.00,
        high=149.00,
        low=145.00,  # Triggers stop
        close=147.00,
        volume=1200000,
    )

    stop_order = Order(
        order_id="stop-1",
        strategy_ts=BAR_TS_DAY1_LATER,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.STOP,
        state=OrderState.SUBMITTED,
        stop_price=Decimal("148.00"),
    )

    engine.submit_order(stop_order)
    fills = engine.on_bar(bar2, symbol="AAPL", ts=BAR_TS_DAY1_LATER)

    assert len(fills) == 1
    # Stop sell: min(stop, close) * 0.9995 = min(148, 147) * 0.9995
    expected_price = Decimal("147.00") * Decimal("0.9995")
    assert fills[0].price == expected_price

    # Position should be flat
    position = engine.portfolio.positions.get_position("AAPL")
    assert position.is_flat()
