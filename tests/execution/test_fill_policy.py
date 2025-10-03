"""Integration tests for FillPolicy - tests core fill evaluation logic."""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.execution.fill_policy import FillPolicy
from qtrader.models.bar import Bar
from qtrader.models.order import Order, OrderSide, OrderState, OrderType

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def fill_policy():
    """Fill policy with 5 bps slippage."""
    return FillPolicy(moc_slip_bps=5)


@pytest.fixture
def sample_bar():
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
def next_bar():
    """Next AAPL bar."""
    return Bar(
        ts=datetime(2023, 1, 15, 9, 31, tzinfo=ET),
        symbol="AAPL",
        open=Decimal("151.50"),
        high=Decimal("153.00"),
        low=Decimal("150.50"),
        close=Decimal("152.00"),
        volume=1200000,
    )


def test_fill_policy_initialization():
    """FillPolicy initializes with slippage setting."""
    policy = FillPolicy(moc_slip_bps=5)
    assert policy.moc_slip_bps == 5


def test_moc_order_fills_immediately(fill_policy, sample_bar):
    """MOC orders should fill at current bar."""
    order = Order(
        order_id="order-1",
        strategy_ts=sample_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    decision = fill_policy.evaluate_order(order, sample_bar, next_bar=None)
    assert decision.should_fill is True
    # BUY adds slippage: 151.00 * 1.0005 = 151.08
    expected = sample_bar.close * Decimal("1.0005")
    assert decision.fill_price == expected.quantize(Decimal("0.01"))


def test_moc_sell_negative_slippage(fill_policy, sample_bar):
    """MOC SELL gets worse price (negative slippage)."""
    order = Order(
        order_id="order-1",
        strategy_ts=sample_bar.ts,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    decision = fill_policy.evaluate_order(order, sample_bar, next_bar=None)
    assert decision.should_fill is True
    # SELL subtracts slippage: 151.00 * 0.9995 = 150.92
    expected = sample_bar.close * Decimal("0.9995")
    assert decision.fill_price == expected.quantize(Decimal("0.01"))
    assert decision.fill_price < sample_bar.close


def test_market_order_needs_next_bar(fill_policy, sample_bar):
    """Market orders without next bar should not fill."""
    order = Order(
        order_id="order-1",
        strategy_ts=sample_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        state=OrderState.SUBMITTED,
        submission_bar_ts=sample_bar.ts,
    )

    decision = fill_policy.evaluate_order(order, sample_bar, next_bar=None)
    assert decision.should_fill is False
    assert decision.next_bar is True  # Needs next bar


def test_limit_order_not_implemented(fill_policy, sample_bar):
    """Limit orders not supported in Stage 3."""
    order = Order(
        order_id="order-1",
        strategy_ts=sample_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        state=OrderState.SUBMITTED,
        limit_price=Decimal("150.00"),
    )

    decision = fill_policy.evaluate_order(order, sample_bar, next_bar=None)
    assert decision.should_fill is False


def test_zero_slippage_moc(sample_bar):
    """MOC with zero slippage fills at exact close."""
    policy = FillPolicy(moc_slip_bps=0)
    order = Order(
        order_id="order-1",
        strategy_ts=sample_bar.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
    )

    decision = policy.evaluate_order(order, sample_bar, next_bar=None)
    assert decision.fill_price == sample_bar.close
