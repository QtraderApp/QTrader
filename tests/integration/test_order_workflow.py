"""
Integration test: Order workflow from submission to portfolio update.

Tests the complete order processing flow:
    Submit Order → Fill Generation → Portfolio Update → Ledger Entry

Uses synthetic bars (not real data) to test the workflow mechanics.
"""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.models.bar import Bar
from qtrader.models.order import Order, OrderSide, OrderState, OrderType, TimeInForce

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def synthetic_bars():
    """Create a sequence of synthetic bars for testing."""
    bars = []
    for day in range(1, 6):  # 5 days
        bar = Bar(
            ts=datetime(2023, 1, day, 16, 0, tzinfo=ET),
            symbol="AAPL",
            open=Decimal("150.00") + Decimal(day),
            high=Decimal("152.00") + Decimal(day),
            low=Decimal("149.00") + Decimal(day),
            close=Decimal("151.00") + Decimal(day),
            volume=1_000_000,
        )
        bars.append(bar)
    return bars


# ============================================================================
# Test: Market Order Complete Workflow
# ============================================================================


def test_market_order_complete_workflow(engine_100k, synthetic_bars):
    """
    Test complete workflow for Market order.

    Flow:
        1. Submit Market BUY order
        2. Process bar (order waits for next bar)
        3. Generate fill at next bar's open
        4. Update portfolio (reduce cash, increase position)
        5. Verify ledger entry
    """
    bar0 = synthetic_bars[0]
    bar1 = synthetic_bars[1]

    # Initial state
    initial_cash = engine_100k.portfolio.cash.get_balance()
    assert initial_cash == Decimal("100000.00")

    # Step 1: Submit order
    order = Order(
        order_id="mkt-buy-1",
        strategy_ts=bar0.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        state=OrderState.SUBMITTED,
        tif=TimeInForce.DAY,
        submission_bar_ts=bar0.ts,
    )

    engine_100k.submit_order(order, bar0.ts)
    assert len(engine_100k.pending_orders) == 1

    # Step 2: Process first bar (order waits)
    fills = engine_100k.on_bar(bar0, bar1)
    assert len(fills) == 1  # Fill happens on bar0 (fills at bar1.open)

    # Step 3: Verify fill
    fill = fills[0]
    assert fill.symbol == "AAPL"
    assert fill.side == OrderSide.BUY
    assert fill.qty == 100
    assert fill.price == bar1.open  # Market fills at next bar's open
    assert fill.fees > 0  # Commission applied

    # Step 4: Verify portfolio updated
    position = engine_100k.portfolio.positions.get_position("AAPL")
    assert position.qty == 100
    assert position.avg_price == bar1.open

    # Cash reduced by (price * qty) + fees
    expected_cash_reduction = (bar1.open * 100) + fill.fees
    final_cash = engine_100k.portfolio.cash.get_balance()
    assert final_cash == initial_cash - expected_cash_reduction

    # Step 5: Verify order state
    orders = engine_100k.get_orders()
    filled_order = next(o for o in orders if o.order_id == "mkt-buy-1")
    assert filled_order.state == OrderState.FILLED
    assert filled_order.filled_qty == 100
    assert filled_order.remaining_qty == 0


# ============================================================================
# Test: MOC Order Complete Workflow
# ============================================================================


def test_moc_order_complete_workflow(engine_100k, synthetic_bars):
    """
    Test complete workflow for Market-On-Close order.

    Flow:
        1. Submit MOC BUY order
        2. Process bar (order fills immediately at close)
        3. Generate fill at bar's close price
        4. Update portfolio
    """
    bar0 = synthetic_bars[0]

    initial_cash = engine_100k.portfolio.cash.get_balance()

    # Submit MOC order
    order = Order(
        order_id="moc-buy-1",
        strategy_ts=bar0.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
        submission_bar_ts=bar0.ts,
    )

    engine_100k.submit_order(order, bar0.ts)
    fills = engine_100k.on_bar(bar0)

    # Should fill immediately at close (with slippage)
    assert len(fills) == 1
    fill = fills[0]
    # MOC has 5 bps slippage by default (buy pays more, sell gets less)
    expected_price = bar0.close * (1 + Decimal("0.0005"))  # 5 bps = 0.05% = 0.0005
    assert fill.price == expected_price
    assert fill.slippage_bps == 5
    assert fill.qty == 100

    # Verify portfolio
    position = engine_100k.portfolio.positions.get_position("AAPL")
    assert position.qty == 100

    # Verify cash (with slippage applied)
    expected_cash = initial_cash - (fill.price * 100) - fill.fees
    assert engine_100k.portfolio.cash.get_balance() == expected_cash


# ============================================================================
# Test: Round-Trip Trade (Buy then Sell)
# ============================================================================


def test_round_trip_trade_workflow(engine_100k, synthetic_bars):
    """
    Test complete round-trip trade workflow.

    Flow:
        1. Buy 100 shares
        2. Hold for a few bars
        3. Sell 100 shares
        4. Verify position flat
        5. Calculate realized PnL
    """
    initial_cash = engine_100k.portfolio.cash.get_balance()

    # Step 1: Buy on bar 0
    buy_order = Order(
        order_id="buy-1",
        strategy_ts=synthetic_bars[0].ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
        submission_bar_ts=synthetic_bars[0].ts,
    )

    engine_100k.submit_order(buy_order, synthetic_bars[0].ts)
    buy_fills = engine_100k.on_bar(synthetic_bars[0])

    assert len(buy_fills) == 1
    buy_fill = buy_fills[0]
    buy_cost = (buy_fill.price * 100) + buy_fill.fees

    # Step 2: Hold for a few bars
    for bar in synthetic_bars[1:3]:
        engine_100k.on_bar(bar)

    # Position should still be 100 shares
    position = engine_100k.portfolio.positions.get_position("AAPL")
    assert position.qty == 100

    # Step 3: Sell on bar 3
    sell_order = Order(
        order_id="sell-1",
        strategy_ts=synthetic_bars[3].ts,
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.MARKET_ON_CLOSE,
        state=OrderState.SUBMITTED,
        submission_bar_ts=synthetic_bars[3].ts,
    )

    engine_100k.submit_order(sell_order, synthetic_bars[3].ts)
    sell_fills = engine_100k.on_bar(synthetic_bars[3])

    assert len(sell_fills) == 1
    sell_fill = sell_fills[0]
    sell_proceeds = (sell_fill.price * 100) - sell_fill.fees

    # Step 4: Verify position flat
    final_position = engine_100k.portfolio.positions.get_position("AAPL")
    assert final_position.is_flat()

    # Step 5: Verify PnL
    final_cash = engine_100k.portfolio.cash.get_balance()
    realized_pnl = final_cash - initial_cash
    expected_pnl = sell_proceeds - buy_cost

    # Should match (small rounding tolerance)
    assert abs(realized_pnl - expected_pnl) < Decimal("0.01")

    # Verify we made/lost money based on price movement
    price_change = sell_fill.price - buy_fill.price
    gross_pnl = price_change * 100
    net_pnl = gross_pnl - (buy_fill.fees + sell_fill.fees)
    assert abs(realized_pnl - net_pnl) < Decimal("0.01")


# ============================================================================
# Test: Multi-Bar Order Processing
# ============================================================================


def test_multi_bar_order_processing(engine_100k, synthetic_bars):
    """
    Test processing multiple bars with pending orders.

    Flow:
        1. Submit multiple orders
        2. Process bars sequentially
        3. Verify fills generated correctly
        4. Verify portfolio state at each step
    """
    # Submit 3 MOC orders on different bars
    orders = []
    for i, bar in enumerate(synthetic_bars[:3]):
        order = Order(
            order_id=f"order-{i}",
            strategy_ts=bar.ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=50,
            order_type=OrderType.MARKET_ON_CLOSE,
            state=OrderState.SUBMITTED,
            submission_bar_ts=bar.ts,
        )
        orders.append(order)

    # Process bars
    total_fills = []
    for i, bar in enumerate(synthetic_bars[:3]):
        engine_100k.submit_order(orders[i], bar.ts)
        fills = engine_100k.on_bar(bar)
        total_fills.extend(fills)

        # Verify cumulative position
        position = engine_100k.portfolio.positions.get_position("AAPL")
        assert position.qty == 50 * (i + 1)

    # Should have 3 fills total
    assert len(total_fills) == 3

    # Final position should be 150 shares
    final_position = engine_100k.portfolio.positions.get_position("AAPL")
    assert final_position.qty == 150


# ============================================================================
# Test: Partial Fill Workflow
# ============================================================================


def test_partial_fill_workflow(engine_10m, synthetic_bars):
    """
    Test partial fill workflow with participation cap.

    Flow:
        1. Submit large order exceeding participation cap
        2. Process bar (partial fill generated)
        3. Verify order state is PARTIALLY_FILLED
        4. Process more bars (residual fills)
        5. Verify final state
    """
    bar0 = synthetic_bars[0]
    bar1 = synthetic_bars[1]

    # Calculate participation cap
    max_participation = engine_10m.config.max_participation
    cap = int(bar0.volume * max_participation)

    # Submit order larger than cap
    large_order = Order(
        order_id="large-1",
        strategy_ts=bar0.ts,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=cap * 3,  # 3x the cap
        order_type=OrderType.MARKET,
        state=OrderState.SUBMITTED,
        tif=TimeInForce.DAY,
        submission_bar_ts=bar0.ts,
    )

    engine_10m.submit_order(large_order, bar0.ts)
    fills = engine_10m.on_bar(bar0, bar1)

    # Should have partial fill
    assert len(fills) == 1
    assert fills[0].qty <= cap
    assert fills[0].partial_index == 1

    # Order should be PARTIALLY_FILLED
    orders = engine_10m.get_orders()
    order_result = next(o for o in orders if o.order_id == "large-1")
    assert order_result.state == OrderState.PARTIALLY_FILLED
    assert order_result.filled_qty == fills[0].qty
    assert order_result.remaining_qty == (cap * 3) - fills[0].qty

    # Process more bars to fill residual
    for i in range(1, len(synthetic_bars)):
        bar = synthetic_bars[i]
        next_bar = synthetic_bars[i + 1] if i + 1 < len(synthetic_bars) else None
        engine_10m.on_bar(bar, next_bar)

    # Should have more fills
    all_fills = engine_10m.get_fills()
    assert len(all_fills) > 1

    # Each fill should have incremental partial_index
    for i, fill in enumerate(all_fills):
        assert fill.partial_index == i + 1
