"""Basic market order example.

Demonstrates the simplest use case: submitting and executing a market order.

This example shows:
1. Creating an execution service with basic configuration
2. Submitting a market order
3. Processing bars to generate fills
4. Inspecting order state and fill details
"""

from datetime import datetime
from decimal import Decimal

from qtrader.models.bar import Bar
from qtrader.services.execution.config import CommissionConfig, ExecutionConfig, SlippageConfig
from qtrader.services.execution.models import Order, OrderSide
from qtrader.services.execution.service import ExecutionService


def main():
    """Run basic market order example."""
    # Step 1: Create execution service with configuration
    print("=" * 70)
    print("STEP 1: Create ExecutionService")
    print("=" * 70)

    config = ExecutionConfig(
        market_order_queue_bars=1,  # Queue for 1 bar before filling
        max_participation_rate=Decimal("0.20"),  # Max 20% of bar volume
        slippage=SlippageConfig(
            model="fixed_bps",
            params={"bps": Decimal("5")},  # 5 basis points slippage
        ),
        commission=CommissionConfig(
            per_share=Decimal("0.005"),  # $0.005 per share
            minimum=Decimal("1.00"),  # Minimum $1.00 commission
        ),
    )

    execution_service = ExecutionService(config)
    print("✓ ExecutionService created")
    print(f"  - Queue bars: {config.market_order_queue_bars}")
    print(f"  - Max participation: {config.max_participation_rate * 100}%")
    print(f"  - Slippage: {config.slippage.params['bps']} bps")
    print(f"  - Commission: ${config.commission.per_share}/share (min ${config.commission.minimum})")

    # Step 2: Create and submit market order
    print("\n" + "=" * 70)
    print("STEP 2: Submit Market Buy Order")
    print("=" * 70)

    order = Order.market_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
    )

    order_id = execution_service.submit_order(order)
    print(f"✓ Order submitted: {order_id}")
    print(f"  - Symbol: {order.symbol}")
    print(f"  - Side: {order.side.value}")
    print(f"  - Quantity: {order.quantity}")
    print(f"  - Type: {order.order_type.value}")
    print(f"  - State: {order.state.value}")

    # Step 3: Process first bar (queueing)
    print("\n" + "=" * 70)
    print("STEP 3: Process First Bar (Queueing)")
    print("=" * 70)

    bar1 = Bar(
        trade_datetime=datetime(2024, 1, 15, 9, 30),
        open=150.00,
        high=151.00,
        low=149.50,
        close=150.50,
        volume=1000000,
    )

    fills = execution_service.on_bar(bar1)
    print(f"Bar 1: {bar1.trade_datetime}")
    print(f"  OHLC: {bar1.open} / {bar1.high} / {bar1.low} / {bar1.close}")
    print(f"  Volume: {bar1.volume:,}")
    print(f"  Fills generated: {len(fills)}")
    print(f"  Order state: {order.state.value}")
    print(f"  Bars queued: {order.bars_queued}")

    # Step 4: Process second bar (filling)
    print("\n" + "=" * 70)
    print("STEP 4: Process Second Bar (Filling)")
    print("=" * 70)

    bar2 = Bar(
        trade_datetime=datetime(2024, 1, 15, 9, 31),
        open=150.50,
        high=151.50,
        low=150.00,
        close=151.00,
        volume=1500000,
    )

    fills = execution_service.on_bar(bar2)
    print(f"Bar 2: {bar2.trade_datetime}")
    print(f"  OHLC: {bar2.open} / {bar2.high} / {bar2.low} / {bar2.close}")
    print(f"  Volume: {bar2.volume:,}")
    print(f"  Fills generated: {len(fills)}")

    if fills:
        fill = fills[0]
        print("\n✓ ORDER FILLED!")
        print(f"  Fill ID: {fill.fill_id}")
        print(f"  Symbol: {fill.symbol}")
        print(f"  Side: {fill.side}")
        print(f"  Quantity: {fill.quantity}")
        print(f"  Price: ${fill.price}")
        print(f"  Commission: ${fill.commission}")
        print(f"  Timestamp: {fill.timestamp}")

        # Calculate total cost
        total_cost = fill.quantity * fill.price + fill.commission
        avg_price = total_cost / fill.quantity
        print(f"\n  Total cost: ${total_cost:,.2f}")
        print(f"  Avg price (incl. commission): ${avg_price:,.2f}")

    # Step 5: Verify final order state
    print("\n" + "=" * 70)
    print("STEP 5: Final Order State")
    print("=" * 70)

    print(f"Order state: {order.state.value}")
    print(f"Filled quantity: {order.filled_quantity}")
    print(f"Remaining quantity: {order.remaining_quantity}")
    print(f"Is complete: {order.is_complete}")
    print(f"Is active: {order.is_active}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Market order successfully executed")
    print("✓ Order queued for 1 bar (as configured)")
    print("✓ Filled on second bar at market price")
    print("✓ Slippage and commission applied")
    print("\nKey takeaways:")
    print("  1. Market orders require queueing (realistic simulation)")
    print("  2. Fills occur at bar open + slippage")
    print("  3. Commission is calculated per-share with minimum")
    print("  4. Order state transitions: PENDING → SUBMITTED → FILLED")


if __name__ == "__main__":
    main()
