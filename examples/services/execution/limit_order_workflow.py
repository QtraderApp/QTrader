"""Limit order workflow example.

Demonstrates limit order submission, price watching, and conditional fills.

This example shows:
1. Submitting a limit buy order
2. Processing bars where price doesn't touch limit
3. Fill when price reaches limit
4. Understanding limit order mechanics
"""

from datetime import datetime
from decimal import Decimal

from qtrader.models.bar import Bar
from qtrader.services.execution.config import ExecutionConfig, SlippageConfig
from qtrader.services.execution.models import Order, OrderSide
from qtrader.services.execution.service import ExecutionService


def main():
    """Run limit order workflow example."""
    print("=" * 70)
    print("LIMIT ORDER WORKFLOW EXAMPLE")
    print("=" * 70)
    print("\nScenario: Buy AAPL at $148.00 or better")
    print("Current market price: ~$150.00\n")

    # Step 1: Create execution service
    config = ExecutionConfig(
        slippage=SlippageConfig(model="zero"),  # No slippage for limit orders
    )
    execution_service = ExecutionService(config)

    # Step 2: Submit limit buy order
    print("=" * 70)
    print("Submitting Limit Buy Order")
    print("=" * 70)

    order = Order.limit_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        limit_price=Decimal("148.00"),
    )

    execution_service.submit_order(order)
    print(f"Order ID: {order.order_id}")
    print(f"Symbol: {order.symbol}")
    print(f"Side: {order.side.value}")
    print(f"Quantity: {order.quantity}")
    print(f"Limit Price: ${order.limit_price}")
    print(f"State: {order.state.value}\n")

    # Step 3: Process bars - price too high
    print("=" * 70)
    print("Processing Bars - Waiting for Price")
    print("=" * 70)

    bars = [
        Bar(
            trade_datetime=datetime(2024, 1, 15, 9, 30),
            open=150.00,
            high=151.00,
            low=149.50,  # Doesn't reach 148.00
            close=150.50,
            volume=1000000,
        ),
        Bar(
            trade_datetime=datetime(2024, 1, 15, 9, 31),
            open=150.50,
            high=151.50,
            low=150.00,  # Still too high
            close=151.00,
            volume=1200000,
        ),
        Bar(
            trade_datetime=datetime(2024, 1, 15, 9, 32),
            open=150.75,
            high=151.00,
            low=149.00,  # Still above 148.00
            close=149.50,
            volume=1500000,
        ),
    ]

    for i, bar in enumerate(bars, 1):
        fills = execution_service.on_bar(bar)
        print(f"\nBar {i}: {bar.trade_datetime.strftime('%H:%M')}")
        print(f"  OHLC: {bar.open} / {bar.high} / {bar.low} / {bar.close}")
        print(f"  Low: ${bar.low} (limit: ${order.limit_price})")
        print(f"  Fills: {len(fills)}")
        print(f"  Order state: {order.state.value}")

        if bar.low <= order.limit_price:
            print("  → Price touched limit!")
        else:
            print(f"  → Waiting (low ${bar.low} > limit ${order.limit_price})")

    # Step 4: Price finally reaches limit
    print("\n" + "=" * 70)
    print("Price Reaches Limit!")
    print("=" * 70)

    fill_bar = Bar(
        trade_datetime=datetime(2024, 1, 15, 9, 33),
        open=149.00,
        high=149.50,
        low=147.50,  # Touches 148.00!
        close=148.25,
        volume=2000000,
    )

    fills = execution_service.on_bar(fill_bar)

    print(f"\nBar 4: {fill_bar.trade_datetime.strftime('%H:%M')}")
    print(f"  OHLC: {fill_bar.open} / {fill_bar.high} / {fill_bar.low} / {fill_bar.close}")
    print(f"  Low: ${fill_bar.low} (limit: ${order.limit_price})")
    print(f"  ✓ Price touched limit at ${order.limit_price}!")

    if fills:
        fill = fills[0]
        print("\n✓ ORDER FILLED!")
        print(f"  Quantity: {fill.quantity}")
        print(f"  Fill Price: ${fill.price}")
        print(f"  Commission: ${fill.commission}")
        print(f"  Total Cost: ${fill.quantity * fill.price + fill.commission:,.2f}")

        # Verify fill price
        assert fill.price <= order.limit_price, "Fill price must be at or below limit!"
        print(f"\n  ✓ Fill price ${fill.price} ≤ limit ${order.limit_price}")

    # Step 5: Understanding limit orders
    print("\n" + "=" * 70)
    print("LIMIT ORDER KEY CONCEPTS")
    print("=" * 70)
    print("\n1. PRICE CONDITION:")
    print("   - Buy limit: Fills when market price ≤ limit price")
    print("   - Sell limit: Fills when market price ≥ limit price")

    print("\n2. FILL PRICE:")
    print("   - Always at or better than limit price")
    print("   - If market gaps through limit, fills at limit (not better)")

    print("\n3. NO GUARANTEE:")
    print("   - Order may never fill if price doesn't reach limit")
    print("   - Use GTC (Good-Till-Cancel) to keep order active")

    print("\n4. NO QUEUEING:")
    print("   - Limit orders don't need queue bars")
    print("   - Fill immediately when price condition met")

    print("\n5. SLIPPAGE:")
    print("   - Generally zero for limit orders")
    print("   - You set the price limit, market comes to you")

    # Comparison example
    print("\n" + "=" * 70)
    print("MARKET VS LIMIT ORDER COMPARISON")
    print("=" * 70)

    print("\nMarket Order (immediate execution):")
    print("  ✓ Guaranteed fill (assuming volume)")
    print("  ✓ Fast execution")
    print("  ✗ Uncertain price")
    print("  ✗ Slippage risk")

    print("\nLimit Order (price control):")
    print("  ✓ Controlled price")
    print("  ✓ No worse than limit")
    print("  ✗ May never fill")
    print("  ✗ Requires price patience")


if __name__ == "__main__":
    main()
