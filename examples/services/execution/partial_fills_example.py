"""Partial fills example.

Demonstrates how large orders fill incrementally across multiple bars due to volume constraints.

This example shows:
1. Submitting a large order
2. Volume-based participation limits
3. Tracking partial fills across bars
4. Understanding market impact
"""

from datetime import datetime
from decimal import Decimal

from qtrader.models.bar import Bar
from qtrader.services.execution.config import ExecutionConfig, SlippageConfig
from qtrader.services.execution.models import Order, OrderSide
from qtrader.services.execution.service import ExecutionService


def main():
    """Run partial fills example."""
    print("=" * 70)
    print("PARTIAL FILLS EXAMPLE - Large Order Execution")
    print("=" * 70)
    print("\nScenario: Buy 10,000 shares of AAPL")
    print("Constraint: Max 20% of bar volume (realistic participation)")
    print()

    # Step 1: Create execution service with participation limit
    config = ExecutionConfig(
        market_order_queue_bars=1,
        max_participation_rate=Decimal("0.20"),  # Max 20% of volume
        slippage=SlippageConfig(model="fixed_bps", params={"bps": Decimal("5")}),
    )

    execution_service = ExecutionService(config)

    print("Configuration:")
    print(f"  Max participation rate: {config.max_participation_rate * 100}%")
    print(f"  Slippage: {config.slippage.params['bps']} bps")
    print()

    # Step 2: Submit large order
    print("=" * 70)
    print("Submitting Large Order")
    print("=" * 70)

    order = Order.market_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10000"),  # Large order
    )

    execution_service.submit_order(order)

    print(f"Order ID: {order.order_id}")
    print(f"Total quantity: {order.quantity:,}")
    print(f"State: {order.state.value}")
    print()

    # Step 3: Queue on first bar
    print("=" * 70)
    print("Bar 1: Queueing")
    print("=" * 70)

    bar1 = Bar(
        trade_datetime=datetime(2024, 1, 15, 9, 30),
        open=150.00,
        high=151.00,
        low=149.50,
        close=150.50,
        volume=100000,  # Moderate volume
    )

    fills = execution_service.on_bar(bar1)
    print(f"Time: {bar1.trade_datetime.strftime('%H:%M')}")
    print(f"Volume: {bar1.volume:,}")
    print(f"Fills: {len(fills)}")
    print("Order queued for next bar")
    print()

    # Step 4: Process bars with varying volume
    print("=" * 70)
    print("Executing Across Multiple Bars")
    print("=" * 70)
    print()

    bars = [
        # Low volume - small fill
        Bar(
            trade_datetime=datetime(2024, 1, 15, 9, 31),
            open=150.50,
            high=151.00,
            low=150.00,
            close=150.75,
            volume=50000,  # Low volume
        ),
        # Medium volume - medium fill
        Bar(
            trade_datetime=datetime(2024, 1, 15, 9, 32),
            open=150.75,
            high=151.50,
            low=150.50,
            close=151.00,
            volume=150000,  # Medium volume
        ),
        # High volume - large fill
        Bar(
            trade_datetime=datetime(2024, 1, 15, 9, 33),
            open=151.00,
            high=152.00,
            low=150.75,
            close=151.50,
            volume=200000,  # High volume
        ),
        # High volume - final fill
        Bar(
            trade_datetime=datetime(2024, 1, 15, 9, 34),
            open=151.50,
            high=152.00,
            low=151.00,
            close=151.75,
            volume=250000,  # Very high volume
        ),
    ]

    total_cost = Decimal("0")
    total_shares = Decimal("0")
    fill_count = 0

    for i, bar in enumerate(bars, 2):
        fills = execution_service.on_bar(bar)

        print(f"Bar {i}: {bar.trade_datetime.strftime('%H:%M')}")
        print(f"  Volume: {bar.volume:,}")
        print(f"  Max fill (20%): {int(bar.volume * config.max_participation_rate):,}")

        if fills:
            fill = fills[0]
            fill_count += 1
            total_shares += fill.quantity
            fill_cost = fill.quantity * fill.price + fill.commission
            total_cost += fill_cost

            print(f"  ✓ FILLED: {fill.quantity:,} shares @ ${fill.price}")
            print(f"    Commission: ${fill.commission}")
            print(f"    Fill cost: ${fill_cost:,.2f}")
            print(f"  Progress: {total_shares:,} / {order.quantity:,} ({total_shares / order.quantity * 100:.1f}%)")
            print(f"  Remaining: {order.remaining_quantity:,}")
        else:
            print("  No fill (shouldn't happen)")

        print(f"  Order state: {order.state.value}")
        print()

        if order.is_complete:
            break

    # Step 5: Summary
    print("=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print()
    print(f"Order Status: {order.state.value}")
    print(f"Total fills: {fill_count}")
    print(f"Total shares: {total_shares:,}")
    print(f"Total cost: ${total_cost:,.2f}")

    if total_shares > 0:
        avg_price_incl_commission = total_cost / total_shares
        print(f"Average price (incl. commission): ${avg_price_incl_commission:.4f}")

    print()
    print("=" * 70)
    print("KEY INSIGHTS: Partial Fills")
    print("=" * 70)
    print()
    print("1. MARKET IMPACT:")
    print("   - Large orders can't fill instantly without moving price")
    print("   - Participation rate limits realistic execution")

    print("\n2. VOLUME DEPENDENCY:")
    print("   - Low volume bars → small fills")
    print("   - High volume bars → large fills")
    print("   - Execution time depends on market liquidity")

    print("\n3. PRICE VARIATION:")
    print("   - Each fill occurs at different price")
    print("   - Average price can drift from initial price")
    print("   - VWAP (volume-weighted average price) matters")

    print("\n4. COMMISSION ACCUMULATION:")
    print("   - Each partial fill incurs commission")
    print("   - Multiple small fills = higher total commission")
    print("   - Trade-off: speed vs. cost")

    print("\n5. REALISTIC SIMULATION:")
    print("   - Real algorithms (TWAP, VWAP, POV) work this way")
    print("   - Crucial for backtesting large strategies")
    print("   - Different from fill-and-forget simplified models")

    # Comparison
    print("\n" + "=" * 70)
    print("INSTANT FILL vs. PARTICIPATION-LIMITED")
    print("=" * 70)

    instant_price = Decimal("150.50")
    instant_cost = order.quantity * instant_price
    print("\nInstant fill model (unrealistic for large orders):")
    print(f"  All 10,000 shares @ ${instant_price}")
    print(f"  Cost: ${instant_cost:,.2f}")

    print("\nParticipation-limited model (realistic):")
    print("  4 partial fills across different prices")
    print(f"  Cost: ${total_cost:,.2f}")
    print(f"  Difference: ${abs(total_cost - instant_cost):,.2f}")

    if total_cost > instant_cost:
        slippage_pct = ((total_cost / instant_cost) - 1) * 100
        print(f"  Slippage: {slippage_pct:.2f}% (adverse price movement)")


if __name__ == "__main__":
    main()
