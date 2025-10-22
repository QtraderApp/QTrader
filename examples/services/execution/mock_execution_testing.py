"""Mock execution for strategy testing.

Demonstrates how to use ExecutionService in strategy backtesting.

This example shows:
1. Creating a simple moving average crossover strategy
2. Integrating ExecutionService for realistic fill simulation
3. Tracking performance with realistic execution costs
4. Comparing with "perfect" instant fills
"""

from datetime import datetime, timedelta
from decimal import Decimal

from qtrader.contracts.data import Bar
from qtrader.services.execution.config import CommissionConfig, ExecutionConfig, SlippageConfig
from qtrader.services.execution.models import Order, OrderSide
from qtrader.services.execution.service import ExecutionService


class SimpleStrategy:
    """Simple moving average crossover strategy for demo purposes."""

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices: list[float] = []
        self.position = Decimal("0")

    def on_bar(self, bar: Bar) -> str | None:
        """Generate trading signals.

        Returns:
            'buy' if fast MA crosses above slow MA
            'sell' if fast MA crosses below slow MA
            None otherwise
        """
        self.prices.append(bar.close)

        if len(self.prices) < self.slow_period:
            return None

        # Calculate moving averages
        fast_ma = sum(self.prices[-self.fast_period :]) / self.fast_period
        slow_ma = sum(self.prices[-self.slow_period :]) / self.slow_period

        # Previous MAs
        if len(self.prices) < self.slow_period + 1:
            return None

        prev_fast = sum(self.prices[-self.fast_period - 1 : -1]) / self.fast_period
        prev_slow = sum(self.prices[-self.slow_period - 1 : -1]) / self.slow_period

        # Detect crossover
        if prev_fast <= prev_slow and fast_ma > slow_ma and self.position == Decimal("0"):
            return "buy"
        elif prev_fast >= prev_slow and fast_ma < slow_ma and self.position > Decimal("0"):
            return "sell"

        return None


def generate_price_bars(days: int = 50) -> list[Bar]:
    """Generate synthetic price data with trend and noise."""
    import random

    random.seed(42)

    bars = []
    base_date = datetime(2024, 1, 1, 9, 30)
    price = 150.0

    for i in range(days):
        # Add trend and noise
        trend = 0.1 if i < days // 2 else -0.1
        noise = random.uniform(-1.0, 1.0)
        price += trend + noise

        # Create OHLCV
        open_price = price
        high_price = price + random.uniform(0, 2.0)
        low_price = price - random.uniform(0, 2.0)
        close_price = price + random.uniform(-1.0, 1.0)
        volume = random.randint(500000, 2000000)

        bar = Bar(
            trade_datetime=base_date + timedelta(days=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        bars.append(bar)
        price = close_price

    return bars


def main():
    """Run mock execution testing example."""
    print("=" * 70)
    print("MOCK EXECUTION FOR STRATEGY TESTING")
    print("=" * 70)
    print("\nStrategy: Simple Moving Average Crossover (5/20)")
    print("Goal: Compare realistic execution vs. instant fills")
    print()

    # Step 1: Create execution service with realistic settings
    print("=" * 70)
    print("Setup")
    print("=" * 70)

    config = ExecutionConfig(
        market_order_queue_bars=1,
        max_participation_rate=Decimal("0.10"),  # Conservative 10%
        slippage=SlippageConfig(
            model="fixed_bps",
            params={"bps": Decimal("10")},  # 10 bps slippage
        ),
        commission=CommissionConfig(
            per_share=Decimal("0.01"),  # $0.01 per share
            minimum=Decimal("1.00"),
        ),
    )

    execution_service = ExecutionService(config)
    strategy = SimpleStrategy(fast_period=5, slow_period=20)

    print("Execution settings:")
    print(f"  Queue bars: {config.market_order_queue_bars}")
    print(f"  Max participation: {config.max_participation_rate * 100}%")
    print(f"  Slippage: {config.slippage.params['bps']} bps")
    print(f"  Commission: ${config.commission.per_share}/share")
    print()

    # Step 2: Generate price data
    print("Generating synthetic price data (50 days)...")
    bars = generate_price_bars(50)
    print(f"Generated {len(bars)} bars")
    print()

    # Step 3: Run backtest with realistic execution
    print("=" * 70)
    print("Running Backtest - Realistic Execution")
    print("=" * 70)
    print()

    cash = Decimal("100000.00")
    position_qty = Decimal("0")
    trades = []

    for i, bar in enumerate(bars):
        # Process existing orders first
        fills = execution_service.on_bar(bar)

        for fill in fills:
            if fill.side == "buy":
                cost = fill.quantity * fill.price + fill.commission
                cash -= cost
                position_qty += fill.quantity
                trades.append(("buy", fill.timestamp, fill.price, fill.quantity, fill.commission))
                print(
                    f"[{i:2d}] BUY FILLED:  {fill.quantity:6,.0f} @ ${fill.price:6.2f} (comm: ${fill.commission:5.2f})"
                )
            else:  # sell
                proceeds = fill.quantity * fill.price - fill.commission
                cash += proceeds
                position_qty -= fill.quantity
                trades.append(("sell", fill.timestamp, fill.price, fill.quantity, fill.commission))
                print(
                    f"[{i:2d}] SELL FILLED: {fill.quantity:6,.0f} @ ${fill.price:6.2f} (comm: ${fill.commission:5.2f})"
                )

        # Generate new signals
        signal = strategy.on_bar(bar)

        if signal == "buy" and position_qty == Decimal("0"):
            # Buy with available cash
            quantity = Decimal(int(cash / Decimal(str(bar.close)) // 100 * 100))  # Round to 100 shares
            if quantity >= 100:
                order = Order.market_order(
                    symbol="SPY",
                    side=OrderSide.BUY,
                    quantity=quantity,
                )
                execution_service.submit_order(order)
                strategy.position = quantity
                print(f"[{i:2d}] BUY SIGNAL:  Submitted order for {quantity:,} shares")

        elif signal == "sell" and position_qty > Decimal("0"):
            # Sell entire position
            order = Order.market_order(
                symbol="SPY",
                side=OrderSide.SELL,
                quantity=position_qty,
            )
            execution_service.submit_order(order)
            strategy.position = Decimal("0")
            print(f"[{i:2d}] SELL SIGNAL: Submitted order for {position_qty:,} shares")

    # Step 4: Calculate final P&L
    print("\n" + "=" * 70)
    print("RESULTS - Realistic Execution")
    print("=" * 70)

    final_value = cash
    if position_qty > 0:
        # Value remaining position at last close
        final_value += position_qty * Decimal(str(bars[-1].close))

    pnl = final_value - Decimal("100000.00")
    pnl_pct = (pnl / Decimal("100000.00")) * 100

    total_commission = sum(t[4] for t in trades)
    num_trades = len(trades)

    print(f"\nInitial capital:    ${100000:,.2f}")
    print(f"Final value:        ${final_value:,.2f}")
    print(f"P&L:                ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    print(f"Total trades:       {num_trades}")
    print(f"Total commissions:  ${total_commission:,.2f}")

    # Step 5: Compare with instant fills (unrealistic)
    print("\n" + "=" * 70)
    print("COMPARISON: Instant Fills (Unrealistic)")
    print("=" * 70)

    # Re-run without execution service
    strategy2 = SimpleStrategy(fast_period=5, slow_period=20)
    cash2 = Decimal("100000.00")
    position_qty2 = Decimal("0")
    trades2 = []

    for bar in bars:
        signal = strategy2.on_bar(bar)

        if signal == "buy" and position_qty2 == Decimal("0"):
            quantity = Decimal(int(cash2 / Decimal(str(bar.close)) // 100 * 100))
            if quantity >= 100:
                # Instant fill at bar close (unrealistic)
                cost = quantity * Decimal(str(bar.close))
                cash2 -= cost
                position_qty2 = quantity
                strategy2.position = quantity
                trades2.append(("buy", bar.trade_datetime, Decimal(str(bar.close)), quantity, Decimal("0")))

        elif signal == "sell" and position_qty2 > Decimal("0"):
            # Instant fill at bar close (unrealistic)
            proceeds = position_qty2 * Decimal(str(bar.close))
            cash2 += proceeds
            position_qty2 = Decimal("0")
            strategy2.position = Decimal("0")
            trades2.append(("sell", bar.trade_datetime, Decimal(str(bar.close)), position_qty2, Decimal("0")))

    final_value2 = cash2
    if position_qty2 > 0:
        final_value2 += position_qty2 * Decimal(str(bars[-1].close))

    pnl2 = final_value2 - Decimal("100000.00")
    pnl_pct2 = (pnl2 / Decimal("100000.00")) * 100

    print(f"\nInstant fills P&L:  ${pnl2:,.2f} ({pnl_pct2:+.2f}%)")
    print(f"Realistic exec P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    print(f"Difference:         ${pnl2 - pnl:,.2f} ({(pnl_pct2 - pnl_pct):+.2f}%)")

    print("\nRealistic execution includes:")
    print("  - Order queueing (1 bar delay)")
    print("  - Slippage (10 bps)")
    print("  - Commission ($0.01/share)")
    print("  - Volume constraints")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)

    print("\n1. EXECUTION MATTERS:")
    print("   - Instant fills overestimate returns")
    print("   - Realistic execution crucial for valid backtests")

    print("\n2. HIDDEN COSTS:")
    print(f"   - Commissions: ${total_commission:,.2f}")
    print("   - Slippage: built into fill prices")
    print("   - Market impact: participation limits")

    print("\n3. SIGNAL-TO-FILL DELAY:")
    print("   - Market orders queue for 1 bar")
    print("   - Price may move against you")
    print("   - More realistic than instant execution")

    print("\n4. PRODUCTION READINESS:")
    print("   - Same ExecutionService used in live trading")
    print("   - Backtest results more accurate")
    print("   - Fewer surprises in production")


if __name__ == "__main__":
    main()
