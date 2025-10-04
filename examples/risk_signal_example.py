"""
Example strategy using Signal-based risk management.

This demonstrates the Phase 2 Signal workflow:
1. Strategy generates Signal objects (trading intent without sizing)
2. Context evaluates signals through RiskManager
3. RiskManager sizes positions and creates Orders
4. Orders are submitted to ExecutionEngine
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from qtrader.api.context import Context
from qtrader.models.bar import Bar
from qtrader.risk import RiskManager, RiskPolicy, Signal, SignalDirection, SignalType, SizingMethod


class SimpleMomentumStrategy:
    """
    Simple momentum strategy using Signal-based API.

    Generates LONG signals when price crosses above 20-bar SMA.
    Generates EXIT signals when position exists and price crosses below SMA.
    """

    def __init__(self, lookback: int = 20):
        """
        Initialize strategy.

        Args:
            lookback: Moving average lookback period
        """
        self.lookback = lookback
        self.price_history: List[Decimal] = []

    def on_start(self, ctx: Context) -> None:
        """Called once before first bar."""
        print(f"Starting SimpleMomentumStrategy with {self.lookback}-bar MA")

    def on_bar(self, bar: Bar, ctx: Context) -> Optional[List[Signal]]:
        """
        Called for each bar. Returns list of signals.

        Args:
            bar: Current bar
            ctx: Context for portfolio access

        Returns:
            List of Signal objects (or None if no signals)
        """
        # Update price history
        self.price_history.append(bar.close)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)

        # Need full lookback before trading
        if len(self.price_history) < self.lookback:
            return None

        # Calculate simple moving average
        sma = sum(self.price_history) / len(self.price_history)

        # Get current position
        position = ctx.get_position(bar.symbol)

        # Generate signals based on price vs SMA
        signals = []

        if position.qty == 0:
            # No position - check for entry signal
            if bar.close > sma:
                # Price above SMA - generate LONG signal
                signal = Signal(
                    signal_id=f"entry_{bar.symbol}_{bar.ts.isoformat()}",
                    strategy_ts=bar.ts,
                    symbol=bar.symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    direction=SignalDirection.LONG,
                    conviction=Decimal("0.75"),  # Medium conviction
                    urgency="normal",
                    metadata={
                        "reason": "Price above SMA",
                        "price": str(bar.close),
                        "sma": str(sma),
                    },
                )
                signals.append(signal)

        elif position.qty > 0:
            # Have long position - check for exit signal
            if bar.close < sma:
                # Price below SMA - generate EXIT signal
                signal = Signal(
                    signal_id=f"exit_{bar.symbol}_{bar.ts.isoformat()}",
                    strategy_ts=bar.ts,
                    symbol=bar.symbol,
                    signal_type=SignalType.EXIT_LONG,
                    direction=SignalDirection.FLAT,
                    conviction=Decimal("1.0"),  # High conviction for exits
                    urgency="high",
                    metadata={
                        "reason": "Price below SMA",
                        "price": str(bar.close),
                        "sma": str(sma),
                    },
                )
                signals.append(signal)

        return signals if signals else None

    def on_fill(self, fill, ctx: Context) -> None:
        """Called after each fill."""
        print(f"Fill: {fill.order_side.value} {fill.qty} @ {fill.fill_price}")

    def on_end(self, ctx: Context) -> None:
        """Called once after last bar."""
        print(f"Strategy complete. Final equity: {ctx.get_equity()}")


def example_usage():
    """
    Example of running a strategy with risk management.

    This demonstrates the complete workflow:
    1. Create RiskPolicy with sizing rules
    2. Create RiskManager with policy
    3. Create Portfolio with initial capital
    4. Create Context linking everything
    5. Strategy generates Signals
    6. Context evaluates signals through RiskManager
    7. RiskManager creates sized Orders
    """
    from qtrader.models import Portfolio

    # Step 1: Configure risk policy
    policy = RiskPolicy(
        sizing_method=SizingMethod.PORTFOLIO_PERCENT,
        default_position_size=Decimal("0.10"),  # 10% of portfolio per position
        max_position_pct=Decimal("0.15"),  # Max 15% in any position
        cash_reserve_pct=Decimal("0.05"),  # Keep 5% cash reserve
        max_positions=5,  # Max 5 concurrent positions
        allow_shorting=False,  # Long only
        reject_on_insufficient_cash=True,
    )

    # Step 2: Create portfolio
    portfolio = Portfolio(initial_cash=Decimal("100000.00"))

    # Step 3: Create risk manager
    risk_manager = RiskManager(policy=policy, portfolio=portfolio)

    # Step 4: Create strategy
    strategy = SimpleMomentumStrategy(lookback=20)

    # Step 5: Create context
    context = Context(
        risk_manager=risk_manager,
        portfolio=portfolio,
        current_symbol="AAPL",
        current_price=Decimal("150.00"),
    )

    # Step 6: Initialize strategy
    strategy.on_start(context)

    # Step 7: Simulate bar and generate signals
    bar = Bar(
        ts=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
        symbol="AAPL",
        open=Decimal("149.50"),
        high=Decimal("151.00"),
        low=Decimal("149.00"),
        close=Decimal("150.50"),
        volume=1000000,
    )

    # Update context for current bar
    context.current_symbol = bar.symbol
    context.current_price = bar.close

    # Generate signals
    signals = strategy.on_bar(bar, context)

    if signals:
        for signal in signals:
            print(f"\nGenerated signal: {signal.signal_type.value} {signal.symbol}")

            # Evaluate signal through risk manager
            decision = context.evaluate_signal(signal)

            print(f"Risk decision: approved={decision.approved}, qty={decision.sized_qty}")

            if decision.approved:
                # Convert to order
                order = context.signal_to_order(signal, decision)
                print(f"Created order: {order.side.value} {order.qty} {order.symbol}")

                # Order would be submitted to ExecutionEngine here
    else:
        print("No signals generated")

    # Step 8: Get risk manager statistics
    stats = risk_manager.get_stats()
    print("\nRisk Manager Stats:")
    print(f"  Signals: {stats['signals_total']}")
    print(f"  Approved: {stats['signals_approved']}")
    print(f"  Rejected: {stats['signals_rejected']}")
    print(f"  Approval Rate: {stats['approval_rate']}")


if __name__ == "__main__":
    example_usage()
