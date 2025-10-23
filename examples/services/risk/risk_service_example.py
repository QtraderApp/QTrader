"""Example: Using RiskService with YAML Configuration

This script demonstrates how to:
1. Load RiskConfig from a YAML file
2. Create and configure a RiskService
3. Simulate signal processing with risk evaluation
4. Handle approvals and rejections

Run this from the project root:
    python examples/services/risk_service_example.py
"""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from qtrader.events.event_bus import EventBus
from qtrader.events.events import OrderApprovedEvent, OrderRejectedEvent, RiskEvaluationTriggerEvent, SignalEvent
from qtrader.services.portfolio_manager.config_loader import load_risk_config
from qtrader.services.portfolio_manager.models import PortfolioState, Position
from qtrader.services.portfolio_manager.service import RiskService


def main():
    """Run RiskService example with YAML config."""
    print("=" * 70)
    print("RiskService Example: Multi-Strategy Risk Management")
    print("=" * 70)
    print()

    # Step 1: Load configuration from YAML
    config_path = Path(__file__).parent / "risk_example.yaml"
    print(f"📄 Loading config from: {config_path}")
    config = load_risk_config(config_path)
    print("✅ Config loaded successfully!")
    print(f"   - Strategies: {[b.strategy_id for b in config.budgets]}")
    print(f"   - Capital allocation: {[f'{b.capital_weight:.0%}' for b in config.budgets]}")
    print(f"   - Concentration limit: {config.concentration.max_position_pct:.0%}")
    print(f"   - Leverage limits: {config.leverage.max_gross:.1f}x gross, {config.leverage.max_net:.1f}x net")
    print()

    # Step 2: Set up event bus and handlers
    event_bus = EventBus()
    approvals = []
    rejections = []

    def on_approval(event: OrderApprovedEvent):
        """Collect approved orders."""
        approvals.append(event)
        print(f"✅ APPROVED: {event.symbol} {event.side} {event.quantity} shares")
        print(f"   Strategy: {event.strategy_id}, Reason: {event.reason}")

    def on_rejection(event: OrderRejectedEvent):
        """Collect rejected signals."""
        rejections.append(event)
        print(f"❌ REJECTED: {event.symbol} {event.side}")
        print(f"   Strategy: {event.strategy_id}, Reason: {event.reason}")

    event_bus.subscribe("order_approved", on_approval)  # pyright: ignore[reportArgumentType]
    event_bus.subscribe("order_rejected", on_rejection)  # pyright: ignore[reportArgumentType]

    # Step 3: Create RiskService
    print("🔧 Creating RiskService...")
    risk_service = RiskService(event_bus=event_bus, config=config)

    # Subscribe RiskService to events
    event_bus.subscribe("signal", risk_service.on_signal)  # pyright: ignore[reportArgumentType]
    event_bus.subscribe("risk_evaluation_trigger", risk_service.on_risk_evaluation_trigger)  # pyright: ignore[reportArgumentType]

    print("✅ RiskService initialized")
    print()

    # Current timestamp for all events
    ts = datetime.now(timezone.utc)

    # Step 4: Set portfolio state (starting position)
    print("💰 Setting portfolio state:")
    initial_equity = Decimal("1000000.00")  # $1M equity
    portfolio_state = PortfolioState(
        ts=ts,
        equity=initial_equity,
        cash=initial_equity,
        gross_exposure=Decimal("0"),
        net_exposure=Decimal("0"),
        positions={},
    )
    risk_service.on_portfolio_state(portfolio_state)
    print(f"   Equity: ${initial_equity:,.2f}")
    print("   Positions: None (starting fresh)")
    print()

    # Step 5: Simulate signal generation
    print("📊 Simulating signal generation...")

    # Momentum strategy signals (60% allocation)
    signals = [
        SignalEvent(
            strategy_id="momentum_v1",
            symbol="AAPL",
            side="BUY",
            strength=0.8,
            metadata={"price": 150.00},
            ts=ts,
        ),
        SignalEvent(
            strategy_id="momentum_v1",
            symbol="MSFT",
            side="BUY",
            strength=0.6,
            metadata={"price": 300.00},
            ts=ts,
        ),
    ]

    # Mean reversion signals (40% allocation)
    signals.extend(
        [
            SignalEvent(
                strategy_id="mean_reversion_v1",
                symbol="GOOGL",
                side="BUY",
                strength=0.9,
                metadata={"price": 140.00},
                ts=ts,
            ),
            SignalEvent(
                strategy_id="mean_reversion_v1",
                symbol="TSLA",
                side="SELL",
                strength=0.7,
                metadata={"price": 250.00},
                ts=ts,
            ),
        ]
    )

    # Publish signals
    for signal in signals:
        event_bus.publish(signal)
        print(f"   📤 Signal: {signal.strategy_id} → {signal.symbol} {signal.side} (strength={signal.strength})")

    print()

    # Step 6: Trigger risk evaluation
    print("🎯 Triggering risk evaluation...")
    trigger = RiskEvaluationTriggerEvent(ts=ts)
    event_bus.publish(trigger)
    print()

    # Step 7: Display results
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"✅ Approved: {len(approvals)}")
    print(f"❌ Rejected: {len(rejections)}")
    print()

    if approvals:
        print("Approved Orders:")
        for event in approvals:
            print(f"  - {event.symbol:6} {event.side:4} {event.quantity:6} shares")
            print(f"    Reason: {event.reason}")
        print()

    if rejections:
        print("Rejected Signals:")
        for event in rejections:
            print(f"  - {event.symbol:6} {event.side:4} → {event.reason}")
    print()

    # Step 8: Demonstrate with existing position (concentration limit)
    print("=" * 70)
    print("SCENARIO 2: Testing Concentration Limit with Existing Position")
    print("=" * 70)
    print()

    # Reset for next scenario
    approvals.clear()
    rejections.clear()

    # Add existing AAPL position (9% of equity)
    position_price = Decimal("150.00")
    position_qty = 600
    position_value = position_price * Decimal(str(position_qty))
    existing_position = Position(
        symbol="AAPL",
        quantity=position_qty,
        market_value=position_value,
    )

    # Update portfolio state with existing position
    updated_portfolio_state = PortfolioState(
        ts=ts,
        equity=initial_equity,
        cash=initial_equity - position_value,
        gross_exposure=position_value,
        net_exposure=position_value,
        positions={"AAPL": existing_position},
    )
    risk_service.on_portfolio_state(updated_portfolio_state)

    print(f"📊 Existing position: AAPL {existing_position.quantity} shares @ ${position_price}")
    print(f"   Position value: ${position_value:,.2f} ({float(position_value / initial_equity):.1%} of equity)")
    print()

    # Try to add more AAPL (should hit 10% concentration limit)
    print("📤 New signal: Add more AAPL...")
    large_signal = SignalEvent(
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=1.0,  # Maximum strength
        metadata={"price": 150.00},
        ts=ts,
    )
    event_bus.publish(large_signal)

    # Trigger evaluation
    trigger2 = RiskEvaluationTriggerEvent(ts=ts)
    event_bus.publish(trigger2)
    print()

    print("Results:")
    if rejections:
        print(f"❌ Signal rejected: {rejections[-1].reason}")
        print("   → Concentration limit protected portfolio from over-exposure!")
    print()

    print("=" * 70)
    print("Example complete! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    main()
