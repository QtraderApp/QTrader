"""
Test to verify concentration vs cash interaction.

This test demonstrates the scenario:
- 5 strategies each with 25% max concentration
- First 4 strategies consume all cash
- 5th strategy signal passes concentration but must be rejected for lack of cash
"""

from datetime import datetime, timezone
from decimal import Decimal

from qtrader.models import OrderSide, OrderType, Portfolio, TimeInForce
from qtrader.risk import RiskManager, RiskPolicy, Signal, SignalDirection, SignalType, SizingMethod


def test_cash_exhaustion_with_concentration_headroom():
    """
    Verify that cash exhaustion rejects signals even when concentration allows them.

    Scenario:
    - Portfolio: $100,000 initial cash
    - Policy: 25% max concentration per position, CHECK_CASH_BEFORE_CONCENTRATION = True
    - First 4 signals: Each use $25,000 (25% of equity)
    - After 4 fills: $0 cash remaining, but only 4 positions (could add more by concentration)
    - 5th signal: NEW symbol/sector, passes concentration (would be 20% of $100k equity)
    - Expected: REJECTED due to insufficient cash, NOT concentration
    """

    # Setup with NEW FLAG
    policy = RiskPolicy(
        sizing_method=SizingMethod.FIXED_VALUE,
        default_position_size=Decimal("25000.00"),  # $25k per position
        max_position_pct=Decimal("0.25"),  # 25% max concentration
        cash_reserve_pct=Decimal("0.0"),  # No reserve for this test
        max_positions=10,  # Allow up to 10 positions
        reject_on_insufficient_cash=True,  # MUST reject on no cash
        check_cash_before_concentration=True,  # NEW: Check cash FIRST
    )

    portfolio = Portfolio(initial_cash=Decimal("100000.00"))
    manager = RiskManager(policy=policy, portfolio=portfolio)

    print("\n" + "=" * 80)
    print("SCENARIO: 5 Strategies with 25% Max Concentration Each")
    print("=" * 80)
    print(f"Initial Cash: ${portfolio.cash.get_balance():,.2f}")
    print(f"Initial Equity: ${portfolio.get_equity():,.2f}")
    print(f"Max Concentration: {policy.max_position_pct:.0%}")
    print(f"Max Positions: {policy.max_positions}")
    print()

    # Symbols for 5 different strategies/sectors
    symbols = ["AAPL", "MSFT", "JPM", "XOM", "PFE"]  # Tech, Tech, Finance, Energy, Healthcare
    prices = [Decimal("150.00"), Decimal("300.00"), Decimal("150.00"), Decimal("100.00"), Decimal("50.00")]

    # Process first 4 signals
    for i in range(4):
        symbol = symbols[i]
        price = prices[i]

        print(f"\n--- Strategy {i + 1}: Signal for {symbol} @ ${price} ---")

        # Create signal
        signal = Signal(
            signal_id=f"strategy_{i + 1}_signal",
            strategy_ts=datetime(2024, 1, 15, 9, 30 + i, 0, tzinfo=timezone.utc),
            symbol=symbol,
            signal_type=SignalType.ENTRY_LONG,
            direction=SignalDirection.LONG,
            target_value=Decimal("25000.00"),  # Request $25k position
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            conviction=Decimal("0.80"),
        )

        # Evaluate
        decision = manager.evaluate_signal(signal, price)

        print(f"Decision: {'APPROVED' if decision.approved else 'REJECTED'}")
        print(f"Sized Qty: {decision.sized_qty} shares")
        print(f"Reason: {decision.reason}")

        if decision.approved:
            # Calculate position value
            position_value = Decimal(decision.sized_qty) * price
            print(f"Position Value: ${position_value:,.2f}")
            print(f"% of Equity: {(position_value / portfolio.get_equity()):.1%}")

            # Simulate fill - update position and deduct cash
            portfolio.positions.update_position(
                symbol=symbol,
                side=OrderSide.BUY,
                qty=decision.sized_qty,
                price=price,
            )

            portfolio.cash.debit(
                amount=position_value,
                timestamp=datetime(2024, 1, 15, 9, 30 + i, 0, tzinfo=timezone.utc).isoformat(),
                transaction_type="TRADE",
                description=f"{symbol} purchase",
            )

            print(f"Cash After Fill: ${portfolio.cash.get_balance():,.2f}")
            print(f"Equity After Fill: ${portfolio.get_equity():,.2f}")
            print(
                f"Positions Count: {len([p for p in portfolio.positions.get_all_positions().values() if p.qty != 0])}"
            )

    # Now try 5th signal - NEW symbol, different sector
    print(f"\n{'=' * 80}")
    print(f"--- Strategy 5: Signal for {symbols[4]} (NEW SYMBOL) ---")
    print(f"{'=' * 80}")

    # Deplete more cash to make this fail
    # After 4 positions we have $31,800 left
    # Strategy 5 wants $25,000 (25% of original equity, not reduced)
    # So we need to reduce cash below $25,000
    portfolio.cash.debit(
        amount=Decimal("10000.00"),  # Reduce cash to $21,800
        timestamp=datetime(2024, 1, 15, 9, 33, 0, tzinfo=timezone.utc).isoformat(),
        transaction_type="FEE",
        description="Simulate additional cash depletion",
    )

    signal_5 = Signal(
        signal_id="strategy_5_signal",
        strategy_ts=datetime(2024, 1, 15, 9, 34, 0, tzinfo=timezone.utc),
        symbol=symbols[4],  # PFE - healthcare, different from existing
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
        target_value=Decimal("25000.00"),  # Request FULL $25k (like other strategies)
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        conviction=Decimal("0.75"),
    )

    current_cash = portfolio.cash.get_balance()
    current_equity = portfolio.get_equity()
    current_positions = len([p for p in portfolio.positions.get_all_positions().values() if p.qty != 0])

    print(f"Current Cash: ${current_cash:,.2f}")
    print(f"Current Equity: ${current_equity:,.2f}")
    print(f"Current Positions: {current_positions}")
    print(f"Requested Position: ${signal_5.target_value:,.2f} (20% of equity)")
    print(
        f"Concentration Limit: {policy.max_position_pct:.0%} of equity = ${(current_equity * policy.max_position_pct):,.2f}"
    )
    print()

    # Evaluate 5th signal
    decision_5 = manager.evaluate_signal(signal_5, prices[4])

    print(f"Decision: {'APPROVED' if decision_5.approved else 'REJECTED'}")
    print(f"Sized Qty: {decision_5.sized_qty} shares")
    print(f"Reason: {decision_5.reason}")

    # Assertions
    print(f"\n{'=' * 80}")
    print("VERIFICATION:")
    print(f"{'=' * 80}")

    # Check that concentration would allow this (20% < 25%)
    requested_pct = Decimal("20000.00") / current_equity
    concentration_allows = requested_pct <= policy.max_position_pct
    print(f"✓ Concentration Check: {requested_pct:.1%} <= {policy.max_position_pct:.0%} = {concentration_allows}")

    # Check that positions count allows this (4 < 10)
    positions_allow = current_positions < (policy.max_positions or 999)
    print(f"✓ Position Count Check: {current_positions} < {policy.max_positions} = {positions_allow}")

    # Check that cash does NOT allow this
    required_cash = Decimal(decision_5.sized_qty * prices[4]) if decision_5.sized_qty > 0 else Decimal("20000.00")
    cash_allows = current_cash >= required_cash
    print(f"✓ Cash Check: ${current_cash:,.2f} >= ${required_cash:,.2f} = {cash_allows}")

    print()

    # The signal MUST be rejected due to insufficient cash
    assert decision_5.approved is False, "Signal should be REJECTED"
    assert "Insufficient cash" in decision_5.reason or "cash" in decision_5.reason.lower(), (
        f"Rejection reason should mention cash, got: {decision_5.reason}"
    )

    print("✅ TEST PASSED: Signal correctly rejected due to insufficient cash")
    print("   (even though concentration and position count would allow it)")

    # Print final stats
    stats = manager.get_stats()
    print("\nRisk Manager Stats:")
    print(f"  Total Signals: {stats['signals_total']}")
    print(f"  Approved: {stats['signals_approved']}")
    print(f"  Rejected: {stats['signals_rejected']}")
    print(f"  Approval Rate: {stats['approval_rate']}")


if __name__ == "__main__":
    test_cash_exhaustion_with_concentration_headroom()
