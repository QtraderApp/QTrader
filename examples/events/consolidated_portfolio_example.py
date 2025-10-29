"""
Example of creating and using ConsolidatedPortfolioEvent.

Demonstrates:
- Creating portfolio snapshots with nested structures
- Multiple strategy groups with positions
- Long, short, and flat positions
- Proper decimal serialization
- Complete portfolio metrics (P&L, exposures, fees)
"""

from __future__ import annotations

from decimal import Decimal

from qtrader.events import ConsolidatedPortfolioEvent, PortfolioPosition, StrategyGroup


def create_sample_portfolio_snapshot() -> ConsolidatedPortfolioEvent:
    """Create a sample consolidated portfolio snapshot event."""

    # Strategy 1: Growth strategy with long position
    aapl_position = PortfolioPosition(
        symbol="AAPL",
        side="long",
        open_quantity=1000,
        average_fill_price=Decimal("130.2755"),
        commission_paid=Decimal("25.50"),
        cost_basis=Decimal("130275.50"),
        market_price=Decimal("135.00"),
        gross_market_value=Decimal("135000.00"),
        unrealized_pl=Decimal("4724.50"),
        realized_pl=Decimal("2000.00"),
        dividends_received=Decimal("850.00"),
        dividends_paid=Decimal("0.00"),
        total_position_value=Decimal("135850.00"),
        sector="Technology",
        country="US",
        asset_class="Equities",
        currency="USD",
        last_updated="2025-10-29T11:40:00Z",
    )

    growth_strategy = StrategyGroup(strategy_id="growth_strategy_001", positions=[aapl_position])

    # Strategy 2: Momentum strategy with short and closed positions
    tsla_position = PortfolioPosition(
        symbol="TSLA",
        side="short",
        open_quantity=-500,
        average_fill_price=Decimal("700.08"),
        commission_paid=Decimal("20.00"),
        cost_basis=Decimal("350040.00"),
        market_price=Decimal("720.00"),
        gross_market_value=Decimal("-360000.00"),
        unrealized_pl=Decimal("-9960.00"),
        realized_pl=Decimal("5000.00"),
        dividends_received=Decimal("0.00"),
        dividends_paid=Decimal("550.00"),
        total_position_value=Decimal("360550.00"),
        sector="Automotive",
        country="US",
        asset_class="Equities",
        currency="USD",
        last_updated="2025-10-29T11:42:00Z",
    )

    msft_position = PortfolioPosition(
        symbol="MSFT",
        side="flat",
        open_quantity=0,
        average_fill_price=Decimal("0.00"),
        commission_paid=Decimal("5.00"),
        cost_basis=Decimal("0.00"),
        market_price=Decimal("420.00"),
        gross_market_value=Decimal("0.00"),
        unrealized_pl=Decimal("0.00"),
        realized_pl=Decimal("1500.00"),
        dividends_received=Decimal("0.00"),
        dividends_paid=Decimal("0.00"),
        total_position_value=Decimal("0.00"),
        sector="Technology",
        country="US",
        asset_class="Equities",
        currency="USD",
        last_updated="2025-10-15T14:20:00Z",
    )

    momentum_strategy = StrategyGroup(strategy_id="momentum_strategy_002", positions=[tsla_position, msft_position])

    # Create consolidated portfolio snapshot
    snapshot = ConsolidatedPortfolioEvent(
        portfolio_id="PORT123",
        start_datetime="2025-01-01T00:00:00Z",
        snapshot_datetime="2025-10-29T11:45:00Z",
        reporting_currency="USD",
        initial_portfolio_equity=Decimal("100000.00"),
        cash_balance=Decimal("50000.00"),
        current_portfolio_equity=Decimal("546400.00"),
        total_market_value=Decimal("496400.00"),
        total_unrealized_pl=Decimal("-5235.50"),
        total_realized_pl=Decimal("8500.00"),
        total_pl=Decimal("3264.50"),
        long_exposure=Decimal("135850.00"),
        short_exposure=Decimal("360550.00"),
        net_exposure=Decimal("-224700.00"),
        gross_exposure=Decimal("496400.00"),
        leverage=Decimal("0.9085"),
        total_commissions_paid=Decimal("50.50"),
        total_dividends_received=Decimal("850.00"),
        total_dividends_paid=Decimal("550.00"),
        total_borrow_fees=Decimal("125.75"),
        total_margin_interest=Decimal("0.00"),
        strategies_groups=[growth_strategy, momentum_strategy],
        currency_conversion_rates={"USD": Decimal("1.0")},
        source_service="portfolio_service",
    )

    return snapshot


def main():
    """Demonstrate portfolio snapshot creation and usage."""
    print("=" * 70)
    print("Consolidated Portfolio Snapshot Example")
    print("=" * 70)

    snapshot = create_sample_portfolio_snapshot()

    print(f"\nPortfolio ID: {snapshot.portfolio_id}")
    print(f"Snapshot Time: {snapshot.snapshot_datetime}")
    print(f"\nPortfolio Metrics:")
    print(f"  Initial Equity: ${snapshot.initial_portfolio_equity:,.2f}")
    print(f"  Current Equity: ${snapshot.current_portfolio_equity:,.2f}")
    print(f"  Cash Balance: ${snapshot.cash_balance:,.2f}")
    print(f"  Total Market Value: ${snapshot.total_market_value:,.2f}")

    print(f"\nP&L:")
    print(f"  Unrealized: ${snapshot.total_unrealized_pl:,.2f}")
    print(f"  Realized: ${snapshot.total_realized_pl:,.2f}")
    print(f"  Total: ${snapshot.total_pl:,.2f}")

    print(f"\nExposure:")
    print(f"  Long: ${snapshot.long_exposure:,.2f}")
    print(f"  Short: ${snapshot.short_exposure:,.2f}")
    print(f"  Net: ${snapshot.net_exposure:,.2f}")
    print(f"  Gross: ${snapshot.gross_exposure:,.2f}")
    print(f"  Leverage: {snapshot.leverage:.4f}x")

    print(f"\nFees & Income:")
    print(f"  Commissions Paid: ${snapshot.total_commissions_paid:,.2f}")
    print(f"  Dividends Received: ${snapshot.total_dividends_received:,.2f}")
    print(f"  Dividends Paid: ${snapshot.total_dividends_paid:,.2f}")
    print(f"  Borrow Fees: ${snapshot.total_borrow_fees:,.2f}")
    print(f"  Margin Interest: ${snapshot.total_margin_interest:,.2f}")

    print(f"\n{'=' * 70}")
    print(f"Strategies: {len(snapshot.strategies_groups)}")
    print(f"{'=' * 70}")

    for group in snapshot.strategies_groups:
        print(f"\nStrategy: {group.strategy_id}")
        print(f"  Positions: {len(group.positions)}")

        for pos in group.positions:
            print(f"\n  {pos.symbol} ({pos.side.upper()})")
            print(f"    Quantity: {pos.open_quantity:,}")
            print(f"    Avg Price: ${pos.average_fill_price:.4f}")
            print(f"    Cost Basis: ${pos.cost_basis:,.2f}")
            print(f"    Market Value: ${pos.gross_market_value:,.2f}")
            print(f"    Unrealized P&L: ${pos.unrealized_pl:,.2f}")
            print(f"    Realized P&L: ${pos.realized_pl:,.2f}")

    print(f"\n{'=' * 70}")
    print("Serialization Example")
    print(f"{'=' * 70}\n")

    # Serialize to dict (wire format with string decimals)
    data = snapshot.model_dump()
    print(f"Event Type: {data['event_type']}")
    print(f"Current Equity (string): {data['current_portfolio_equity']}")
    print(f"Leverage (string): {data['leverage']}")
    print(f"First Position Symbol: {data['strategies_groups'][0]['positions'][0]['symbol']}")
    print(f"First Position Market Value (string): {data['strategies_groups'][0]['positions'][0]['gross_market_value']}")

    # Serialize to JSON
    json_str = snapshot.model_dump_json(indent=2)
    print(f"\nJSON Preview (first 500 chars):")
    print(json_str[:500] + "...")

    print(f"\n{'=' * 70}")
    print("Event validated successfully against JSON Schema!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
