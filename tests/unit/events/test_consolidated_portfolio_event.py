"""
Tests for ConsolidatedPortfolioEvent and nested models.

Validates:
- Event creation with nested structures
- Schema validation against portfolio/consolidated_portfolio.v1.json
- Decimal serialization for all numeric fields
- Optional fields handling
- Envelope validation
- Immutability of nested models
"""

from decimal import Decimal

import pytest

from qtrader.events import ConsolidatedPortfolioEvent, PortfolioPosition, StrategyGroup


class TestPortfolioPosition:
    """Test PortfolioPosition nested model."""

    def test_create_long_position(self):
        """Test creating a long position."""
        position = PortfolioPosition(
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

        assert position.symbol == "AAPL"
        assert position.side == "long"
        assert position.open_quantity == 1000
        assert position.average_fill_price == Decimal("130.2755")

    def test_create_short_position(self):
        """Test creating a short position with negative market value."""
        position = PortfolioPosition(
            symbol="TSLA",
            side="short",
            open_quantity=-500,
            average_fill_price=Decimal("700.04"),
            commission_paid=Decimal("20.00"),
            cost_basis=Decimal("350020.00"),
            market_price=Decimal("720.00"),
            gross_market_value=Decimal("-360000.00"),
            unrealized_pl=Decimal("-9980.00"),
            realized_pl=Decimal("5000.00"),
            dividends_received=Decimal("0.00"),
            dividends_paid=Decimal("550.00"),
            total_position_value=Decimal("360550.00"),
            currency="USD",
            last_updated="2025-10-29T11:42:00Z",
        )

        assert position.side == "short"
        assert position.open_quantity == -500
        assert position.gross_market_value == Decimal("-360000.00")
        assert position.unrealized_pl == Decimal("-9980.00")

    def test_create_flat_position(self):
        """Test creating a closed/flat position."""
        position = PortfolioPosition(
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
            currency="USD",
            last_updated="2025-10-15T14:20:00Z",
        )

        assert position.side == "flat"
        assert position.open_quantity == 0
        assert position.realized_pl == Decimal("1500.00")

    def test_position_serialization(self):
        """Test position serializes decimals to strings."""
        position = PortfolioPosition(
            symbol="AAPL",
            side="long",
            open_quantity=100,
            average_fill_price=Decimal("150.25"),
            commission_paid=Decimal("1.50"),
            cost_basis=Decimal("15026.50"),
            market_price=Decimal("155.00"),
            gross_market_value=Decimal("15500.00"),
            unrealized_pl=Decimal("473.50"),
            realized_pl=Decimal("0.00"),
            dividends_received=Decimal("0.00"),
            dividends_paid=Decimal("0.00"),
            total_position_value=Decimal("15500.00"),
            currency="USD",
            last_updated="2025-10-29T12:00:00Z",
        )

        data = position.model_dump()
        assert data["average_fill_price"] == "150.25"
        assert data["commission_paid"] == "1.50"
        assert data["cost_basis"] == "15026.50"

    def test_position_immutable(self):
        """Test position is immutable."""
        position = PortfolioPosition(
            symbol="AAPL",
            side="long",
            open_quantity=100,
            average_fill_price=Decimal("150.00"),
            commission_paid=Decimal("1.00"),
            cost_basis=Decimal("15001.00"),
            market_price=Decimal("155.00"),
            gross_market_value=Decimal("15500.00"),
            unrealized_pl=Decimal("499.00"),
            realized_pl=Decimal("0.00"),
            dividends_received=Decimal("0.00"),
            dividends_paid=Decimal("0.00"),
            total_position_value=Decimal("15500.00"),
            currency="USD",
            last_updated="2025-10-29T12:00:00Z",
        )

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            position.market_price = Decimal("160.00")


class TestStrategyGroup:
    """Test StrategyGroup nested model."""

    def test_create_strategy_group(self):
        """Test creating a strategy group with positions."""
        positions = [
            PortfolioPosition(
                symbol="AAPL",
                side="long",
                open_quantity=100,
                average_fill_price=Decimal("150.00"),
                commission_paid=Decimal("1.00"),
                cost_basis=Decimal("15001.00"),
                market_price=Decimal("155.00"),
                gross_market_value=Decimal("15500.00"),
                unrealized_pl=Decimal("499.00"),
                realized_pl=Decimal("0.00"),
                dividends_received=Decimal("0.00"),
                dividends_paid=Decimal("0.00"),
                total_position_value=Decimal("15500.00"),
                currency="USD",
                last_updated="2025-10-29T12:00:00Z",
            )
        ]

        group = StrategyGroup(strategy_id="growth_strategy_001", positions=positions)

        assert group.strategy_id == "growth_strategy_001"
        assert len(group.positions) == 1
        assert group.positions[0].symbol == "AAPL"

    def test_strategy_group_immutable(self):
        """Test strategy group is immutable."""
        group = StrategyGroup(strategy_id="test_strategy", positions=[])

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            group.strategy_id = "new_strategy"


class TestConsolidatedPortfolioEvent:
    """Test ConsolidatedPortfolioEvent."""

    def test_create_minimal_portfolio_event(self):
        """Test creating portfolio event with required fields only."""
        event = ConsolidatedPortfolioEvent(
            portfolio_id="PORT123",
            start_datetime="2025-01-01T00:00:00Z",
            snapshot_datetime="2025-10-29T11:45:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            current_portfolio_equity=Decimal("150000.00"),
            total_market_value=Decimal("100000.00"),
            total_unrealized_pl=Decimal("5000.00"),
            total_realized_pl=Decimal("3000.00"),
            total_pl=Decimal("8000.00"),
            long_exposure=Decimal("100000.00"),
            short_exposure=Decimal("0.00"),
            net_exposure=Decimal("100000.00"),
            gross_exposure=Decimal("100000.00"),
            leverage=Decimal("0.67"),
            strategies_groups=[],
            source_service="portfolio_service",
        )

        assert event.portfolio_id == "PORT123"
        assert event.event_type == "consolidated_portfolio"
        assert event.SCHEMA_BASE == "portfolio/consolidated_portfolio"
        assert event.current_portfolio_equity == Decimal("150000.00")

    def test_create_complete_portfolio_event(self):
        """Test creating portfolio event with all fields."""
        positions = [
            PortfolioPosition(
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
        ]

        groups = [StrategyGroup(strategy_id="growth_strategy_001", positions=positions)]

        event = ConsolidatedPortfolioEvent(
            portfolio_id="PORT123",
            start_datetime="2025-01-01T00:00:00Z",
            snapshot_datetime="2025-10-29T11:45:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            current_portfolio_equity=Decimal("546400.00"),
            total_market_value=Decimal("496400.00"),
            total_unrealized_pl=Decimal("-5255.50"),
            total_realized_pl=Decimal("8500.00"),
            total_pl=Decimal("3244.50"),
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
            strategies_groups=groups,
            currency_conversion_rates={"USD": Decimal("1.0")},
            source_service="portfolio_service",
        )

        assert event.portfolio_id == "PORT123"
        assert event.total_commissions_paid == Decimal("50.50")
        assert len(event.strategies_groups) == 1
        assert event.strategies_groups[0].strategy_id == "growth_strategy_001"

    def test_portfolio_event_serialization(self):
        """Test portfolio event serializes decimals to strings."""
        event = ConsolidatedPortfolioEvent(
            portfolio_id="PORT123",
            start_datetime="2025-01-01T00:00:00Z",
            snapshot_datetime="2025-10-29T11:45:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            current_portfolio_equity=Decimal("150000.00"),
            total_market_value=Decimal("100000.00"),
            total_unrealized_pl=Decimal("5000.00"),
            total_realized_pl=Decimal("3000.00"),
            total_pl=Decimal("8000.00"),
            long_exposure=Decimal("100000.00"),
            short_exposure=Decimal("0.00"),
            net_exposure=Decimal("100000.00"),
            gross_exposure=Decimal("100000.00"),
            leverage=Decimal("0.67"),
            strategies_groups=[],
            currency_conversion_rates={"USD": Decimal("1.0"), "EUR": Decimal("0.85")},
            source_service="portfolio_service",
        )

        data = event.model_dump()
        assert data["initial_portfolio_equity"] == "100000.00"
        assert data["cash_balance"] == "50000.00"
        assert data["leverage"] == "0.67"
        assert data["currency_conversion_rates"]["USD"] == "1.0"
        assert data["currency_conversion_rates"]["EUR"] == "0.85"

    def test_portfolio_event_with_multiple_strategies(self):
        """Test portfolio event with multiple strategy groups."""
        positions1 = [
            PortfolioPosition(
                symbol="AAPL",
                side="long",
                open_quantity=100,
                average_fill_price=Decimal("150.00"),
                commission_paid=Decimal("1.00"),
                cost_basis=Decimal("15001.00"),
                market_price=Decimal("155.00"),
                gross_market_value=Decimal("15500.00"),
                unrealized_pl=Decimal("499.00"),
                realized_pl=Decimal("0.00"),
                dividends_received=Decimal("0.00"),
                dividends_paid=Decimal("0.00"),
                total_position_value=Decimal("15500.00"),
                currency="USD",
                last_updated="2025-10-29T12:00:00Z",
            )
        ]

        positions2 = [
            PortfolioPosition(
                symbol="TSLA",
                side="short",
                open_quantity=-50,
                average_fill_price=Decimal("700.00"),
                commission_paid=Decimal("2.00"),
                cost_basis=Decimal("35002.00"),
                market_price=Decimal("720.00"),
                gross_market_value=Decimal("-36000.00"),
                unrealized_pl=Decimal("-998.00"),
                realized_pl=Decimal("0.00"),
                dividends_received=Decimal("0.00"),
                dividends_paid=Decimal("50.00"),
                total_position_value=Decimal("36050.00"),
                currency="USD",
                last_updated="2025-10-29T12:00:00Z",
            )
        ]

        groups = [
            StrategyGroup(strategy_id="growth_strategy", positions=positions1),
            StrategyGroup(strategy_id="momentum_strategy", positions=positions2),
        ]

        event = ConsolidatedPortfolioEvent(
            portfolio_id="PORT123",
            start_datetime="2025-01-01T00:00:00Z",
            snapshot_datetime="2025-10-29T11:45:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            current_portfolio_equity=Decimal("101550.00"),
            total_market_value=Decimal("51550.00"),
            total_unrealized_pl=Decimal("-499.00"),
            total_realized_pl=Decimal("0.00"),
            total_pl=Decimal("-499.00"),
            long_exposure=Decimal("15500.00"),
            short_exposure=Decimal("36050.00"),
            net_exposure=Decimal("-20550.00"),
            gross_exposure=Decimal("51550.00"),
            leverage=Decimal("0.51"),
            strategies_groups=groups,
            source_service="portfolio_service",
        )

        assert len(event.strategies_groups) == 2
        assert event.strategies_groups[0].strategy_id == "growth_strategy"
        assert event.strategies_groups[1].strategy_id == "momentum_strategy"
        assert len(event.strategies_groups[0].positions) == 1
        assert len(event.strategies_groups[1].positions) == 1

    def test_portfolio_event_immutable(self):
        """Test portfolio event is immutable."""
        event = ConsolidatedPortfolioEvent(
            portfolio_id="PORT123",
            start_datetime="2025-01-01T00:00:00Z",
            snapshot_datetime="2025-10-29T11:45:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            current_portfolio_equity=Decimal("150000.00"),
            total_market_value=Decimal("100000.00"),
            total_unrealized_pl=Decimal("5000.00"),
            total_realized_pl=Decimal("3000.00"),
            total_pl=Decimal("8000.00"),
            long_exposure=Decimal("100000.00"),
            short_exposure=Decimal("0.00"),
            net_exposure=Decimal("100000.00"),
            gross_exposure=Decimal("100000.00"),
            leverage=Decimal("0.67"),
            strategies_groups=[],
            source_service="portfolio_service",
        )

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            event.cash_balance = Decimal("60000.00")

    def test_portfolio_event_envelope_validation(self):
        """Test portfolio event validates envelope fields."""
        event = ConsolidatedPortfolioEvent(
            portfolio_id="PORT123",
            start_datetime="2025-01-01T00:00:00Z",
            snapshot_datetime="2025-10-29T11:45:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            current_portfolio_equity=Decimal("150000.00"),
            total_market_value=Decimal("100000.00"),
            total_unrealized_pl=Decimal("5000.00"),
            total_realized_pl=Decimal("3000.00"),
            total_pl=Decimal("8000.00"),
            long_exposure=Decimal("100000.00"),
            short_exposure=Decimal("0.00"),
            net_exposure=Decimal("100000.00"),
            gross_exposure=Decimal("100000.00"),
            leverage=Decimal("0.67"),
            strategies_groups=[],
            source_service="portfolio_service",
        )

        # Envelope fields should be present
        assert event.event_id is not None
        assert event.event_type == "consolidated_portfolio"
        assert event.event_version == 1
        assert event.occurred_at is not None
        assert event.source_service == "portfolio_service"

    def test_schema_base_matches_event_type(self):
        """Test SCHEMA_BASE contract name matches event_type."""
        event = ConsolidatedPortfolioEvent(
            portfolio_id="PORT123",
            start_datetime="2025-01-01T00:00:00Z",
            snapshot_datetime="2025-10-29T11:45:00Z",
            reporting_currency="USD",
            initial_portfolio_equity=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            current_portfolio_equity=Decimal("150000.00"),
            total_market_value=Decimal("100000.00"),
            total_unrealized_pl=Decimal("5000.00"),
            total_realized_pl=Decimal("3000.00"),
            total_pl=Decimal("8000.00"),
            long_exposure=Decimal("100000.00"),
            short_exposure=Decimal("0.00"),
            net_exposure=Decimal("100000.00"),
            gross_exposure=Decimal("100000.00"),
            leverage=Decimal("0.67"),
            strategies_groups=[],
            source_service="portfolio_service",
        )

        # event_type should match the contract name from SCHEMA_BASE
        assert event.SCHEMA_BASE == "portfolio/consolidated_portfolio"
        assert event.event_type == "consolidated_portfolio"
