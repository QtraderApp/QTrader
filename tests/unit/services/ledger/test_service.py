"""Unit tests for PortfolioService - basic fill processing (Week 1).

Tests cover:
- Opening long positions (buy with no existing short)
- Opening short positions (sell with no existing long)
- Cash flow calculations
- Position tracking
- Query methods
- Input validation
- Ledger entry creation
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.services.ledger.models import LedgerEntryType, PortfolioConfig
from qtrader.services.ledger.service import PortfolioService


@pytest.fixture
def timestamp() -> datetime:
    """Standard timestamp for tests."""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def basic_config() -> PortfolioConfig:
    """Standard portfolio configuration."""
    return PortfolioConfig(
        initial_cash=Decimal("100000.00"),
        lot_method_long="fifo",
        lot_method_short="lifo",
    )


@pytest.fixture
def service(basic_config: PortfolioConfig) -> PortfolioService:
    """Standard portfolio service instance."""
    return PortfolioService(config=basic_config)


class TestServiceInitialization:
    """Test service initialization."""

    def test_initialize_with_config(self, basic_config: PortfolioConfig) -> None:
        """Test service initializes with configuration."""
        service = PortfolioService(config=basic_config)

        assert service.get_cash() == Decimal("100000.00")
        assert service.get_positions() == {}
        assert service.get_equity() == Decimal("100000.00")


class TestOpenLongPosition:
    """Test opening long positions (buy with no existing short)."""

    def test_buy_deducts_cash(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test buying stock deducts cash correctly."""
        # Buy 100 AAPL @ $150.00 with $10 commission
        # Cost = 100 * $150.00 + $10 = $15,010.00
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        expected_cash = Decimal("100000.00") - Decimal("15010.00")
        assert service.get_cash() == expected_cash

    def test_buy_creates_position(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test buying stock creates position entry."""
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        position = service.get_position("AAPL")
        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        # total_cost excludes commission (tracked separately in lot)
        assert position.total_cost == Decimal("15000.00")

    def test_buy_creates_ledger_entry(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test buying stock creates ledger entry."""
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        entries = service.get_ledger()
        assert len(entries) == 1
        assert entries[0].entry_type == LedgerEntryType.FILL
        assert entries[0].symbol == "AAPL"
        assert entries[0].quantity == Decimal("100")
        # cash_flow is negative for buy (cash out), includes commission
        assert entries[0].cash_flow == Decimal("-15010.00")
        assert entries[0].commission == Decimal("10.00")

    def test_buy_multiple_symbols(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test buying multiple symbols."""
        # Buy 100 AAPL @ $150.00
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Buy 50 TSLA @ $200.00
        service.apply_fill(
            fill_id="fill_002",
            timestamp=timestamp,
            symbol="TSLA",
            side="buy",
            quantity=Decimal("50"),
            price=Decimal("200.00"),
            commission=Decimal("5.00"),
        )

        positions = service.get_positions()
        assert len(positions) == 2
        assert positions["AAPL"].quantity == Decimal("100")
        assert positions["TSLA"].quantity == Decimal("50")

    def test_buy_adds_to_existing_position(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test adding to existing long position."""
        # First buy: 100 AAPL @ $150.00
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Second buy: 50 AAPL @ $155.00
        service.apply_fill(
            fill_id="fill_002",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("50"),
            price=Decimal("155.00"),
            commission=Decimal("5.00"),
        )

        position = service.get_position("AAPL")
        assert position is not None
        assert position.quantity == Decimal("150")  # 100 + 50
        # Cost: $15,000 + $7,750 = $22,750 (excludes commissions)
        assert position.total_cost == Decimal("22750.00")


class TestOpenShortPosition:
    """Test opening short positions (sell with no existing long)."""

    def test_short_sell_adds_cash(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test short selling credits cash correctly."""
        # Sell short 100 AAPL @ $150.00 with $10 commission
        # Proceeds = 100 * $150.00 - $10 = $14,990.00
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        expected_cash = Decimal("100000.00") + Decimal("14990.00")
        assert service.get_cash() == expected_cash

    def test_short_sell_creates_position(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test short selling creates position entry."""
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        position = service.get_position("AAPL")
        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("-100")  # Negative quantity
        # total_cost excludes commission (tracked separately)
        assert position.total_cost == Decimal("-15000.00")

    def test_short_sell_creates_ledger_entry(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test short selling creates ledger entry."""
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        entries = service.get_ledger()
        assert len(entries) == 1
        assert entries[0].entry_type == LedgerEntryType.FILL
        assert entries[0].symbol == "AAPL"
        assert entries[0].quantity == Decimal("-100")
        # cash_flow is positive for sell (cash in), minus commission
        assert entries[0].cash_flow == Decimal("14990.00")
        assert entries[0].commission == Decimal("10.00")


class TestInputValidation:
    """Test input validation for apply_fill."""

    def test_reject_zero_quantity(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test that zero quantity raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            service.apply_fill(
                fill_id="fill_001",
                timestamp=timestamp,
                symbol="AAPL",
                side="buy",
                quantity=Decimal("0"),
                price=Decimal("150.00"),
                commission=Decimal("10.00"),
            )

    def test_reject_negative_price(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test that negative price raises ValueError."""
        with pytest.raises(ValueError, match="Price must be positive"):
            service.apply_fill(
                fill_id="fill_001",
                timestamp=timestamp,
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                price=Decimal("-150.00"),
                commission=Decimal("10.00"),
            )

    def test_reject_negative_commission(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test that negative commission raises ValueError."""
        with pytest.raises(ValueError, match="Commission cannot be negative"):
            service.apply_fill(
                fill_id="fill_001",
                timestamp=timestamp,
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                price=Decimal("150.00"),
                commission=Decimal("-10.00"),
            )

    def test_reject_duplicate_fill_id(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test that duplicate fill_id raises ValueError."""
        # First fill succeeds
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Second fill with same ID fails
        with pytest.raises(ValueError, match="Fill ID fill_001 already exists"):
            service.apply_fill(
                fill_id="fill_001",
                timestamp=timestamp,
                symbol="AAPL",
                side="buy",
                quantity=Decimal("50"),
                price=Decimal("155.00"),
                commission=Decimal("5.00"),
            )


class TestUpdatePrices:
    """Test update_prices method."""

    def test_update_prices_changes_market_value(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test that update_prices changes position market values."""
        # Open position: 100 AAPL @ $150.00
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Update price to $160.00
        service.update_prices({"AAPL": Decimal("160.00")})

        position = service.get_position("AAPL")
        assert position is not None
        assert position.market_value == Decimal("16000.00")  # 100 * $160.00
        # Unrealized P&L = $16,000 - $15,000 (commission excluded from cost basis)
        assert position.unrealized_pnl == Decimal("1000.00")

    def test_update_prices_affects_equity(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test that price updates affect total equity."""
        # Open position: 100 AAPL @ $150.00
        service.apply_fill(
            fill_id="fill_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Update price to $160.00
        service.update_prices({"AAPL": Decimal("160.00")})

        # New equity = cash + market_value = $84,990 + $16,000
        new_equity = service.get_equity()
        assert new_equity == Decimal("100990.00")


class TestCloseLongPosition:
    """Test closing long positions (FIFO)."""

    def test_close_full_position(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test closing entire long position."""
        # Open: Buy 100 @ $150
        service.apply_fill(
            fill_id="buy_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        initial_cash = service.get_cash()

        # Close: Sell 100 @ $160
        service.apply_fill(
            fill_id="sell_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("160.00"),
            commission=Decimal("10.00"),
        )

        # Position should be flat
        position = service.get_position("AAPL")
        assert position is None or position.quantity == Decimal("0")

        # Cash flow:
        # Initial buy: -$15,010 (100 * $150 + $10)
        # Sell: +$15,990 (100 * $160 - $10)
        # Net: +$980
        final_cash = service.get_cash()
        assert final_cash == initial_cash + Decimal("15990.00")

        # Realized P&L: (160 - 150) * 100 - commissions
        # = $1,000 - $20 = $980
        realized_pnl = service.get_realized_pnl()
        assert realized_pnl == Decimal("980.00")

    def test_close_partial_position(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test partial close of long position."""
        # Open: Buy 100 @ $150
        service.apply_fill(
            fill_id="buy_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Close partial: Sell 60 @ $160
        service.apply_fill(
            fill_id="sell_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("60"),
            price=Decimal("160.00"),
            commission=Decimal("6.00"),
        )

        # Position should have 40 shares remaining
        position = service.get_position("AAPL")
        assert position is not None
        assert position.quantity == Decimal("40")

        # Realized P&L on 60 shares: (160 - 150) * 60 - (entry_comm + exit_comm)
        # Entry commission for 60 shares: $10 * (60/100) = $6
        # Exit commission: $6
        # = $600 - $12 = $588
        realized_pnl = service.get_realized_pnl(symbol="AAPL")
        assert realized_pnl == Decimal("588.00")

    def test_close_fifo_multiple_lots(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test FIFO closes oldest lots first."""
        # Buy 100 @ $150 (oldest)
        service.apply_fill(
            fill_id="buy_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Buy 50 @ $155 (middle)
        service.apply_fill(
            fill_id="buy_002",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("50"),
            price=Decimal("155.00"),
            commission=Decimal("5.00"),
        )

        # Buy 75 @ $160 (newest)
        service.apply_fill(
            fill_id="buy_003",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("75"),
            price=Decimal("160.00"),
            commission=Decimal("7.50"),
        )

        # Sell 130 @ $165 (should close oldest 100 + 30 from middle lot)
        service.apply_fill(
            fill_id="sell_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("130"),
            price=Decimal("165.00"),
            commission=Decimal("13.00"),
        )

        # Remaining position: 95 shares (20 from lot2 + 75 from lot3)
        position = service.get_position("AAPL")
        assert position is not None
        assert position.quantity == Decimal("95")

        # Realized P&L:
        # Lot 1 (100 shares): (165-150)*100 - ($10 entry + $10 exit) = $1,480
        #   Exit commission allocation: $13 * (100/130) = $10
        # Lot 2 (30 shares): (165-155)*30 - ($3 entry + $3 exit) = $294
        #   Entry commission allocation: $5 * (30/50) = $3
        #   Exit commission allocation: $13 * (30/130) = $3
        # Total = $1,480 + $294 = $1,774
        realized_pnl = service.get_realized_pnl(symbol="AAPL")
        assert realized_pnl == Decimal("1774.00")

    def test_close_creates_ledger_entry(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test that closing creates proper ledger entry."""
        # Open position
        service.apply_fill(
            fill_id="buy_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Close position
        service.apply_fill(
            fill_id="sell_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("160.00"),
            commission=Decimal("10.00"),
        )

        # Check ledger entries
        entries = service.get_ledger()
        assert len(entries) == 2

        # Second entry should be the close
        close_entry = entries[1]
        assert close_entry.entry_type == LedgerEntryType.FILL
        assert close_entry.symbol == "AAPL"
        assert close_entry.quantity == Decimal("-100")  # Negative for sell
        assert close_entry.price == Decimal("160.00")
        assert close_entry.cash_flow == Decimal("15990.00")  # 100 * 160 - 10
        assert close_entry.realized_pnl == Decimal("980.00")
        assert "FIFO" in close_entry.description


class TestCloseShortPosition:
    """Test closing short positions (LIFO)."""

    def test_close_full_short_position(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test closing entire short position."""
        # Open: Sell short 100 @ $150
        service.apply_fill(
            fill_id="sell_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        initial_cash = service.get_cash()

        # Close: Buy to cover 100 @ $140
        service.apply_fill(
            fill_id="buy_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("140.00"),
            commission=Decimal("10.00"),
        )

        # Position should be flat
        position = service.get_position("AAPL")
        assert position is None or position.quantity == Decimal("0")

        # Cash flow:
        # Initial short: +$14,990 (100 * $150 - $10)
        # Buy to cover: -$14,010 (100 * $140 + $10)
        # Net: +$980
        final_cash = service.get_cash()
        assert final_cash == initial_cash - Decimal("14010.00")

        # Realized P&L: (150 - 140) * 100 - commissions
        # = $1,000 - $20 = $980
        realized_pnl = service.get_realized_pnl()
        assert realized_pnl == Decimal("980.00")

    def test_close_partial_short_position(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test partial close of short position."""
        # Open: Sell short 100 @ $150
        service.apply_fill(
            fill_id="sell_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Close partial: Buy to cover 60 @ $140
        service.apply_fill(
            fill_id="buy_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("60"),
            price=Decimal("140.00"),
            commission=Decimal("6.00"),
        )

        # Position should have -40 shares remaining
        position = service.get_position("AAPL")
        assert position is not None
        assert position.quantity == Decimal("-40")

        # Realized P&L on 60 shares: (150 - 140) * 60 - (entry_comm + exit_comm)
        # Entry commission for 60 shares: $10 * (60/100) = $6
        # Exit commission: $6
        # = $600 - $12 = $588
        realized_pnl = service.get_realized_pnl(symbol="AAPL")
        assert realized_pnl == Decimal("588.00")

    def test_close_lifo_multiple_lots(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test LIFO closes newest lots first."""
        # Sell short 100 @ $150 (oldest)
        service.apply_fill(
            fill_id="sell_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Sell short 50 @ $155 (middle)
        service.apply_fill(
            fill_id="sell_002",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("50"),
            price=Decimal("155.00"),
            commission=Decimal("5.00"),
        )

        # Sell short 75 @ $160 (newest)
        service.apply_fill(
            fill_id="sell_003",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("75"),
            price=Decimal("160.00"),
            commission=Decimal("7.50"),
        )

        # Buy to cover 100 @ $145 (should close newest 75 + 25 from middle lot)
        service.apply_fill(
            fill_id="buy_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("145.00"),
            commission=Decimal("10.00"),
        )

        # Remaining position: -125 shares (oldest 100 + 25 from middle)
        position = service.get_position("AAPL")
        assert position is not None
        assert position.quantity == Decimal("-125")

        # Realized P&L:
        # Lot 3 (75 shares LIFO newest): (160-145)*75 - ($7.50 entry + $7.50 exit) = $1,110
        #   Exit commission allocation: $10 * (75/100) = $7.50
        # Lot 2 (25 shares): (155-145)*25 - ($2.50 entry + $2.50 exit) = $245
        #   Entry commission allocation: $5 * (25/50) = $2.50
        #   Exit commission allocation: $10 * (25/100) = $2.50
        # Total = $1,110 + $245 = $1,355
        realized_pnl = service.get_realized_pnl(symbol="AAPL")
        assert realized_pnl == Decimal("1355.00")

    def test_short_loss_scenario(
        self,
        service: PortfolioService,
        timestamp: datetime,
    ) -> None:
        """Test closing short at a loss."""
        # Sell short 100 @ $150
        service.apply_fill(
            fill_id="sell_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="sell",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Buy to cover at higher price @ $160 (loss)
        service.apply_fill(
            fill_id="buy_001",
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("160.00"),
            commission=Decimal("10.00"),
        )

        # Realized P&L: (150 - 160) * 100 - $20
        # = -$1,000 - $20 = -$1,020
        realized_pnl = service.get_realized_pnl()
        assert realized_pnl == Decimal("-1020.00")
