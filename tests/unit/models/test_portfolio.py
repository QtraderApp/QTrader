"""Tests for Portfolio (unified account state)."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.models.order import OrderSide
from qtrader.models.portfolio import Portfolio


@pytest.fixture
def portfolio():
    """Create portfolio with $100k initial cash."""
    return Portfolio(initial_cash=Decimal("100000.00"))


def test_portfolio_initialization():
    """Portfolio should initialize with starting cash."""
    portfolio = Portfolio(initial_cash=Decimal("50000.00"))

    assert portfolio.cash.get_balance() == Decimal("50000.00")
    assert len(portfolio.positions.get_all_positions()) == 0
    assert portfolio.get_equity() == Decimal("50000.00")


def test_portfolio_apply_fill_buy(portfolio):
    """Apply BUY fill should debit cash and create long position."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.05"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    # Cash should be debited (qty * price + commission)
    expected_cash = Decimal("100000.00") - (Decimal("100") * Decimal("150.00") + Decimal("1.05"))
    assert portfolio.cash.get_balance() == Decimal("84998.95")
    assert portfolio.cash.get_balance() == expected_cash

    # Position should be created
    pos = portfolio.get_position("AAPL")
    assert pos is not None
    assert pos.qty == 100
    assert pos.avg_price == Decimal("150.00")


def test_portfolio_apply_fill_sell(portfolio):
    """Apply SELL fill should credit cash and create short position."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    portfolio.apply_fill(
        symbol="MSFT",
        side=OrderSide.SELL,
        qty=50,
        fill_price=Decimal("300.00"),
        commission=Decimal("1.25"),
        ts=ts,
        order_id="order-2",
        fill_id="fill-2",
    )

    # Cash should be credited (qty * price - commission)
    expected_cash = Decimal("100000.00") + (Decimal("50") * Decimal("300.00") - Decimal("1.25"))
    assert portfolio.cash.get_balance() == Decimal("114998.75")
    assert portfolio.cash.get_balance() == expected_cash

    # Position should be short
    pos = portfolio.get_position("MSFT")
    assert pos is not None
    assert pos.qty == -50
    assert pos.avg_price == Decimal("300.00")


def test_portfolio_multiple_fills_same_symbol(portfolio):
    """Multiple fills on same symbol should average cost."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # First buy: 100 @ $150
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    # Second buy: 50 @ $160
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=50,
        fill_price=Decimal("160.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-2",
        fill_id="fill-2",
    )

    # Position should have averaged cost
    pos = portfolio.get_position("AAPL")
    assert pos.qty == 150

    # Average cost: (100 * 150 + 50 * 160) / 150 = 23000 / 150 = 153.33...
    expected_avg = (Decimal("100") * Decimal("150.00") + Decimal("50") * Decimal("160.00")) / Decimal("150")
    assert pos.avg_price == expected_avg


def test_portfolio_fill_validation(portfolio):
    """Fill validation should reject invalid inputs."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # qty <= 0 should fail
    with pytest.raises(ValueError, match="Fill qty must be > 0"):
        portfolio.apply_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=0,
            fill_price=Decimal("150.00"),
            commission=Decimal("1.00"),
            ts=ts,
            order_id="order-1",
            fill_id="fill-1",
        )

    # Negative commission should fail
    with pytest.raises(ValueError, match="Commission must be >= 0"):
        portfolio.apply_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,
            fill_price=Decimal("150.00"),
            commission=Decimal("-1.00"),
            ts=ts,
            order_id="order-1",
            fill_id="fill-1",
        )


def test_portfolio_short_dividend(portfolio):
    """Short dividend should debit cash when short."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Create short position
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    initial_cash = portfolio.cash.get_balance()

    # Apply dividend
    portfolio.apply_short_dividend(
        symbol="AAPL",
        dividend_per_share=Decimal("0.50"),
        ts=ts,
    )

    # Should debit cash (100 shares * $0.50/share = $50)
    expected_cash = initial_cash - Decimal("50.00")
    assert portfolio.cash.get_balance() == expected_cash


def test_portfolio_no_dividend_when_long(portfolio):
    """No dividend debit when position is long (apply_short_dividend)."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Create long position
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    initial_cash = portfolio.cash.get_balance()

    # Apply dividend (should be no-op for long)
    portfolio.apply_short_dividend(
        symbol="AAPL",
        dividend_per_share=Decimal("0.50"),
        ts=ts,
    )

    # Cash should be unchanged
    assert portfolio.cash.get_balance() == initial_cash


def test_portfolio_long_dividend(portfolio):
    """Long positions should receive dividend credits."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Create long position: 200 shares @ $150
    portfolio.apply_fill(
        symbol="MSFT",
        side=OrderSide.BUY,
        qty=200,
        fill_price=Decimal("400.00"),
        commission=Decimal("1.50"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    initial_cash = portfolio.cash.get_balance()

    # Apply dividend: 200 shares * $0.50/share = $100 credit
    portfolio.apply_long_dividend(
        symbol="MSFT",
        dividend_per_share=Decimal("0.50"),
        ts=ts,
    )

    # Should credit cash (200 shares * $0.50/share = $100)
    expected_cash = initial_cash + Decimal("100.00")
    assert portfolio.cash.get_balance() == expected_cash

    # Verify transaction recorded
    transactions = portfolio.cash.get_transactions()
    dividend_txns = [t for t in transactions if t.transaction_type == "DIVIDEND_RECEIVED"]
    assert len(dividend_txns) == 1
    assert dividend_txns[0].amount == Decimal("100.00")
    assert "MSFT" in dividend_txns[0].description
    assert "200 shares" in dividend_txns[0].description


def test_portfolio_long_dividend_requires_long_position(portfolio):
    """Long dividend should only apply when position is net long."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Create short position
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    initial_cash = portfolio.cash.get_balance()

    # Apply long dividend (should be no-op for short)
    portfolio.apply_long_dividend(
        symbol="AAPL",
        dividend_per_share=Decimal("0.50"),
        ts=ts,
    )

    # Cash should be unchanged
    assert portfolio.cash.get_balance() == initial_cash


def test_portfolio_long_dividend_no_position(portfolio):
    """Long dividend should be no-op when no position exists."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    initial_cash = portfolio.cash.get_balance()

    # Apply dividend with no position (should be no-op)
    portfolio.apply_long_dividend(
        symbol="AAPL",
        dividend_per_share=Decimal("0.50"),
        ts=ts,
    )

    # Cash should be unchanged
    assert portfolio.cash.get_balance() == initial_cash


def test_portfolio_long_dividend_partial_position(portfolio):
    """Long dividend should handle partial positions correctly."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Create position with 50 shares
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=50,
        fill_price=Decimal("180.00"),
        commission=Decimal("0.75"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    initial_cash = portfolio.cash.get_balance()

    # Apply dividend: 50 shares * $0.45/share = $22.50 credit
    portfolio.apply_long_dividend(
        symbol="AAPL",
        dividend_per_share=Decimal("0.45"),
        ts=ts,
    )

    expected_cash = initial_cash + Decimal("22.50")
    assert portfolio.cash.get_balance() == expected_cash


def test_portfolio_borrow_cost(portfolio):
    """Borrow cost should accrue on short positions."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Create short position: 100 shares @ $150 = $15,000 short MV
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    # Update prices for valuation
    portfolio.update_prices({"AAPL": Decimal("150.00")})

    initial_cash = portfolio.cash.get_balance()

    # Apply borrow cost (3% annual)
    portfolio.apply_borrow_cost(
        borrow_rate_annual=Decimal("0.03"),
        ts=ts,
    )

    # Daily cost = $15,000 * (0.03 / 252) = $1.7857...
    expected_daily_cost = Decimal("15000.00") * (Decimal("0.03") / Decimal("252"))
    expected_cash = initial_cash - expected_daily_cost

    assert portfolio.cash.get_balance() == expected_cash


def test_portfolio_equity_calculation(portfolio):
    """Equity should include cash + unrealized PnL."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Buy 100 AAPL @ $150
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    # Update price to $160 (profit!)
    portfolio.update_prices({"AAPL": Decimal("160.00")})

    # Equity = cash + unrealized PnL
    # Cash = 100k - 15k - 1 = 84,999
    # Unrealized PnL = (160 - 150) * 100 = 1,000
    # Equity = 84,999 + 1,000 = 85,999
    expected_equity = Decimal("85999.00")
    assert portfolio.get_equity() == expected_equity


def test_portfolio_margin_usage(portfolio):
    """Margin usage should be sum of long + abs(short) market value."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Long position: 100 AAPL @ $150 = $15,000
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    # Short position: 50 MSFT @ $300 = $15,000
    portfolio.apply_fill(
        symbol="MSFT",
        side=OrderSide.SELL,
        qty=50,
        fill_price=Decimal("300.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-2",
        fill_id="fill-2",
    )

    # Update prices
    portfolio.update_prices({"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")})

    # Margin usage = $15,000 (long) + $15,000 (abs short) = $30,000
    expected_margin = Decimal("30000.00")
    assert portfolio.get_margin_usage() == expected_margin


def test_portfolio_can_afford(portfolio):
    """can_afford should check cash balance."""
    assert portfolio.can_afford(Decimal("50000.00"))
    assert portfolio.can_afford(Decimal("100000.00"))
    assert not portfolio.can_afford(Decimal("100001.00"))


def test_portfolio_snapshot(portfolio):
    """Snapshot should include all portfolio state."""
    ts = datetime(2023, 1, 15, 9, 30, tzinfo=timezone.utc)

    # Create position
    portfolio.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        ts=ts,
        order_id="order-1",
        fill_id="fill-1",
    )

    # Update price
    portfolio.update_prices({"AAPL": Decimal("160.00")})

    # Get snapshot
    snapshot = portfolio.snapshot(ts)

    assert snapshot["ts"] == ts.isoformat()
    assert snapshot["cash"] == float(portfolio.cash.get_balance())
    assert snapshot["equity"] == float(portfolio.get_equity())
    assert len(snapshot["positions"]) == 1
    assert snapshot["positions"][0]["symbol"] == "AAPL"
    assert snapshot["positions"][0]["qty"] == 100
