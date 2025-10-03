"""Tests for Cash Ledger."""

from decimal import Decimal

import pytest

from qtrader.models.ledger import CashLedger, CashTransaction


def test_ledger_initialization_default():
    """Ledger should initialize with default cash."""
    ledger = CashLedger()

    assert ledger.get_balance() == Decimal("100000.0")
    transactions = ledger.get_transactions()
    assert len(transactions) == 1
    assert transactions[0].transaction_type == "DEPOSIT"
    assert transactions[0].amount == Decimal("100000.0")


def test_ledger_initialization_custom():
    """Ledger should initialize with custom cash."""
    ledger = CashLedger(initial_cash=Decimal("50000.0"))

    assert ledger.get_balance() == Decimal("50000.0")


def test_ledger_initialization_negative_fails():
    """Ledger should reject negative initial cash."""
    with pytest.raises(ValueError, match="initial_cash must be >= 0"):
        CashLedger(initial_cash=Decimal("-1000.0"))


def test_ledger_debit():
    """Debit should subtract from balance."""
    ledger = CashLedger(initial_cash=Decimal("10000.0"))

    new_balance = ledger.debit(
        amount=Decimal("500.0"),
        timestamp="2023-01-15T09:30:00",
        transaction_type="FILL",
        description="Buy 100 AAPL @ $5.00",
    )

    assert new_balance == Decimal("9500.0")
    assert ledger.get_balance() == Decimal("9500.0")


def test_ledger_credit():
    """Credit should add to balance."""
    ledger = CashLedger(initial_cash=Decimal("10000.0"))

    new_balance = ledger.credit(
        amount=Decimal("500.0"),
        timestamp="2023-01-15T09:30:00",
        transaction_type="FILL",
        description="Sell 100 AAPL @ $5.00",
    )

    assert new_balance == Decimal("10500.0")
    assert ledger.get_balance() == Decimal("10500.0")


def test_ledger_debit_negative_fails():
    """Debit with negative amount should fail."""
    ledger = CashLedger()

    with pytest.raises(ValueError, match="Debit amount must be >= 0"):
        ledger.debit(
            amount=Decimal("-500.0"),
            timestamp="2023-01-15T09:30:00",
            transaction_type="FILL",
            description="Invalid",
        )


def test_ledger_credit_negative_fails():
    """Credit with negative amount should fail."""
    ledger = CashLedger()

    with pytest.raises(ValueError, match="Credit amount must be >= 0"):
        ledger.credit(
            amount=Decimal("-500.0"),
            timestamp="2023-01-15T09:30:00",
            transaction_type="FILL",
            description="Invalid",
        )


def test_ledger_multiple_transactions():
    """Ledger should track multiple transactions correctly."""
    ledger = CashLedger(initial_cash=Decimal("10000.0"))

    # Debit 1500
    ledger.debit(Decimal("1500.0"), "2023-01-15T09:30:00", "FILL", "Buy AAPL")
    assert ledger.get_balance() == Decimal("8500.0")

    # Credit 2000
    ledger.credit(Decimal("2000.0"), "2023-01-15T10:00:00", "FILL", "Sell MSFT")
    assert ledger.get_balance() == Decimal("10500.0")

    # Debit 500
    ledger.debit(Decimal("500.0"), "2023-01-15T11:00:00", "BORROW_COST", "Short borrow")
    assert ledger.get_balance() == Decimal("10000.0")

    # Check transactions
    transactions = ledger.get_transactions()
    assert len(transactions) == 4  # initial + 3 new


def test_ledger_can_afford():
    """can_afford should check balance correctly."""
    ledger = CashLedger(initial_cash=Decimal("10000.0"))

    assert ledger.can_afford(Decimal("5000.0"))
    assert ledger.can_afford(Decimal("10000.0"))
    assert not ledger.can_afford(Decimal("10001.0"))


def test_ledger_can_afford_with_cushion():
    """can_afford with cushion should work correctly."""
    ledger = CashLedger(initial_cash=Decimal("10000.0"))

    # Need 5000 + 1000 cushion = 6000 total
    assert ledger.can_afford(Decimal("5000.0"), cushion=Decimal("1000.0"))

    # Need 9500 + 1000 cushion = 10500 total
    assert not ledger.can_afford(Decimal("9500.0"), cushion=Decimal("1000.0"))


def test_ledger_transactions_immutable():
    """get_transactions should return copy (immutable)."""
    ledger = CashLedger(initial_cash=Decimal("10000.0"))

    transactions1 = ledger.get_transactions()
    ledger.debit(Decimal("500.0"), "2023-01-15T09:30:00", "FILL", "Test")
    transactions2 = ledger.get_transactions()

    # Original list unchanged
    assert len(transactions1) == 1
    assert len(transactions2) == 2


def test_ledger_transaction_details():
    """Transaction should record all details correctly."""
    ledger = CashLedger(initial_cash=Decimal("10000.0"))

    ledger.debit(
        amount=Decimal("1500.50"),
        timestamp="2023-01-15T09:30:00",
        transaction_type="FILL",
        description="Buy 100 AAPL @ $15.00 + fees",
    )

    transactions = ledger.get_transactions()
    tx = transactions[1]  # Skip initial deposit

    assert tx.timestamp == "2023-01-15T09:30:00"
    assert tx.transaction_type == "FILL"
    assert tx.amount == Decimal("-1500.50")  # Negative for debit
    assert tx.description == "Buy 100 AAPL @ $15.00 + fees"
    assert tx.balance_after == Decimal("8499.50")


def test_ledger_get_net_flow():
    """get_net_flow should calculate net flow excluding initial deposit."""
    ledger = CashLedger(initial_cash=Decimal("10000.0"))

    # Credit 1000 (sell)
    ledger.credit(Decimal("1000.0"), "2023-01-15T09:30:00", "FILL", "Sell")

    # Debit 500 (buy)
    ledger.debit(Decimal("500.0"), "2023-01-15T10:00:00", "FILL", "Buy")

    # Debit 100 (fees)
    ledger.debit(Decimal("100.0"), "2023-01-15T11:00:00", "FEES", "Commissions")

    # Net flow = 1000 - 500 - 100 = 400
    assert ledger.get_net_flow() == Decimal("400.0")


def test_ledger_negative_balance_allowed():
    """Ledger should allow negative balance (margin account)."""
    ledger = CashLedger(initial_cash=Decimal("1000.0"))

    # Debit more than balance
    ledger.debit(Decimal("1500.0"), "2023-01-15T09:30:00", "FILL", "Large buy")

    assert ledger.get_balance() == Decimal("-500.0")


def test_cash_transaction_immutable():
    """CashTransaction should be immutable."""
    tx = CashTransaction(
        timestamp="2023-01-15T09:30:00",
        transaction_type="FILL",
        amount=Decimal("100.0"),
        description="Test",
        balance_after=Decimal("10100.0"),
    )

    # Verify it's a NamedTuple (immutable)
    assert isinstance(tx, tuple)
    assert tx.amount == Decimal("100.0")
