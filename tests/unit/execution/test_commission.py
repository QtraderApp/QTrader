"""Tests for CommissionCalculator."""

from decimal import Decimal

import pytest

from qtrader.execution.commission import CommissionCalculator, CommissionResult


def test_commission_calculator_initialization():
    """CommissionCalculator should initialize with defaults."""
    calc = CommissionCalculator()

    assert calc.per_share == Decimal("0.0005")
    assert calc.ticket_min == Decimal("1.00")


def test_commission_calculator_custom_rates():
    """CommissionCalculator should accept custom rates."""
    calc = CommissionCalculator(
        per_share=Decimal("0.001"),
        ticket_min=Decimal("5.00"),
    )

    assert calc.per_share == Decimal("0.001")
    assert calc.ticket_min == Decimal("5.00")


def test_commission_calculator_validation():
    """CommissionCalculator should validate rates."""
    # Negative per_share should fail
    with pytest.raises(ValueError, match="per_share must be >= 0"):
        CommissionCalculator(per_share=Decimal("-0.001"))

    # Negative ticket_min should fail
    with pytest.raises(ValueError, match="ticket_min must be >= 0"):
        CommissionCalculator(ticket_min=Decimal("-1.00"))


def test_commission_calculator_below_minimum():
    """Commission below ticket minimum should be enforced."""
    calc = CommissionCalculator(
        per_share=Decimal("0.0005"),
        ticket_min=Decimal("1.00"),
    )

    # 100 shares * $0.0005 = $0.05 < $1.00 minimum
    result = calc.calculate(qty=100)

    assert result.commission == Decimal("1.00")
    assert result.per_share_cost == Decimal("0.05")
    assert result.ticket_minimum == Decimal("1.00")
    assert result.minimum_enforced is True


def test_commission_calculator_above_minimum():
    """Commission above ticket minimum should not be enforced."""
    calc = CommissionCalculator(
        per_share=Decimal("0.0005"),
        ticket_min=Decimal("1.00"),
    )

    # 5000 shares * $0.0005 = $2.50 > $1.00 minimum
    result = calc.calculate(qty=5000)

    assert result.commission == Decimal("2.50")
    assert result.per_share_cost == Decimal("2.50")
    assert result.ticket_minimum == Decimal("0")
    assert result.minimum_enforced is False


def test_commission_calculator_exact_minimum():
    """Commission exactly at minimum should not enforce minimum."""
    calc = CommissionCalculator(
        per_share=Decimal("0.0005"),
        ticket_min=Decimal("1.00"),
    )

    # 2000 shares * $0.0005 = $1.00 = minimum
    result = calc.calculate(qty=2000)

    assert result.commission == Decimal("1.00")
    assert result.per_share_cost == Decimal("1.00")
    assert result.ticket_minimum == Decimal("0")
    assert result.minimum_enforced is False


def test_commission_calculator_zero_rates():
    """CommissionCalculator should handle zero rates."""
    calc = CommissionCalculator(
        per_share=Decimal("0"),
        ticket_min=Decimal("0"),
    )

    result = calc.calculate(qty=100)

    assert result.commission == Decimal("0")
    assert result.per_share_cost == Decimal("0")
    assert result.minimum_enforced is False


def test_commission_calculator_large_order():
    """CommissionCalculator should handle large orders."""
    calc = CommissionCalculator()

    # 1,000,000 shares * $0.0005 = $500
    result = calc.calculate(qty=1000000)

    assert result.commission == Decimal("500.00")
    assert result.per_share_cost == Decimal("500.00")
    assert result.minimum_enforced is False


def test_commission_calculator_qty_validation():
    """Calculate should reject invalid qty."""
    calc = CommissionCalculator()

    # qty <= 0 should fail
    with pytest.raises(ValueError, match="qty must be > 0"):
        calc.calculate(qty=0)

    with pytest.raises(ValueError, match="qty must be > 0"):
        calc.calculate(qty=-100)


def test_commission_result_is_immutable():
    """CommissionResult should be immutable."""
    result = CommissionResult(
        commission=Decimal("1.00"),
        per_share_cost=Decimal("0.50"),
        ticket_minimum=Decimal("1.00"),
        minimum_enforced=True,
    )

    # Verify it's a tuple (immutable)
    assert isinstance(result, tuple)
    assert result.commission == Decimal("1.00")
