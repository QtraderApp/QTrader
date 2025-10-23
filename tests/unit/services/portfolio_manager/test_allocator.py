"""Tests for capital allocation logic."""

from decimal import Decimal

import pytest

from qtrader.services.portfolio_manager.allocator import (
    allocate_capital,
    get_allocation_summary,
    get_unallocated_capital,
    validate_allocations,
)
from qtrader.services.portfolio_manager.models import StrategyBudget

# ============================================================================
# allocate_capital() Tests
# ============================================================================


def test_allocate_capital_single_strategy():
    """Test allocation with single strategy."""
    budgets = [StrategyBudget("momentum_v1", 0.3)]
    equity = Decimal("1000000")

    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.02)

    assert len(allocations) == 1
    # 1M * 0.98 (after 2% buffer) * 0.3 = 294,000
    assert allocations["momentum_v1"] == Decimal("294000")


def test_allocate_capital_multiple_strategies():
    """Test allocation with multiple strategies."""
    budgets = [
        StrategyBudget("momentum_v1", 0.3),
        StrategyBudget("mean_reversion_v1", 0.2),
        StrategyBudget("pairs_v1", 0.1),
    ]
    equity = Decimal("1000000")

    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.02)

    assert len(allocations) == 3
    # Deployable: 1M * 0.98 = 980,000
    assert allocations["momentum_v1"] == Decimal("294000")  # 980k * 0.3
    assert allocations["mean_reversion_v1"] == Decimal("196000")  # 980k * 0.2
    assert allocations["pairs_v1"] == Decimal("98000")  # 980k * 0.1


def test_allocate_capital_zero_weight():
    """Test allocation with zero weight strategy."""
    budgets = [
        StrategyBudget("momentum_v1", 0.3),
        StrategyBudget("disabled_strat", 0.0),
    ]
    equity = Decimal("1000000")

    allocations = allocate_capital(budgets, equity)

    assert allocations["momentum_v1"] > 0
    assert allocations["disabled_strat"] == Decimal("0")


def test_allocate_capital_full_allocation():
    """Test allocation at 100% (no unallocated capital)."""
    budgets = [
        StrategyBudget("strat1", 0.5),
        StrategyBudget("strat2", 0.5),
    ]
    equity = Decimal("1000000")

    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.0)

    # Total should equal equity (100% allocated)
    total = sum(allocations.values())
    assert total == equity


def test_allocate_capital_partial_allocation():
    """Test allocation with weights summing to < 1.0."""
    budgets = [
        StrategyBudget("momentum_v1", 0.2),
        StrategyBudget("mean_reversion_v1", 0.1),
    ]
    equity = Decimal("1000000")

    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.02)

    # Total allocated should be < deployable capital
    deployable = equity * Decimal("0.98")
    total_allocated = sum(allocations.values())
    assert total_allocated < deployable
    # Specifically: 0.3 * 980k = 294k allocated, 686k unallocated


def test_allocate_capital_custom_cash_buffer():
    """Test allocation with custom cash buffer."""
    budgets = [StrategyBudget("momentum_v1", 0.5)]
    equity = Decimal("1000000")

    # 5% cash buffer
    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.05)

    # Deployable: 1M * 0.95 = 950k
    # Allocated: 950k * 0.5 = 475k
    assert allocations["momentum_v1"] == Decimal("475000")


def test_allocate_capital_zero_cash_buffer():
    """Test allocation with zero cash buffer."""
    budgets = [StrategyBudget("momentum_v1", 0.3)]
    equity = Decimal("1000000")

    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.0)

    # All equity deployable: 1M * 0.3 = 300k
    assert allocations["momentum_v1"] == Decimal("300000")


def test_allocate_capital_small_equity():
    """Test allocation with small equity."""
    budgets = [StrategyBudget("momentum_v1", 0.5)]
    equity = Decimal("10000")  # $10k

    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.02)

    # 10k * 0.98 * 0.5 = 4,900
    assert allocations["momentum_v1"] == Decimal("4900")


def test_allocate_capital_large_equity():
    """Test allocation with large equity."""
    budgets = [StrategyBudget("momentum_v1", 0.25)]
    equity = Decimal("100000000")  # $100M

    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.02)

    # 100M * 0.98 * 0.25 = 24.5M
    assert allocations["momentum_v1"] == Decimal("24500000")


def test_allocate_capital_precision():
    """Test allocation maintains decimal precision."""
    budgets = [StrategyBudget("momentum_v1", 0.333333)]  # Repeating decimal
    equity = Decimal("1000000")

    allocations = allocate_capital(budgets, equity, cash_buffer_pct=0.02)

    # Should maintain precision
    expected = Decimal("1000000") * Decimal("0.98") * Decimal("0.333333")
    assert allocations["momentum_v1"] == expected


def test_allocate_capital_invalid_equity_zero():
    """Test allocation rejects zero equity."""
    budgets = [StrategyBudget("momentum_v1", 0.3)]

    with pytest.raises(ValueError, match="Equity must be positive"):
        allocate_capital(budgets, Decimal("0"))


def test_allocate_capital_invalid_equity_negative():
    """Test allocation rejects negative equity."""
    budgets = [StrategyBudget("momentum_v1", 0.3)]

    with pytest.raises(ValueError, match="Equity must be positive"):
        allocate_capital(budgets, Decimal("-1000"))


def test_allocate_capital_invalid_cash_buffer_negative():
    """Test allocation rejects negative cash buffer."""
    budgets = [StrategyBudget("momentum_v1", 0.3)]

    with pytest.raises(ValueError, match="cash_buffer_pct must be in"):
        allocate_capital(budgets, Decimal("1000000"), cash_buffer_pct=-0.05)


def test_allocate_capital_invalid_cash_buffer_too_high():
    """Test allocation rejects cash buffer > 50%."""
    budgets = [StrategyBudget("momentum_v1", 0.3)]

    with pytest.raises(ValueError, match="cash_buffer_pct must be in"):
        allocate_capital(budgets, Decimal("1000000"), cash_buffer_pct=0.6)


# ============================================================================
# validate_allocations() Tests
# ============================================================================


def test_validate_allocations_valid():
    """Test validation passes for valid allocations."""
    allocations = {
        "momentum_v1": Decimal("300000"),
        "mean_reversion_v1": Decimal("200000"),
    }
    equity = Decimal("1000000")

    assert validate_allocations(allocations, equity, cash_buffer_pct=0.02) is True


def test_validate_allocations_at_limit():
    """Test validation passes when at exact limit."""
    equity = Decimal("1000000")
    deployable = equity * Decimal("0.98")
    allocations = {"momentum_v1": deployable}

    assert validate_allocations(allocations, equity, cash_buffer_pct=0.02) is True


def test_validate_allocations_exceeds_limit():
    """Test validation fails when exceeding deployable capital."""
    allocations = {
        "momentum_v1": Decimal("600000"),
        "mean_reversion_v1": Decimal("500000"),
    }
    equity = Decimal("1000000")

    # Total 1.1M > 980k deployable
    assert validate_allocations(allocations, equity, cash_buffer_pct=0.02) is False


# ============================================================================
# get_unallocated_capital() Tests
# ============================================================================


def test_get_unallocated_capital_partial_allocation():
    """Test unallocated capital with partial allocation."""
    allocations = {
        "momentum_v1": Decimal("300000"),
        "mean_reversion_v1": Decimal("200000"),
    }
    equity = Decimal("1000000")

    unallocated = get_unallocated_capital(allocations, equity, cash_buffer_pct=0.02)

    # Deployable: 980k, Allocated: 500k, Unallocated: 480k
    assert unallocated == Decimal("480000")


def test_get_unallocated_capital_full_allocation():
    """Test unallocated capital when fully allocated."""
    equity = Decimal("1000000")
    deployable = equity * Decimal("0.98")
    allocations = {"momentum_v1": deployable}

    unallocated = get_unallocated_capital(allocations, equity, cash_buffer_pct=0.02)

    assert unallocated == Decimal("0")


def test_get_unallocated_capital_zero_allocation() -> None:
    """Test unallocated capital with zero allocation."""
    allocations: dict[str, Decimal] = {}
    equity = Decimal("1000000")

    unallocated = get_unallocated_capital(allocations, equity, cash_buffer_pct=0.02)

    # All deployable capital unallocated
    assert unallocated == equity * Decimal("0.98")


# ============================================================================
# get_allocation_summary() Tests
# ============================================================================


def test_get_allocation_summary_complete():
    """Test allocation summary contains all fields."""
    allocations = {"momentum_v1": Decimal("300000")}
    equity = Decimal("1000000")

    summary = get_allocation_summary(allocations, equity, cash_buffer_pct=0.02)

    assert "equity" in summary
    assert "cash_buffer" in summary
    assert "deployable" in summary
    assert "allocated" in summary
    assert "unallocated" in summary
    assert "utilization" in summary


def test_get_allocation_summary_values():
    """Test allocation summary calculates correct values."""
    allocations = {
        "momentum_v1": Decimal("300000"),
        "mean_reversion_v1": Decimal("200000"),
    }
    equity = Decimal("1000000")

    summary = get_allocation_summary(allocations, equity, cash_buffer_pct=0.02)

    assert summary["equity"] == Decimal("1000000")
    assert summary["cash_buffer"] == Decimal("20000")  # 2%
    assert summary["deployable"] == Decimal("980000")
    assert summary["allocated"] == Decimal("500000")
    assert summary["unallocated"] == Decimal("480000")
    assert abs(float(summary["utilization"]) - 0.5102) < 0.0001  # ~51.02%


def test_get_allocation_summary_zero_allocation() -> None:
    """Test allocation summary with zero allocation."""
    allocations: dict[str, Decimal] = {}
    equity = Decimal("1000000")

    summary = get_allocation_summary(allocations, equity, cash_buffer_pct=0.02)

    assert summary["allocated"] == Decimal("0")
    assert summary["utilization"] == 0.0


def test_get_allocation_summary_full_utilization():
    """Test allocation summary with 100% utilization."""
    equity = Decimal("1000000")
    deployable = equity * Decimal("0.98")
    allocations = {"momentum_v1": deployable}

    summary = get_allocation_summary(allocations, equity, cash_buffer_pct=0.02)

    assert summary["unallocated"] == Decimal("0")
    assert summary["utilization"] == 1.0
