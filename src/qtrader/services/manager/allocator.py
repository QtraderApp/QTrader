"""
Capital Allocation Logic.

Pure functions for allocating capital across strategies.
Phase 4 MVP: Fixed budgets only (no dynamic rebalancing or drawdown throttling).

Architecture:
- Pure functions (no state, no side effects)
- Deterministic (same inputs → same outputs)
- Testable (easy to unit test)
"""

from decimal import Decimal

from qtrader.services.manager.models import StrategyBudget


def allocate_capital(
    budgets: list[StrategyBudget],
    equity: Decimal,
    cash_buffer_pct: float = 0.02,
) -> dict[str, Decimal]:
    """
    Allocate capital across strategies using fixed budgets.

    Phase 4 MVP: Simple fixed allocation based on configured weights.
    Phase 11: Add drawdown throttling, dynamic rebalancing.

    Algorithm:
        1. Calculate deployable capital = equity * (1 - cash_buffer_pct)
        2. Allocate per strategy: allocated = deployable * strategy_weight
        3. Return dict mapping strategy_id → allocated capital

    Args:
        budgets: List of strategy budgets with capital weights
        equity: Current portfolio equity
        cash_buffer_pct: Cash buffer to reserve (default 2%)

    Returns:
        Dictionary mapping strategy_id to allocated capital (Decimal)

    Example:
        >>> budgets = [
        ...     StrategyBudget("momentum_v1", 0.3),
        ...     StrategyBudget("mean_reversion_v1", 0.2),
        ... ]
        >>> allocate_capital(budgets, Decimal("1000000"), 0.02)
        {'momentum_v1': Decimal('294000'), 'mean_reversion_v1': Decimal('196000')}

    Notes:
        - Weights can sum to < 1.0 (unallocated capital stays as cash)
        - Cash buffer is deducted before allocation
        - Returns 0 for strategies with 0 weight
        - Deterministic: same inputs always produce same outputs
    """
    # Validate inputs
    if equity <= 0:
        raise ValueError(f"Equity must be positive, got {equity}")

    if not 0.0 <= cash_buffer_pct <= 0.5:
        raise ValueError(f"cash_buffer_pct must be in [0, 0.5], got {cash_buffer_pct}")

    # Calculate deployable capital (after cash buffer)
    deployable_capital = equity * Decimal(str(1.0 - cash_buffer_pct))

    # Allocate to each strategy
    allocations: dict[str, Decimal] = {}
    for budget in budgets:
        # Calculate allocation for this strategy
        allocated = deployable_capital * Decimal(str(budget.capital_weight))
        allocations[budget.strategy_id] = allocated

    return allocations


def validate_allocations(
    allocations: dict[str, Decimal],
    equity: Decimal,
    cash_buffer_pct: float = 0.02,
) -> bool:
    """
    Validate that allocations don't exceed available capital.

    Ensures total allocated capital ≤ equity * (1 - cash_buffer_pct).

    Args:
        allocations: Dictionary of strategy_id → allocated capital
        equity: Current portfolio equity
        cash_buffer_pct: Cash buffer percentage

    Returns:
        True if allocations are valid, False otherwise

    Example:
        >>> allocations = {
        ...     "momentum_v1": Decimal("300000"),
        ...     "mean_reversion_v1": Decimal("200000"),
        ... }
        >>> validate_allocations(allocations, Decimal("1000000"), 0.02)
        True
    """
    deployable_capital = equity * Decimal(str(1.0 - cash_buffer_pct))
    total_allocated = sum(allocations.values())
    return total_allocated <= deployable_capital


def get_unallocated_capital(
    allocations: dict[str, Decimal],
    equity: Decimal,
    cash_buffer_pct: float = 0.02,
) -> Decimal:
    """
    Calculate unallocated capital remaining after allocations.

    Returns the amount of deployable capital not allocated to any strategy.

    Args:
        allocations: Dictionary of strategy_id → allocated capital
        equity: Current portfolio equity
        cash_buffer_pct: Cash buffer percentage

    Returns:
        Unallocated capital (Decimal)

    Example:
        >>> allocations = {
        ...     "momentum_v1": Decimal("300000"),
        ...     "mean_reversion_v1": Decimal("200000"),
        ... }
        >>> get_unallocated_capital(allocations, Decimal("1000000"), 0.02)
        Decimal('480000')  # 980k deployable - 500k allocated
    """
    deployable_capital = equity * Decimal(str(1.0 - cash_buffer_pct))
    total_allocated = sum(allocations.values())
    return deployable_capital - total_allocated


def get_allocation_summary(
    allocations: dict[str, Decimal],
    equity: Decimal,
    cash_buffer_pct: float = 0.02,
) -> dict[str, Decimal | float]:
    """
    Get summary of capital allocation.

    Returns breakdown of how capital is distributed.

    Args:
        allocations: Dictionary of strategy_id → allocated capital
        equity: Current portfolio equity
        cash_buffer_pct: Cash buffer percentage

    Returns:
        Dictionary with keys:
        - 'equity': Total equity
        - 'cash_buffer': Reserved cash buffer
        - 'deployable': Capital available for deployment
        - 'allocated': Total allocated to strategies
        - 'unallocated': Remaining deployable capital
        - 'utilization': Allocation as % of deployable (0-1)

    Example:
        >>> allocations = {"momentum_v1": Decimal("300000")}
        >>> summary = get_allocation_summary(allocations, Decimal("1000000"))
        >>> summary['utilization']
        Decimal('0.306122...')  # ~30.6% of deployable capital used
    """
    cash_buffer = equity * Decimal(str(cash_buffer_pct))
    deployable = equity - cash_buffer
    allocated = sum(allocations.values())
    unallocated = deployable - allocated
    utilization = allocated / deployable if deployable > 0 else Decimal("0")

    return {
        "equity": equity,
        "cash_buffer": cash_buffer,
        "deployable": deployable,
        "allocated": allocated,
        "unallocated": unallocated,
        "utilization": float(utilization),  # Convert to float for convenience
    }
