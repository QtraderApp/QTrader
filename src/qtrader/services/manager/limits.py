"""Risk limit checking for the RiskService.

This module provides functions to check whether proposed orders would violate
risk limits. The MVP implementation supports two types of limits:

1. Concentration Limits: Per-symbol exposure as % of equity
2. Leverage Limits: Portfolio gross/net exposure as % of equity

Limit Flow:
1. Calculate proposed position after order execution
2. Calculate proposed exposures (per-symbol, gross, net)
3. Compare against configured limits
4. Return violation details if any limit exceeded

Key Principles:
- Pure functions (no side effects)
- Decimal precision for financial calculations
- Returns detailed violation reasons for audit trail
- Considers both existing positions and proposed orders

Thread Safety:
- All functions are pure and thread-safe
- No shared mutable state
"""

from dataclasses import dataclass
from decimal import Decimal

from qtrader.services.manager.models import ConcentrationLimit, LeverageLimit, OrderBase, Position


@dataclass(frozen=True)
class LimitViolation:
    """Details about a limit violation.

    Immutable record of why an order was rejected due to limit breach.
    Used in audit trails and rejection reasons.
    """

    limit_type: str  # "concentration" or "leverage"
    symbol: str  # symbol that violated (or "PORTFOLIO" for leverage)
    proposed_exposure: Decimal  # proposed exposure in USD
    proposed_pct: float  # proposed exposure as % of equity
    limit_pct: float  # configured limit as % of equity
    message: str  # human-readable violation message


def check_concentration_limit(
    order: OrderBase,
    current_positions: list[Position],
    equity: Decimal,
    current_price: Decimal,
    limit: ConcentrationLimit,
) -> LimitViolation | None:
    """Check if order would violate concentration limit for its symbol.

    Concentration limit restricts per-symbol exposure as a percentage of equity.
    This prevents over-concentration in a single security.

    The check:
    1. Find current position in the symbol (if any)
    2. Calculate proposed position after order execution
    3. Calculate proposed exposure: |proposed_qty| * current_price
    4. Calculate proposed percentage: proposed_exposure / equity
    5. Compare against limit.max_pct

    Args:
        order: Proposed order to check
        current_positions: List of current positions
        equity: Current portfolio equity
        current_price: Current market price for order.symbol
        limit: Concentration limit configuration

    Returns:
        LimitViolation if limit exceeded, None if within limit

    Example:
        >>> order = OrderBase(
        ...     strategy_id="A", symbol="AAPL", side="BUY",
        ...     quantity=100, reason="..."
        ... )
        >>> positions = [Position(symbol="AAPL", quantity=50, market_value=...)]
        >>> violation = check_concentration_limit(
        ...     order=order,
        ...     current_positions=positions,
        ...     equity=Decimal("100000"),
        ...     current_price=Decimal("150"),
        ...     limit=ConcentrationLimit(max_pct=0.10)
        ... )
        >>> if violation:
        ...     print(violation.message)
        Concentration limit exceeded for AAPL: 2.25% > 10.00%
    """
    # Find current position in this symbol
    current_qty = 0
    for pos in current_positions:
        if pos.symbol == order.symbol:
            current_qty = pos.quantity
            break

    # Calculate proposed position after order execution
    if order.side == "BUY":
        proposed_qty = current_qty + order.quantity
    else:  # SELL
        proposed_qty = current_qty - order.quantity

    # Calculate proposed exposure (absolute value for long or short)
    proposed_exposure = abs(proposed_qty) * current_price

    # Calculate proposed percentage of equity
    if equity == 0:
        # Avoid division by zero; if equity is 0, any position is a violation
        if proposed_qty != 0:
            return LimitViolation(
                limit_type="concentration",
                symbol=order.symbol,
                proposed_exposure=proposed_exposure,
                proposed_pct=float("inf"),
                limit_pct=limit.max_position_pct,
                message=(f"Concentration limit exceeded for {order.symbol}: proposed position with zero equity"),
            )
        return None

    proposed_pct = float(proposed_exposure / equity)

    # Check limit
    if proposed_pct > limit.max_position_pct:
        return LimitViolation(
            limit_type="concentration",
            symbol=order.symbol,
            proposed_exposure=proposed_exposure,
            proposed_pct=proposed_pct,
            limit_pct=limit.max_position_pct,
            message=(
                f"Concentration limit exceeded for {order.symbol}: {proposed_pct:.2%} > {limit.max_position_pct:.2%}"
            ),
        )

    return None


def check_leverage_limits(
    order: OrderBase,
    current_positions: list[Position],
    equity: Decimal,
    current_price: Decimal,
    limit: LeverageLimit,
) -> LimitViolation | None:
    """Check if order would violate leverage limits (gross or net).

    Leverage limits restrict portfolio-wide exposure:
    - Gross leverage: sum of absolute exposures (long + short)
    - Net leverage: net exposure (long - short)

    The check:
    1. Calculate current gross and net exposure
    2. Simulate order execution (add/remove position)
    3. Calculate proposed gross and net exposure
    4. Check both against configured limits

    Args:
        order: Proposed order to check
        current_positions: List of current positions
        equity: Current portfolio equity
        current_price: Current market price for order.symbol
        limit: Leverage limit configuration

    Returns:
        LimitViolation if any limit exceeded, None if within limits
        If both gross and net exceeded, returns gross violation (more severe)

    Example:
        >>> order = OrderBase(
        ...     strategy_id="A", symbol="AAPL", side="BUY",
        ...     quantity=1000, reason="..."
        ... )
        >>> positions = [
        ...     Position(symbol="SPY", quantity=500, market_value=Decimal("225000")),
        ...     Position(symbol="AAPL", quantity=100, market_value=Decimal("15000")),
        ... ]
        >>> violation = check_leverage_limits(
        ...     order=order,
        ...     current_positions=positions,
        ...     equity=Decimal("100000"),
        ...     current_price=Decimal("150"),
        ...     limit=LeverageLimit(max_gross_leverage=2.0, max_net_leverage=1.0)
        ... )
        >>> if violation:
        ...     print(violation.message)
        Gross leverage limit exceeded: 3.90 > 2.00
    """
    # Step 1: Calculate current exposures (excluding the order's symbol)
    current_gross = Decimal("0")
    current_net = Decimal("0")
    current_qty_in_symbol = 0

    for pos in current_positions:
        if pos.symbol == order.symbol:
            current_qty_in_symbol = pos.quantity
        else:
            # Add to current exposures (for other symbols)
            current_gross += abs(pos.market_value)
            current_net += pos.market_value  # signed (long=+, short=-)

    # Step 2: Calculate proposed position in order's symbol
    if order.side == "BUY":
        proposed_qty = current_qty_in_symbol + order.quantity
    else:  # SELL
        proposed_qty = current_qty_in_symbol - order.quantity

    # Calculate proposed exposure for order's symbol
    proposed_symbol_exposure = Decimal(str(proposed_qty)) * current_price

    # Step 3: Calculate proposed portfolio exposures
    proposed_gross = current_gross + abs(proposed_symbol_exposure)
    proposed_net = current_net + proposed_symbol_exposure

    # Step 4: Calculate proposed leverage ratios
    if equity == 0:
        # Avoid division by zero; if equity is 0, any position is a violation
        if proposed_gross > 0:
            return LimitViolation(
                limit_type="leverage",
                symbol="PORTFOLIO",
                proposed_exposure=proposed_gross,
                proposed_pct=float("inf"),
                limit_pct=limit.max_gross,
                message="Leverage limit exceeded: proposed position with zero equity",
            )
        return None

    proposed_gross_leverage = float(proposed_gross / equity)
    proposed_net_leverage = float(abs(proposed_net) / equity)

    # Step 5: Check limits (gross first, then net)
    if proposed_gross_leverage > limit.max_gross:
        return LimitViolation(
            limit_type="leverage",
            symbol="PORTFOLIO",
            proposed_exposure=proposed_gross,
            proposed_pct=proposed_gross_leverage,
            limit_pct=limit.max_gross,
            message=(f"Gross leverage limit exceeded: {proposed_gross_leverage:.2f} > {limit.max_gross:.2f}"),
        )

    if proposed_net_leverage > limit.max_net:
        return LimitViolation(
            limit_type="leverage",
            symbol="PORTFOLIO",
            proposed_exposure=abs(proposed_net),
            proposed_pct=proposed_net_leverage,
            limit_pct=limit.max_net,
            message=(f"Net leverage limit exceeded: {proposed_net_leverage:.2f} > {limit.max_net:.2f}"),
        )

    return None


def check_all_limits(
    order: OrderBase,
    current_positions: list[Position],
    equity: Decimal,
    current_price: Decimal,
    concentration_limit: ConcentrationLimit | None,
    leverage_limit: LeverageLimit | None,
) -> list[LimitViolation]:
    """Check all configured limits for an order.

    Convenience function that checks both concentration and leverage limits.
    Returns all violations found (empty list if no violations).

    Args:
        order: Proposed order to check
        current_positions: List of current positions
        equity: Current portfolio equity
        current_price: Current market price for order.symbol
        concentration_limit: Concentration limit config (None = skip check)
        leverage_limit: Leverage limit config (None = skip check)

    Returns:
        List of all violations (empty if order passes all limits)

    Example:
        >>> violations = check_all_limits(
        ...     order=order,
        ...     current_positions=positions,
        ...     equity=equity,
        ...     current_price=price,
        ...     concentration_limit=ConcentrationLimit(max_pct=0.10),
        ...     leverage_limit=LeverageLimit(max_gross_leverage=2.0, max_net_leverage=1.0)
        ... )
        >>> if violations:
        ...     for v in violations:
        ...         print(f"REJECT: {v.message}")
    """
    violations: list[LimitViolation] = []

    if concentration_limit is not None:
        violation = check_concentration_limit(
            order=order,
            current_positions=current_positions,
            equity=equity,
            current_price=current_price,
            limit=concentration_limit,
        )
        if violation:
            violations.append(violation)

    if leverage_limit is not None:
        violation = check_leverage_limits(
            order=order,
            current_positions=current_positions,
            equity=equity,
            current_price=current_price,
            limit=leverage_limit,
        )
        if violation:
            violations.append(violation)

    return violations
