"""
Position sizing methods.

Implements Phase 1 sizing methods:
- FIXED_QUANTITY: Fixed number of shares
- FIXED_VALUE: Fixed dollar amount
- PORTFOLIO_PERCENT: Percentage of equity (default)
- RISK_PERCENT: Percentage at risk (requires stop loss)
"""

from decimal import Decimal

import structlog

from qtrader.models.portfolio import Portfolio
from qtrader.risk.policy import RiskPolicy, SizingMethod
from qtrader.risk.signal import Signal

logger = structlog.get_logger()


def calculate_position_size(
    signal: Signal,
    policy: RiskPolicy,
    portfolio: Portfolio,
    current_price: Decimal,
) -> int:
    """
    Calculate position size based on sizing method.

    Args:
        signal: Trading signal with sizing hints
        policy: Risk policy configuration
        portfolio: Current portfolio state
        current_price: Current market price

    Returns:
        Position size in shares (integer)

    Raises:
        ValueError: If sizing method not supported or invalid parameters
    """
    method = policy.sizing_method

    if method == SizingMethod.FIXED_QUANTITY:
        return _size_fixed_quantity(signal, policy)

    elif method == SizingMethod.FIXED_VALUE:
        return _size_fixed_value(signal, policy, current_price)

    elif method == SizingMethod.PORTFOLIO_PERCENT:
        return _size_portfolio_percent(signal, policy, portfolio, current_price)

    elif method == SizingMethod.RISK_PERCENT:
        return _size_risk_percent(signal, policy, portfolio, current_price)

    else:
        # Phase 2 methods - fallback to PORTFOLIO_PERCENT
        logger.warning(
            "risk.sizing.unsupported_method",
            method=method.value,
            fallback="portfolio_percent",
            signal_id=signal.signal_id,
        )
        # Use PORTFOLIO_PERCENT as fallback
        return _size_portfolio_percent(signal, policy, portfolio, current_price)


def _size_fixed_quantity(signal: Signal, policy: RiskPolicy) -> int:
    """
    Fixed quantity sizing - use signal hint or policy default.

    Formula: qty = signal.target_qty OR int(policy.default_position_size)

    Args:
        signal: Trading signal (may have target_qty hint)
        policy: Risk policy (has default_position_size)

    Returns:
        Fixed quantity in shares
    """
    if signal.target_qty is not None:
        return signal.target_qty

    # Use default_position_size as fixed quantity (assumes it's configured as shares count)
    return int(policy.default_position_size)


def _size_fixed_value(signal: Signal, policy: RiskPolicy, current_price: Decimal) -> int:
    """
    Fixed value sizing - fixed dollar amount.

    Formula: qty = target_value / current_price

    Args:
        signal: Trading signal (may have target_value hint)
        policy: Risk policy (has default_position_size as dollar amount)
        current_price: Current market price

    Returns:
        Position size in shares

    Raises:
        ValueError: If current_price is zero or negative
    """
    if current_price <= Decimal("0"):
        raise ValueError(f"Invalid current_price: {current_price}")

    if signal.target_value is not None:
        target_value = signal.target_value
    else:
        # Use default_position_size as dollar amount
        target_value = policy.default_position_size

    # Calculate shares
    qty = target_value / current_price

    return int(qty)


def _size_portfolio_percent(
    signal: Signal,
    policy: RiskPolicy,
    portfolio: Portfolio,
    current_price: Decimal,
) -> int:
    """
    Portfolio percent sizing - percentage of equity.

    Formula: qty = (equity × weight) / current_price

    Args:
        signal: Trading signal (may have target_weight hint)
        policy: Risk policy (has default_position_size as %)
        portfolio: Current portfolio state
        current_price: Current market price

    Returns:
        Position size in shares

    Raises:
        ValueError: If current_price is zero or negative
    """
    if current_price <= Decimal("0"):
        raise ValueError(f"Invalid current_price: {current_price}")

    # Get equity
    equity = portfolio.get_equity()

    if equity <= Decimal("0"):
        logger.warning(
            "risk.sizing.zero_equity",
            equity=str(equity),
            signal_id=signal.signal_id,
        )
        return 0

    # Get weight (from signal or policy)
    if signal.target_weight is not None:
        weight = signal.target_weight
    else:
        weight = policy.default_position_size

    # Calculate target value
    target_value = equity * weight

    # Calculate shares
    qty = target_value / current_price

    return int(qty)


def _size_risk_percent(
    signal: Signal,
    policy: RiskPolicy,
    portfolio: Portfolio,
    current_price: Decimal,
) -> int:
    """
    Risk percent sizing - percentage of equity at risk.

    Formula: qty = (equity × risk_pct) / (current_price - stop_price)

    Args:
        signal: Trading signal (MUST have stop_price)
        policy: Risk policy (has default_position_size as risk %)
        portfolio: Current portfolio state
        current_price: Current market price

    Returns:
        Position size in shares

    Raises:
        ValueError: If stop_price missing, or invalid price relationship
    """
    if signal.stop_price is None:
        raise ValueError("RISK_PERCENT sizing requires stop_price in signal")

    if current_price <= Decimal("0"):
        raise ValueError(f"Invalid current_price: {current_price}")

    # Get equity
    equity = portfolio.get_equity()

    if equity <= Decimal("0"):
        logger.warning(
            "risk.sizing.zero_equity",
            equity=str(equity),
            signal_id=signal.signal_id,
        )
        return 0

    # Get risk percentage (from signal or policy)
    if signal.target_weight is not None:
        risk_pct = signal.target_weight
    else:
        risk_pct = policy.default_position_size

    # Calculate risk amount ($ at risk)
    risk_amount = equity * risk_pct

    # Calculate risk per share
    # For LONG: risk = current_price - stop_price (stop below entry)
    # For SHORT: risk = stop_price - current_price (stop above entry)
    if signal.signal_type.value.startswith("entry_long"):
        risk_per_share = current_price - signal.stop_price
        if risk_per_share <= Decimal("0"):
            raise ValueError(
                f"Invalid stop for LONG: stop_price ({signal.stop_price}) must be below current_price ({current_price})"
            )
    else:  # entry_short
        risk_per_share = signal.stop_price - current_price
        if risk_per_share <= Decimal("0"):
            raise ValueError(
                f"Invalid stop for SHORT: stop_price ({signal.stop_price}) must be above current_price ({current_price})"
            )

    # Calculate shares
    qty = risk_amount / risk_per_share

    return int(qty)
