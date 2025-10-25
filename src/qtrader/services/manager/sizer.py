"""Position sizing logic for the RiskService.

This module provides position sizing functionality for translating trading signals
into concrete order quantities. The MVP implementation supports the fixed-fraction
sizing model only.

Sizing Flow:
1. Receive signal with strength [-1, 1] and allocated capital for the strategy
2. Calculate target notional: fraction * allocated_capital * |strength|
3. Convert notional to quantity using current price
4. Round to lot size (default: 1 share)
5. Enforce minimum quantity (if specified)

Key Principles:
- Pure functions (no side effects)
- Decimal precision for financial calculations
- Signal strength acts as a scaling factor (0 = no position, 1 = full fraction)
- Negative strength not used in sizing (direction determined by signal.side)

Thread Safety:
- All functions are pure and thread-safe
- No shared mutable state
"""

from decimal import Decimal

from qtrader.services.manager.models import Signal


def size_position(
    signal: Signal,
    allocated_capital: Decimal,
    current_price: Decimal,
    fraction: Decimal,
    lot_size: int = 1,
    min_quantity: int = 0,
) -> int:
    """Calculate order quantity for a signal using fixed-fraction sizing.

    Formula:
        target_notional = fraction * allocated_capital * |strength|
        quantity = floor(target_notional / current_price / lot_size) * lot_size

    The signal strength scales the position size:
    - strength = 0.0 → 0% of max position (no order)
    - strength = 0.5 → 50% of max position
    - strength = 1.0 → 100% of max position (full fraction)

    Args:
        signal: Trading signal with strength [-1, 1]
        allocated_capital: Capital allocated to this signal's strategy
        current_price: Current market price for the symbol
        fraction: Position sizing fraction (e.g., 0.02 = 2% of capital)
        lot_size: Minimum trading unit (default: 1 share)
        min_quantity: Minimum order quantity (default: 0, no minimum)

    Returns:
        Order quantity in shares (always >= 0, never negative)
        Returns 0 if:
        - strength is zero
        - calculated quantity < min_quantity
        - calculated quantity < lot_size
        - current_price is zero

    Raises:
        ValueError: If allocated_capital, current_price, or fraction is negative
        ValueError: If lot_size or min_quantity is <= 0

    Examples:
        >>> signal = Signal(ts=..., strategy_id="A", symbol="AAPL",
        ...                 side="BUY", strength=Decimal("1.0"))
        >>> size_position(
        ...     signal=signal,
        ...     allocated_capital=Decimal("10000"),
        ...     current_price=Decimal("150"),
        ...     fraction=Decimal("0.02"),
        ...     lot_size=1,
        ...     min_quantity=0
        ... )
        1  # 10000 * 0.02 * 1.0 = 200 notional / 150 price = 1.33 shares → 1 share

        >>> # Signal with 50% strength
        >>> signal2 = Signal(ts=..., strategy_id="A", symbol="AAPL",
        ...                  side="BUY", strength=Decimal("0.5"))
        >>> size_position(
        ...     signal=signal2,
        ...     allocated_capital=Decimal("10000"),
        ...     current_price=Decimal("150"),
        ...     fraction=Decimal("0.02"),
        ... )
        0  # 10000 * 0.02 * 0.5 = 100 notional / 150 price = 0.67 shares → 0 shares

        >>> # Lot size = 100 (e.g., options contracts)
        >>> signal3 = Signal(ts=..., strategy_id="A", symbol="SPY",
        ...                  side="BUY", strength=Decimal("1.0"))
        >>> size_position(
        ...     signal=signal3,
        ...     allocated_capital=Decimal("100000"),
        ...     current_price=Decimal("450"),
        ...     fraction=Decimal("0.10"),
        ...     lot_size=100,
        ... )
        200  # 100000 * 0.10 * 1.0 = 10000 notional / 450 = 22.2 shares → 200 shares
    """
    # Validation
    if allocated_capital < 0:
        raise ValueError(f"allocated_capital must be non-negative, got {allocated_capital}")
    if current_price < 0:
        raise ValueError(f"current_price must be non-negative, got {current_price}")
    if fraction < 0:
        raise ValueError(f"fraction must be non-negative, got {fraction}")
    if lot_size <= 0:
        raise ValueError(f"lot_size must be positive, got {lot_size}")
    if min_quantity < 0:
        raise ValueError(f"min_quantity must be non-negative, got {min_quantity}")

    # Early exit: zero strength or zero price
    if signal.strength == 0 or current_price == 0:
        return 0

    # Step 1: Calculate target notional using absolute strength
    # Convert strength (float) to Decimal for precision
    abs_strength = Decimal(str(abs(signal.strength)))
    target_notional = fraction * allocated_capital * abs_strength

    # Step 2: Convert to quantity (raw shares)
    raw_quantity = target_notional / current_price

    # Step 3: Round down to lot size
    # Formula: floor(raw_quantity / lot_size) * lot_size
    quantity = int(raw_quantity / lot_size) * lot_size

    # Step 4: Enforce minimum quantity
    if quantity < min_quantity:
        return 0

    return quantity


class FixedFractionSizer:
    """Position sizer using fixed-fraction capital allocation.

    This sizer calculates order quantities by allocating a fixed fraction of
    the strategy's capital to each position, scaled by signal strength.

    The sizing formula is:
        target_notional = fraction * allocated_capital * |signal.strength|
        quantity = floor(target_notional / current_price / lot_size) * lot_size

    Configuration:
    - fraction: Fraction of capital per position (from RiskConfig.sizing.fraction)
    - lot_size: Minimum trading unit (default: 1 share)
    - min_quantity: Minimum order quantity (default: 0)

    Thread Safety:
        Immutable after construction, safe for concurrent use.

    Example:
        >>> config = SizingConfig(model="fixed_fraction", fraction=Decimal("0.02"))
        >>> sizer = FixedFractionSizer(
        ...     fraction=config.fraction,
        ...     lot_size=1,
        ...     min_quantity=0
        ... )
        >>> signal = Signal(...)
        >>> quantity = sizer.size_position(
        ...     signal=signal,
        ...     allocated_capital=Decimal("10000"),
        ...     current_price=Decimal("150")
        ... )
    """

    def __init__(
        self,
        fraction: Decimal,
        lot_size: int = 1,
        min_quantity: int = 0,
    ):
        """Initialize the fixed-fraction sizer.

        Args:
            fraction: Position sizing fraction (0 < fraction <= 1)
            lot_size: Minimum trading unit (default: 1 share)
            min_quantity: Minimum order quantity (default: 0)

        Raises:
            ValueError: If fraction is not in (0, 1] or lot_size/min_quantity invalid
        """
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {lot_size}")
        if min_quantity < 0:
            raise ValueError(f"min_quantity must be non-negative, got {min_quantity}")

        self._fraction = fraction
        self._lot_size = lot_size
        self._min_quantity = min_quantity

    @property
    def fraction(self) -> Decimal:
        """Position sizing fraction."""
        return self._fraction

    @property
    def lot_size(self) -> int:
        """Minimum trading unit."""
        return self._lot_size

    @property
    def min_quantity(self) -> int:
        """Minimum order quantity."""
        return self._min_quantity

    def size_position(
        self,
        signal: Signal,
        allocated_capital: Decimal,
        current_price: Decimal,
    ) -> int:
        """Calculate order quantity for a signal.

        Args:
            signal: Trading signal with strength [-1, 1]
            allocated_capital: Capital allocated to this signal's strategy
            current_price: Current market price for the symbol

        Returns:
            Order quantity in shares (always >= 0)

        Raises:
            ValueError: If allocated_capital or current_price is negative

        Example:
            >>> sizer = FixedFractionSizer(fraction=Decimal("0.02"))
            >>> signal = Signal(
            ...     ts=datetime(2024, 1, 1),
            ...     strategy_id="A",
            ...     symbol="AAPL",
            ...     side="BUY",
            ...     strength=Decimal("1.0")
            ... )
            >>> sizer.size_position(
            ...     signal=signal,
            ...     allocated_capital=Decimal("10000"),
            ...     current_price=Decimal("150")
            ... )
            1  # 10000 * 0.02 = 200 notional / 150 price = 1.33 shares → 1 share
        """
        return size_position(
            signal=signal,
            allocated_capital=allocated_capital,
            current_price=current_price,
            fraction=self._fraction,
            lot_size=self._lot_size,
            min_quantity=self._min_quantity,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"FixedFractionSizer("
            f"fraction={self._fraction}, "
            f"lot_size={self._lot_size}, "
            f"min_quantity={self._min_quantity})"
        )
