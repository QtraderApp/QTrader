"""Configuration for execution service.

Defines commission rates, slippage models, and execution constraints.
"""

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class CommissionConfig:
    """Commission calculation settings.

    Attributes:
        per_share: Cost per share (e.g., $0.005)
        minimum: Minimum commission per trade (e.g., $1.00)

    Example:
        >>> config = CommissionConfig(
        ...     per_share=Decimal("0.005"),
        ...     minimum=Decimal("1.00")
        ... )
        >>> # For 100 shares: max(100 * 0.005, 1.00) = $1.00
        >>> # For 500 shares: max(500 * 0.005, 1.00) = $2.50
    """

    per_share: Decimal = Decimal("0.005")
    minimum: Decimal = Decimal("1.00")


@dataclass
class ExecutionConfig:
    """Configuration for execution service.

    Attributes:
        slippage_bps: Slippage in basis points (5 = 0.05%)
        max_participation_rate: Max % of bar volume (0.10 = 10%)
        market_order_queue_bars: Bars to queue market orders (1 = next bar)
        commission: Commission calculation settings

    Example:
        >>> config = ExecutionConfig(
        ...     slippage_bps=Decimal("5"),
        ...     max_participation_rate=Decimal("0.10"),
        ...     market_order_queue_bars=1
        ... )
    """

    slippage_bps: Decimal = Decimal("5")
    max_participation_rate: Decimal = Decimal("0.10")
    market_order_queue_bars: int = 1
    commission: CommissionConfig = field(default_factory=CommissionConfig)
