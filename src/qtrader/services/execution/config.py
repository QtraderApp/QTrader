"""Configuration for execution service.

Defines commission rates, slippage models, and execution constraints.
"""

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class CommissionConfig:
    """Commission calculation settings.

    Supports three commission models:
    1. Per-share: max(qty * per_share, minimum)
    2. Per-trade: flat_fee OR percentage * notional
    3. Tiered: Volume-based brackets with different per-share rates

    Exactly one model must be specified. Commission cap applies to all models.

    Attributes:
        per_share: Cost per share for per-share model (e.g., $0.005)
        minimum: Minimum commission per trade (default: $0)
        flat_fee: Fixed commission per trade for per-trade model
        percentage: Percentage of notional for per-trade model (e.g., 0.001 = 0.1%)
        tiers: Volume brackets for tiered model [(max_qty, per_share_rate), ...]
        cap: Maximum commission per order (optional)

    Examples:
        Per-share model (existing):
        >>> config = CommissionConfig(
        ...     per_share=Decimal("0.005"),
        ...     minimum=Decimal("1.00")
        ... )
        >>> # 100 shares: max(100 * 0.005, 1.00) = $1.00
        >>> # 500 shares: max(500 * 0.005, 1.00) = $2.50

        Per-trade flat fee:
        >>> config = CommissionConfig(
        ...     flat_fee=Decimal("5.00"),
        ...     minimum=Decimal("1.00")
        ... )
        >>> # Any size: $5.00

        Per-trade percentage:
        >>> config = CommissionConfig(
        ...     percentage=Decimal("0.001"),  # 0.1%
        ...     minimum=Decimal("1.00")
        ... )
        >>> # 100 shares @ $50: max(100 * 50 * 0.001, 1.00) = max(5.00, 1.00) = $5.00

        Tiered model:
        >>> config = CommissionConfig(
        ...     tiers=[
        ...         (Decimal("1000"), Decimal("0.01")),   # 0-1000: $0.01/share
        ...         (Decimal("5000"), Decimal("0.005")),  # 1001-5000: $0.005/share
        ...         (Decimal("inf"), Decimal("0.003"))    # 5001+: $0.003/share
        ...     ],
        ...     minimum=Decimal("1.00")
        ... )
        >>> # 500 shares: 500 * 0.01 = $5.00
        >>> # 3000 shares: 1000 * 0.01 + 2000 * 0.005 = $20.00

        With commission cap:
        >>> config = CommissionConfig(
        ...     per_share=Decimal("0.01"),
        ...     cap=Decimal("50.00")
        ... )
        >>> # 10000 shares: min(10000 * 0.01, 50.00) = $50.00 (capped)
    """

    per_share: Decimal | None = None
    minimum: Decimal = Decimal("0")
    flat_fee: Decimal | None = None
    percentage: Decimal | None = None
    tiers: list[tuple[Decimal, Decimal]] | None = None
    cap: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one commission model is specified."""
        models_specified = [
            self.per_share is not None,
            self.flat_fee is not None or self.percentage is not None,
            self.tiers is not None,
        ]

        if sum(models_specified) == 0:
            raise ValueError("Must specify at least one commission model")
        if sum(models_specified) > 1:
            raise ValueError("Cannot specify multiple commission models (per_share, flat_fee/percentage, or tiers)")

        # Validate per-trade model
        if self.flat_fee is not None and self.percentage is not None:
            raise ValueError("Cannot specify both flat_fee and percentage")

        # Validate tiers
        if self.tiers is not None:
            if len(self.tiers) == 0:
                raise ValueError("Tiers cannot be empty")
            for max_qty, rate in self.tiers:
                if max_qty <= 0:
                    raise ValueError(f"Tier max_qty must be positive, got {max_qty}")
                if rate < 0:
                    raise ValueError(f"Tier rate cannot be negative, got {rate}")

        # Validate cap
        if self.cap is not None and self.cap < 0:
            raise ValueError(f"Commission cap cannot be negative, got {self.cap}")

        # Validate minimum
        if self.minimum < 0:
            raise ValueError(f"Minimum commission cannot be negative, got {self.minimum}")


def _default_commission_config() -> CommissionConfig:
    """Create default commission configuration.

    Returns:
        CommissionConfig with per-share model ($0.005 per share, $1.00 minimum)
    """
    return CommissionConfig(per_share=Decimal("0.005"), minimum=Decimal("1.00"))


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
    commission: CommissionConfig = field(default_factory=_default_commission_config)
