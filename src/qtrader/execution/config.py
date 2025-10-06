"""Execution engine configuration."""

from decimal import Decimal
from typing import NamedTuple, Optional


class ExecutionConfig(NamedTuple):
    """Configuration for execution engine."""

    # Commission settings
    per_share: Decimal = Decimal("0.0005")  # Per-share commission
    ticket_min: Decimal = Decimal("1.00")  # Minimum commission per ticket

    # Slippage settings (basis points)
    moc_slip_bps: int = 5  # MOC slippage in bps (default 5 bps)
    stop_slip_bps: int = 5  # Stop order slippage in bps (default 5 bps)

    # Fill mode settings (Stage 4)
    limit_mode: str = "conservative"  # conservative | optimistic
    stop_mode: str = "conservative"  # conservative | optimistic

    # Borrow cost settings
    borrow_rate_annual: Decimal = Decimal("0.03")  # 3% annual borrow cost for shorts

    # Participation settings (Stage 5)
    max_participation: Decimal = Decimal("0.10")  # Max 10% of bar volume
    queue_bars: int = 3  # Number of bars to keep residuals before expiration
    allow_high_participation: bool = False  # Allow max_participation > 0.20

    # Fill price safeguards
    max_fill_price_deviation_pct: Optional[Decimal] = Decimal(
        "0.10"
    )  # Cancel order if fill price deviates >10% from signal price (None = disabled)

    # Warmup settings (Stage 6A)
    warmup: bool = False  # Enable warmup phase for indicators
    warmup_bars: Optional[int] = None  # Number of warmup bars (None = auto-detect)

    def __post_init__(self):
        """Validate configuration."""
        if self.per_share < Decimal("0"):
            raise ValueError(f"per_share must be >= 0, got {self.per_share}")
        if self.ticket_min < Decimal("0"):
            raise ValueError(f"ticket_min must be >= 0, got {self.ticket_min}")
        if self.moc_slip_bps < 0:
            raise ValueError(f"moc_slip_bps must be >= 0, got {self.moc_slip_bps}")
        if self.stop_slip_bps < 0:
            raise ValueError(f"stop_slip_bps must be >= 0, got {self.stop_slip_bps}")
        if self.limit_mode not in ("conservative", "optimistic"):
            raise ValueError(f"limit_mode must be 'conservative' or 'optimistic', got {self.limit_mode}")
        if self.stop_mode not in ("conservative", "optimistic"):
            raise ValueError(f"stop_mode must be 'conservative' or 'optimistic', got {self.stop_mode}")
        if self.borrow_rate_annual < Decimal("0"):
            raise ValueError(f"borrow_rate_annual must be >= 0, got {self.borrow_rate_annual}")
        if not (Decimal("0") < self.max_participation <= Decimal("1.0")):
            raise ValueError(f"max_participation must be (0, 1], got {self.max_participation}")
        if self.queue_bars < 1:
            raise ValueError(f"queue_bars must be >= 1, got {self.queue_bars}")
        # Note: warmup_bars validation omitted - NamedTuple doesn't call __post_init__
        # Validation can be added in warmup processor if needed
