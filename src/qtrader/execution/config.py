"""Execution engine configuration."""

from decimal import Decimal
from typing import NamedTuple


class ExecutionConfig(NamedTuple):
    """Configuration for execution engine."""

    # Commission settings
    per_share: Decimal = Decimal("0.0005")  # Per-share commission
    ticket_min: Decimal = Decimal("1.00")  # Minimum commission per ticket

    # Slippage settings (basis points)
    moc_slip_bps: int = 5  # MOC slippage in bps (default 5 bps)

    # Borrow cost settings
    borrow_rate_annual: Decimal = Decimal("0.03")  # 3% annual borrow cost for shorts

    # Participation settings (Stage 5)
    max_participation: Decimal = Decimal("0.10")  # Max 10% of bar volume
    high_participation_warn: Decimal = Decimal("0.05")  # Warn above 5%

    def __post_init__(self):
        """Validate configuration."""
        if self.per_share < Decimal("0"):
            raise ValueError(f"per_share must be >= 0, got {self.per_share}")
        if self.ticket_min < Decimal("0"):
            raise ValueError(f"ticket_min must be >= 0, got {self.ticket_min}")
        if self.moc_slip_bps < 0:
            raise ValueError(f"moc_slip_bps must be >= 0, got {self.moc_slip_bps}")
        if self.borrow_rate_annual < Decimal("0"):
            raise ValueError(f"borrow_rate_annual must be >= 0, got {self.borrow_rate_annual}")
        if not (Decimal("0") < self.max_participation <= Decimal("1.0")):
            raise ValueError(f"max_participation must be (0, 1], got {self.max_participation}")
        if not (Decimal("0") < self.high_participation_warn <= self.max_participation):
            raise ValueError(
                f"high_participation_warn must be (0, max_participation], got {self.high_participation_warn}"
            )
