"""Commission calculation for order fills."""

from decimal import Decimal
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)


class CommissionResult(NamedTuple):
    """Result of commission calculation."""

    commission: Decimal  # Total commission charged
    per_share_cost: Decimal  # Per-share portion
    ticket_minimum: Decimal  # Ticket minimum (if enforced)
    minimum_enforced: bool  # Whether ticket minimum was applied


class CommissionCalculator:
    """
    Calculate commissions for order fills.

    Formula:
    - Base commission = qty * per_share
    - Final commission = max(base_commission, ticket_min)
    """

    def __init__(
        self,
        per_share: Decimal = Decimal("0.0005"),
        ticket_min: Decimal = Decimal("1.00"),
    ):
        """
        Initialize commission calculator.

        Args:
            per_share: Per-share commission rate (default $0.0005/share)
            ticket_min: Minimum commission per ticket (default $1.00)
        """
        if per_share < Decimal("0"):
            raise ValueError(f"per_share must be >= 0, got {per_share}")
        if ticket_min < Decimal("0"):
            raise ValueError(f"ticket_min must be >= 0, got {ticket_min}")

        self.per_share = per_share
        self.ticket_min = ticket_min

        logger.info(
            "commission_calculator.initialized",
            per_share=float(per_share),
            ticket_min=float(ticket_min),
        )

    def calculate(self, qty: int) -> CommissionResult:
        """
        Calculate commission for a fill.

        Args:
            qty: Fill quantity (positive)

        Returns:
            CommissionResult with commission details

        Raises:
            ValueError: If qty <= 0
        """
        if qty <= 0:
            raise ValueError(f"qty must be > 0, got {qty}")

        # Calculate per-share cost
        per_share_cost = Decimal(qty) * self.per_share

        # Apply ticket minimum
        if per_share_cost < self.ticket_min:
            final_commission = self.ticket_min
            minimum_enforced = True
        else:
            final_commission = per_share_cost
            minimum_enforced = False

        logger.debug(
            "commission.calculated",
            qty=qty,
            per_share_cost=float(per_share_cost),
            final_commission=float(final_commission),
            minimum_enforced=minimum_enforced,
        )

        return CommissionResult(
            commission=final_commission,
            per_share_cost=per_share_cost,
            ticket_minimum=self.ticket_min if minimum_enforced else Decimal("0"),
            minimum_enforced=minimum_enforced,
        )
