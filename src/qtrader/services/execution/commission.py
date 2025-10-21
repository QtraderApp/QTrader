"""Commission calculation utilities.

Calculates trading commissions based on configuration.
"""

from decimal import Decimal

from qtrader.services.execution.config import CommissionConfig


class CommissionCalculator:
    """Calculates commissions for fills.

    Supports per-share commission model with minimum.

    Attributes:
        config: Commission configuration
    """

    def __init__(self, config: CommissionConfig) -> None:
        """Initialize commission calculator.

        Args:
            config: Commission configuration
        """
        self.config = config

    def calculate(self, quantity: Decimal) -> Decimal:
        """Calculate commission for a fill.

        Uses per-share model with minimum:
            commission = max(quantity * per_share, minimum)

        Args:
            quantity: Number of shares filled

        Returns:
            Commission amount

        Raises:
            ValueError: If quantity is negative

        Example:
            >>> config = CommissionConfig(per_share=Decimal("0.005"), minimum=Decimal("1.00"))
            >>> calc = CommissionCalculator(config)
            >>> calc.calculate(Decimal("100"))  # 100 * 0.005 = 0.50, min is 1.00
            Decimal('1.00')
            >>> calc.calculate(Decimal("500"))  # 500 * 0.005 = 2.50
            Decimal('2.50')
        """
        if quantity < 0:
            raise ValueError(f"Quantity cannot be negative, got {quantity}")

        per_share_cost = quantity * self.config.per_share
        return max(per_share_cost, self.config.minimum)
