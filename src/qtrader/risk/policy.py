"""
Risk policy configuration.

Defines position sizing methods, concentration limits, leverage constraints,
and safety margins for risk management.
"""

from decimal import Decimal
from enum import Enum
from typing import NamedTuple, Optional


class SizingMethod(Enum):
    """
    Position sizing methods.

    Phase 1 methods (implemented):
    - FIXED_QUANTITY: Fixed number of shares
    - FIXED_VALUE: Fixed dollar amount
    - PORTFOLIO_PERCENT: Percentage of equity
    - RISK_PERCENT: Percentage at risk (requires stop)

    Phase 2 methods (deferred):
    - VOLATILITY_TARGET: Size based on volatility
    - KELLY_CRITERION: Optimal Kelly sizing
    - EQUAL_RISK_CONTRIBUTION: Risk parity
    """

    FIXED_QUANTITY = "fixed_quantity"
    FIXED_VALUE = "fixed_value"
    PORTFOLIO_PERCENT = "portfolio_percent"
    RISK_PERCENT = "risk_percent"
    VOLATILITY_TARGET = "volatility_target"  # Phase 2
    KELLY_CRITERION = "kelly_criterion"  # Phase 2
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"  # Phase 2


class RiskPolicy(NamedTuple):
    """
    Risk management policy configuration.

    Attributes:
        sizing_method: Position sizing method (default PORTFOLIO_PERCENT)
        default_position_size: Default position size for FIXED_QUANTITY or base % for PORTFOLIO_PERCENT
        max_position_pct: Maximum position size as % of equity (default 0.20 = 20%)
        max_positions: Maximum concurrent positions (None = unlimited)
        max_gross_exposure: Maximum gross exposure (default 1.0 = 100%)
        max_net_exposure: Maximum net exposure (default 1.0 = 100%)
        allow_shorting: Allow short positions (default False)
        cash_reserve_pct: Minimum cash reserve as % of equity (default 0.05 = 5%)
        max_fill_price_deviation_pct: Reject orders if fill price deviates >X% from signal price (default 0.10 = 10%, None = disabled)
        reject_on_insufficient_cash: Reject signals if insufficient cash (default True)
        reject_on_concentration_breach: Reject signals if concentration limits breached (default True)
        reject_on_leverage_breach: Reject signals if leverage limits breached (default True)
        check_cash_before_concentration: Check cash BEFORE concentration adjustment for multi-strategy fairness (default False)
    """

    # Position sizing (Phase 1)
    sizing_method: SizingMethod = SizingMethod.PORTFOLIO_PERCENT
    default_position_size: Decimal = Decimal("0.05")

    # Concentration limits
    max_position_pct: Decimal = Decimal("0.20")
    max_positions: Optional[int] = None

    # Leverage & exposure
    max_gross_exposure: Decimal = Decimal("1.0")
    max_net_exposure: Decimal = Decimal("1.0")
    allow_shorting: bool = False

    # Safety margins
    cash_reserve_pct: Decimal = Decimal("0.05")

    # Fill price safeguards
    max_fill_price_deviation_pct: Optional[Decimal] = Decimal(
        "0.10"
    )  # Reject if fill price deviates >10% from signal price

    # Validation
    reject_on_insufficient_cash: bool = True
    reject_on_concentration_breach: bool = True
    reject_on_leverage_breach: bool = True
    check_cash_before_concentration: bool = False  # Check cash before concentration adjustment

    def validate(self) -> None:
        """
        Validate policy configuration.

        Raises:
            ValueError: If policy is invalid
        """
        # Validate default_position_size based on sizing method
        # For FIXED_QUANTITY and FIXED_VALUE, can be any positive number
        # For percentage-based methods, must be 0.0-1.0
        if self.sizing_method in (SizingMethod.PORTFOLIO_PERCENT, SizingMethod.RISK_PERCENT):
            if not (Decimal("0.0") < self.default_position_size <= Decimal("1.0")):
                raise ValueError(
                    f"default_position_size must be 0.0-1.0 for {self.sizing_method.value}, got {self.default_position_size}"
                )
        else:
            # For FIXED_QUANTITY and FIXED_VALUE, just check positive
            if self.default_position_size <= Decimal("0"):
                raise ValueError(f"default_position_size must be positive, got {self.default_position_size}")

        # Validate percentages

        if not (Decimal("0.0") < self.max_position_pct <= Decimal("1.0")):
            raise ValueError(f"max_position_pct must be 0.0-1.0, got {self.max_position_pct}")

        if not (Decimal("0.0") <= self.cash_reserve_pct < Decimal("1.0")):
            raise ValueError(f"cash_reserve_pct must be 0.0-1.0, got {self.cash_reserve_pct}")

        if not (Decimal("0.0") < self.max_gross_exposure <= Decimal("10.0")):
            raise ValueError(f"max_gross_exposure must be 0.0-10.0, got {self.max_gross_exposure}")

        if not (Decimal("0.0") <= self.max_net_exposure <= self.max_gross_exposure):
            raise ValueError(f"max_net_exposure must be 0.0-{self.max_gross_exposure}, got {self.max_net_exposure}")

        # Validate max_positions
        if self.max_positions is not None and self.max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {self.max_positions}")

        # Validate sizing method is supported in Phase 1
        phase1_methods = {
            SizingMethod.FIXED_QUANTITY,
            SizingMethod.FIXED_VALUE,
            SizingMethod.PORTFOLIO_PERCENT,
            SizingMethod.RISK_PERCENT,
        }

        if self.sizing_method not in phase1_methods:
            # Phase 2 methods - validation only (will be handled with fallback in RiskManager)
            pass
