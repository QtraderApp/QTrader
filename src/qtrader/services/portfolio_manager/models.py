"""
Risk Service Data Models.

Defines immutable data structures for risk management:
- Signal: Trading signal from strategy
- OrderBase: Risk-approved order to be executed
- PortfolioState: Snapshot of portfolio for risk checks
- RiskConfig: Configuration for capital allocation, sizing, and limits

All models follow LEGO principles:
- Immutable (frozen dataclasses where appropriate)
- Pure data (no business logic)
- Type-safe (full type hints)
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal


@dataclass
class Signal:
    """
    Trading signal from strategy.

    Represents a strategy's intent to trade, with confidence strength.
    Not yet sized or approved - that's RiskService's job.
    """

    strategy_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    strength: float  # [-1, 1] signal confidence, 0 = no conviction
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal fields."""
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be in [-1, 1], got {self.strength}")
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got {self.side}")


@dataclass
class OrderBase:
    """
    Order to be sent to ExecutionService.

    Risk-approved order with quantity determined by RiskService.
    Includes audit trail in reason field.
    """

    strategy_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    reason: str  # audit trail: "Approved: 500 shares, 2% of allocated capital"

    def __post_init__(self) -> None:
        """Validate order fields."""
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got {self.side}")
        if not self.reason:
            raise ValueError("reason cannot be empty")


@dataclass
class Position:
    """
    Position snapshot for risk calculations.

    Immutable snapshot from PortfolioService.
    """

    symbol: str
    quantity: int  # signed: positive = long, negative = short
    market_value: Decimal  # current market value (qty * price)

    def __post_init__(self) -> None:
        """Validate position fields."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")


@dataclass
class PortfolioState:
    """
    Portfolio snapshot from PortfolioService.

    Immutable snapshot used for risk checks.
    RiskService caches latest state, never mutates it.
    """

    ts: datetime
    equity: Decimal
    cash: Decimal
    gross_exposure: Decimal  # sum of |position_value| for all positions
    net_exposure: Decimal  # sum of position_value (signed)
    positions: dict[str, Position]  # symbol -> Position

    def __post_init__(self) -> None:
        """Validate portfolio state fields."""
        if self.equity < 0:
            raise ValueError(f"equity cannot be negative, got {self.equity}")
        # Note: cash can be negative (margin account)
        if self.gross_exposure < 0:
            raise ValueError(f"gross_exposure cannot be negative, got {self.gross_exposure}")


# ============================================================================
# Configuration Models
# ============================================================================


@dataclass
class StrategyBudget:
    """
    Capital allocation for a strategy.

    Fixed budget allocation (Phase 4 MVP).
    Dynamic rebalancing deferred to Phase 11.
    """

    strategy_id: str
    capital_weight: float  # 0.0 to 1.0, e.g., 0.3 = 30% of equity

    def __post_init__(self) -> None:
        """Validate budget fields."""
        if not 0.0 <= self.capital_weight <= 1.0:
            raise ValueError(f"capital_weight must be in [0, 1], got {self.capital_weight}")
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")


@dataclass
class SizingConfig:
    """
    Position sizing configuration.

    Phase 4 MVP: fixed_fraction only.
    Phase 11: vol_target, equal_weight, kelly.
    """

    model: Literal["fixed_fraction"]
    fraction: float  # e.g., 0.02 = 2% of allocated capital per signal
    min_quantity: int = 1
    round_to_lot: bool = True

    def __post_init__(self) -> None:
        """Validate sizing config."""
        if self.model != "fixed_fraction":
            raise ValueError(f"Phase 4 MVP only supports 'fixed_fraction', got {self.model}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.min_quantity < 1:
            raise ValueError(f"min_quantity must be >= 1, got {self.min_quantity}")


@dataclass
class ConcentrationLimit:
    """
    Concentration limit per symbol.

    Prevents blow-ups from single position.
    """

    max_position_pct: float  # e.g., 0.10 = 10% of equity per symbol

    def __post_init__(self) -> None:
        """Validate concentration limit."""
        if not 0.0 < self.max_position_pct <= 1.0:
            raise ValueError(f"max_position_pct must be in (0, 1], got {self.max_position_pct}")


@dataclass
class LeverageLimit:
    """
    Portfolio leverage limits.

    Controls total portfolio exposure.
    """

    max_gross: float  # e.g., 2.0 = 200% gross exposure
    max_net: float  # e.g., 1.0 = 100% net exposure

    def __post_init__(self) -> None:
        """Validate leverage limits."""
        if self.max_gross <= 0.0:
            raise ValueError(f"max_gross must be positive, got {self.max_gross}")
        if self.max_net <= 0.0:
            raise ValueError(f"max_net must be positive, got {self.max_net}")
        # Note: max_net can be > max_gross for long-short portfolios


@dataclass
class RiskConfig:
    """
    Complete risk management configuration.

    Loaded from YAML, validated on init.
    """

    budgets: list[StrategyBudget]
    sizing: dict[str, SizingConfig]  # strategy_id -> SizingConfig
    concentration: ConcentrationLimit
    leverage: LeverageLimit
    cash_buffer_pct: float = 0.02  # reserve 2% cash for safety

    def __post_init__(self) -> None:
        """Validate risk config."""
        # Check budgets sum to <= 1.0
        total_weight = sum(b.capital_weight for b in self.budgets)
        if total_weight > 1.0:
            raise ValueError(f"Budget weights sum to {total_weight:.2%}, must be <= 100%")

        # Check all budgets have sizing config
        strategy_ids = {b.strategy_id for b in self.budgets}
        sizing_ids = set(self.sizing.keys())
        missing = strategy_ids - sizing_ids
        if missing:
            raise ValueError(f"Strategies {missing} have budgets but no sizing config")

        # Validate cash buffer
        if not 0.0 <= self.cash_buffer_pct <= 0.5:
            raise ValueError(f"cash_buffer_pct must be in [0, 0.5], got {self.cash_buffer_pct}")
