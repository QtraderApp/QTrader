"""
Signal model - Trading intent before position sizing.

Signals represent WHAT to trade (symbol, direction) but not HOW MUCH.
RiskManager converts Signal → Order with appropriate position size.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional

from qtrader.models.order import OrderType, TimeInForce


class SignalType(Enum):
    """Signal type - trading action intent."""

    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    REBALANCE = "rebalance"


class SignalDirection(Enum):
    """Signal direction - position target."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Signal(NamedTuple):
    """
    Trading signal from strategy (pre-sizing).

    Represents INTENT, not sized order.
    RiskManager converts Signal → Order with appropriate qty.

    Attributes:
        signal_id: Unique identifier for this signal
        strategy_ts: Strategy timestamp (when signal generated)
        symbol: Trading symbol
        signal_type: Type of signal (entry/exit/rebalance)
        direction: Desired direction (long/short/flat)
        target_qty: Optional quantity hint (strategy preference)
        target_weight: Optional portfolio weight (0.0-1.0)
        target_value: Optional dollar value
        order_type: Preferred order type (default MARKET)
        limit_price: Limit price for LIMIT orders
        stop_price: Stop price for STOP orders or risk sizing
        tif: Time in force (default DAY)
        conviction: Signal confidence (0.0-1.0, default 1.0)
        urgency: Signal urgency (normal|high|low, default normal)
        metadata: Additional signal metadata
    """

    signal_id: str
    strategy_ts: datetime
    symbol: str
    signal_type: SignalType
    direction: SignalDirection

    # Sizing hints (strategy preference, not final)
    target_qty: Optional[int] = None
    target_weight: Optional[Decimal] = None
    target_value: Optional[Decimal] = None

    # Order preferences
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    tif: TimeInForce = TimeInForce.DAY

    # Risk context
    conviction: Decimal = Decimal("1.0")
    urgency: str = "normal"
    metadata: Dict[str, Any] = {}

    def validate(self) -> None:
        """
        Validate signal consistency.

        Raises:
            ValueError: If signal is invalid
        """
        # Validate conviction range
        if not (Decimal("0.0") <= self.conviction <= Decimal("1.0")):
            raise ValueError(f"Conviction must be 0.0-1.0, got {self.conviction}")

        # Validate urgency
        if self.urgency not in ("normal", "high", "low"):
            raise ValueError(f"Urgency must be normal|high|low, got {self.urgency}")

        # Validate signal_type and direction consistency
        if self.signal_type == SignalType.ENTRY_LONG and self.direction != SignalDirection.LONG:
            raise ValueError("ENTRY_LONG requires LONG direction")

        if self.signal_type == SignalType.ENTRY_SHORT and self.direction != SignalDirection.SHORT:
            raise ValueError("ENTRY_SHORT requires SHORT direction")

        if self.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
            if self.direction != SignalDirection.FLAT:
                raise ValueError(f"EXIT signals should have FLAT direction, got {self.direction}")

        # Validate order type requirements
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders require limit_price")

        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP orders require stop_price")

        # Validate sizing hints
        if self.target_qty is not None and self.target_qty <= 0:
            raise ValueError(f"target_qty must be positive, got {self.target_qty}")

        if self.target_weight is not None:
            if not (Decimal("0.0") < self.target_weight <= Decimal("1.0")):
                raise ValueError(f"target_weight must be 0.0-1.0, got {self.target_weight}")

        if self.target_value is not None and self.target_value <= Decimal("0"):
            raise ValueError(f"target_value must be positive, got {self.target_value}")
