"""Order models for trading engine."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import NamedTuple, Optional


class OrderSide(Enum):
    """Order side (buy or sell)."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    MARKET_ON_CLOSE = "MOC"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderState(Enum):
    """Order state in lifecycle."""

    SUBMITTED = "SUBMITTED"
    TRIGGERED = "TRIGGERED"  # For stop orders only
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"


class TimeInForce(Enum):
    """Time-in-force for orders."""

    DAY = "DAY"  # Expires at end of day
    IOC = "IOC"  # Immediate or cancel (Market/MOC)
    GTC = "GTC"  # Good till canceled (Phase 2)


class OrderBase(NamedTuple):
    """Base order fields without validation."""

    order_id: str
    strategy_ts: datetime  # When strategy submitted the order
    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType
    state: OrderState
    limit_price: Optional[Decimal] = None  # For LIMIT orders
    stop_price: Optional[Decimal] = None  # For STOP orders
    tif: TimeInForce = TimeInForce.DAY
    remaining_qty: int = 0  # Unfilled quantity (for partials)
    filled_qty: int = 0  # Filled quantity so far
    avg_fill_price: Optional[Decimal] = None  # Average fill price across all slices
    submission_bar_ts: Optional[datetime] = None  # Bar timestamp when submitted
    expiry_bar_ts: Optional[datetime] = None  # When order expires (for DAY orders)
    signal_price: Optional[Decimal] = None  # Price when signal was generated (for deviation checks)

    def with_state(self, new_state: OrderState) -> "OrderBase":
        """Create new order with updated state."""
        return self._replace(state=new_state)

    def with_partial_fill(self, fill_qty: int, fill_price: Decimal, remaining: int) -> "OrderBase":
        """
        Create new order after partial fill.

        Args:
            fill_qty: Quantity filled in this slice
            fill_price: Price of this fill
            remaining: Remaining unfilled quantity

        Returns:
            New Order with updated filled_qty, remaining_qty, avg_fill_price
        """
        new_filled = self.filled_qty + fill_qty

        # Calculate new average fill price
        if self.avg_fill_price is None:
            new_avg_price = fill_price
        else:
            # Weighted average: (old_total + new_fill_value) / new_total_qty
            old_value = self.avg_fill_price * self.filled_qty
            new_value = fill_price * fill_qty
            new_avg_price = (old_value + new_value) / new_filled

        # Determine new state
        if remaining == 0:
            new_state = OrderState.FILLED
        else:
            new_state = OrderState.PARTIALLY_FILLED

        return self._replace(
            filled_qty=new_filled,
            remaining_qty=remaining,
            avg_fill_price=new_avg_price,
            state=new_state,
        )

    def is_terminal(self) -> bool:
        """Check if order is in terminal state (no further updates)."""
        return self.state in {OrderState.FILLED, OrderState.EXPIRED, OrderState.CANCELED}

    def is_fillable(self) -> bool:
        """Check if order can still be filled."""
        return self.state in {
            OrderState.SUBMITTED,
            OrderState.TRIGGERED,
            OrderState.PARTIALLY_FILLED,
        }


def Order(
    order_id: str,
    strategy_ts: datetime,
    symbol: str,
    side: OrderSide,
    qty: int,
    order_type: OrderType,
    state: OrderState,
    limit_price: Optional[Decimal] = None,
    stop_price: Optional[Decimal] = None,
    tif: TimeInForce = TimeInForce.DAY,
    remaining_qty: int = 0,
    filled_qty: int = 0,
    avg_fill_price: Optional[Decimal] = None,
    submission_bar_ts: Optional[datetime] = None,
    expiry_bar_ts: Optional[datetime] = None,
    signal_price: Optional[Decimal] = None,
) -> OrderBase:
    """
    Create an immutable order with validation.

    Order objects are immutable - state changes create new Order instances.
    This ensures order history is preserved and makes testing easier.

    Args:
        order_id: Unique order identifier
        strategy_ts: When strategy submitted the order
        symbol: Trading symbol
        side: BUY or SELL
        qty: Order quantity
        order_type: MARKET, MARKET_ON_CLOSE, LIMIT, or STOP
        state: Current order state
        limit_price: Required for LIMIT orders
        stop_price: Required for STOP orders
        tif: Time-in-force (DAY, IOC, GTC)
        remaining_qty: Unfilled quantity (auto-set for SUBMITTED orders)
        filled_qty: Filled quantity so far
        avg_fill_price: Average fill price across all slices
        submission_bar_ts: Bar timestamp when submitted
        expiry_bar_ts: When order expires
        signal_price: Price when signal was generated (for deviation checks)

    Returns:
        Validated Order instance

    Raises:
        ValueError: If validation fails
    """
    # Validate qty > 0
    if qty <= 0:
        raise ValueError(f"Order qty must be > 0, got {qty}")

    # Validate limit price for LIMIT orders
    if order_type == OrderType.LIMIT and limit_price is None:
        raise ValueError("LIMIT orders must have limit_price")

    # Validate stop price for STOP orders
    if order_type == OrderType.STOP and stop_price is None:
        raise ValueError("STOP orders must have stop_price")

    # Validate remaining_qty
    if remaining_qty < 0:
        raise ValueError(f"remaining_qty cannot be negative: {remaining_qty}")

    # Validate filled_qty
    if filled_qty < 0:
        raise ValueError(f"filled_qty cannot be negative: {filled_qty}")

    if filled_qty > qty:
        raise ValueError(f"filled_qty ({filled_qty}) cannot exceed qty ({qty})")

    # Set remaining_qty automatically for SUBMITTED orders if not provided
    if state == OrderState.SUBMITTED and remaining_qty == 0 and filled_qty == 0:
        remaining_qty = qty

    return OrderBase(
        order_id=order_id,
        strategy_ts=strategy_ts,
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        state=state,
        limit_price=limit_price,
        stop_price=stop_price,
        tif=tif,
        remaining_qty=remaining_qty,
        filled_qty=filled_qty,
        avg_fill_price=avg_fill_price,
        submission_bar_ts=submission_bar_ts,
        expiry_bar_ts=expiry_bar_ts,
        signal_price=signal_price,
    )


class Fill(NamedTuple):
    """
    Immutable fill (execution) record.

    Each fill represents a single execution slice.
    For partial fills, there will be multiple Fill objects per Order.
    """

    fill_id: str
    order_id: str
    execution_ts: datetime  # When fill occurred (bar timestamp)
    symbol: str
    side: OrderSide
    qty: int
    price: Decimal  # Execution price (before fees)
    slippage_bps: int  # Slippage in basis points
    fees: Decimal  # Commissions + fees
    participation: float  # Fraction of bar volume (0.0 - 1.0)
    partial_index: int  # 0 for first fill, 1 for second, etc.

    def gross_value(self) -> Decimal:
        """Calculate gross cash impact (before fees)."""
        return self.price * self.qty

    def net_value(self) -> Decimal:
        """Calculate net cash impact (after fees)."""
        gross = self.gross_value()
        # Buys are negative cash impact, sells are positive
        if self.side == OrderSide.BUY:
            return -(gross + self.fees)
        else:
            return gross - self.fees
