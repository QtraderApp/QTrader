"""Fill policy for order execution."""

from decimal import Decimal
from typing import NamedTuple, Optional

import structlog

from qtrader.models.bar import Bar
from qtrader.models.order import OrderBase, OrderSide, OrderType

logger = structlog.get_logger(__name__)


class FillDecision(NamedTuple):
    """Decision on whether and how to fill an order."""

    should_fill: bool  # Whether order should be filled
    fill_price: Decimal  # Execution price (including slippage)
    reason: str  # Reason for fill or no-fill decision
    next_bar: bool = False  # If True, schedule for next bar instead


class FillPolicy:
    """
    Determine when and how to fill orders.

    Phase 1 (Stage 3) supports:
    - Market orders: Fill at next bar open
    - MOC orders: Fill at current bar close with slippage

    Conservative mode (default):
    - Market: next bar open (guaranteed available)
    - MOC: current bar close ± slippage bps
    """

    def __init__(self, moc_slip_bps: int = 5):
        """
        Initialize fill policy.

        Args:
            moc_slip_bps: Slippage in basis points for MOC orders (default 5)
        """
        if moc_slip_bps < 0:
            raise ValueError(f"moc_slip_bps must be >= 0, got {moc_slip_bps}")

        self.moc_slip_bps = moc_slip_bps
        logger.info("fill_policy.initialized", moc_slip_bps=moc_slip_bps)

    def evaluate_market_order(
        self,
        order: OrderBase,
        current_bar: Bar,
        next_bar: Optional[Bar] = None,
    ) -> FillDecision:
        """
        Evaluate Market order for fill.

        Market orders fill at NEXT bar open (conservative).

        Args:
            order: Market order to evaluate
            current_bar: Current bar being processed
            next_bar: Next bar (if available)

        Returns:
            FillDecision with fill details or schedule for next bar
        """
        if order.order_type != OrderType.MARKET:
            raise ValueError(f"Expected MARKET order, got {order.order_type}")

        # Market orders fill at next bar open
        if next_bar is None:
            return FillDecision(
                should_fill=False,
                fill_price=Decimal("0"),
                reason="No next bar available (end of data)",
                next_bar=False,
            )

        # Fill at next bar open
        fill_price = next_bar.open
        logger.debug(
            "fill_policy.market_fill",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            fill_price=float(fill_price),
            next_bar_ts=next_bar.ts.isoformat(),
        )

        return FillDecision(
            should_fill=True,
            fill_price=fill_price,
            reason=f"Market order fills at next bar open: ${fill_price}",
            next_bar=True,  # Mark that this fills on next bar
        )

    def evaluate_moc_order(
        self,
        order: OrderBase,
        current_bar: Bar,
    ) -> FillDecision:
        """
        Evaluate MOC (Market-On-Close) order for fill.

        MOC orders fill at current bar close with slippage.

        Slippage calculation:
        - BUY: close * (1 + slip_bps/10000)
        - SELL: close * (1 - slip_bps/10000)

        Args:
            order: MOC order to evaluate
            current_bar: Current bar being processed

        Returns:
            FillDecision with fill details
        """
        if order.order_type != OrderType.MARKET_ON_CLOSE:
            raise ValueError(f"Expected MOC order, got {order.order_type}")

        # Calculate slippage
        slip_factor = Decimal(self.moc_slip_bps) / Decimal("10000")

        # Apply slippage based on side
        if order.side == OrderSide.BUY:
            # Buys pay slippage (price increases)
            fill_price = current_bar.close * (Decimal("1") + slip_factor)
        else:  # SELL
            # Sells pay slippage (price decreases)
            fill_price = current_bar.close * (Decimal("1") - slip_factor)

        logger.debug(
            "fill_policy.moc_fill",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            close_price=float(current_bar.close),
            slip_bps=self.moc_slip_bps,
            fill_price=float(fill_price),
        )

        return FillDecision(
            should_fill=True,
            fill_price=fill_price,
            reason=f"MOC order fills at close ± {self.moc_slip_bps} bps: ${fill_price}",
            next_bar=False,
        )

    def evaluate_order(
        self,
        order: OrderBase,
        current_bar: Bar,
        next_bar: Optional[Bar] = None,
    ) -> FillDecision:
        """
        Evaluate any order type for fill.

        Routes to appropriate evaluation method based on order type.

        Args:
            order: Order to evaluate
            current_bar: Current bar being processed
            next_bar: Next bar (if available, for Market orders)

        Returns:
            FillDecision with fill details

        Raises:
            ValueError: If order type not supported in Stage 3
        """
        if not order.is_fillable():
            return FillDecision(
                should_fill=False,
                fill_price=Decimal("0"),
                reason=f"Order not fillable (state={order.state.value})",
                next_bar=False,
            )

        if order.order_type == OrderType.MARKET:
            return self.evaluate_market_order(order, current_bar, next_bar)
        elif order.order_type == OrderType.MARKET_ON_CLOSE:
            return self.evaluate_moc_order(order, current_bar)
        else:
            # Limit and Stop orders handled in Stage 4
            return FillDecision(
                should_fill=False,
                fill_price=Decimal("0"),
                reason=f"Order type {order.order_type.value} not supported in Stage 3",
                next_bar=False,
            )
