"""Fill policy - determines if/when orders fill."""

from decimal import Decimal
from typing import NamedTuple, Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.models.canonical_bar import CanonicalBar
from qtrader.models.order import OrderBase, OrderSide, OrderType

logger = LoggerFactory.get_logger()


class FillDecision(NamedTuple):
    """Decision on whether and how to fill an order."""

    should_fill: bool  # Whether order should be filled
    fill_price: Decimal  # Execution price (including slippage)
    reason: str  # Reason for fill or no-fill decision
    next_bar: bool = False  # If True, schedule for next bar instead


class FillPolicy:
    """
    Determine when and how to fill orders.

    Phase 1 supports:
    - Market orders: Fill at next bar open
    - MOC orders: Fill at current bar close with slippage
    - Limit orders: Fill with conservative touch rules (Stage 4)
    - Stop orders: Fill with conservative touch rules (Stage 4)

    Conservative mode (default):
    - Market: next bar open (guaranteed available)
    - MOC: current bar close ± slippage bps
    - Limit Buy: if low ≤ limit, fill at min(limit, close)
    - Limit Sell: if high ≥ limit, fill at max(limit, close)
    - Stop Buy: if high ≥ stop, fill at max(stop, close) ± slippage
    - Stop Sell: if low ≤ stop, fill at min(stop, close) ± slippage
    """

    def __init__(
        self,
        moc_slip_bps: int = 5,
        stop_slip_bps: int = 5,
        limit_mode: str = "conservative",
        stop_mode: str = "conservative",
    ):
        """
        Initialize fill policy.

        Args:
            moc_slip_bps: Slippage in basis points for MOC orders (default 5)
            stop_slip_bps: Slippage in basis points for Stop orders (default 5)
            limit_mode: Fill mode for limit orders - "conservative" or "optimistic" (default "conservative")
            stop_mode: Fill mode for stop orders - "conservative" or "optimistic" (default "conservative")
        """
        if moc_slip_bps < 0:
            raise ValueError(f"moc_slip_bps must be >= 0, got {moc_slip_bps}")
        if stop_slip_bps < 0:
            raise ValueError(f"stop_slip_bps must be >= 0, got {stop_slip_bps}")
        if limit_mode not in ("conservative", "optimistic"):
            raise ValueError(f"limit_mode must be 'conservative' or 'optimistic', got {limit_mode}")
        if stop_mode not in ("conservative", "optimistic"):
            raise ValueError(f"stop_mode must be 'conservative' or 'optimistic', got {stop_mode}")

        self.moc_slip_bps = moc_slip_bps
        self.stop_slip_bps = stop_slip_bps
        self.limit_mode = limit_mode
        self.stop_mode = stop_mode

        logger.info(
            "fill_policy.initialized",
            moc_slip_bps=moc_slip_bps,
            stop_slip_bps=stop_slip_bps,
            limit_mode=limit_mode,
            stop_mode=stop_mode,
        )

    def evaluate_market_order(
        self,
        order: OrderBase,
        current_bar: CanonicalBar,
        next_bar: Optional[CanonicalBar] = None,
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
        fill_price = Decimal(str(next_bar.open))
        logger.debug(
            "fill_policy.market_fill",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            fill_price=float(fill_price),
            next_bar_datetime=next_bar.trade_datetime,  # CanonicalBar has trade_datetime (ISO string)
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
        current_bar: CanonicalBar,
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
            fill_price = Decimal(str(current_bar.close)) * (Decimal("1") + slip_factor)
        else:  # SELL
            # Sells pay slippage (price decreases)
            fill_price = Decimal(str(current_bar.close)) * (Decimal("1") - slip_factor)

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

    def evaluate_limit_order(
        self,
        order: OrderBase,
        current_bar: CanonicalBar,
    ) -> FillDecision:
        """
        Evaluate Limit order for fill using conservative touch rules.

        Conservative rules:
        - Limit Buy: if low ≤ limit, fill at min(limit, close)
        - Limit Sell: if high ≥ limit, fill at max(limit, close)

        Args:
            order: Limit order to evaluate
            current_bar: Current bar being processed

        Returns:
            FillDecision with fill details or no-fill reason
        """
        if order.order_type != OrderType.LIMIT:
            raise ValueError(f"Expected LIMIT order, got {order.order_type}")

        if order.limit_price is None:
            raise ValueError(f"Limit order {order.order_id} missing limit_price")

        limit_price = order.limit_price

        # Conservative touch rules
        if order.side == OrderSide.BUY:
            # Buy: need price to touch or go below limit
            if Decimal(str(current_bar.low)) <= limit_price:
                # Fill at min(limit, close) - best price we could get
                fill_price = min(limit_price, Decimal(str(current_bar.close)))

                logger.debug(
                    "fill_policy.limit_buy_fill",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    qty=order.qty,
                    limit_price=float(limit_price),
                    bar_low=float(current_bar.low),
                    bar_close=float(current_bar.close),
                    fill_price=float(fill_price),
                )

                return FillDecision(
                    should_fill=True,
                    fill_price=fill_price,
                    reason=f"Limit Buy touched (low={current_bar.low} ≤ limit={limit_price}), fill at ${fill_price}",
                    next_bar=False,
                )
            else:
                return FillDecision(
                    should_fill=False,
                    fill_price=Decimal("0"),
                    reason=f"Limit Buy not touched (low={current_bar.low} > limit={limit_price})",
                    next_bar=False,
                )

        else:  # SELL
            # Sell: need price to touch or go above limit
            if Decimal(str(current_bar.high)) >= limit_price:
                # Fill at max(limit, close) - best price we could get
                fill_price = max(limit_price, Decimal(str(current_bar.close)))

                logger.debug(
                    "fill_policy.limit_sell_fill",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    qty=order.qty,
                    limit_price=float(limit_price),
                    bar_high=float(current_bar.high),
                    bar_close=float(current_bar.close),
                    fill_price=float(fill_price),
                )

                return FillDecision(
                    should_fill=True,
                    fill_price=fill_price,
                    reason=f"Limit Sell touched (high={current_bar.high} ≥ limit={limit_price}), fill at ${fill_price}",
                    next_bar=False,
                )
            else:
                return FillDecision(
                    should_fill=False,
                    fill_price=Decimal("0"),
                    reason=f"Limit Sell not touched (high={current_bar.high} < limit={limit_price})",
                    next_bar=False,
                )

    def evaluate_stop_order(
        self,
        order: OrderBase,
        current_bar: CanonicalBar,
    ) -> FillDecision:
        """
        Evaluate Stop order for fill using conservative touch rules.

        Stop orders become market orders when triggered.
        Conservative rules:
        - Stop Buy: if high ≥ stop, fill at max(stop, close) ± slippage
        - Stop Sell: if low ≤ stop, fill at min(stop, close) ± slippage

        Args:
            order: Stop order to evaluate
            current_bar: Current bar being processed

        Returns:
            FillDecision with fill details or no-fill reason
        """
        if order.order_type != OrderType.STOP:
            raise ValueError(f"Expected STOP order, got {order.order_type}")

        if order.stop_price is None:
            raise ValueError(f"Stop order {order.order_id} missing stop_price")

        stop_price = order.stop_price
        slip_factor = Decimal(self.stop_slip_bps) / Decimal("10000")

        # Conservative touch rules
        if order.side == OrderSide.BUY:
            # Stop Buy: triggered when price goes up to or above stop
            if Decimal(str(current_bar.high)) >= stop_price:
                # Fill at max(stop, close) - worst price we'd get
                base_price = max(stop_price, Decimal(str(current_bar.close)))
                # Add slippage (buys pay more)
                fill_price = base_price * (Decimal("1") + slip_factor)

                logger.debug(
                    "fill_policy.stop_buy_fill",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    qty=order.qty,
                    stop_price=float(stop_price),
                    bar_high=float(current_bar.high),
                    bar_close=float(current_bar.close),
                    base_price=float(base_price),
                    slip_bps=self.stop_slip_bps,
                    fill_price=float(fill_price),
                )

                return FillDecision(
                    should_fill=True,
                    fill_price=fill_price,
                    reason=f"Stop Buy triggered (high={current_bar.high} ≥ stop={stop_price}), fill at ${fill_price}",
                    next_bar=False,
                )
            else:
                return FillDecision(
                    should_fill=False,
                    fill_price=Decimal("0"),
                    reason=f"Stop Buy not triggered (high={current_bar.high} < stop={stop_price})",
                    next_bar=False,
                )

        else:  # SELL
            # Stop Sell: triggered when price goes down to or below stop
            if Decimal(str(current_bar.low)) <= stop_price:
                # Fill at min(stop, close) - worst price we'd get
                base_price = min(stop_price, Decimal(str(current_bar.close)))
                # Subtract slippage (sells get less)
                fill_price = base_price * (Decimal("1") - slip_factor)

                logger.debug(
                    "fill_policy.stop_sell_fill",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    qty=order.qty,
                    stop_price=float(stop_price),
                    bar_low=float(current_bar.low),
                    bar_close=float(current_bar.close),
                    base_price=float(base_price),
                    slip_bps=self.stop_slip_bps,
                    fill_price=float(fill_price),
                )

                return FillDecision(
                    should_fill=True,
                    fill_price=fill_price,
                    reason=f"Stop Sell triggered (low={current_bar.low} ≤ stop={stop_price}), fill at ${fill_price}",
                    next_bar=False,
                )
            else:
                return FillDecision(
                    should_fill=False,
                    fill_price=Decimal("0"),
                    reason=f"Stop Sell not triggered (low={current_bar.low} > stop={stop_price})",
                    next_bar=False,
                )

    def evaluate_order(
        self,
        order: OrderBase,
        current_bar: CanonicalBar,
        next_bar: Optional[CanonicalBar] = None,
        is_close_only: bool = False,
    ) -> FillDecision:
        """
        Evaluate any order type for fill.

        Routes to appropriate evaluation method based on order type.

        Args:
            order: Order to evaluate
            current_bar: Current bar being processed
            next_bar: Next bar (if available, for Market orders)
            is_close_only: If True, skip limit/stop evaluation (malformed bar)

        Returns:
            FillDecision with fill details

        Raises:
            ValueError: If order type not supported
        """
        if not order.is_fillable():
            return FillDecision(
                should_fill=False,
                fill_price=Decimal("0"),
                reason=f"Order not fillable (state={order.state.value})",
                next_bar=False,
            )

        # Close-only bars skip limit/stop (high/low not trustworthy)
        if is_close_only and order.order_type in (OrderType.LIMIT, OrderType.STOP):
            return FillDecision(
                should_fill=False,
                fill_price=Decimal("0"),
                reason="Close-only bar: limit/stop evaluation disabled (malformed OHLC)",
                next_bar=False,
            )

        if order.order_type == OrderType.MARKET:
            return self.evaluate_market_order(order, current_bar, next_bar)
        elif order.order_type == OrderType.MARKET_ON_CLOSE:
            return self.evaluate_moc_order(order, current_bar)
        elif order.order_type == OrderType.LIMIT:
            return self.evaluate_limit_order(order, current_bar)
        elif order.order_type == OrderType.STOP:
            return self.evaluate_stop_order(order, current_bar)

        # Unreachable: all OrderType enum values handled above
        raise AssertionError(f"Unhandled OrderType: {order.order_type}")  # pragma: no cover
