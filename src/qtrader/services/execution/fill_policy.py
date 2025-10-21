"""Fill policy for order execution simulation.

Implements conservative fill logic based on bar data and order types.
"""

from decimal import Decimal

from qtrader.models.bar import Bar
from qtrader.services.execution.config import ExecutionConfig
from qtrader.services.execution.models import FillDecision, Order, OrderType


class FillPolicy:
    """Fill policy implementing conservative execution simulation.

    Evaluates orders against bar data and determines if/how they should fill.
    Uses conservative assumptions to avoid overly optimistic backtests.

    Conservative Rules:
    - Market: Fill at next bar's open (queued for 1 bar)
    - Limit: Fill at min(limit, close) for buy, max(limit, close) for sell
    - Stop: Fill at max(stop, close) + slippage for buy, min(stop, close) - slippage for sell
    - MOC: Fill at current bar's close + slippage

    Attributes:
        config: Execution configuration with slippage, participation limits
    """

    def __init__(self, config: ExecutionConfig) -> None:
        """Initialize fill policy.

        Args:
            config: Execution configuration
        """
        self.config = config

    def evaluate_order(self, order: Order, bar: Bar) -> FillDecision:
        """Evaluate order against bar data.

        Routes to appropriate evaluation method based on order type.
        Applies volume participation limits for all order types.

        Args:
            order: Order to evaluate
            bar: Bar data with OHLCV

        Returns:
            FillDecision with fill instructions

        Raises:
            ValueError: If order type not supported
        """
        # Skip if order not active
        if not order.is_active:
            return FillDecision(
                should_fill=False, reason=f"Order in terminal state: {order.state}", queue_for_next_bar=False
            )

        # Route to appropriate handler
        if order.order_type == OrderType.MARKET:
            return self._evaluate_market(order, bar)
        elif order.order_type == OrderType.LIMIT:
            return self._evaluate_limit(order, bar)
        elif order.order_type == OrderType.STOP:
            return self._evaluate_stop(order, bar)
        elif order.order_type == OrderType.MARKET_ON_CLOSE:
            return self._evaluate_moc(order, bar)
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

    def _evaluate_market(self, order: Order, bar: Bar) -> FillDecision:
        """Evaluate market order.

        Market orders fill at next bar's open after queueing.

        Args:
            order: Market order
            bar: Current bar

        Returns:
            FillDecision
        """
        # Market orders must be queued for N bars
        if order.bars_queued < self.config.market_order_queue_bars:
            return FillDecision(
                should_fill=False,
                reason=f"Market order queued ({order.bars_queued}/{self.config.market_order_queue_bars} bars)",
                queue_for_next_bar=True,
            )

        # Fill at bar open with slippage
        fill_price = self._apply_slippage(Decimal(str(bar.open)), order.side.value)

        # Calculate fillable quantity based on volume participation
        fill_quantity = self._calculate_fillable_quantity(order, bar.volume)

        # Cannot fill if zero quantity
        if fill_quantity == 0:
            return FillDecision(
                should_fill=False, reason="Market order cannot fill (zero volume bar)", queue_for_next_bar=True
            )

        return FillDecision(
            should_fill=True,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            reason="Market order filled at next bar open",
            queue_for_next_bar=(fill_quantity < order.remaining_quantity),
        )

    def _evaluate_limit(self, order: Order, bar: Bar) -> FillDecision:
        """Evaluate limit order.

        Buy limit: Fill if bar.low <= limit_price
        Sell limit: Fill if bar.high >= limit_price
        Fill at min(limit, close) for buy, max(limit, close) for sell.

        Args:
            order: Limit order
            bar: Current bar

        Returns:
            FillDecision
        """
        if order.limit_price is None:
            raise ValueError("Limit order must have limit_price")

        # Convert bar prices to Decimal
        bar_low = Decimal(str(bar.low))
        bar_high = Decimal(str(bar.high))
        bar_close = Decimal(str(bar.close))

        # Check if limit price touched
        if order.side.value == "buy":
            if bar_low > order.limit_price:
                return FillDecision(
                    should_fill=False, reason="Buy limit not touched (bar.low > limit)", queue_for_next_bar=True
                )
            # Fill at min(limit, close) - conservative
            fill_price = min(order.limit_price, bar_close)
        else:  # sell
            if bar_high < order.limit_price:
                return FillDecision(
                    should_fill=False, reason="Sell limit not touched (bar.high < limit)", queue_for_next_bar=True
                )
            # Fill at max(limit, close) - conservative
            fill_price = max(order.limit_price, bar_close)

        # Calculate fillable quantity
        fill_quantity = self._calculate_fillable_quantity(order, bar.volume)

        # Cannot fill if zero quantity
        if fill_quantity == 0:
            return FillDecision(
                should_fill=False,
                reason=f"{order.side.value.capitalize()} limit touched but cannot fill (zero volume)",
                queue_for_next_bar=True,
            )

        return FillDecision(
            should_fill=True,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            reason=f"{order.side.value.capitalize()} limit touched and filled",
            queue_for_next_bar=(fill_quantity < order.remaining_quantity),
        )

    def _evaluate_stop(self, order: Order, bar: Bar) -> FillDecision:
        """Evaluate stop order.

        Buy stop: Trigger if bar.high >= stop_price
        Sell stop: Trigger if bar.low <= stop_price
        Fill at max(stop, close) + slippage for buy, min(stop, close) - slippage for sell.

        Args:
            order: Stop order
            bar: Current bar

        Returns:
            FillDecision
        """
        if order.stop_price is None:
            raise ValueError("Stop order must have stop_price")

        # Convert bar prices to Decimal
        bar_low = Decimal(str(bar.low))
        bar_high = Decimal(str(bar.high))
        bar_close = Decimal(str(bar.close))

        # Check if stop triggered
        if order.side.value == "buy":
            if bar_high < order.stop_price:
                return FillDecision(
                    should_fill=False, reason="Buy stop not triggered (bar.high < stop)", queue_for_next_bar=True
                )
            # Fill at max(stop, close) + slippage
            base_price = max(order.stop_price, bar_close)
            fill_price = self._apply_slippage(base_price, "buy")
        else:  # sell
            if bar_low > order.stop_price:
                return FillDecision(
                    should_fill=False, reason="Sell stop not triggered (bar.low > stop)", queue_for_next_bar=True
                )
            # Fill at min(stop, close) - slippage
            base_price = min(order.stop_price, bar_close)
            fill_price = self._apply_slippage(base_price, "sell")

        # Calculate fillable quantity
        fill_quantity = self._calculate_fillable_quantity(order, bar.volume)

        # Cannot fill if zero quantity
        if fill_quantity == 0:
            return FillDecision(
                should_fill=False,
                reason=f"{order.side.value.capitalize()} stop triggered but cannot fill (zero volume)",
                queue_for_next_bar=True,
            )

        return FillDecision(
            should_fill=True,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            reason=f"{order.side.value.capitalize()} stop triggered and filled",
            queue_for_next_bar=(fill_quantity < order.remaining_quantity),
        )

    def _evaluate_moc(self, order: Order, bar: Bar) -> FillDecision:
        """Evaluate market-on-close order.

        Fills at current bar's close price with slippage.

        Args:
            order: MOC order
            bar: Current bar

        Returns:
            FillDecision
        """
        # Fill at close with slippage
        base_price = Decimal(str(bar.close))
        fill_price = self._apply_slippage(base_price, order.side.value)

        # Calculate fillable quantity
        fill_quantity = self._calculate_fillable_quantity(order, bar.volume)

        # Cannot fill if zero quantity
        if fill_quantity == 0:
            return FillDecision(
                should_fill=False, reason="MOC order cannot fill (zero volume bar)", queue_for_next_bar=True
            )

        return FillDecision(
            should_fill=True,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            reason="MOC order filled at close",
            queue_for_next_bar=(fill_quantity < order.remaining_quantity),
        )

    def _apply_slippage(self, price: Decimal, side: str) -> Decimal:
        """Apply slippage to price.

        Buy: Pay MORE (price * (1 + bps/10000))
        Sell: Receive LESS (price * (1 - bps/10000))

        Args:
            price: Base price
            side: "buy" or "sell"

        Returns:
            Price with slippage applied
        """
        multiplier = Decimal("1") + (self.config.slippage_bps / Decimal("10000"))

        if side == "buy":
            return price * multiplier
        else:  # sell
            return price * (Decimal("2") - multiplier)

    def _calculate_fillable_quantity(self, order: Order, bar_volume: int) -> Decimal:
        """Calculate how much can fill this bar based on volume participation.

        Args:
            order: Order to fill
            bar_volume: Bar's total volume

        Returns:
            Quantity that can fill (may be partial)
        """
        # Handle zero/negative volume bars
        if bar_volume <= 0:
            return Decimal("0")

        # Max fillable based on participation limit
        max_fillable = Decimal(str(bar_volume)) * self.config.max_participation_rate

        # Return minimum of remaining quantity and max fillable
        return min(order.remaining_quantity, max_fillable)
