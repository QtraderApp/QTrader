"""Execution service implementation.

Simulates realistic order execution for backtesting.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from qtrader.config import LoggerFactory
from qtrader.events.event_bus import EventBus
from qtrader.events.events import FillEvent, OrderApprovedEvent, PriceBarEvent
from qtrader.models.bar import Bar
from qtrader.services.execution.commission import CommissionCalculator
from qtrader.services.execution.config import ExecutionConfig
from qtrader.services.execution.fill_policy import FillPolicy
from qtrader.services.execution.models import Fill, Order, OrderSide, OrderState, OrderType

logger = LoggerFactory.get_logger()


class ExecutionService:
    """Execution service for order simulation.

    Accepts orders, evaluates them against bar data, and generates fills.
    Does NOT modify portfolio - returns Fill objects for external application.

    Attributes:
        config: Execution configuration
        fill_policy: Fill evaluation logic
        commission_calculator: Commission calculation
        _orders: All orders (by order_id)
        _pending_orders_by_symbol: Active orders grouped by symbol

    Example:
        >>> config = ExecutionConfig()
        >>> execution = ExecutionService(config)
        >>>
        >>> # Submit order
        >>> order = Order(
        ...     symbol="AAPL",
        ...     side=OrderSide.BUY,
        ...     quantity=Decimal("100"),
        ...     order_type=OrderType.MARKET,
        ...     created_at=datetime.now()
        ... )
        >>> order_id = execution.submit_order(order)
        >>>
        >>> # Process bar
        >>> bar = Bar(...)
        >>> fills = execution.on_bar(bar)
        >>>
        >>> # Apply fills to portfolio
        >>> for fill in fills:
        ...     portfolio.apply_fill(**fill.__dict__)
    """

    def __init__(self, config: ExecutionConfig, event_bus: Optional[EventBus] = None) -> None:
        """Initialize execution service.

        Args:
            config: Execution configuration
            event_bus: Optional event bus for Phase 5 event-driven mode
        """
        self.config = config
        self.fill_policy = FillPolicy(config)
        self.commission_calculator = CommissionCalculator(config.commission)
        self._event_bus = event_bus

        # Order tracking
        self._orders: dict[str, Order] = {}
        self._pending_orders_by_symbol: dict[str, list[str]] = {}

        # Subscribe to events if event bus provided
        if self._event_bus:
            self._event_bus.subscribe("price_bar", self.on_bar_event)  # type: ignore[arg-type]
            self._event_bus.subscribe("order_approved", self.on_order_approved)  # type: ignore[arg-type]

    def submit_order(self, order: Order) -> str:
        """Submit a new order for execution.

        Order is validated (by Order.__post_init__) and tracked.

        Args:
            order: Order object with all parameters

        Returns:
            order_id: Unique identifier for tracking

        Raises:
            ValueError: If duplicate order_id

        Example:
            >>> order = Order(
            ...     symbol="AAPL",
            ...     side=OrderSide.BUY,
            ...     quantity=Decimal("100"),
            ...     order_type=OrderType.MARKET,
            ...     created_at=datetime.now()
            ... )
            >>> order_id = execution.submit_order(order)
        """
        # Check for duplicate
        if order.order_id in self._orders:
            error_msg = f"Order ID already exists: {order.order_id}"
            logger.error(
                "execution.order.duplicate",
                order_id=order.order_id,
                symbol=order.symbol,
                error=error_msg,
            )
            raise ValueError(error_msg)

        # Mark as submitted
        order.submit(order.created_at)

        # Track order
        self._orders[order.order_id] = order

        # Log submission
        logger.debug(
            "execution.order.submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=str(order.quantity),
            order_type=order.order_type.value,
            limit_price=str(order.limit_price) if order.limit_price else None,
            stop_price=str(order.stop_price) if order.stop_price else None,
        )

        # Add to pending list for this symbol
        if order.symbol not in self._pending_orders_by_symbol:
            self._pending_orders_by_symbol[order.symbol] = []
        self._pending_orders_by_symbol[order.symbol].append(order.order_id)

        return order.order_id

    def on_bar(self, bar: Bar) -> list[Fill]:
        """Process bar and generate fills for eligible orders.

        Evaluates all pending orders for this symbol against bar data.
        Generates Fill objects for successful executions.

        Args:
            bar: Price/volume data for single symbol

        Returns:
            List of Fill objects (empty if no fills)

        Example:
            >>> bar = Bar(
            ...     trade_datetime=datetime(2020, 1, 2, 9, 30),
            ...     open=150.0,
            ...     high=151.0,
            ...     low=149.5,
            ...     close=150.5,
            ...     volume=1000000
            ... )
            >>> fills = execution.on_bar(bar)
        """
        # Extract symbol from bar (assuming it's available via some mechanism)
        # For now, we need to iterate through pending orders for all symbols
        # and check if they match. In practice, the caller would provide symbol.

        fills: list[Fill] = []

        # Process all symbols (caller should filter by symbol before calling)
        for symbol in list(self._pending_orders_by_symbol.keys()):
            order_ids = self._pending_orders_by_symbol.get(symbol, [])

            for order_id in list(order_ids):
                order = self._orders.get(order_id)

                # Defensive: handle missing order
                if order is None:
                    logger.error(
                        "execution.on_bar.missing_order",
                        order_id=order_id,
                        symbol=symbol,
                    )
                    continue

                # Skip if order not active
                if not order.is_active:
                    self._remove_from_pending(order)
                    continue

                # Evaluate order against bar
                decision = self.fill_policy.evaluate_order(order, bar)

                # Handle expiry
                if decision.should_expire:
                    order.expire(bar.trade_datetime)
                    self._remove_from_pending(order)
                    logger.info(
                        "execution.order.expired",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        reason=decision.reason,
                    )
                    continue

                # Queue market orders (before processing fill)
                if decision.queue_for_next_bar:
                    order.bars_queued += 1

                # Generate fill if should_fill (BEFORE cancellation)
                if decision.should_fill:
                    # Calculate commission (pass price for percentage-based models)
                    commission = self.commission_calculator.calculate(decision.fill_quantity, decision.fill_price)

                    # Create fill
                    fill = Fill(
                        order_id=order.order_id,
                        timestamp=bar.trade_datetime,
                        symbol=order.symbol,
                        side=order.side.value,  # Convert enum to string
                        quantity=decision.fill_quantity,
                        price=decision.fill_price,
                        commission=commission,
                    )
                    fills.append(fill)

                    logger.info(
                        "execution.fill.generated",
                        order_id=order.order_id,
                        fill_id=fill.fill_id,
                        symbol=fill.symbol,
                        side=fill.side,
                        quantity=fill.quantity,
                        price=fill.price,
                        commission=fill.commission,
                    )

                    # Update order state
                    order.update_fill(decision.fill_quantity, decision.fill_price, bar.trade_datetime)

                    # Remove from pending if complete
                    if order.is_complete:
                        self._remove_from_pending(order)
                        logger.debug(
                            "execution.order.completed",
                            order_id=order.order_id,
                            symbol=order.symbol,
                            filled_quantity=order.filled_quantity,
                        )

                # Handle cancellation AFTER fill (IOC/FOK partial fills)
                if decision.should_cancel and not order.is_complete:
                    order.cancel(bar.trade_datetime)
                    self._remove_from_pending(order)
                    logger.warning(
                        "execution.order.cancelled",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        reason=decision.reason or "policy_decision",
                        filled_quantity=order.filled_quantity,
                    )

        return fills

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled, False if not found or already complete

        Example:
            >>> execution.cancel_order(order.order_id)
        """
        order = self._orders.get(order_id)

        if order is None:
            logger.warning(
                "execution.cancel.not_found",
                order_id=order_id,
            )
            return False

        if order.is_complete:
            logger.warning(
                "execution.cancel.already_complete",
                order_id=order_id,
                symbol=order.symbol,
                state=order.state.value,
            )
            return False

        # Cancel order
        order.cancel(datetime.now())

        # Remove from pending
        self._remove_from_pending(order)

        logger.info(
            "execution.order.cancelled_manual",
            order_id=order_id,
            symbol=order.symbol,
            filled_quantity=order.filled_quantity,
        )

        return True

    def get_order(self, order_id: str) -> Order | None:
        """Retrieve order by ID.

        Args:
            order_id: Order identifier

        Returns:
            Order object or None if not found
        """
        return self._orders.get(order_id)

    def get_pending_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all pending/partially filled orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active orders

        Example:
            >>> pending = execution.get_pending_orders(symbol="AAPL")
            >>> print(f"Active AAPL orders: {len(pending)}")
        """
        if symbol is not None:
            order_ids = self._pending_orders_by_symbol.get(symbol, [])
            return [self._orders[oid] for oid in order_ids if self._orders[oid].is_active]
        else:
            # Return all pending orders across all symbols
            return [order for order in self._orders.values() if order.is_active]

    def get_filled_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all filled orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of filled orders

        Example:
            >>> filled = execution.get_filled_orders(symbol="AAPL")
            >>> print(f"Filled AAPL orders: {len(filled)}")
        """
        if symbol is not None:
            return [
                order for order in self._orders.values() if order.symbol == symbol and order.state == OrderState.FILLED
            ]
        else:
            return [order for order in self._orders.values() if order.state == OrderState.FILLED]

    def _remove_from_pending(self, order: Order) -> None:
        """Remove order from pending tracking.

        Args:
            order: Order to remove
        """
        if order.symbol in self._pending_orders_by_symbol:
            order_ids = self._pending_orders_by_symbol[order.symbol]
            if order.order_id in order_ids:
                order_ids.remove(order.order_id)

            # Clean up empty lists
            if not order_ids:
                del self._pending_orders_by_symbol[order.symbol]

    # ==================== Phase 5: Event Handlers ====================

    def on_bar_event(self, event: PriceBarEvent) -> None:
        """
        Handle price bar event - attempt to fill pending orders for this symbol.

        Args:
            event: Price bar event with symbol and bar data
        """
        if event.bar is None:
            return

        # Process bar for this symbol's pending orders
        fills = self.on_bar(event.bar)

        # Publish fill events
        if self._event_bus:
            for fill in fills:
                fill_event = FillEvent(
                    fill_id=fill.fill_id,
                    order_id=fill.order_id,
                    timestamp=fill.timestamp,
                    symbol=fill.symbol,
                    side=fill.side,
                    quantity=fill.quantity,
                    price=fill.price,
                    commission=fill.commission,
                )
                self._event_bus.publish(fill_event)

                logger.debug(
                    "execution.fill_event_published",
                    fill_id=fill.fill_id,
                    symbol=fill.symbol,
                    side=fill.side,
                    quantity=str(fill.quantity),
                )

    def on_order_approved(self, event: OrderApprovedEvent) -> None:
        """
        Handle order approved event - create and submit order.

        Args:
            event: Order approved event from RiskService
        """
        # Create order from approved event
        order = Order(
            symbol=event.symbol,
            side=OrderSide.BUY if event.side == "BUY" else OrderSide.SELL,
            quantity=Decimal(str(event.quantity)),
            order_type=OrderType.MARKET,  # For Phase 5, always use market orders
            created_at=event.ts,
        )

        # Submit order
        try:
            order_id = self.submit_order(order)
            logger.debug(
                "execution.order_from_approved",
                order_id=order_id,
                strategy_id=event.strategy_id,
                symbol=event.symbol,
                side=event.side,
                quantity=event.quantity,
            )
        except ValueError as e:
            logger.error(
                "execution.order_submission_failed",
                strategy_id=event.strategy_id,
                symbol=event.symbol,
                error=str(e),
            )

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], event_bus: EventBus) -> "ExecutionService":
        """
        Factory method to create service from configuration.

        Args:
            config_dict: Execution configuration dictionary
            event_bus: Event bus for communication

        Returns:
            Configured ExecutionService instance
        """
        # For Phase 5, we use a simplified ExecutionConfig
        # Fill policy is specified in backtest config, but ExecutionConfig
        # uses other parameters. We'll create a default config.
        execution_config = ExecutionConfig()

        return cls(config=execution_config, event_bus=event_bus)
