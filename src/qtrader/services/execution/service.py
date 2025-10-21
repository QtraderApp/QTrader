"""Execution service implementation.

Simulates realistic order execution for backtesting.
"""

from datetime import datetime

from qtrader.models.bar import Bar
from qtrader.services.execution.commission import CommissionCalculator
from qtrader.services.execution.config import ExecutionConfig
from qtrader.services.execution.fill_policy import FillPolicy
from qtrader.services.execution.models import Fill, Order, OrderState


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

    def __init__(self, config: ExecutionConfig) -> None:
        """Initialize execution service.

        Args:
            config: Execution configuration
        """
        self.config = config
        self.fill_policy = FillPolicy(config)
        self.commission_calculator = CommissionCalculator(config.commission)

        # Order tracking
        self._orders: dict[str, Order] = {}
        self._pending_orders_by_symbol: dict[str, list[str]] = {}

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
            raise ValueError(f"Order ID already exists: {order.order_id}")

        # Mark as submitted
        order.submit(order.created_at)

        # Track order
        self._orders[order.order_id] = order

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
                order = self._orders[order_id]

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

                    # Update order state
                    order.update_fill(decision.fill_quantity, decision.fill_price, bar.trade_datetime)

                    # Remove from pending if complete
                    if order.is_complete:
                        self._remove_from_pending(order)

                # Handle cancellation AFTER fill (IOC/FOK partial fills)
                if decision.should_cancel and not order.is_complete:
                    order.cancel(bar.trade_datetime)
                    self._remove_from_pending(order)

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
            return False

        if order.is_complete:
            return False

        # Cancel order
        order.cancel(datetime.now())

        # Remove from pending
        self._remove_from_pending(order)

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
