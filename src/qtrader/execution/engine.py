"""Execution engine - processes bars and fills orders."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import structlog

from qtrader.execution.commission import CommissionCalculator
from qtrader.execution.config import ExecutionConfig
from qtrader.execution.fill_policy import FillDecision, FillPolicy
from qtrader.models.bar import Bar
from qtrader.models.order import Fill, OrderBase, OrderState, TimeInForce
from qtrader.models.portfolio import Portfolio

logger = structlog.get_logger(__name__)


class ExecutionEngine:
    """
    Execution engine processes bars and fills orders.

    Event loop:
    1. Receive bar
    2. Update portfolio prices
    3. Evaluate pending orders
    4. Generate fills
    5. Apply fills to portfolio
    6. Update order states
    7. Handle EOD accruals (borrow costs)

    Stage 3 supports:
    - Market orders (fill at next bar open)
    - MOC orders (fill at current bar close with slippage)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        config: Optional[ExecutionConfig] = None,
    ):
        """
        Initialize execution engine.

        Args:
            portfolio: Portfolio to manage
            config: Execution configuration (uses defaults if None)
        """
        self.portfolio = portfolio
        self.config = config or ExecutionConfig()

        # Initialize components
        self.commission_calc = CommissionCalculator(
            per_share=self.config.per_share,
            ticket_min=self.config.ticket_min,
        )
        self.fill_policy = FillPolicy(
            moc_slip_bps=self.config.moc_slip_bps,
            stop_slip_bps=self.config.stop_slip_bps,
            limit_mode=self.config.limit_mode,
            stop_mode=self.config.stop_mode,
        )

        # Order tracking
        self.pending_orders: Dict[str, OrderBase] = {}  # order_id -> Order
        self.filled_orders: Dict[str, OrderBase] = {}  # order_id -> Order
        self.expired_orders: Dict[str, OrderBase] = {}  # order_id -> Order
        self.all_fills: List[Fill] = []

        # Bar tracking (for Market orders that need next bar)
        self.current_bar: Optional[Bar] = None
        self.next_bar: Optional[Bar] = None

        logger.info(
            "execution_engine.initialized",
            config=str(self.config),
        )

    def submit_order(self, order: OrderBase) -> None:
        """
        Submit order to execution engine.

        Args:
            order: Order to submit

        Raises:
            ValueError: If order already exists
        """
        if order.order_id in self.pending_orders:
            raise ValueError(f"Order {order.order_id} already exists")

        # Set order to SUBMITTED state if not already
        if order.state != OrderState.SUBMITTED:
            order = order.with_state(OrderState.SUBMITTED)

        # Set submission_bar_ts for DAY orders (for expiration tracking)
        if order.tif == TimeInForce.DAY and order.submission_bar_ts is None and self.current_bar is not None:
            order = order._replace(submission_bar_ts=self.current_bar.ts)

        self.pending_orders[order.order_id] = order

        logger.info(
            "execution_engine.order_submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            order_type=order.order_type.value,
            tif=order.tif.value,
        )

    def on_bar(self, bar: Bar, next_bar: Optional[Bar] = None, is_close_only: bool = False) -> List[Fill]:
        """
        Process bar and generate fills.

        Args:
            bar: Current bar to process
            next_bar: Next bar (needed for Market orders)
            is_close_only: If True, skip limit/stop evaluation (malformed OHLC bar)

        Returns:
            List of fills generated on this bar
        """
        self.current_bar = bar
        self.next_bar = next_bar

        # Update portfolio with current prices
        self.portfolio.update_prices({bar.symbol: bar.close})

        logger.debug(
            "execution_engine.on_bar",
            symbol=bar.symbol,
            ts=bar.ts.isoformat(),
            close=float(bar.close),
            pending_orders=len(self.pending_orders),
            is_close_only=is_close_only,
        )

        # Evaluate pending orders for this symbol
        fills = []
        orders_to_remove = []
        orders_to_expire = []

        for order_id, order in list(self.pending_orders.items()):
            # Only evaluate orders for this symbol
            if order.symbol != bar.symbol:
                continue

            # Check DAY order expiration
            # DAY orders expire at end of day (after submission bar has passed)
            # For intraday bars: expires when date changes
            # For daily bars: expires after 1 bar (next day)
            if order.tif == TimeInForce.DAY and order.submission_bar_ts is not None:
                # Get submission date and current date
                submission_date = order.submission_bar_ts.date()
                current_date = bar.ts.date()

                # Expire if we're past the submission date
                if current_date > submission_date:
                    orders_to_expire.append(order_id)
                    updated_order = order.with_state(OrderState.EXPIRED)
                    self.expired_orders[order_id] = updated_order

                    logger.info(
                        "execution_engine.order_expired",
                        order_id=order_id,
                        symbol=order.symbol,
                        order_type=order.order_type.value,
                        submission_date=submission_date.isoformat(),
                        current_date=current_date.isoformat(),
                        reason="DAY order expired (new day)",
                    )
                    continue

            # Evaluate order
            decision = self.fill_policy.evaluate_order(order, bar, next_bar, is_close_only)

            if decision.should_fill:
                # Generate fill
                fill = self._generate_fill(order, decision, bar)
                fills.append(fill)

                # Apply fill to portfolio
                self._apply_fill(order, fill, bar)

                # Update order state
                updated_order = order.with_state(OrderState.FILLED)
                self.filled_orders[order_id] = updated_order
                orders_to_remove.append(order_id)

                logger.info(
                    "execution_engine.order_filled",
                    order_id=order_id,
                    fill_id=fill.fill_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    qty=order.qty,
                    fill_price=float(fill.price),
                    fees=float(fill.fees),
                )

        # Remove filled orders from pending
        for order_id in orders_to_remove:
            del self.pending_orders[order_id]

        # Remove expired orders from pending
        for order_id in orders_to_expire:
            del self.pending_orders[order_id]

        return fills

    def _generate_fill(
        self,
        order: OrderBase,
        decision: FillDecision,
        bar: Bar,
    ) -> Fill:
        """
        Generate fill from order and decision.

        Args:
            order: Order being filled
            decision: Fill decision with price
            bar: Current bar

        Returns:
            Fill object
        """
        # Calculate commission
        commission_result = self.commission_calc.calculate(order.qty)

        # Generate unique fill ID
        fill_id = f"fill-{uuid.uuid4().hex[:8]}"

        # Calculate slippage in bps
        if order.order_type.value == "MOC":
            slippage_bps = self.config.moc_slip_bps
        elif order.order_type.value == "STOP":
            slippage_bps = self.config.stop_slip_bps
        else:
            slippage_bps = 0  # Market/Limit orders have no slippage

        # Create fill
        fill = Fill(
            fill_id=fill_id,
            order_id=order.order_id,
            execution_ts=bar.ts,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=decision.fill_price,
            slippage_bps=slippage_bps,
            fees=commission_result.commission,
            participation=0.0,  # Stage 5 feature
            partial_index=0,  # Stage 5 feature (no partials in Stage 3)
        )

        self.all_fills.append(fill)
        return fill

    def _apply_fill(
        self,
        order: OrderBase,
        fill: Fill,
        bar: Bar,
    ) -> None:
        """
        Apply fill to portfolio.

        Args:
            order: Order being filled
            fill: Fill details
            bar: Current bar
        """
        self.portfolio.apply_fill(
            symbol=fill.symbol,
            side=fill.side,
            qty=fill.qty,
            fill_price=fill.price,
            commission=fill.fees,
            ts=fill.execution_ts,
            order_id=fill.order_id,
            fill_id=fill.fill_id,
        )

    def on_end_of_day(self, ts: datetime) -> None:
        """
        Handle end-of-day processing.

        - Accrue borrow costs on short positions

        Args:
            ts: End-of-day timestamp
        """
        # Accrue borrow costs
        self.portfolio.apply_borrow_cost(
            borrow_rate_annual=self.config.borrow_rate_annual,
            ts=ts,
        )

        logger.debug("execution_engine.eod", ts=ts.isoformat())

    def get_pending_orders(self) -> Dict[str, OrderBase]:
        """Get all pending orders."""
        return self.pending_orders.copy()

    def get_filled_orders(self) -> Dict[str, OrderBase]:
        """Get all filled orders."""
        return self.filled_orders.copy()

    def get_all_fills(self) -> List[Fill]:
        """Get all fills generated."""
        return self.all_fills.copy()

    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        return {
            "pending_orders": len(self.pending_orders),
            "filled_orders": len(self.filled_orders),
            "total_fills": len(self.all_fills),
            "portfolio_equity": float(self.portfolio.get_equity()),
            "portfolio_cash": float(self.portfolio.cash.get_balance()),
        }
